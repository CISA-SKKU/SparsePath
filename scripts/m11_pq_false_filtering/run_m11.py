#!/usr/bin/env python
"""M11 — PQ false-filtering measurement driver (single node, no RDMA).

Drives the `pq_search_bindings` harness to answer reviewer R7·D4 / R3·D7:
  * false filtering rate = fraction of would-be-useful candidates the
    d_worst*(1+alpha) PQ filter drops before full-vector fetch, and
  * its recall impact = recall(no filter oracle) - recall(filter on),
  swept over alpha, on every dataset incl. LAION (512-d).

For each dataset the harness computes, per candidate seen while
converged & beam-full: the PQ approximate distance AND (for dropped ones)
the exact squared-L2 distance. A drop is a *false* filter iff exact <= worst
(it would have entered the beam). recall(ON) vs recall(OFF) is the exact
recall cost of false filtering.

Nothing here fetches over the network. It loads the real global Vamana graph +
the real PQ codebook/codes the system ships with (so the filter measured here
is the filter the paper describes; only the transport is dropped).

Usage:
    python run_m11.py --config datasets.yaml [--dataset laion100M] [--out results/]

See datasets.example.yaml for the per-dataset path schema.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

# pq_search_bindings.so is written to <repo>/lib by the build.
# this file lives at <repo>/scripts/m11_pq_false_filtering/run_m11.py
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "lib"))

try:
    import pq_search_bindings as pqsb
except ImportError as e:
    sys.exit(f"cannot import pq_search_bindings (build it first: docker exec danns ./build_all.sh): {e}")

try:
    import yaml
except ImportError:
    yaml = None

_HARNESS = {
    "float32": pqsb.PQSearchHarnessFloat,
    "float":   pqsb.PQSearchHarnessFloat,
    "uint8":   pqsb.PQSearchHarnessUInt8,
    "int8":    pqsb.PQSearchHarnessInt8,
}
_NP_DTYPE = {"float32": np.float32, "float": np.float32, "uint8": np.uint8, "int8": np.int8}


# ---------------------------------------------------------------------------
# file readers (DiskANN bin conventions)
# ---------------------------------------------------------------------------
def load_bin(path, dtype):
    """(npts, dim) queries. Supports .npy and DiskANN bin
    ([int32 npts][int32 dim][ data npts*dim ])."""
    if path.endswith(".npy"):
        return np.load(path).astype(dtype)
    with open(path, "rb") as f:
        npts, dim = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=dtype, count=int(npts) * int(dim))
    return data.reshape(int(npts), int(dim))


def load_gt(path):
    """Ground-truth neighbor ids -> (npts, K) uint32.
    Supports .npy (np.save of an (npts,K) id array) and DiskANN bin
    ([int32 npts][int32 K][u32 ids][f32 dists])."""
    if path.endswith(".npy"):
        ids = np.load(path)
        return ids.astype(np.uint32)
    with open(path, "rb") as f:
        npts, K = np.fromfile(f, dtype=np.int32, count=2)
        ids = np.fromfile(f, dtype=np.uint32, count=int(npts) * int(K)).reshape(int(npts), int(K))
    return ids


# ---------------------------------------------------------------------------
# per-dataset run
# ---------------------------------------------------------------------------
def run_dataset(name, cfg, out_dir):
    dtype = cfg["dtype"]
    npd = _NP_DTYPE[dtype]
    Harness = _HARNESS[dtype]
    dim = int(cfg["dim"])
    K = int(cfg.get("K", 10))
    Ls = int(cfg.get("L_search", 100))
    conv_window = int(cfg.get("conv_window", 3))
    conv_thr = float(cfg.get("conv_threshold_pct", 1.0))
    alphas = [float(a) for a in cfg.get("alphas", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])]
    id_offset = int(cfg.get("id_offset", 0))          # mapped_id = (tag-1) + id_offset
    max_cap = int(cfg["max_capacity"])
    max_points = int(cfg.get("max_points", max_cap))
    n_query = cfg.get("n_query", None)                # sub-sample queries for speed
    max_drops = int(cfg.get("max_drops", 2_000_000))

    print(f"\n=== {name} (dim={dim}, dtype={dtype}) ===", flush=True)
    h = Harness(dim=dim, max_points=max_points, R=int(cfg.get("R", 64)),
                L_build=int(cfg.get("L_build", Ls)), num_threads=int(cfg.get("num_threads", 0)))
    t0 = time.time()
    h.load_index(cfg["graph_file"], cfg["base_file"], cfg["tags_file"], search_L=Ls)
    h.load_pq(cfg["pq_codebook"], cfg["pq_codes"], max_capacity=max_cap, alpha=alphas[-1])
    print(f"  loaded: num_points={h.num_points()}  M={h.M()}  ({time.time()-t0:.1f}s)", flush=True)

    queries = load_bin(cfg["query_file"], npd)
    gt = load_gt(cfg["gt_file"])
    if n_query:
        queries = queries[:int(n_query)]
        gt = gt[:int(n_query)]
    print(f"  queries={queries.shape}  gt={gt.shape}", flush=True)

    # ---- L-independent setup: partition configs + nearest-centroid coordinators ----
    Nq = queries.shape[0]
    L_sweep = [int(x) for x in cfg.get("L_search_sweep", [Ls])]
    L_ref = Ls if Ls in L_sweep else L_sweep[0]     # L at which to record the drop CDF

    def compute_centroids(pid, M):
        active_lo = int(cfg.get("active_gid_lo", 0))     # iteration150 active gids start at 150M
        active_n = int(cfg.get("active_n", max_points))
        stride = int(cfg.get("centroid_sample_stride", 97))
        mm = np.memmap(cfg["base_file"], dtype=npd, mode="r", offset=8).reshape(-1, dim)
        idx = np.arange(active_lo, active_lo + active_n, stride, dtype=np.int64)
        sub = np.asarray(mm[idx], dtype=np.float64)
        parts = pid[idx]
        C = np.zeros((M, dim), dtype=np.float64)
        for p in range(M):
            msk = parts == p
            if msk.any():
                C[p] = sub[msk].mean(axis=0)
        return C.astype(np.float32)

    def coord_nc(centroids):
        # REAL routing: query -> nearest partition centroid (content-routed).
        q = queries.astype(np.float32)
        d = (q * q).sum(1)[:, None] - 2.0 * (q @ centroids.T) + (centroids * centroids).sum(1)[None, :]
        return d.argmin(axis=1).astype(np.uint32)

    configs = []   # (cfg_name, pid, coord) — nearest-centroid routing (the real model)
    for ps in cfg.get("partitions", []):
        pid = np.load(ps["path"]).astype(np.uint32)
        M = int(ps["M"])
        coord = coord_nc(compute_centroids(pid, M))
        configs.append((ps["name"], pid, coord))
        print(f"  config[{ps['name']}] M={M} routed-hist={np.bincount(coord, minlength=M)[:M].tolist()}", flush=True)

    def one(mode, L, a, part_of, coord, recall_off_L, rec):
        h.set_alpha(a)
        r = h.search_batch(queries, K=K, L_search=L, conv_window=conv_window,
                           conv_threshold_pct=conv_thr, filter_on=True,
                           record_drops=rec, max_drops=max_drops,
                           partition_of=part_of, coordinator=coord)
        recall_on = _recall_mapped(r["result_tags"], gt, K, id_offset)
        elig = int(r["n_filter_eligible"]); drop = int(r["n_pq_filtered"]); false_ = int(r["n_false_local"])
        ff_e = (false_ / elig) if elig else 0.0
        row = dict(dataset=name, mode=mode, dim=dim, L=L, alpha=a,
                   recall_on=round(recall_on, 4), recall_off=round(recall_off_L, 4),
                   recall_delta=round(recall_off_L - recall_on, 4),
                   n_filter_eligible=elig, n_pq_filtered=drop, n_false_local=false_,
                   false_filter_rate_of_eligible=round(ff_e, 6),
                   false_filter_rate_of_dropped=round((false_ / drop) if drop else 0.0, 6),
                   drop_rate=round((drop / elig) if elig else 0.0, 4),
                   avg_hops=round(int(r["sum_hops"]) / max(1, int(r["n_queries"])), 1))
        print(f"  [L={L}][{mode}] a={a:.2f} ON={recall_on:.4f} Δ={recall_off_L - recall_on:+.4f} "
              f"elig={elig} false={false_} ({ff_e * 100:.3f}%)", flush=True)
        return row, r

    # ---- gid-mapping sanity (alpha=0, filter-all) ----
    probe = h.search_batch(queries[: min(1000, Nq)], K=K, L_search=min(100, max(L_sweep)),
                           conv_window=conv_window, conv_threshold_pct=conv_thr,
                           filter_on=True, record_drops=True, max_drops=200000)
    h.set_alpha(0.0)
    corr = _drop_corr(probe)
    print(f"  [sanity] pq-vs-exact corr = {corr:.3f} ({'OK' if corr > 0.5 else 'SUSPECT gid mapping!'})", flush=True)

    # ---- L sweep x {filter_all, nc per config} x alpha ----
    all_rows, drop_samples = [], {}
    for L in L_sweep:
        off = h.search_batch(queries, K=K, L_search=L, conv_window=conv_window,
                             conv_threshold_pct=conv_thr, filter_on=False)
        recall_off_L = _recall_mapped(off["result_tags"], gt, K, id_offset)
        print(f"  --- L={L}: recall(OFF)={recall_off_L:.4f} ---", flush=True)
        for a in alphas:
            row, r = one("filter_all", L, a, None, None, recall_off_L, rec=(L == L_ref))
            all_rows.append(row)
            if L == L_ref and len(r["drop_pqdist"]):
                w = np.asarray(r["drop_worst"], dtype=np.float64); w[w == 0] = np.nan
                drop_samples[a] = dict(pq_over_worst=np.asarray(r["drop_pqdist"], dtype=np.float64) / w,
                                       exact_over_worst=np.asarray(r["drop_exact"], dtype=np.float64) / w)
        for (cname, pid, coord) in configs:
            for a in alphas:
                row, _ = one(f"nc_{cname}", L, a, pid, coord, recall_off_L, rec=False)
                all_rows.append(row)

    # ---- persist ----
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_rows.json"), "w") as f:
        json.dump({"sanity_corr": corr, "rows": all_rows}, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"{name}_drop_samples.npz"),
                        **{f"a{int(round(a*100)):03d}_{kk}": vv
                           for a, d in drop_samples.items() for kk, vv in d.items()})
    _try_plot(name, [r for r in all_rows if r["mode"] == "filter_all" and r["L"] == L_ref], drop_samples, out_dir)
    return all_rows


def _recall_mapped(found_tags, gt, K, id_offset):
    ft = np.asarray(found_tags)
    mapped = ft.astype(np.int64) - 1 + id_offset
    mapped[ft == np.iinfo(np.uint32).max] = -1
    nq = min(mapped.shape[0], gt.shape[0]); hits = 0
    for i in range(nq):
        f = set(int(x) for x in mapped[i, :K] if x >= 0)
        g = set(int(x) for x in gt[i, :K])
        hits += len(f & g)
    return hits / (nq * K)


def _drop_corr(probe):
    pq = np.asarray(probe["drop_pqdist"], dtype=np.float64)
    ex = np.asarray(probe["drop_exact"], dtype=np.float64)
    if len(pq) < 3:
        return float("nan")
    return float(np.corrcoef(pq, ex)[0, 1])


def _try_plot(name, rows, drop_samples, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = [r["alpha"] for r in rows]
    ax[0].plot(a, [r["false_filter_rate_of_eligible"] * 100 for r in rows], "o-", label="false-filter %")
    ax[0].plot(a, [r["recall_delta"] for r in rows], "s--", label="recall Δ")
    ax[0].axvline(0.15, color="gray", ls=":"); ax[0].set_xlabel("alpha"); ax[0].legend()
    ax[0].set_title(f"{name}: false-filter & recall Δ vs α")
    for a_, d in sorted(drop_samples.items()):
        x = np.sort(d["pq_over_worst"][np.isfinite(d["pq_over_worst"])])
        if len(x):
            ax[1].plot(x, np.linspace(0, 1, len(x)), label=f"α={a_:.2f}")
    ax[1].axvline(1.15, color="gray", ls=":", label="1+0.15")
    ax[1].set_xlabel("pq_dist / d_worst  (dropped candidates)"); ax[1].set_ylabel("CDF")
    ax[1].set_xlim(0.9, 2.0); ax[1].legend(fontsize=7); ax[1].set_title(f"{name}: PQ error vs filter boundary")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{ts}_{name}_m11_false_filtering.png")
    fig.savefig(p, dpi=130); print(f"  plot -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="datasets YAML")
    ap.add_argument("--dataset", default=None, help="run only this dataset key")
    ap.add_argument("--out", default="results_m11")
    args = ap.parse_args()
    if yaml is None:
        sys.exit("pyyaml not available")
    with open(args.config) as f:
        conf = yaml.safe_load(f)
    datasets = conf["datasets"]
    keys = [args.dataset] if args.dataset else list(datasets.keys())
    all_rows = []
    for k in keys:
        all_rows += run_dataset(k, datasets[k], args.out)
    # combined markdown table
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "M11_table.md"), "w") as f:
        f.write("| dataset | mode | L | dim | α | false-filter% (elig) | recall(ON) | recall(OFF) | recall Δ | drop_rate |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            f.write(f"| {r['dataset']} | {r.get('mode','-')} | {r.get('L','-')} | {r['dim']} | {r['alpha']:.2f} | "
                    f"{r['false_filter_rate_of_eligible']*100:.3f} | {r['recall_on']:.4f} | "
                    f"{r['recall_off']:.4f} | {r['recall_delta']:+.4f} | {r['drop_rate']:.3f} |\n")
    print(f"\nwrote {args.out}/M11_table.md", flush=True)


if __name__ == "__main__":
    main()
