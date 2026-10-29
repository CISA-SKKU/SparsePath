#!/usr/bin/env python
"""M10 — Convergence-Aware Prefetching semantics driver (single node, no RDMA).

Answers reviewer R7·D3: after convergence the search batches B unvisited
candidates, which changes the order nodes are visited vs. standard best-first.
This driver measures, on the real Vamana graph:

  1. recall preservation  — recall(B>0) vs recall(B=0 best-first oracle) across
     B and the convergence threshold. If Δ≈0, batching does not hurt recall.
  2. threshold selection   — how recall and the fetch-round count move with
     `conv_threshold_pct` per dataset (the recall/QPS trade-off + selection guide).
  3. QPS proxy             — avg fetch rounds per query. Best-first issues one
     fetch per hop; CAP issues one per batch round, so round_reduction =
     rounds(B=0)/rounds(B) is the RDMA-round saving that becomes QPS on a cluster.

It drives the SAME converge-gated beam search the paper ships
(`pq_search_bindings`, which mirrors `search/src/beam_search.cpp`
ColocationBeamSearch); only the transport (RDMA/colocation) is dropped. The
batch path here is the harness's `batch_size`>0 branch, a line-for-line mirror
of beam_search.cpp's batch expansion + >5% fallback. The PQ filter is OFF
(filter_on=False) so this isolates the *batching* effect (PQ filtering is M11).

Why single-node is faithful: batching only changes WHICH candidates are expanded
together (recall), never how the fetch is transported (performance). Recall is
layout-independent; every vector is local here, so recall(B) is exact.

Usage:
    python run_m10.py --config datasets.yaml [--dataset deep100M] [--out results/]

See datasets.example.yaml for the per-dataset path schema (same shape as M11's,
minus the PQ fields — M10 needs no PQ codebook).
"""
import argparse
import json
import os
import sys
import time
import numpy as np

# pq_search_bindings.so is written to <repo>/lib by the build.
# this file lives at <repo>/scripts/m10_cap_semantics/run_m10.py
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
# file readers (DiskANN bin conventions) — identical to run_m11.py
# ---------------------------------------------------------------------------
def load_bin(path, dtype):
    """(npts, dim). Supports .npy and DiskANN bin ([int32 npts][int32 dim][data])."""
    if path.endswith(".npy"):
        return np.load(path).astype(dtype)
    with open(path, "rb") as f:
        npts, dim = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=dtype, count=int(npts) * int(dim))
    return data.reshape(int(npts), int(dim))


def load_gt(path):
    """Ground-truth neighbor ids -> (npts, K) uint32. Supports .npy and DiskANN bin.
    Returns None if path is empty/absent (conv-stats-only mode: recall reported as nan)."""
    if not path:
        return None
    if path.endswith(".npy"):
        return np.load(path).astype(np.uint32)
    with open(path, "rb") as f:
        npts, K = np.fromfile(f, dtype=np.int32, count=2)
        ids = np.fromfile(f, dtype=np.uint32, count=int(npts) * int(K)).reshape(int(npts), int(K))
    return ids


def _recall_mapped(found_tags, gt, K, id_offset):
    """recall@K with tag->global-id mapping (mapped_id = tag-1+id_offset)."""
    ft = np.asarray(found_tags)
    mapped = ft.astype(np.int64) - 1 + id_offset
    mapped[ft == np.iinfo(np.uint32).max] = -1
    nq = min(mapped.shape[0], gt.shape[0]); hits = 0
    for i in range(nq):
        f = set(int(x) for x in mapped[i, :K] if x >= 0)
        g = set(int(x) for x in gt[i, :K])
        hits += len(f & g)
    return hits / (nq * K)


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
    id_offset = int(cfg.get("id_offset", 0))
    # M10 loads no PQ, so max_capacity is irrelevant; only the ctor's max_points
    # (= graph header start / frozen loc) matters. Accept either key.
    max_points = int(cfg.get("max_points", cfg.get("max_capacity")))
    n_query = cfg.get("n_query", None)

    # sweep axes (paper default: B=8, conv_threshold_pct=1.0 in code / see M10 gate)
    batch_sizes = [int(b) for b in cfg.get("batch_sizes", [0, 2, 4, 8, 16])]  # 0 = best-first baseline
    conv_thrs = [float(t) for t in cfg.get("conv_threshold_pcts", [1.0])]
    L_sweep = [int(x) for x in cfg.get("L_search_sweep", [Ls])]
    # threshold slice runs only at these L's (operating region); default = all L.
    thr_slice_L = set(int(x) for x in cfg.get("thr_slice_L", L_sweep))

    print(f"\n=== {name} (dim={dim}, dtype={dtype}) ===", flush=True)
    h = Harness(dim=dim, max_points=max_points, R=int(cfg.get("R", 64)),
                L_build=int(cfg.get("L_build", Ls)), num_threads=int(cfg.get("num_threads", 0)))
    t0 = time.time()
    h.load_index(cfg["graph_file"], cfg["base_file"], cfg["tags_file"], search_L=Ls)
    print(f"  loaded: num_points={h.num_points()}  ({time.time()-t0:.1f}s)", flush=True)
    # M10 isolates batching: PQ filter OFF, so no codebook needed.

    queries = load_bin(cfg["query_file"], npd)
    gt = load_gt(cfg.get("gt_file"))
    if n_query:
        queries = queries[:int(n_query)]
        if gt is not None:
            gt = gt[:int(n_query)]
    print(f"  queries={queries.shape}  gt={gt.shape if gt is not None else None}", flush=True)

    def one(L, thr, B):
        # filter_on=False -> pure batching effect (no PQ filter). B=0 -> best-first.
        r = h.search_batch(queries, K=K, L_search=L, conv_window=conv_window,
                           conv_threshold_pct=thr, filter_on=False,
                           record_drops=False, batch_size=B)
        nq = max(1, int(r["n_queries"]))
        recall = _recall_mapped(r["result_tags"], gt, K, id_offset) if gt is not None else float("nan")
        return dict(
            dataset=name, dim=dim, L=L, conv_threshold_pct=thr, batch_size=B,
            recall=round(recall, 4),
            avg_hops=round(int(r["sum_hops"]) / nq, 2),
            avg_fetch_rounds=round(int(r["sum_fetch_rounds"]) / nq, 2),
            avg_conv_hop=round(int(r["sum_conv_hop"]) / max(1, int(r["n_converged"])), 2),
            frac_converged=round(int(r["n_converged"]) / nq, 4),
            n_batch_rounds=int(r["n_batch_rounds"]),
            n_fallbacks=int(r["n_fallbacks"]),
        )

    # Two-slice design (not a full B×thr cross-product):
    #   recall slice    — sweep B at the reference threshold  (R7·D3 part 1: does batching hurt recall)
    #   threshold slice — sweep threshold at the reference B   (R7·D3 part 2: threshold selection)
    thr_ref = float(cfg.get("thr_ref", 1.0))
    B_ref = int(cfg.get("B_ref", 8))
    rows = []

    def emit(L, thr, B, base_recall, base_rounds, slc):
        row = one(L, thr, B)
        row["slice"] = slc
        row["recall_base"] = base_recall
        row["recall_delta"] = round(base_recall - row["recall"], 4)
        row["round_reduction"] = round(base_rounds / row["avg_fetch_rounds"], 3) \
            if row["avg_fetch_rounds"] else 0.0
        rows.append(row)
        print(f"      [{slc:<9}] thr={thr:<4} B={B:<3d} recall={row['recall']:.4f} "
              f"Δ={row['recall_delta']:+.4f} rounds={row['avg_fetch_rounds']:.1f} "
              f"({row['round_reduction']:.2f}x fewer) conv_hop={row['avg_conv_hop']:.0f} "
              f"fallbacks={row['n_fallbacks']}", flush=True)

    for L in L_sweep:
        # Baseline = best-first (B=0), threshold-INDEPENDENT -> measure once per L.
        base = one(L, thr_ref, 0)
        base_rounds = base["avg_fetch_rounds"]
        base_recall = base["recall"]
        rows.append(dict(base, slice="baseline", recall_base=base_recall,
                         recall_delta=0.0, round_reduction=1.0))
        print(f"  --- L={L}: baseline B=0 recall={base_recall:.4f} rounds={base_rounds:.1f} ---",
              flush=True)
        # recall slice: sweep B at thr_ref
        for B in batch_sizes:
            if B == 0:
                continue
            emit(L, thr_ref, B, base_recall, base_rounds, "recall")
        # threshold slice: sweep threshold at B_ref, only at operating-region L's
        # (skip thr_ref — already measured in the recall slice at B_ref).
        if L in thr_slice_L:
            for thr in conv_thrs:
                if abs(thr - thr_ref) < 1e-9:
                    continue
                emit(L, thr, B_ref, base_recall, base_rounds, "threshold")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_rows.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    _try_plot(name, rows, out_dir)
    return rows


def _try_plot(name, rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Use the first L present; plot recall Δ and round-reduction vs B, one line per threshold.
    Ls = sorted({r["L"] for r in rows})
    L0 = Ls[0]
    thrs = sorted({r["conv_threshold_pct"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for thr in thrs:
        sub = sorted([r for r in rows if r["L"] == L0 and r["conv_threshold_pct"] == thr
                      and r["batch_size"] > 0], key=lambda r: r["batch_size"])
        if not sub:
            continue
        B = [r["batch_size"] for r in sub]
        ax[0].plot(B, [r["recall_delta"] for r in sub], "o-", label=f"thr={thr}")
        ax[1].plot(B, [r["round_reduction"] for r in sub], "s-", label=f"thr={thr}")
    ax[0].axhline(0.0, color="gray", ls=":"); ax[0].set_xlabel("batch size B")
    ax[0].set_ylabel("recall Δ (best-first − CAP)"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"{name}: recall preservation vs B (L={L0})")
    ax[1].set_xlabel("batch size B"); ax[1].set_ylabel("fetch-round reduction (×)")
    ax[1].legend(fontsize=8); ax[1].set_title(f"{name}: QPS proxy — fewer RDMA rounds")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{ts}_{name}_m10_cap.png")
    fig.savefig(p, dpi=130); print(f"  plot -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="datasets YAML")
    ap.add_argument("--dataset", default=None, help="run only this dataset key")
    ap.add_argument("--out", default="results_m10")
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
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "M10_table.md"), "w") as f:
        f.write("| dataset | L | conv_thr% | B | recall | recall(base) | Δ | avg_rounds | round_reduc | conv_hop | fallbacks |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            f.write(f"| {r['dataset']} | {r['L']} | {r['conv_threshold_pct']:.2f} | {r['batch_size']} | "
                    f"{r['recall']:.4f} | {r['recall_base']:.4f} | {r['recall_delta']:+.4f} | "
                    f"{r['avg_fetch_rounds']:.1f} | {r['round_reduction']:.2f} | "
                    f"{r['avg_conv_hop']:.1f} | {r['n_fallbacks']} |\n")
    print(f"\nwrote {args.out}/M10_table.md", flush=True)


if __name__ == "__main__":
    main()
