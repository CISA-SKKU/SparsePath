#!/usr/bin/env python
"""M10 CAP — partition-aware REMOTE-fetch-round reduction (the honest QPS metric).

The plain fetch-round count treats every round the same, but only rounds that
issue a REMOTE fetch pay an RDMA round-trip; local fetches are ~free. So the
QPS-relevant quantity is the number of *remote* fetch rounds, and how much
batching cuts it — which is placement-dependent (balanced ~75% remote edges,
centroid ~25%, lits ~7.5%).

This driver measures, per placement (nearest-centroid query routing, mirroring
M11): remote_fetch_rounds for best-first (B=0) vs CAP (B=8), at the operating
L's. `remote_round_reduction = remote_rounds(B0) / remote_rounds(B8)` is the
honest, layout-aware version of the fetch-round speedup.

A candidate is REMOTE iff its partition != the query's coordinator partition
(nearest-centroid). partition_of passed to the harness is GID-indexed:
  - vamana partition_ids.npy  -> used directly (gid-indexed).
  - lits .labels (tag-indexed, entry0=start) -> labels[1:] (since gid = tag-1,
    partition_of[gid] = labels[gid+1]).

Usage:
    python run_m10_remote.py --config datasets_remote.yaml [--dataset deep100M] [--out results/]
"""
import argparse
import json
import os
import sys
import time
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "lib"))
try:
    import pq_search_bindings as pqsb
except ImportError as e:
    sys.exit(f"cannot import pq_search_bindings (build it): {e}")
import yaml

_HARNESS = {"float32": pqsb.PQSearchHarnessFloat, "float": pqsb.PQSearchHarnessFloat,
            "uint8": pqsb.PQSearchHarnessUInt8, "int8": pqsb.PQSearchHarnessInt8}
_NP = {"float32": np.float32, "float": np.float32, "uint8": np.uint8, "int8": np.int8}


def load_bin(path, dtype):
    if path.endswith(".npy"):
        return np.load(path).astype(dtype)
    with open(path, "rb") as f:
        npts, dim = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=dtype, count=int(npts) * int(dim))
    return data.reshape(int(npts), int(dim))


def load_gt(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.uint32)
    with open(path, "rb") as f:
        npts, K = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.uint32, count=int(npts) * int(K)).reshape(int(npts), int(K))


def recall_mapped(found_tags, gt, K, id_offset):
    ft = np.asarray(found_tags)
    mapped = ft.astype(np.int64) - 1 + id_offset
    mapped[ft == np.iinfo(np.uint32).max] = -1
    nq = min(mapped.shape[0], gt.shape[0]); hits = 0
    for i in range(nq):
        f = set(int(x) for x in mapped[i, :K] if x >= 0)
        hits += len(f & set(int(x) for x in gt[i, :K]))
    return hits / (nq * K)


def load_partition_gid(spec):
    """Return gid-indexed partition array (uint32)."""
    if spec.get("kind", "partition_ids") == "labels":
        # lits .labels: [n u32][K u32] header, tag-indexed. gid = tag-1 -> labels[1:].
        arr = np.fromfile(spec["file"], dtype=np.uint32, offset=8)[1:]
    else:
        arr = np.load(spec["file"]).astype(np.uint32)
    return np.ascontiguousarray(arr, dtype=np.uint32)


def run_dataset(name, cfg, out_dir):
    npd = _NP[cfg["dtype"]]; Harness = _HARNESS[cfg["dtype"]]
    dim = int(cfg["dim"]); max_points = int(cfg.get("max_points", 100000000))
    K = int(cfg.get("K", 10)); conv_window = int(cfg.get("conv_window", 3))
    conv_thr = float(cfg.get("conv_threshold_pct", 1.0))
    id_offset = int(cfg.get("id_offset", 0))
    Bs = [int(b) for b in cfg.get("batch_sizes", [0, 8])]
    Ls = [int(x) for x in cfg.get("L_search", [100, 200])]
    n_query = cfg.get("n_query", None)
    active_lo = int(cfg.get("active_gid_lo", 0))
    active_n = int(cfg.get("active_n", max_points))
    stride = int(cfg.get("centroid_sample_stride", 97))

    queries = load_bin(cfg["query_file"], npd); gt = load_gt(cfg["gt_file"])
    if n_query:
        queries = queries[:int(n_query)]; gt = gt[:int(n_query)]
    qf = queries.astype(np.float32)

    def compute_centroids(pid, M):
        mm = np.memmap(cfg["base_file"], dtype=npd, mode="r", offset=8).reshape(-1, dim)
        idx = np.arange(active_lo, active_lo + active_n, stride, dtype=np.int64)
        sub = np.asarray(mm[idx], dtype=np.float64); parts = pid[idx]
        C = np.zeros((M, dim), dtype=np.float64)
        for p in range(M):
            msk = parts == p
            if msk.any():
                C[p] = sub[msk].mean(axis=0)
        return C.astype(np.float32)

    def coord_nc(C):
        d = (qf * qf).sum(1)[:, None] - 2.0 * (qf @ C.T) + (C * C).sum(1)[None, :]
        return d.argmin(axis=1).astype(np.uint32)

    rows = []
    print(f"\n=== {name} (dim={dim}, {cfg['dtype']}) ===", flush=True)
    for g in cfg["graphs"]:
        h = Harness(dim=dim, max_points=max_points)
        t0 = time.time()
        h.load_index(g["graph_file"], cfg["base_file"], g["tags_file"], search_L=max(Ls))
        print(f"[{name}/{g['name']}] loaded ({time.time()-t0:.1f}s)", flush=True)
        for pl in g["placements"]:
            pid = load_partition_gid(pl); M = int(pl["M"])
            coord = coord_nc(compute_centroids(pid, M))
            print(f"  placement {pl['name']} (M={M}) routed-hist={np.bincount(coord, minlength=M).tolist()}", flush=True)
            base = {}
            for L in Ls:
                for B in Bs:
                    r = h.search_batch(queries, K=K, L_search=L, conv_window=conv_window,
                                       conv_threshold_pct=conv_thr, filter_on=False, batch_size=B,
                                       partition_of=pid, coordinator=coord)
                    nq = max(1, int(r["n_queries"]))
                    row = dict(dataset=name, graph=g["name"], placement=pl["name"], L=L, B=B,
                               recall=round(recall_mapped(r["result_tags"], gt, K, id_offset), 4),
                               avg_rounds=round(int(r["sum_fetch_rounds"]) / nq, 2),
                               avg_remote_rounds=round(int(r["sum_remote_fetch_rounds"]) / nq, 2))
                    if B == 0:
                        base[L] = row
                    else:
                        b0 = base[L]
                        row["remote_round_reduction"] = round(
                            b0["avg_remote_rounds"] / row["avg_remote_rounds"], 3) if row["avg_remote_rounds"] else 0.0
                        row["all_round_reduction"] = round(
                            b0["avg_rounds"] / row["avg_rounds"], 3) if row["avg_rounds"] else 0.0
                    rows.append(row)
                    tag = "B0(best-first)" if B == 0 else f"B{B}(CAP)"
                    extra = ""
                    if B != 0:
                        extra = f"  remote-round↓={row['remote_round_reduction']}x (all-round↓={row['all_round_reduction']}x)"
                    print(f"    [{pl['name']:<9}] L={L} {tag:<15} recall={row['recall']:.4f} "
                          f"rounds={row['avg_rounds']:.1f} remote_rounds={row['avg_remote_rounds']:.1f}{extra}", flush=True)
        del h
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_remote_rounds.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results_m10_remote")
    args = ap.parse_args()
    conf = yaml.safe_load(open(args.config))
    keys = [args.dataset] if args.dataset else list(conf["datasets"].keys())
    allrows = []
    for k in keys:
        allrows += run_dataset(k, conf["datasets"][k], args.out)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "M10_remote_rounds_table.md"), "w") as f:
        f.write("| dataset | graph | placement | L | B | recall | rounds | remote_rounds | remote-round↓ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in allrows:
            rr = r.get("remote_round_reduction", "-")
            f.write(f"| {r['dataset']} | {r['graph']} | {r['placement']} | {r['L']} | {r['B']} | "
                    f"{r['recall']:.4f} | {r['avg_rounds']:.1f} | {r['avg_remote_rounds']:.1f} | {rr} |\n")
    print(f"\nwrote {args.out}/M10_remote_rounds_table.md", flush=True)


if __name__ == "__main__":
    main()
