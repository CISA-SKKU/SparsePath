#!/usr/bin/env python
"""Q1 (M08 / R5.W2 ask #2) — SparsePath memory-recall Pareto under quantized vectors.

For a FIXED SparsePath graph, run beam search ranking the beam by:
  * exact (raw fp32/int8) distance -> recall upper bound, full per-vector memory
  * PQ-estimated distance          -> "SparsePath under quantized vectors" (M bytes/vec)
and report recall@10 vs GT at each L, plus per-vector memory. This is the honest
memory-vs-recall Pareto the reviewer (R5.W2 ask #2) asked for; the throughput axis
is argued separately (bytes/vec accounting, see design doc).

Filter is OFF (filter_on=False): Q1 isolates quantized *search* recall, not the
cross-partition filter (that is M11). Reuses the M11 harness + configs; the only
new capability is search_batch(search_dist="pq"), added additively.

Usage:
    python run_q1.py --config ../m11_pq_false_filtering/datasets_deep100M.yaml \
                     --dataset deep100M --out results_q1_deep100M
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
    import single_engine as pqsb          # unified single-node engine (preferred)
except ImportError:
    try:
        import pq_search_bindings as pqsb  # legacy module (same classes)
    except ImportError as e:
        sys.exit(f"cannot import single_engine/pq_search_bindings "
                 f"(build first: ./build_all.sh): {e}")
try:
    import yaml
except ImportError:
    yaml = None

_HARNESS = {"float32": pqsb.PQSearchHarnessFloat, "float": pqsb.PQSearchHarnessFloat,
            "uint8": pqsb.PQSearchHarnessUInt8, "int8": pqsb.PQSearchHarnessInt8}
_NP = {"float32": np.float32, "float": np.float32, "uint8": np.uint8, "int8": np.int8}
_BYTES = {"float32": 4, "float": 4, "uint8": 1, "int8": 1}


def load_bin(path, dtype):
    if path.endswith(".npy"):
        return np.load(path).astype(dtype)
    with open(path, "rb") as f:
        n, d = np.fromfile(f, np.int32, 2)
        data = np.fromfile(f, dtype, int(n) * int(d))
    return data.reshape(int(n), int(d))


def load_gt(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.uint32)
    with open(path, "rb") as f:
        n, K = np.fromfile(f, np.int32, 2)
        ids = np.fromfile(f, np.uint32, int(n) * int(K)).reshape(int(n), int(K))
    return ids


def recall_at_k(found_tags, gt, K, id_offset):
    ft = np.asarray(found_tags)
    mapped = ft.astype(np.int64) - 1 + id_offset
    mapped[ft == np.iinfo(np.uint32).max] = -1
    nq = min(mapped.shape[0], gt.shape[0])
    hits = 0
    for i in range(nq):
        f = set(int(x) for x in mapped[i, :K] if x >= 0)
        g = set(int(x) for x in gt[i, :K])
        hits += len(f & g)
    return hits / (nq * K)


def load_partition(cfg, part_name, queries):
    """Load partition_ids + nearest-centroid coordinator (mirrors run_m11).

    Returns (partition_of, coordinator) or (None, None) when part_name is None.
    Enables local/remote fetch accounting in search_batch (bytes/query axis).
    """
    if not part_name:
        return None, None
    entry = next((p for p in cfg.get("partitions", []) if p["name"] == part_name), None)
    if entry is None:
        sys.exit(f"partition '{part_name}' not in config partitions list")
    pid = np.load(entry["path"]).astype(np.uint32)
    M = int(entry["M"])
    # Partition centroids: prefer the build-time artifact (centroids.fbin next to
    # partition_ids.npy, [M x dim] float32). Fallback: recompute from a gid-strided
    # base subsample (clamped to base size — partition_ids covers the FULL pool,
    # e.g. 300M incl. future inserts, while the base file holds only active gids).
    cfile = os.path.join(os.path.dirname(entry["path"]), "centroids.fbin")
    if os.path.exists(cfile):
        with open(cfile, "rb") as f:
            cn, cd = np.fromfile(f, np.int32, 2)
            C = np.fromfile(f, np.float32, int(cn) * int(cd)).reshape(int(cn), int(cd))
        if int(cn) != M:
            sys.exit(f"centroids.fbin has {cn} rows, expected M={M}: {cfile}")
        print(f"  partition[{part_name}] centroids from artifact: {cfile}", flush=True)
    else:
        cfg_base = cfg["base_file"]
        with open(cfg_base, "rb") as f:
            n, d = np.fromfile(f, np.int32, 2)
        limit = min(pid.shape[0], int(n))
        sub_idx = np.arange(0, limit, max(1, limit // 2000000))
        mm = np.memmap(cfg_base, dtype=_NP[cfg["dtype"]], mode="r", offset=8,
                       shape=(int(n), int(d)))
        parts = pid[sub_idx]
        sub = mm[sub_idx].astype(np.float32)
        C = np.zeros((M, int(d)), dtype=np.float64)
        for p in range(M):
            msk = parts == p
            if msk.any():
                C[p] = sub[msk].mean(axis=0)
        C = C.astype(np.float32)
    q = queries.astype(np.float32)
    dmat = (q * q).sum(1)[:, None] - 2.0 * (q @ C.T) + (C * C).sum(1)[None, :]
    coord = dmat.argmin(axis=1).astype(np.uint32)
    print(f"  partition[{part_name}] M={M} routed-hist="
          f"{np.bincount(coord, minlength=M)[:M].tolist()}", flush=True)
    return pid, coord


def run(name, cfg, out_dir):
    dtype = cfg["dtype"]
    npd = _NP[dtype]
    H = _HARNESS[dtype]
    dim = int(cfg["dim"])
    K = int(cfg.get("K", 10))
    Ls = int(cfg.get("L_search", 100))
    cw = int(cfg.get("conv_window", 3))
    ct = float(cfg.get("conv_threshold_pct", 1.0))
    id_off = int(cfg.get("id_offset", 0))
    max_cap = int(cfg["max_capacity"])
    max_pts = int(cfg.get("max_points", max_cap))
    nq = cfg.get("n_query", None)
    L_sweep = [int(x) for x in cfg.get("L_search_sweep", [Ls])]

    print(f"\n=== Q1 {name} (dim={dim}, {dtype}) ===", flush=True)
    h = H(dim=dim, max_points=max_pts, R=int(cfg.get("R", 64)),
          L_build=int(cfg.get("L_build", Ls)), num_threads=int(cfg.get("num_threads", 0)))
    t0 = time.time()
    h.load_index(cfg["graph_file"], cfg["base_file"], cfg["tags_file"], search_L=max(L_sweep))
    # q1_load_pq: false skips the (multi-GB) code load for runs whose modes never
    # touch PQ — e.g. the LVQ arm on laion, where pq_all_codes.bin is 32 GB and the
    # node is already at the swap boundary. Default true = unchanged behaviour.
    if bool(cfg.get("q1_load_pq", True)):
        h.load_pq(cfg["pq_codebook"], cfg["pq_codes"], max_capacity=max_cap, alpha=0.0)
    else:
        print("  q1_load_pq=false -> PQ codes NOT loaded (no pq/pq_rerank mode)", flush=True)
    print(f"  loaded num_points={h.num_points()} M={h.M()} ({time.time()-t0:.1f}s)", flush=True)

    queries = load_bin(cfg["query_file"], npd)
    gt = load_gt(cfg["gt_file"])
    if nq:
        queries = queries[:int(nq)]
        gt = gt[:int(nq)]
    print(f"  queries={queries.shape} gt={gt.shape}", flush=True)

    # multi-label accounting: all listed partitions are counted in ONE traversal
    # (labels only affect counters, not the search itself).
    pnames = cfg.get("q1_partitions") or \
        ([cfg["q1_partition"]] if cfg.get("q1_partition") else [])
    pids, coords = [], []
    for pn in pnames:
        pid, coord = load_partition(cfg, pn, queries)
        pids.append(pid)
        coords.append(coord)
    rerank_scope = str(cfg.get("q1_rerank_scope", "full"))

    # OPQ: PQ codes live in a rotated frame; the harness rotates the QUERY for the
    # PQ LUT only (exact distances stay in the original frame with the original
    # base). Mathematically identical to the 08-06 rotated-base protocol, but no
    # rotated copy of the base is needed on disk.
    opq_R = None
    opq_path = cfg.get("opq_rotation")
    if opq_path:
        opq_R = np.load(opq_path).astype(np.float32)
        orth = float(np.abs(opq_R @ opq_R.T - np.eye(opq_R.shape[0], dtype=np.float32)).max())
        if opq_R.shape != (dim, dim):
            sys.exit(f"opq_rotation shape {opq_R.shape} != ({dim},{dim}): {opq_path}")
        if orth > 1e-4:
            sys.exit(f"opq_rotation not orthonormal (max |RR^T - I| = {orth:g}): {opq_path}")
        print(f"  OPQ rotation {opq_path} shape={opq_R.shape} orth_err={orth:.3e}", flush=True)

    M = int(h.M())
    # per-vector storage bytes for each precision (Pareto x-axis).
    # LVQ (VLDB'23) bytes = codes + per-vector (scale, offset) floats:
    #   lvq8   = D   + 8            (8-bit primary)
    #   lvq4   = D/2 + 8            (4-bit primary, 2 dims/byte — max SQ compression)
    #   lvq4x8 = D/2 + D + 16       (4-bit primary + 8-bit residual, two scale/offset pairs)
    BYTES = {"exact": dim * _BYTES[dtype], "pq": M, "pq_rerank": M,
             "sq8": dim, "sq8_rerank": dim,                     # 1 byte/dim scalar (no centering)
             "lvq8": dim + 8, "lvq4": (dim + 1) // 2 + 8,
             "lvq4x8": (dim + 1) // 2 + dim + 16,
             "rabitq": (dim + 7) // 8, "rabitq_rerank": (dim + 7) // 8}  # 1 bit/dim
    KEY = {"exact": "raw"}   # row-key alias (plot expects recall_raw/bytes_raw)
    modes = [m.strip() for m in str(cfg.get(
        "q1_modes", "exact,pq,pq_rerank,sq8,sq8_rerank")).split(",") if m.strip()]
    # fetch-accounting counters exported per (L, mode, label) — bytes/query is
    # derived in the analysis (mode-dependent bytes/fetch model, Q4-anchored).
    rows = []
    for L in L_sweep:
        rec, ctrs = {}, {}
        for m in modes:
            t_s = time.time()
            r = h.search_batch(queries, K=K, L_search=L, conv_window=cw, conv_threshold_pct=ct,
                               filter_on=False, search_dist=m,
                               partition_of=(pids if pids else None),
                               coordinator=(coords if coords else None),
                               rerank_scope=rerank_scope,
                               opq_rotation=opq_R)
            dt = time.time() - t_s
            rec[m] = recall_at_k(r["result_tags"], gt, K, id_off)
            nq_run = max(1, int(r["n_queries"]))
            ctr_time = {f"qps_1node": round(nq_run / dt, 1), "search_s": round(dt, 2)}
            ctrs[m] = {"sum_hops": int(r["sum_hops"]) / nq_run}
            ctrs[m].update(ctr_time)
            for i, pn in enumerate(pnames):
                lc = r["label_counters"][i]
                for c, v in lc.items():
                    ctrs[m][f"{c}_{pn}"] = int(v) / nq_run
        parts = "  ".join(f"{m}={rec[m]:.4f}({BYTES.get(m, '?')}B)" for m in modes)
        print(f"  L={L}: recall@{K}  {parts}", flush=True)
        row = dict(dataset=name, dim=dim, dtype=dtype, L=L, K=K, pq_M=M,
                   partitions=pnames, rerank_scope=rerank_scope,
                   opq=bool(opq_R is not None))
        if opq_R is not None:
            row["opq_rotation"] = opq_path
        for m in modes:
            k = KEY.get(m, m)
            row[f"recall_{k}"] = round(rec[m], 4)
            row[f"bytes_{k}"] = BYTES.get(m)
            for c, v in ctrs[m].items():
                row[f"{c}_{k}"] = round(v, 2)   # per-query averages
        rows.append(row)

    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"{name}_q1_pareto.json")
    with open(outp, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  -> {outp}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="datasets YAML (reuse M11 configs)")
    ap.add_argument("--dataset", default=None, help="run only this dataset key")
    ap.add_argument("--out", default="results_q1")
    ap.add_argument("--modes", default=None,
                    help="comma list of search_dist modes (overrides cfg q1_modes)")
    ap.add_argument("--partition", default=None,
                    help="comma list of partition names from cfg partitions list "
                         "(e.g. balanced,centroid,lits) — enables multi-label "
                         "local/remote fetch accounting in one traversal")
    ap.add_argument("--opq-rotation", default=None,
                    help="path to an OPQ rotation .npy (D x D float32); overrides "
                         "cfg opq_rotation. PQ LUT is built from R*q, exact "
                         "distances stay in the original frame")
    ap.add_argument("--rerank-scope", default=None, choices=["full", "beam"],
                    help="rerank pool: full=DiskANN full_retset, beam=final beam only "
                         "(fetch count <= L; distributed-reproducible)")
    a = ap.parse_args()
    if yaml is None:
        sys.exit("pyyaml not available")
    with open(a.config) as f:
        docs = yaml.safe_load(f)
    ds = docs["datasets"]
    keys = [a.dataset] if a.dataset else list(ds.keys())
    all_rows = []
    for k in keys:
        if a.modes:
            ds[k]["q1_modes"] = a.modes
        if a.partition:
            ds[k]["q1_partitions"] = [p.strip() for p in a.partition.split(",") if p.strip()]
        if a.rerank_scope:
            ds[k]["q1_rerank_scope"] = a.rerank_scope
        if a.opq_rotation:
            ds[k]["opq_rotation"] = a.opq_rotation
        all_rows += run(k, ds[k], a.out)
    print("\n== Q1 summary (memory-recall Pareto points) ==")
    for r in all_rows:
        print(r)


if __name__ == "__main__":
    main()
