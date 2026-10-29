#!/usr/bin/env python3
"""
gen_gt_chunked.py - exact brute-force k-NN ground truth for very large active windows.

Why this exists
---------------
pipeline/generate_gt.py materialises the whole active set (`vectors[active_locs]`)
and a full (B, N_active) distance matrix.  For a 200M x 512 fp32 window that is
~410 GB resident plus ~800 GB of distances -- infeasible.  This script computes
the *same* result while streaming the base file, chunking over BOTH the database
axis and the query axis, so peak RSS stays in the ~10-20 GB range.

Algorithm
---------
For every DB chunk (contiguous gid range) and every query batch:

    dists = (-2 * Q) @ V^T  +  |V|^2                     (rank-equivalent sq-L2;
                                                          the per-query |Q|^2 term
                                                          is a row constant and is
                                                          added back at the end)

Top-K selection uses a *tile-min* prefilter instead of a full-width argpartition,
which is ~50x faster:

    partition each row into tiles of `--tile` columns and take each tile's min.
    A tile containing one of the K smallest elements has min <= v_K, and at most
    K elements are <= v_K, so at most K tiles can have min <= v_K.  Hence the K
    smallest tile-minima provably cover all K smallest elements.  Gather those
    K tiles (K*tile candidates) and argpartition just those.

Selection + the `+|V|^2` broadcast run in a thread pool (numpy releases the GIL
on both), the BLAS sgemm is multi-threaded, and the next DB chunk is prefetched
with os.preadv on a background thread so NFS reads overlap compute.

Usage
-----
    python gen_gt_chunked.py \
        --base    /path/to/base.fbin \
        --queries /path/to/query.fbin \
        --gid-lo  200000000 --gid-n 200000000 \
        --K 10 \
        --out     /path/to/gt_lits_k8_iter150_K10_Q10000.npy

`--gid-lo/--gid-n` describe the active window as *absolute row indices* into the
base file; emitted ids are absolute gids (row index), matching the convention
`gid = tag - 1 + id_offset` used by the M11 harness.

Writes alongside `--out`:
    <out>                       (Q, K) int64 absolute gids
    <out stem>_queries.npy      (Q, D) float32 queries  (harness loads this)
    <out stem>.json             metadata
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_NP_DTYPE = {
    "float32": np.float32, "float": np.float32,
    "uint8": np.uint8, "int8": np.int8,
}


# ---------------------------------------------------------------------------
# bin file helpers ([int32 npts][int32 dim] then row-major data)
# ---------------------------------------------------------------------------
def read_header(path, dtype):
    size = os.path.getsize(path)
    npts_hdr, dim = (int(x) for x in np.fromfile(path, dtype=np.int32, count=2))
    itemsize = np.dtype(dtype).itemsize
    row = dim * itemsize
    if (size - 8) % row != 0:
        raise SystemExit(
            f"{path}: (size-8)={size-8} not divisible by dim*itemsize={row}; "
            f"wrong --dim/--dtype?")
    npts_true = (size - 8) // row
    return npts_hdr, npts_true, dim


def load_queries(path, dtype, dim_expect=None):
    if path.endswith(".npy"):
        q = np.load(path)
        return np.ascontiguousarray(q, dtype=np.float32)
    npts_hdr, npts_true, dim = read_header(path, dtype)
    if npts_hdr != npts_true:
        print(f"  [warn] query header npts={npts_hdr} != file-derived {npts_true}; "
              f"using {npts_true}", flush=True)
    q = np.fromfile(path, dtype=dtype, offset=8, count=npts_true * dim)
    q = q.reshape(npts_true, dim)
    if dim_expect is not None and dim != dim_expect:
        raise SystemExit(f"query dim {dim} != base dim {dim_expect}")
    return np.ascontiguousarray(q, dtype=np.float32)


# ---------------------------------------------------------------------------
# top-K over one (rows, c) distance block, with tile-min prefilter
# ---------------------------------------------------------------------------
def _select_rows(dblk, vn, K, tile, s, e, out_v, out_i):
    """Rows [s:e) of dblk: add |V|^2, then exact top-K. Runs on a worker thread."""
    sub = dblk[s:e]
    sub += vn                                        # broadcast, releases GIL
    r, c = sub.shape
    k = min(K, c)

    if tile <= 0 or c <= tile or c % tile != 0:
        # small / ragged block: plain argpartition is fine here
        p = np.argpartition(sub, k - 1, axis=1)[:, :k]
        out_v[s:e, :k] = np.take_along_axis(sub, p, 1)
        out_i[s:e, :k] = p
        if k < K:
            out_v[s:e, k:] = np.inf
            out_i[s:e, k:] = -1
        return

    nt = c // tile
    tmin = sub.reshape(r, nt, tile).min(axis=2)      # (r, nt) tile minima
    kt = min(K, nt)
    tsel = np.argpartition(tmin, kt - 1, axis=1)[:, :kt].astype(np.int64)
    off = np.arange(tile, dtype=np.int64)
    for i in range(r):
        cols = (tsel[i, :, None] * tile + off).ravel()   # (kt*tile,) candidates
        vals = sub[i, cols]
        p = np.argpartition(vals, k - 1)[:k]
        out_v[s + i, :k] = vals[p]
        out_i[s + i, :k] = cols[p]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base .fbin/.bin (row-major, 8B header)")
    ap.add_argument("--queries", required=True, help="query .fbin/.bin/.npy")
    ap.add_argument("--gid-lo", type=int, required=True, help="first active gid (row index)")
    ap.add_argument("--gid-n", type=int, required=True, help="number of active vectors")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--out", required=True, help="output .npy for (Q,K) gids")
    ap.add_argument("--dtype", default="float32", choices=sorted(_NP_DTYPE))
    ap.add_argument("--num-queries", type=int, default=None, help="use only first N queries")
    ap.add_argument("--query-offset", type=int, default=0, help="skip first N queries")
    ap.add_argument("--db-chunk", type=int, default=2_000_000, help="vectors per DB chunk")
    ap.add_argument("--qb", type=int, default=500, help="queries per batch")
    ap.add_argument("--tile", type=int, default=2000, help="tile width for top-K prefilter (0=off)")
    ap.add_argument("--select-threads", type=int, default=16)
    ap.add_argument("--rows-per-task", type=int, default=32)
    ap.add_argument("--report-every", type=int, default=1, help="log every N DB chunks")
    ap.add_argument("--no-queries-npy", action="store_true")
    ap.add_argument("--dry-run-chunks", type=int, default=0,
                    help="stop after N DB chunks (timing probe; does not write output)")
    args = ap.parse_args()

    t_start = time.time()
    src_dt = _NP_DTYPE[args.dtype]

    npts_hdr, npts_true, dim = read_header(args.base, src_dt)
    print(f"base    : {args.base}", flush=True)
    print(f"          dim={dim} dtype={args.dtype} npts(header)={npts_hdr} "
          f"npts(size-derived)={npts_true}"
          + ("  [HEADER MISMATCH -> using size-derived]" if npts_hdr != npts_true else ""),
          flush=True)

    lo, n = args.gid_lo, args.gid_n
    if lo < 0 or n <= 0 or lo + n > npts_true:
        raise SystemExit(f"active window [{lo}, {lo+n}) does not fit in base "
                         f"({npts_true} vectors)")

    queries = load_queries(args.queries, src_dt, dim_expect=dim)
    if args.query_offset:
        queries = queries[args.query_offset:]
    if args.num_queries:
        queries = queries[:args.num_queries]
    queries = np.ascontiguousarray(queries)
    Q = queries.shape[0]
    K = args.K
    print(f"queries : {args.queries}  -> {queries.shape} float32", flush=True)
    print(f"window  : gid [{lo}, {lo+n})  n={n}  K={K}", flush=True)

    # rank-equivalent distance: (-2Q) @ V^T + |V|^2 ; |Q|^2 added back at the end
    qs = (queries * -2.0).astype(np.float32)
    q2 = np.einsum("ij,ij->i", queries, queries).astype(np.float64)

    db_chunk = min(args.db_chunk, n)
    qb = min(args.qb, Q)
    nchunks = (n + db_chunk - 1) // db_chunk

    # running top-K (distances lack the +|Q|^2 row constant; ordering is unaffected)
    best_v = np.full((Q, K), np.inf, dtype=np.float32)
    best_g = np.full((Q, K), -1, dtype=np.int64)

    # double-buffered DB chunk + reusable distance block
    raw_bufs = [np.empty((db_chunk, dim), dtype=src_dt) for _ in range(2)]
    if src_dt is np.float32:
        fp_bufs = raw_bufs
    else:
        fp_bufs = [np.empty((db_chunk, dim), dtype=np.float32) for _ in range(2)]
    dblk = np.empty((qb, db_chunk), dtype=np.float32)
    cand_v = np.empty((qb, K), dtype=np.float32)
    cand_i = np.empty((qb, K), dtype=np.int64)

    itemsize = np.dtype(src_dt).itemsize
    fd = os.open(args.base, os.O_RDONLY)
    pool = ThreadPoolExecutor(max(1, args.select_threads))
    reader = ThreadPoolExecutor(1)

    def read_into(buf, first, count):
        nbytes = count * dim * itemsize
        off = 8 + first * dim * itemsize
        mv = memoryview(buf.reshape(-1)[:count * dim].view(np.uint8))
        got = 0
        while got < nbytes:
            step = min(1 << 30, nbytes - got)
            got_now = os.preadv(fd, [mv[got:got + step]], off + got)
            if got_now == 0:
                raise SystemExit(f"unexpected EOF reading {args.base} at {off+got}")
            got += got_now
        return count

    gemm_s = sel_s = prep_s = io_wait_s = 0.0
    peak_chunks = nchunks if not args.dry_run_chunks else min(nchunks, args.dry_run_chunks)

    pending = reader.submit(read_into, raw_bufs[0], lo, min(db_chunk, n))
    for ci in range(peak_chunks):
        c0 = ci * db_chunk
        c = min(db_chunk, n - c0)
        cur = ci % 2

        t0 = time.time()
        pending.result()                              # wait for this chunk's read
        io_wait_s += time.time() - t0

        if ci + 1 < peak_chunks:                      # prefetch the next one
            nxt0 = (ci + 1) * db_chunk
            pending = reader.submit(read_into, raw_bufs[(ci + 1) % 2],
                                    lo + nxt0, min(db_chunk, n - nxt0))

        t0 = time.time()
        if src_dt is np.float32:
            V = fp_bufs[cur][:c]
        else:
            V = fp_bufs[cur][:c]
            np.copyto(V, raw_bufs[cur][:c])           # widen uint8/int8 -> float32
        # |V|^2, threaded over row slabs
        vn = np.empty(c, dtype=np.float32)
        slab = max(1, (c + args.select_threads - 1) // args.select_threads)
        list(pool.map(lambda s: np.einsum("ij,ij->i", V[s:s + slab], V[s:s + slab],
                                          out=vn[s:s + slab]),
                      range(0, c, slab)))
        prep_s += time.time() - t0

        VT = V.T                                      # (dim, c) view -> BLAS transb='T'
        for qs0 in range(0, Q, qb):
            qs1 = min(qs0 + qb, Q)
            r = qs1 - qs0
            D = dblk[:r, :c]

            t0 = time.time()
            np.matmul(qs[qs0:qs1], VT, out=D)
            gemm_s += time.time() - t0

            t0 = time.time()
            rp = max(1, args.rows_per_task)
            list(pool.map(lambda s: _select_rows(D, vn, K, args.tile, s,
                                                 min(s + rp, r), cand_v, cand_i),
                          range(0, r, rp)))
            # merge chunk-local top-K into the running top-K
            av = np.concatenate([best_v[qs0:qs1], cand_v[:r]], axis=1)
            ag = np.concatenate([best_g[qs0:qs1],
                                 cand_i[:r] + (lo + c0)], axis=1)
            p = np.argpartition(av, K - 1, axis=1)[:, :K]
            best_v[qs0:qs1] = np.take_along_axis(av, p, 1)
            best_g[qs0:qs1] = np.take_along_axis(ag, p, 1)
            sel_s += time.time() - t0

        if (ci + 1) % args.report_every == 0 or ci + 1 == peak_chunks:
            el = time.time() - t_start
            done = ci + 1
            eta = el / done * (peak_chunks - done)
            print(f"  chunk {done}/{peak_chunks}  gid<{lo+c0+c}  "
                  f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m  "
                  f"[gemm {gemm_s:.0f}s sel {sel_s:.0f}s prep {prep_s:.0f}s "
                  f"iowait {io_wait_s:.0f}s]", flush=True)

    reader.shutdown(wait=True)
    pool.shutdown(wait=True)
    os.close(fd)

    if args.dry_run_chunks:
        print(f"dry run: stopped after {peak_chunks} chunks, nothing written", flush=True)
        return

    # final ordering: distance asc, gid asc as a deterministic tie-break
    order = np.lexsort((best_g, best_v), axis=1)
    gt = np.take_along_axis(best_g, order, 1)
    dist = np.take_along_axis(best_v, order, 1).astype(np.float64) + q2[:, None]
    dist = np.maximum(dist, 0.0)

    bad = int((gt < lo).sum() + (gt >= lo + n).sum())
    if bad:
        raise SystemExit(f"BUG: {bad} ids outside the active window")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stem = out[:-4] if out.endswith(".npy") else out
    np.save(out, gt)
    if not args.no_queries_npy:
        np.save(stem + "_queries.npy", queries)

    elapsed = time.time() - t_start
    try:
        import resource
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    except Exception:
        peak_gb = None

    meta = {
        "K": K, "num_queries": int(Q), "num_active": int(n),
        "active_window": [int(lo), int(lo + n)],
        "base_file": os.path.abspath(args.base),
        "query_file": os.path.abspath(args.queries),
        "dim": int(dim), "dtype": args.dtype,
        "method": "brute_force_chunked_tilemin_exact",
        "db_chunk": int(db_chunk), "qb": int(qb), "tile": int(args.tile),
        "elapsed_seconds": round(elapsed, 2),
        "peak_rss_gb": round(peak_gb, 2) if peak_gb else None,
        "gid_min": int(gt.min()), "gid_max": int(gt.max()),
        "mean_dist1": float(dist[:, 0].mean()), "mean_distK": float(dist[:, -1].mean()),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {out}  {gt.shape} {gt.dtype}", flush=True)
    if not args.no_queries_npy:
        print(f"wrote {stem}_queries.npy  {queries.shape} {queries.dtype}", flush=True)
    print(f"wrote {stem}.json", flush=True)
    print(f"gid range [{meta['gid_min']}, {meta['gid_max']}]  "
          f"inside window [{lo}, {lo+n}): yes", flush=True)
    print(f"mean d1={meta['mean_dist1']:.4f}  mean d{K}={meta['mean_distK']:.4f}", flush=True)
    print(f"elapsed {elapsed/60:.1f} min   peak RSS "
          f"{meta['peak_rss_gb']} GB", flush=True)
    print(f"breakdown: gemm {gemm_s:.0f}s  select {sel_s:.0f}s  "
          f"prep {prep_s:.0f}s  io-wait {io_wait_s:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
