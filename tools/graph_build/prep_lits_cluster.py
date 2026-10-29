#!/usr/bin/env python3
"""prep_lits_cluster.py — LITS build assets from a raw DiskANN bin pool.

Port of diskann-lits scripts/update/prep_cluster.py, scale-adapted for the
SparsePath update-cycle builder (200M/400M initial windows: streaming
MiniBatchKMeans over a memmap instead of in-RAM KMeans). Produces, for the
first `aw` (= initial_size) rows of the pool:

  <out-prefix>.labels     {aw+1 u32, K u32} + (aw+1) x u32   tag-indexed:
                          entry 0 = start point's cluster, entry i = row i-1
  <out-prefix>.centroids  {K i32, dim i32} + K*dim float32   (member means)
  <out-prefix>.start.bin  {1 i32, dim i32} + 1 vector        (window mean, ref)

The start point is the per-dim mean of the window (float64 accumulate, then
dtype round/clip) — the same formula as diskann_bindings set_mean_frozen_point,
which installs the frozen point at build time. Entry 0 of .labels is that
vector's predicted cluster, so tag-0 lookups in the engine agree with the
actual frozen point. Centroids are float32 member means (start point included
in its cluster), matching diskann-lits' .centroids convention.

Feed the outputs to build_updatecycle_lits.py:
  --lits-labels <prefix>.labels --lits-centroids <prefix>.centroids --lits-clusters K
"""
import argparse, os, struct, sys, time
import numpy as np

_DT = {"float32": np.float32, "uint8": np.uint8, "int8": np.int8}


def window_mean(vectors, aw, dim, dtype, chunk):
    """Per-dim mean of vectors[:aw] in float64, cast back to dtype
    (round + clip for integer types) — mirrors set_mean_frozen_point."""
    acc = np.zeros(dim, dtype=np.float64)
    for s in range(0, aw, chunk):
        e = min(s + chunk, aw)
        acc += vectors[s:e].astype(np.float64).sum(axis=0)
    mean = acc / float(aw)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        mean = np.clip(np.rint(mean), info.min, info.max)
    return mean.astype(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pool .bin/.fbin (DiskANN bin: {n,dim} i32 header)")
    ap.add_argument("--dtype", required=True, choices=list(_DT))
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--active-window", type=int, required=True, help="aw = initial_size of the build")
    ap.add_argument("--k", type=int, required=True, help="cluster count (= target partition count Np)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-sample", type=int, default=20_000_000,
                    help="rows subsampled (evenly strided) for the kmeans fit")
    ap.add_argument("--chunk-rows", type=int, default=5_000_000)
    ap.add_argument("--batch-size", type=int, default=262_144, help="MiniBatchKMeans batch size")
    ap.add_argument("--no-balanced", action="store_true",
                    help="plain nearest-centroid labels (default: capacity-balanced "
                         "assignment, equal shares like the earlier 100M runs)")
    args = ap.parse_args()

    from sklearn.cluster import MiniBatchKMeans

    dt = _DT[args.dtype]
    aw, K, dim = args.active_window, args.k, args.dim

    hdr = np.fromfile(args.base, dtype=np.uint32, count=2)
    n_total, d = int(hdr[0]), int(hdr[1])
    assert d == dim, f"dim mismatch: file {d} != --dim {dim}"
    assert aw <= n_total, f"active window {aw:,} > pool {n_total:,}"
    vectors = np.memmap(args.base, dtype=dt, mode="r", offset=8, shape=(n_total, d))
    print(f"[prep] pool={args.base} n={n_total:,} d={d} dtype={args.dtype} aw={aw:,} K={K}", flush=True)

    t0 = time.time()
    # --- start point = window mean (dtype-rounded) ---
    start_vec = window_mean(vectors, aw, dim, dt, args.chunk_rows)
    print(f"[prep] window mean done ({time.time()-t0:.1f}s)", flush=True)

    # --- fit: MiniBatchKMeans on an evenly strided subsample of the window ---
    stride = max(1, aw // max(1, args.fit_sample))
    km = MiniBatchKMeans(n_clusters=K, random_state=args.seed,
                         batch_size=args.batch_size, n_init=3)
    t1 = time.time()
    fitted_rows = 0
    for s in range(0, aw, args.chunk_rows):
        e = min(s + args.chunk_rows, aw)
        sub = np.ascontiguousarray(vectors[s:e:stride]).astype(np.float32)
        if len(sub) == 0:
            continue
        km.partial_fit(sub)
        fitted_rows += len(sub)
    print(f"[prep] fit done: {fitted_rows:,} rows (stride {stride}) ({time.time()-t1:.1f}s)", flush=True)

    # --- assignment pass over the full window; accumulate member sums for centroids ---
    # CAPACITY-BALANCED assignment (default): per-cluster caps split aw+1 evenly
    # (cap diff <= 1), rows go to the nearest centroid with remaining capacity,
    # over-cap rows spill to their next-nearest. Matches the earlier 100M runs
    # (initial counts were exactly 25M+-1 per cluster = balanced labels) and the
    # SparsePath partition stage (balanced streaming kmeans). The runtime
    # balanced allocator only balances NEW inserts; the initial window must be
    # balanced here or partitions inherit the natural cluster skew.
    labels = np.zeros(aw + 1, dtype=np.uint32)
    sums = np.zeros((K, dim), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    total = aw + 1  # window rows + start point
    if args.no_balanced:
        rem = None
    else:
        cap = np.full(K, total // K, dtype=np.int64)
        cap[: total % K] += 1
        rem = cap.copy()
        rem[0] -= 1  # reserve the start point's slot up front (assigned to its
        # nearest cluster below; slot returned there if it lands elsewhere)

    t2 = time.time()
    for s in range(0, aw, args.chunk_rows):
        e = min(s + args.chunk_rows, aw)
        chunk = np.ascontiguousarray(vectors[s:e]).astype(np.float32)
        if rem is None:
            lab = km.predict(chunk).astype(np.uint32)
        else:
            d = km.transform(chunk).astype(np.float32)   # (n, K) distances
            pref = np.argsort(d, axis=1)
            lab = np.full(len(chunk), -1, dtype=np.int64)
            unassigned = np.arange(len(chunk))
            for p in range(K):
                if len(unassigned) == 0:
                    break
                choice = pref[unassigned, p]
                for k in range(K):
                    rows = unassigned[choice == k]
                    if len(rows) == 0 or rem[k] <= 0:
                        continue
                    if len(rows) > rem[k]:
                        # keep the closest rem[k] rows; rest spill to next pref
                        keep = rows[np.argsort(d[rows, k])[: rem[k]]]
                    else:
                        keep = rows
                    lab[keep] = k
                    rem[k] -= len(keep)
                unassigned = unassigned[lab[unassigned] < 0]
            assert len(unassigned) == 0, "balanced assignment failed to place all rows"
            lab = lab.astype(np.uint32)
        labels[1 + s:1 + e] = lab  # tag i = row i-1
        counts += np.bincount(lab, minlength=K)
        for k in range(K):
            m = lab == k
            if m.any():
                sums[k] += chunk[m].astype(np.float64).sum(axis=0)
        if (s // args.chunk_rows) % 10 == 0:
            print(f"[prep] assign {e:,}/{aw:,} ({time.time()-t2:.1f}s)", flush=True)

    # entry 0 = start point's cluster (start included in its cluster's mean).
    # Nearest centroid regardless of capacity (single point); the slot reserved
    # in cluster 0 is returned if the start lands elsewhere.
    start_lab = int(km.predict(start_vec.astype(np.float32).reshape(1, -1))[0])
    labels[0] = start_lab
    sums[start_lab] += start_vec.astype(np.float64)
    counts[start_lab] += 1
    if rem is not None and start_lab != 0:
        rem[0] += 1
        rem[start_lab] -= 1  # may go to -1: 1-row overflow, harmless

    if (counts == 0).any():
        sys.exit(f"[prep] FATAL: empty cluster(s): {np.where(counts == 0)[0].tolist()}")
    centroids = (sums / counts[:, None]).astype(np.float32)

    dist = counts / counts.sum()
    print(f"[prep] cluster sizes: {counts.tolist()}  (min/max share {dist.min():.3f}/{dist.max():.3f})",
          flush=True)

    # --- outputs (formats per diskann-lits prep_cluster.py) ---
    out = args.out_prefix
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".labels", "wb") as f:            # {aw+1 u32, K u32} + u32[]
        f.write(struct.pack("<II", aw + 1, K))
        labels.tofile(f)
    with open(out + ".centroids", "wb") as f:         # {K i32, dim i32} + f32[]
        f.write(struct.pack("<ii", K, dim))
        centroids.tofile(f)
    with open(out + ".start.bin", "wb") as f:         # {1, dim} + 1 vector (reference)
        f.write(struct.pack("<ii", 1, dim))
        start_vec.tofile(f)
    print(f"[prep] wrote {out}.labels / .centroids / .start.bin "
          f"(start cluster={start_lab}, total {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
