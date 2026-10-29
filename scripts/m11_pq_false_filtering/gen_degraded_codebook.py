#!/usr/bin/env python
"""Tier B — degraded PQ codebook generator (stale-codebook drift-stress).

Produces a *deliberately degraded* PQ codebook + re-encoded codes, in the exact
binary format the shipped pipeline emits (so the M11 harness reads them with
`load_pq` unchanged). Used to trace filter recall vs codebook-quality (QE):
recall should stay flat until the codebook's distance error crosses the alpha
margin — the "safety margin" that rebuts the i.i.d.-benchmark objection.

Levers (controlled drift, monotone-ish QE dial):
  * noise      : shipped centroids + N(0, (sigma_rel * per-chunk centroid std)^2).
                 sigma_rel=0 reproduces the shipped codebook (control).
  * undertrain : retrain faiss k-means on a small sample (starve the codebook).
  * biased     : retrain on ONE spatial partition's active vectors (narrow
                 initial distribution) then apply to the full active set.

Formats (must match search/src/pq_filter.cpp::load and pipeline/stages.py):
  codebook.bin : [u32 M][u32 chunk_offsets[M+1]][u32 ksub=256]
                 [float32 centroids: for m in M -> 256 * chunk_size_m, row-major (k,di)]
  all_codes.bin: [u32 N][u32 M][u8 codes[N*M]]  (gid-indexed, code[gid*M+m])

This module is pure numpy/faiss (no harness) so it can be exercised no-fire.
run_tierB.py imports these functions and drives load_pq per variant.
"""
import struct
import numpy as np

KSUB = 256


# ---------------------------------------------------------------------------
# codebook / codes binary I/O  (byte-compatible with pq_filter.cpp + stages.py)
# ---------------------------------------------------------------------------
def load_codebook(path):
    """-> (centroids_list[M] of (256, ds) float32, chunk_offsets int[M+1])."""
    with open(path, "rb") as f:
        (M,) = struct.unpack("<I", f.read(4))
        chunk_offsets = np.frombuffer(f.read((M + 1) * 4), dtype=np.uint32).astype(np.int64)
        (ksub,) = struct.unpack("<I", f.read(4))
        if ksub != KSUB:
            raise ValueError(f"ksub must be 256, got {ksub}")
        centroids_list = []
        for m in range(M):
            ds = int(chunk_offsets[m + 1] - chunk_offsets[m])
            buf = np.frombuffer(f.read(KSUB * ds * 4), dtype=np.float32)
            centroids_list.append(np.ascontiguousarray(buf.reshape(KSUB, ds)))
    return centroids_list, chunk_offsets


def write_codebook(path, centroids_list, chunk_offsets):
    M = len(centroids_list)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", M))
        np.asarray(chunk_offsets, dtype=np.uint32).tofile(f)
        f.write(struct.pack("<I", KSUB))
        for m in range(M):
            np.ascontiguousarray(centroids_list[m], dtype=np.float32).tofile(f)


def write_codes(path, codes_active, active_lo, N):
    """Write gid-indexed codes: [0,active_lo) zero, [active_lo, active_lo+len) = codes_active.

    N (file header) is kept identical across all variants so the harness's
    load_pq (which resizes+reads the first N entries) overwrites cleanly on
    every re-load — no stale region survives between variants.
    """
    M = codes_active.shape[1]
    n_active = codes_active.shape[0]
    if active_lo + n_active > N:
        raise ValueError(f"active window [{active_lo},{active_lo+n_active}) exceeds N={N}")
    with open(path, "wb") as f:
        f.write(struct.pack("<II", N, M))
        _write_zeros(f, active_lo * M)                        # zero prefix [0, active_lo)
        codes_active.astype(np.uint8).tofile(f)
        _write_zeros(f, (N - active_lo - n_active) * M)       # zero suffix


def _write_zeros(f, nbytes, chunk=1 << 26):
    if nbytes <= 0:
        return
    z = np.zeros(min(nbytes, chunk), dtype=np.uint8)
    while nbytes > 0:
        w = min(nbytes, chunk)
        f.write(z[:w].tobytes())
        nbytes -= w


# ---------------------------------------------------------------------------
# encode + quantization error  (nearest centroid per chunk, squared L2)
# ---------------------------------------------------------------------------
def encode_and_qe(base_slice, centroids_list, chunk_offsets, batch=1_000_000):
    """base_slice: (n, dim) float32. -> (codes (n,M) uint8, qe_mean float).

    qe = mean over vectors of sum_m ||x_chunk_m - centroid_m[code_m]||^2
    (total squared reconstruction error; same distance the filter LUT uses).
    """
    n, dim = base_slice.shape
    M = len(centroids_list)
    codes = np.empty((n, M), dtype=np.uint8)
    qe_sum = 0.0
    cnorm = [(c.astype(np.float32) ** 2).sum(1) for c in centroids_list]   # (256,) per chunk
    for s in range(0, n, batch):
        e = min(s + batch, n)
        xb = base_slice[s:e]
        for m in range(M):
            o0, o1 = int(chunk_offsets[m]), int(chunk_offsets[m + 1])
            sub = np.ascontiguousarray(xb[:, o0:o1], dtype=np.float32)      # (b, ds)
            c = centroids_list[m]                                           # (256, ds)
            # d(x,k) = ||x||^2 - 2 x.c_k + ||c_k||^2 ; argmin ignores ||x||^2
            d = (-2.0) * (sub @ c.T) + cnorm[m][None, :]                    # (b,256)
            km = np.argmin(d, axis=1).astype(np.uint8)
            codes[s:e, m] = km
            xsq = (sub * sub).sum(1)
            qe_sum += float((xsq + d[np.arange(e - s), km]).sum())
    return codes, qe_sum / n


# ---------------------------------------------------------------------------
# degradation levers
# ---------------------------------------------------------------------------
def degrade_noise(centroids_list, sigma_rel, seed):
    """Add N(0, (sigma_rel * per-chunk centroid coordinate std)^2) to centroids.
    sigma_rel = 0 returns an exact copy (control == shipped codebook)."""
    rng = np.random.RandomState(seed)
    out = []
    for c in centroids_list:
        if sigma_rel <= 0:
            out.append(c.copy())
            continue
        std = float(c.std())
        noise = rng.normal(0.0, sigma_rel * std, size=c.shape).astype(np.float32)
        out.append((c + noise).astype(np.float32))
    return out


def train_codebook(train_vecs, chunk_offsets, seed=42, niter=15):
    """faiss per-chunk k-means (mirrors pipeline/stages.py stage_pq_build)."""
    import faiss
    M = len(chunk_offsets) - 1
    out = []
    for m in range(M):
        o0, o1 = int(chunk_offsets[m]), int(chunk_offsets[m + 1])
        sub = np.ascontiguousarray(train_vecs[:, o0:o1], dtype=np.float32)
        km = faiss.Kmeans(o1 - o0, KSUB, niter=niter, verbose=False, seed=seed)
        km.train(sub)
        out.append(np.ascontiguousarray(km.centroids, dtype=np.float32))
    return out


def degrade_undertrain(base_slice, chunk_offsets, sample_size, seed=42):
    rng = np.random.RandomState(seed)
    n = base_slice.shape[0]
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    return train_codebook(np.ascontiguousarray(base_slice[idx]), chunk_offsets, seed=seed)


def degrade_biased(base_slice, chunk_offsets, partition_of_active, part_id,
                   sample_size=100_000, seed=42):
    """Train on ONE partition's active vectors only (narrow spatial coverage)."""
    rng = np.random.RandomState(seed)
    sel = np.where(partition_of_active == part_id)[0]
    if len(sel) == 0:
        raise ValueError(f"partition {part_id} has no active vectors")
    if len(sel) > sample_size:
        sel = rng.choice(sel, size=sample_size, replace=False)
    return train_codebook(np.ascontiguousarray(base_slice[sel]), chunk_offsets, seed=seed)


# ---------------------------------------------------------------------------
# self-test (no-fire): round-trip the shipped codebook format
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="round-trip / inspect a codebook")
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--out", default=None, help="if set, rewrite and byte-diff")
    args = ap.parse_args()
    cl, co = load_codebook(args.codebook)
    print(f"M={len(cl)} chunk_offsets={co.tolist()} shapes={[c.shape for c in cl]}")
    print(f"centroid stats: mean={np.mean([c.mean() for c in cl]):.4f} "
          f"std={np.mean([c.std() for c in cl]):.4f}")
    if args.out:
        write_codebook(args.out, cl, co)
        a = open(args.codebook, "rb").read(); b = open(args.out, "rb").read()
        print(f"round-trip byte-identical: {a == b}  ({len(a)} vs {len(b)} bytes)")
        os.remove(args.out)
