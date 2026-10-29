#!/usr/bin/env python
"""Is the OPQ D/4 regression on spacev/sift intrinsic, or a RANDOM-INIT artifact?

diag_opq_vs_pq.py established that the shipped M*_opq codes reconstruct WORSE
than plain PQ on spacev (1.25x) and sift (1.56x), with no harness bug. faiss's
OPQMatrix starts from a RANDOM rotation ("making random dxd rotation" in the
build log) and runs alternating minimisation from there. On a dataset whose
NATURAL axis order already groups correlated dimensions (SIFT = 16 spatial cells
x 8 orientation bins, so a consecutive-4 chunk stays inside one cell's
histogram), the identity is already a good rotation and a random start lands in
a worse local optimum that 50 iterations never escape.

This script re-runs the same alternating optimisation from R = IDENTITY, which
by construction cannot end up worse than plain PQ (iteration 0 IS plain PQ).
Comparing identity-init OPQ against plain PQ on the SAME training sample tells
us whether a rebuild can recover anything:

  identity-OPQ << plain PQ  -> rebuild with identity init, real gain available
  identity-OPQ ~= plain PQ  -> OPQ offers nothing here; report as a finding

READ-ONLY: trains in memory, writes no artifacts.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "m11_pq_false_filtering"))
from gen_degraded_codebook import train_codebook, encode_and_qe

ART = "/experiments/artifacts"
GID_LO, GID_N = 150_000_000, 100_000_000
TRAIN_SAMPLE, TRAIN_STRIDE = 200_000, 500
HELD_N, HELD_STRIDE = 50_000, 2000
OUTER = 25          # alternating iterations (faiss default niter is 50)
KM_ITER = 8         # inner k-means iterations per outer step (faiss niter_pq=4)

DS = [
    ("deep100M",   96,  24, np.float32,
     "/workspace/datasets_nfs/DEEP1B/deep300M_base.fbin"),
    ("spacev100M", 100, 25, np.int8,
     "/workspace/datasets_nfs/SPACEV1B/spacev300M_base.i8bin"),
    ("sift100M",   128, 32, np.uint8,
     "/workspace/datasets_nfs/SIFT1B/sift300M_base.bin"),
]


def decode(codes, cl, co):
    out = np.empty((codes.shape[0], int(co[-1])), dtype=np.float32)
    for m in range(len(cl)):
        out[:, co[m]:co[m + 1]] = cl[m][codes[:, m]]
    return out


def sq_err(x, xhat):
    d = x - xhat
    return float((d * d).sum(1).mean())


def opq_identity_init(train, co, outer=OUTER, seed=42):
    """Alternating OPQ from R = I.  Convention: Y = X @ Rt  (Rt = R.T)."""
    dim = int(co[-1])
    Rt = np.eye(dim, dtype=np.float32)
    hist = []
    for it in range(outer):
        Y = np.ascontiguousarray(train @ Rt, dtype=np.float32)
        cl = train_codebook(Y, co, seed=seed, niter=KM_ITER)
        codes, _ = encode_and_qe(Y, cl, co)
        Yhat = decode(codes, cl, co)
        hist.append(sq_err(Y, Yhat))
        # orthogonal Procrustes: minimise ||X @ Rt - Yhat||_F  over orthogonal Rt
        U, _, Vt = np.linalg.svd(train.T @ Yhat, full_matrices=True)
        Rt = np.ascontiguousarray((U @ Vt).astype(np.float32))
    Y = np.ascontiguousarray(train @ Rt, dtype=np.float32)
    cl = train_codebook(Y, co, seed=seed, niter=KM_ITER)
    return Rt, cl, hist


def run(name, dim, M, dt, base):
    print(f"\n{'='*72}\n=== {name}  dim={dim} M={M} {np.dtype(dt).name}\n{'='*72}",
          flush=True)
    t0 = time.time()
    ds0, rem = divmod(dim, M)
    sizes = [ds0 + 1] * rem + [ds0] * (M - rem)
    co = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    with open(base, "rb") as f:
        n, d = np.fromfile(f, np.int32, 2)
    mm = np.memmap(base, dtype=dt, mode="r", offset=8, shape=(int(n), int(d)))
    tr_idx = np.arange(GID_LO, GID_LO + GID_N, TRAIN_STRIDE)[:TRAIN_SAMPLE]
    hv_idx = np.arange(GID_LO + 1, GID_LO + GID_N, HELD_STRIDE)[:HELD_N]
    train = np.ascontiguousarray(mm[tr_idx], dtype=np.float32)
    held = np.ascontiguousarray(mm[hv_idx], dtype=np.float32)
    print(f"[data] train={train.shape} held={held.shape} ({time.time()-t0:.0f}s)",
          flush=True)

    # arm A: plain PQ (no rotation) — the identity baseline
    cl_p = train_codebook(train, co, seed=42, niter=20)
    codes_p, _ = encode_and_qe(held, cl_p, co)
    e_pq = sq_err(held, decode(codes_p, cl_p, co))

    # arm B: shipped random-init OPQ rotation, re-trained PQ (isolates the rotation)
    R_rand = np.load(f"{ART}/{name}/pq/M{M}_opq/opq_rotation.npy").astype(np.float32)
    Rt_rand = np.ascontiguousarray(R_rand.T)
    held_r = np.ascontiguousarray(held @ Rt_rand, dtype=np.float32)
    cl_r = train_codebook(np.ascontiguousarray(train @ Rt_rand, dtype=np.float32),
                          co, seed=42, niter=20)
    codes_r, _ = encode_and_qe(held_r, cl_r, co)
    e_rand = sq_err(held_r, decode(codes_r, cl_r, co))

    # arm C: identity-init OPQ
    t1 = time.time()
    Rt_id, cl_id, hist = opq_identity_init(train, co)
    held_i = np.ascontiguousarray(held @ Rt_id, dtype=np.float32)
    codes_i, _ = encode_and_qe(held_i, cl_id, co)
    e_id = sq_err(held_i, decode(codes_i, cl_id, co))
    dev = float(np.abs(Rt_id - np.eye(dim, dtype=np.float32)).max())
    print(f"[identity-OPQ] {OUTER} outer iters in {time.time()-t1:.0f}s; "
          f"train-obj {hist[0]:.2f} -> {hist[-1]:.2f}; |Rt - I|max={dev:.3f}",
          flush=True)

    print(f"\n[held-out] mean ||x - x_hat||^2 (native units)")
    print(f"    plain PQ (identity)          {e_pq:12.4f}   1.0000")
    print(f"    OPQ, shipped RANDOM init     {e_rand:12.4f}   {e_rand/e_pq:.4f}")
    print(f"    OPQ, IDENTITY init           {e_id:12.4f}   {e_id/e_pq:.4f}", flush=True)
    return dict(name=name, e_pq=e_pq, e_rand=e_rand, e_id=e_id)


def main():
    only = sys.argv[1:] or None
    out = [run(*s) for s in DS if not only or s[0] in only]
    print(f"\n\n{'='*72}\n=== SUMMARY: held-out recon error relative to plain PQ\n{'='*72}")
    print(f"{'dataset':12s} {'random-init OPQ':>18s} {'identity-init OPQ':>20s}")
    for r in out:
        print(f"{r['name']:12s} {r['e_rand']/r['e_pq']:18.4f} {r['e_id']/r['e_pq']:20.4f}")


if __name__ == "__main__":
    main()
