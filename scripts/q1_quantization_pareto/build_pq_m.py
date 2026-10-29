#!/usr/bin/env python
"""Build a fresh PQ codebook + gid-indexed codes at a given M (A1-M sweep, M08).

Reuses the byte-compatible I/O + faiss per-chunk k-means from the Tier-B
generator (gen_degraded_codebook.py — verified byte-identical to the shipped
pipeline formats). Trains on a strided sample of the ACTIVE gid window and
encodes only that window (zero-padded to the full pool N in the codes header,
same convention the harness load_pq expects).

Example (inside danns container):
  python build_pq_m.py --base /workspace/datasets_nfs/DEEP1B/deep300M_base.fbin \
      --out /experiments/artifacts/deep100M/pq/M32 --M 32 --dim 96 \
      --gid-lo 150000000 --gid-n 100000000 --pool-n 300000000

OPQ variant (--opq): learns a faiss OPQMatrix(dim, M) rotation on the same train
sample, writes opq_rotation.npy + opq_rotation_meta.json next to the codebook, and
trains/encodes PQ in the rotated frame. The base is rotated CHUNK-WISE IN MEMORY —
no rotated base file is ever written to disk. The search harness applies the
matching QUERY-side rotation (search_batch(opq_rotation=...)), which is exact
because R is orthogonal.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "m11_pq_false_filtering"))
from gen_degraded_codebook import (KSUB, write_codebook, write_codes,
                                   train_codebook, encode_and_qe)


_G = {}


def _encode_slab(rng):
    s, e = rng
    xb = np.ascontiguousarray(_G["mm"][_G["lo"] + s:_G["lo"] + e], dtype=np.float32)
    if _G.get("Rt") is not None:                    # OPQ: rotate in memory, never on disk
        xb = np.ascontiguousarray(xb @ _G["Rt"], dtype=np.float32)
    c, qe = encode_and_qe(xb, _G["cl"], _G["co"])
    return s, c, qe * (e - s)


def _decode(codes, cl, co):
    out = np.empty((codes.shape[0], int(co[-1])), dtype=np.float32)
    for m in range(len(cl)):
        out[:, co[m]:co[m + 1]] = cl[m][codes[:, m]]
    return out


def _recon_err(x, cl, co):
    """Held-out mean ||x - x_hat||^2 under the given codebook."""
    codes, _ = encode_and_qe(x, cl, co)
    d = x - _decode(codes, cl, co)
    return float((d * d).sum(1).mean())


def _opq_random(train, dim, M, niter):
    """faiss OPQMatrix — random-rotation init (the 2026-08-06 protocol)."""
    import faiss
    opq = faiss.OPQMatrix(dim, M)
    opq.niter = niter
    opq.max_train_points = max(int(train.shape[0]), 65536)
    opq.verbose = True
    t1 = time.time()
    opq.train(train)
    R = faiss.vector_to_array(opq.A).reshape(opq.d_out, opq.d_in).astype(np.float32)
    b = faiss.vector_to_array(opq.b)
    if b.size and np.abs(b).max() > 0:
        sys.exit(f"OPQMatrix returned a non-zero bias (|b|max={np.abs(b).max():g}); "
                 "the rotation-only shortcut assumes a pure linear transform")
    return R, time.time() - t1, faiss.__version__


def _opq_identity(train, co, niter, km_iter=8, seed=42):
    """Ge et al. OPQ_NP alternating minimisation, started from R = IDENTITY.

    faiss initialises from a RANDOM rotation. On datasets whose natural axis
    order already groups correlated dimensions (SIFT: 16 spatial cells x 8
    orientation bins, so a consecutive-4 chunk stays inside one cell's
    histogram) the identity is already near-optimal, and a random start lands in
    a strictly worse local optimum that more iterations do not escape — measured
    2026-08-21: random-init OPQ reconstructs 1.25x (spacev) / 1.56x (sift) WORSE
    than plain PQ, while identity init lands at 1.00x. Identity init cannot be
    worse than plain PQ by construction: iteration 0 IS plain PQ.
    Convention: y = x @ Rt, returns R = Rt.T to match the rest of this file.
    """
    dim = int(co[-1])
    Rt = np.eye(dim, dtype=np.float32)
    t1 = time.time()
    for it in range(niter):
        Y = np.ascontiguousarray(train @ Rt, dtype=np.float32)
        cl = train_codebook(Y, co, seed=seed, niter=km_iter)
        codes, _ = encode_and_qe(Y, cl, co)
        Yhat = _decode(codes, cl, co)
        # orthogonal Procrustes: min ||X @ Rt - Yhat||_F over orthogonal Rt
        U, _, Vt = np.linalg.svd(train.T @ Yhat, full_matrices=True)
        Rt = np.ascontiguousarray((U @ Vt).astype(np.float32))
        if it % 5 == 0 or it == niter - 1:
            d = Y - Yhat
            print(f"[opq/identity] iter {it:3d} train-obj={float((d*d).sum(1).mean()):.4f}",
                  flush=True)
    return np.ascontiguousarray(Rt.T), time.time() - t1


def train_opq(train, dim, M, niter, out_dir, base_path, gid_lo, gid_n,
              stride, chunk_offsets, init="random", holdout=None):
    """Learn the OPQ rotation and persist it. Returns R with y = x @ R.T.

    init: "random"   faiss OPQMatrix (2026-08-06 protocol)
          "identity" Ge et al. alternating minimisation from R = I
          "best"     train both, keep whichever reconstructs the held-out slice
                     better (OPQ is a refinement of PQ and must never lose to it)
    """
    import faiss
    co = np.asarray(chunk_offsets)
    cands, timings = {}, {}
    if init in ("random", "best"):
        R, s, _ = _opq_random(train, dim, M, niter)
        cands["random"], timings["random"] = R, s
        print(f"[opq] random-init rotation trained in {s:.0f}s", flush=True)
    if init in ("identity", "best"):
        R, s = _opq_identity(train, co, niter=min(niter, 25))
        cands["identity"], timings["identity"] = R, s
        print(f"[opq] identity-init rotation trained in {s:.0f}s", flush=True)
    if not cands:
        sys.exit(f"unknown --opq-init {init}")

    scores = {}
    if len(cands) > 1:
        if holdout is None:
            sys.exit("--opq-init best requires a held-out sample")
        base_cl = train_codebook(train, co, seed=42, niter=20)
        scores["plain_pq"] = _recon_err(holdout, base_cl, co)
        for k, R in cands.items():
            Rt = np.ascontiguousarray(R.T)
            cl = train_codebook(np.ascontiguousarray(train @ Rt, dtype=np.float32),
                                co, seed=42, niter=20)
            scores[k] = _recon_err(np.ascontiguousarray(holdout @ Rt, dtype=np.float32),
                                   cl, co)
        for k in sorted(scores):
            print(f"[opq/select] held-out recon err  {k:10s} {scores[k]:14.4f}  "
                  f"({scores[k]/scores['plain_pq']:.4f} x plain PQ)", flush=True)
        chosen = min(cands, key=lambda k: scores[k])
        if scores[chosen] > scores["plain_pq"]:
            print(f"[opq/select] WARNING: every rotation loses to plain PQ; "
                  f"keeping {chosen} anyway (report as OPQ ~= PQ)", flush=True)
    else:
        chosen = next(iter(cands))
    R = cands[chosen]
    print(f"[opq] selected init = {chosen}", flush=True)

    err = float(np.abs(R @ R.T - np.eye(dim, dtype=np.float32)).max())
    print(f"[opq] orth_err={err:.3e}", flush=True)
    if err > 1e-4:
        sys.exit("trained matrix is not orthonormal — abort")

    # sanity: the rotation preserves pairwise L2 on a held-out slice
    s = train[:2000]
    sr = s @ R.T
    d0 = ((s[:500, None, :] - s[None, 500:1000, :]) ** 2).sum(-1)
    d1 = ((sr[:500, None, :] - sr[None, 500:1000, :]) ** 2).sum(-1)
    rel = float(np.abs(d1 - d0).max() / max(1e-9, d0.max()))
    print(f"[opq] pairwise-L2 max rel err after rotation = {rel:.3e}", flush=True)
    if rel > 1e-4:
        sys.exit("rotation is not distance preserving — abort")

    rot_path = os.path.join(out_dir, "opq_rotation.npy")
    np.save(rot_path, R)
    meta = dict(dim=dim, M=M, niter=niter, gid_lo=gid_lo, gid_n=gid_n,
                train_sample=int(train.shape[0]), stride=int(stride),
                opq_init_requested=init, opq_init_selected=chosen,
                holdout_recon_err=scores or None,
                faiss_version=faiss.__version__, orth_err=err,
                pairwise_l2_rel_err=rel,
                train_seconds={k: round(v, 1) for k, v in timings.items()},
                base=base_path,
                note="OPQ rotation only; the PQ codebook below is trained on the "
                     "ROTATED sample and the codes are encoded from chunk-wise "
                     "in-memory rotation (no rotated base file on disk). Query-side "
                     "rotation is applied by the harness (search_batch opq_rotation).")
    with open(os.path.join(out_dir, "opq_rotation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[opq] -> {rot_path}", flush=True)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pool base .fbin (float32)")
    ap.add_argument("--out", required=True, help="output dir (pq/M{M})")
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--gid-lo", type=int, required=True, help="active window start gid")
    ap.add_argument("--gid-n", type=int, required=True, help="active window size")
    ap.add_argument("--pool-n", type=int, required=True, help="codes header N (pool size)")
    ap.add_argument("--train-sample", type=int, default=200000)
    ap.add_argument("--niter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel encode processes (fork). Launch with "
                         "OMP_NUM_THREADS=2 to avoid BLAS oversubscription.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "int8", "uint8"],
                    help="base file element type (converted to float32 internally)")
    ap.add_argument("--opq", action="store_true",
                    help="OPQ: learn a faiss OPQMatrix(dim, M) rotation on the train "
                         "sample, save it as opq_rotation.npy, then train/encode PQ in "
                         "the rotated frame. The base is rotated CHUNK-WISE IN MEMORY — "
                         "no rotated base file is written. int8/uint8 bases are promoted "
                         "to float32 first (OPQ is a float-space transform).")
    ap.add_argument("--opq-niter", type=int, default=50,
                    help="OPQ outer iterations (08-06 protocol used 50; the "
                         "identity-init optimiser caps at 25)")
    ap.add_argument("--opq-init", default="random",
                    choices=["random", "identity", "best"],
                    help="rotation initialisation. random = faiss OPQMatrix "
                         "(08-06 protocol). identity = Ge et al. alternating "
                         "minimisation from R=I, which cannot lose to plain PQ. "
                         "best = train both and keep the lower held-out "
                         "reconstruction error (recommended: random init is "
                         "1.25x/1.56x WORSE than plain PQ on spacev/sift).")
    ap.add_argument("--opq-train-sample", type=int, default=200000,
                    help="rows used to learn the ROTATION (a dim x dim matrix "
                         "needs far fewer rows than the codebook). Taken as an "
                         "even subsample of the codebook training set.")
    a = ap.parse_args()
    if a.opq and a.dim % a.M != 0:
        sys.exit(f"--opq requires dim % M == 0 (faiss OPQMatrix); dim={a.dim} M={a.M}")

    # uneven chunks when dim % M != 0 (larger chunks first, DiskANN-style);
    # everything downstream reads chunk_offsets from the codebook header.
    ds0, rem = divmod(a.dim, a.M)
    sizes = [ds0 + 1] * rem + [ds0] * (a.M - rem)
    chunk_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    np_dtype = {"float32": np.float32, "int8": np.int8, "uint8": np.uint8}[a.dtype]
    with open(a.base, "rb") as f:
        n, d = np.fromfile(f, np.int32, 2)
    if int(d) != a.dim or a.gid_lo + a.gid_n > int(n):
        sys.exit(f"base header n={n} d={d} incompatible with args")
    mm = np.memmap(a.base, dtype=np_dtype, mode="r", offset=8,
                   shape=(int(n), int(d)))

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # 1) train on a strided sample of the active window
    stride = max(1, a.gid_n // a.train_sample)
    idx = np.arange(a.gid_lo, a.gid_lo + a.gid_n, stride)[:a.train_sample]
    train = np.ascontiguousarray(mm[idx], dtype=np.float32)
    print(f"[train] sample={train.shape} stride={stride} niter={a.niter}", flush=True)

    # 1a) OPQ (optional): learn the rotation on the SAME sample, then move the
    # sample into the rotated frame so the PQ codebook below is an OPQ codebook.
    Rt = None
    if a.opq:
        sub = max(1, train.shape[0] // max(1, a.opq_train_sample))
        rot_train = np.ascontiguousarray(train[::sub][:a.opq_train_sample])
        holdout = None
        if a.opq_init == "best":
            # disjoint from every training grid (offset +1, coarser stride)
            hv = np.arange(a.gid_lo + 1, a.gid_lo + a.gid_n, 2000)[:50000]
            holdout = np.ascontiguousarray(mm[hv], dtype=np.float32)
            print(f"[opq] holdout={holdout.shape} for init selection", flush=True)
        print(f"[opq] rotation train rows={rot_train.shape[0]} "
              f"(subsample 1/{sub} of the codebook sample)", flush=True)
        R = train_opq(rot_train, a.dim, a.M, a.opq_niter, a.out, a.base,
                      a.gid_lo, a.gid_n, stride, chunk_offsets,
                      init=a.opq_init, holdout=holdout)
        Rt = np.ascontiguousarray(R.T)          # y = x @ R.T
        train = np.ascontiguousarray(train @ Rt, dtype=np.float32)

    cl = train_codebook(train, chunk_offsets, seed=a.seed, niter=a.niter)
    write_codebook(os.path.join(a.out, "pq_codebook.bin"), cl, chunk_offsets)
    print(f"[train] codebook written ({time.time()-t0:.0f}s)", flush=True)

    # 2) encode the active window in slabs (memmap -> encode_and_qe batches inside)
    codes = np.empty((a.gid_n, a.M), dtype=np.uint8)
    qe_acc = 0.0
    if a.workers <= 1:
        slab = 10_000_000
        for s in range(0, a.gid_n, slab):
            e = min(s + slab, a.gid_n)
            xb = np.ascontiguousarray(mm[a.gid_lo + s:a.gid_lo + e], dtype=np.float32)
            if Rt is not None:
                xb = np.ascontiguousarray(xb @ Rt, dtype=np.float32)
            c, qe = encode_and_qe(xb, cl, chunk_offsets)
            codes[s:e] = c
            qe_acc += qe * (e - s)
            print(f"[encode] {e/1e6:.0f}M/{a.gid_n/1e6:.0f}M "
                  f"({time.time()-t0:.0f}s, qe so far {qe_acc/e:.2f})", flush=True)
    else:
        # fork workers inherit mm/cl via globals; slabs returned and assembled.
        global _G
        _G = dict(mm=mm, cl=cl, co=chunk_offsets, lo=a.gid_lo, Rt=Rt)
        slab = 2_000_000
        tasks = [(s, min(s + slab, a.gid_n)) for s in range(0, a.gid_n, slab)]
        done = 0
        with Pool(a.workers) as pool:
            for s, c, qe_w in pool.imap_unordered(_encode_slab, tasks):
                codes[s:s + c.shape[0]] = c
                qe_acc += qe_w
                done += c.shape[0]
                print(f"[encode/{a.workers}w] {done/1e6:.0f}M/{a.gid_n/1e6:.0f}M "
                      f"({time.time()-t0:.0f}s)", flush=True)
    write_codes(os.path.join(a.out, "pq_all_codes.bin"), codes, a.gid_lo, a.pool_n)

    meta = dict(num_chunks=a.M, chunk_offsets=chunk_offsets.tolist(), ksub=KSUB,
                dim=a.dim, total_vectors=a.pool_n,
                encoded_gid_range=[a.gid_lo, a.gid_lo + a.gid_n],
                trained_on="strided sample of active window",
                train_sample=a.train_sample, niter=a.niter, seed=a.seed,
                qe_mean=qe_acc / a.gid_n,
                opq=bool(a.opq),
                opq_rotation=(os.path.join(a.out, "opq_rotation.npy")
                              if a.opq else None),
                opq_niter=(a.opq_niter if a.opq else None),
                opq_init=(a.opq_init if a.opq else None),
                opq_train_sample=(a.opq_train_sample if a.opq else None),
                note="A1-M sweep (M08) — built by build_pq_m.py, NOT the "
                     "pipeline stage; codes cover ONLY encoded_gid_range"
                     + (" | OPQ: codes are in the rotated frame; the harness must "
                        "be given opq_rotation (query-side rotation)" if a.opq else ""))
    with open(os.path.join(a.out, "pq_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {a.out} qe_mean={meta['qe_mean']:.2f} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
