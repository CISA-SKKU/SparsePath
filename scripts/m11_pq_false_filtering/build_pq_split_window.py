#!/usr/bin/env python
"""Build a PQ codebook whose TRAINING window is decoupled from its ENCODING window.

Why: the shipped `pq/M32` codebook (built by q1_quantization_pareto/build_pq_m.py)
trains on the *active* window [150M,250M) — i.e. it is a FRESH codebook — while the
older `pq/M12` trains on the *initial* window [0,100M) — i.e. STALE. Comparing them
therefore confounds two variables at once (M: 12->32, and vintage: stale->fresh).

This script builds the missing cell: **M=32 trained on the INITIAL window** ("stale-M32"),
encoding the same active window. With it the 2x2 becomes separable:

              trained on [0,100M) (stale)   trained on [150M,250M) (fresh)
  M=12        pq/M12          (have)         --
  M=32        pq/M32_stale    (this script)  pq/M32          (have)

  M effect        = M32_stale - M12       (vintage held at stale)
  vintage effect  = M32       - M32_stale (M held at 32)

Everything else (ksub, train_sample, niter, seed, chunk layout, output format) is kept
identical to build_pq_m.py so the only difference is the training window.

Example (inside danns container on worker-host):
  python build_pq_split_window.py \
    --base /workspace/datasets_nfs/DEEP1B/deep300M_base.fbin \
    --out /experiments/artifacts/deep100M/pq/M32_stale --M 32 --dim 96 \
    --train-lo 0          --train-n 100000000 \
    --enc-lo   150000000  --enc-n   100000000 \
    --pool-n 300000000 --workers 8
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_degraded_codebook import (KSUB, write_codebook, write_codes,   # noqa: E402
                                   train_codebook, encode_and_qe)

_G = {}


def _encode_slab(rng):
    s, e = rng
    xb = np.ascontiguousarray(_G["mm"][_G["lo"] + s:_G["lo"] + e], dtype=np.float32)
    c, qe = encode_and_qe(xb, _G["cl"], _G["co"])
    return s, c, qe * (e - s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--train-lo", type=int, required=True, help="codebook TRAINING window start gid")
    ap.add_argument("--train-n", type=int, required=True, help="codebook TRAINING window size")
    ap.add_argument("--enc-lo", type=int, required=True, help="ENCODING window start gid")
    ap.add_argument("--enc-n", type=int, required=True, help="ENCODING window size")
    ap.add_argument("--pool-n", type=int, required=True, help="codes header N (pool size)")
    ap.add_argument("--train-sample", type=int, default=200000)
    ap.add_argument("--niter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dtype", default="float32", choices=["float32", "int8", "uint8"])
    a = ap.parse_args()

    # identical chunk layout rule to build_pq_m.py (larger chunks first when dim % M != 0)
    ds0, rem = divmod(a.dim, a.M)
    sizes = [ds0 + 1] * rem + [ds0] * (a.M - rem)
    chunk_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    np_dtype = {"float32": np.float32, "int8": np.int8, "uint8": np.uint8}[a.dtype]
    with open(a.base, "rb") as f:
        n, d = np.fromfile(f, np.int32, 2)
    n, d = int(n), int(d)
    if d != a.dim:
        sys.exit(f"base dim {d} != --dim {a.dim}")
    for name, lo, cnt in (("train", a.train_lo, a.train_n), ("enc", a.enc_lo, a.enc_n)):
        if lo + cnt > n:
            sys.exit(f"{name} window [{lo},{lo+cnt}) exceeds base n={n}")
    mm = np.memmap(a.base, dtype=np_dtype, mode="r", offset=8, shape=(n, d))

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # 1) train on a strided sample of the TRAINING window (may differ from encode window)
    stride = max(1, a.train_n // a.train_sample)
    idx = np.arange(a.train_lo, a.train_lo + a.train_n, stride)[:a.train_sample]
    train = np.ascontiguousarray(mm[idx], dtype=np.float32)
    print(f"[train] window=[{a.train_lo},{a.train_lo+a.train_n}) sample={train.shape} "
          f"stride={stride} niter={a.niter} seed={a.seed}", flush=True)
    cl = train_codebook(train, chunk_offsets, seed=a.seed, niter=a.niter)
    write_codebook(os.path.join(a.out, "pq_codebook.bin"), cl, chunk_offsets)
    print(f"[train] codebook written ({time.time()-t0:.0f}s)", flush=True)

    # 2) encode the ENCODING window
    codes = np.empty((a.enc_n, a.M), dtype=np.uint8)
    qe_acc = 0.0
    if a.workers <= 1:
        slab = 10_000_000
        for s in range(0, a.enc_n, slab):
            e = min(s + slab, a.enc_n)
            xb = np.ascontiguousarray(mm[a.enc_lo + s:a.enc_lo + e], dtype=np.float32)
            c, qe = encode_and_qe(xb, cl, chunk_offsets)
            codes[s:e] = c
            qe_acc += qe * (e - s)
            print(f"[encode] {e/1e6:.0f}M/{a.enc_n/1e6:.0f}M ({time.time()-t0:.0f}s)", flush=True)
    else:
        global _G
        _G = dict(mm=mm, cl=cl, co=chunk_offsets, lo=a.enc_lo)
        slab = 2_000_000
        tasks = [(s, min(s + slab, a.enc_n)) for s in range(0, a.enc_n, slab)]
        done = 0
        with Pool(a.workers) as pool:
            for s, c, qe_w in pool.imap_unordered(_encode_slab, tasks):
                codes[s:s + c.shape[0]] = c
                qe_acc += qe_w
                done += c.shape[0]
                print(f"[encode/{a.workers}w] {done/1e6:.0f}M/{a.enc_n/1e6:.0f}M "
                      f"({time.time()-t0:.0f}s)", flush=True)
    write_codes(os.path.join(a.out, "pq_all_codes.bin"), codes, a.enc_lo, a.pool_n)

    meta = dict(num_chunks=a.M, chunk_offsets=chunk_offsets.tolist(), ksub=KSUB,
                dim=a.dim, total_vectors=a.pool_n,
                encoded_gid_range=[a.enc_lo, a.enc_lo + a.enc_n],
                trained_gid_range=[a.train_lo, a.train_lo + a.train_n],
                trained_on="strided sample of an EXPLICIT training window "
                           "(decoupled from the encoding window)",
                train_sample=a.train_sample, niter=a.niter, seed=a.seed,
                qe_mean=qe_acc / a.enc_n,
                note="built by build_pq_split_window.py (M11/M08 stale-vs-fresh isolation); "
                     "codes cover ONLY encoded_gid_range")
    with open(os.path.join(a.out, "pq_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {a.out} qe_mean={meta['qe_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
