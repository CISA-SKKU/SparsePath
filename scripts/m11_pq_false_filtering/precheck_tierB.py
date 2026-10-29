#!/usr/bin/env python
"""Tier B pre-check (no graph load): validate the generator against ground truth
and confirm the degradation ladder actually spans QE past the alpha margin.

  1. encode a strided sample of the active window with the SHIPPED codebook and
     compare to the shipped pq_all_codes -> proves our encode is pipeline-faithful
     (and that shipped codes were produced by the shipped codebook).
  2. QE ladder for the noise lever -> confirms sigma pushes QE monotonically past
     the ~alpha margin so the recall-vs-QE knee will be visible.
"""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_degraded_codebook as gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--active-lo", type=int, required=True)
    ap.add_argument("--active-n", type=int, required=True)
    ap.add_argument("--nsample", type=int, default=200000)
    args = ap.parse_args()

    cl, co = gen.load_codebook(args.codebook)
    M = len(cl)
    lo = args.active_lo
    n = min(args.nsample, args.active_n)
    hi = lo + n                                    # CONTIGUOUS slab (fast sequential NFS read)
    print(f"M={M} sample={n} gids [{lo}..{hi-1}] (contiguous)", flush=True)

    mm = np.memmap(args.base, dtype=np.float32, mode="r", offset=8).reshape(-1, args.dim)
    X = np.ascontiguousarray(mm[lo:hi], dtype=np.float32)

    # (1) encode with shipped codebook, compare to shipped codes
    my_codes, qe_ship = gen.encode_and_qe(X, cl, co, batch=200000)
    cm = np.memmap(args.codes, dtype=np.uint8, mode="r", offset=8).reshape(-1, M)
    ship_codes = np.ascontiguousarray(cm[lo:hi])
    match = float((my_codes == ship_codes).mean())
    exact_row = float((my_codes == ship_codes).all(1).mean())
    print(f"[1] encode-vs-shipped: per-byte match={match*100:.3f}%  exact-row match={exact_row*100:.3f}%", flush=True)
    print(f"    shipped-codebook QE on sample = {qe_ship:.5f}", flush=True)

    # (2) QE ladder for noise lever
    print("[2] noise QE ladder (sigma_rel -> QE, QE ratio):", flush=True)
    for s in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]:
        cents = gen.degrade_noise(cl, s, seed=42)
        _, qe = gen.encode_and_qe(X, cents, co, batch=200000)
        print(f"    sigma={s:<4} QE={qe:.5f}  ratio={qe/qe_ship:.4f}", flush=True)


if __name__ == "__main__":
    main()
