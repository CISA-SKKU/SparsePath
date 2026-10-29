#!/usr/bin/env python
"""Apples-to-apples held-out reconstruction error: shipped PQ vs OPQ v1 vs v2.

pq_meta qe_mean is an ENCODE-TIME average over the full 100M active window and
the shipped M*_active meta has no qe_mean at all, so the metas cannot be
compared across arms. This measures all three arms with the SAME held-out slice
(50K rows, offset +1 / stride 2000 — disjoint from every training grid) and the
SAME code path, which is the quantity that actually predicts pq-only recall.
READ-ONLY.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "m11_pq_false_filtering"))
from gen_degraded_codebook import load_codebook

ART = "/experiments/artifacts"
GID_LO, GID_N = 150_000_000, 100_000_000
HELD_N, HELD_STRIDE = 50_000, 2000

DS = [
    ("deep100M",   96,  24, np.float32,
     "/workspace/datasets_nfs/DEEP1B/deep300M_base.fbin", "M24_active"),
    ("spacev100M", 100, 25, np.int8,
     "/workspace/datasets_nfs/SPACEV1B/spacev300M_base.i8bin", "M25_active"),
    ("sift100M",   128, 32, np.uint8,
     "/workspace/datasets_nfs/SIFT1B/sift300M_base.bin", "M32_active"),
]


def read_codes(path, gids, M):
    with open(path, "rb") as f:
        N, Mf = np.fromfile(f, np.uint32, 2)
    assert int(Mf) == M
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(int(N), M))
    return np.ascontiguousarray(mm[gids])


def decode(codes, cl, co):
    out = np.empty((codes.shape[0], int(co[-1])), dtype=np.float32)
    for m in range(len(cl)):
        out[:, co[m]:co[m + 1]] = cl[m][codes[:, m]]
    return out


def main():
    only = sys.argv[1:] or None
    summary = []
    for name, dim, M, dt, base, fam in DS:
        if only and name not in only:
            continue
        t0 = time.time()
        with open(base, "rb") as f:
            n, d = np.fromfile(f, np.int32, 2)
        mm = np.memmap(base, dtype=dt, mode="r", offset=8, shape=(int(n), int(d)))
        hv = np.arange(GID_LO + 1, GID_LO + GID_N, HELD_STRIDE)[:HELD_N]
        held = np.ascontiguousarray(mm[hv], dtype=np.float32)

        arms = [("shipped PQ", f"{ART}/{name}/pq/{fam}", None),
                ("OPQ v1", f"{ART}/{name}/pq/M{M}_opq",
                 f"{ART}/{name}/pq/M{M}_opq/opq_rotation.npy"),
                ("OPQ v2", f"{ART}/{name}/pq/M{M}_opq_v2",
                 f"{ART}/{name}/pq/M{M}_opq_v2/opq_rotation.npy")]
        print(f"\n=== {name} (held-out {held.shape[0]} rows, {time.time()-t0:.0f}s)")
        base_e, row = None, {}
        for lab, pqdir, rotp in arms:
            cl, co = load_codebook(f"{pqdir}/pq_codebook.bin")
            codes = read_codes(f"{pqdir}/pq_all_codes.bin", hv, M)
            x = held
            if rotp:
                R = np.load(rotp).astype(np.float32)
                x = np.ascontiguousarray(held @ np.ascontiguousarray(R.T),
                                         dtype=np.float32)
            dd = x - decode(codes, cl, co)
            e = float((dd * dd).sum(1).mean())
            base_e = base_e if base_e is not None else e
            row[lab] = e
            print(f"    {lab:12s} {e:14.4f}   {e/base_e:7.4f} x shipped", flush=True)
        summary.append((name, row))

    print(f"\n{'='*64}\n{'dataset':12s} {'OPQ v1 / shipped':>18s} {'OPQ v2 / shipped':>18s}")
    for name, row in summary:
        s = row["shipped PQ"]
        print(f"{name:12s} {row['OPQ v1']/s:18.4f} {row['OPQ v2']/s:18.4f}")


if __name__ == "__main__":
    main()
