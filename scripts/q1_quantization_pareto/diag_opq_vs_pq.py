#!/usr/bin/env python
"""Diagnose the OPQ D/4 regression on spacev/sift (2026-08-21). READ-ONLY.

deep100M (float32) gained ~+3pp pq-only from OPQ; spacev100M (int8) and
sift100M (uint8) LOST 3.0-3.5pp at every L. raw recall is bit-identical, so the
exact path is untouched — the question is whether the OPQ codes are genuinely
worse (protocol / intrinsic) or whether the harness reads them wrong
(rotation direction / dtype / LUT bug).

Three tests, all on a HELD-OUT slice of the active window:

 (1) reconstruction error  E[||x - x_hat||^2] in NATIVE units, four arms:
       shipped   : M*_active codebook + its stored codes      (train 1M, niter 15)
       opq       : M*_opq codebook + stored codes, un-rotated (train 200K, niter 20)
       ctrl_pq200k: plain PQ freshly trained on the EXACT sample the OPQ used
                    -> isolates the ROTATION effect from the TRAIN-SIZE effect
       opq_reenc : re-encode x@R.T with the shipped OPQ codebook
                    -> if this differs from `opq`, the stored codes are corrupt
 (2) distance quality: LUT-estimated vs exact distance on real queries
     (bias / MAE / Spearman / top-10 overlap), plain PQ vs OPQ. Also asserts
     the OPQ LUT distance equals ||q - x_hat_original_frame||^2, which is the
     decisive test of the rotation DIRECTION and of the int8/uint8 promotion.
 (3) rotation + dtype sanity: orthonormality, (x@R.T)@R == x, query value
     ranges (uint8 > 127 must survive promotion unsigned).

Run inside the danns container (faiss + /workspace/datasets_nfs mounts).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "m11_pq_false_filtering"))
from gen_degraded_codebook import load_codebook, train_codebook, encode_and_qe

ART = "/experiments/artifacts"
GID_LO, GID_N = 150_000_000, 100_000_000
TRAIN_SAMPLE, TRAIN_STRIDE = 200_000, 500      # exactly what build_pq_m --opq used
HELD_N, HELD_STRIDE = 50_000, 2000             # disjoint from the training grid
NQ, NCAND = 1000, 5000

DS = [
    # name, dim, M, dtype, base, active_family, query_npy
    ("deep100M",   96,  24, np.float32,
     "/workspace/datasets_nfs/DEEP1B/deep300M_base.fbin", "M24_active",
     f"{ART}/deep100M/gt/gt_iteration150_K10_Q10000_queries.npy"),
    ("spacev100M", 100, 25, np.int8,
     "/workspace/datasets_nfs/SPACEV1B/spacev300M_base.i8bin", "M25_active",
     f"{ART}/spacev100M/gt/gt_iteration150_K10_Q10000_queries.npy"),
    ("sift100M",   128, 32, np.uint8,
     "/workspace/datasets_nfs/SIFT1B/sift300M_base.bin", "M32_active",
     f"{ART}/sift100M/gt/gt_iteration150_K10_Q10000_queries.npy"),
]


def read_codes(path, gids, M):
    with open(path, "rb") as f:
        N, Mf = np.fromfile(f, np.uint32, 2)
    assert int(Mf) == M, f"{path}: M={Mf} != {M}"
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(int(N), M))
    return np.ascontiguousarray(mm[gids])


def decode(codes, cl, co):
    n = codes.shape[0]
    dim = int(co[-1])
    out = np.empty((n, dim), dtype=np.float32)
    for m in range(len(cl)):
        out[:, co[m]:co[m + 1]] = cl[m][codes[:, m]]
    return out


def sq_err(x, xhat):
    d = x - xhat
    return float((d * d).sum(1).mean())


def lut_dists(q, codes, cl, co):
    """q (nq,dim) float32, codes (nc,M) -> (nq,nc) PQ-estimated squared L2."""
    nq, nc = q.shape[0], codes.shape[0]
    acc = np.zeros((nq, nc), dtype=np.float32)
    for m in range(len(cl)):
        qm = q[:, co[m]:co[m + 1]]                       # (nq, ds)
        c = cl[m]                                        # (256, ds)
        lut = ((qm * qm).sum(1)[:, None]
               - 2.0 * (qm @ c.T)
               + (c * c).sum(1)[None, :]).astype(np.float32)   # (nq,256)
        acc += lut[:, codes[:, m]]
    return acc


def spearman(a, b):
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))
    return float((num / np.maximum(den, 1e-12)).mean())


def topk_overlap(est, exact, k=10):
    ie = np.argsort(est, axis=1)[:, :k]
    ix = np.argsort(exact, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ie, ix)]))


def run(name, dim, M, dt, base, fam, qpath):
    print(f"\n{'='*78}\n=== {name}  dim={dim} M={M} dtype={np.dtype(dt).name}\n{'='*78}",
          flush=True)
    t0 = time.time()
    opq_dir = f"{ART}/{name}/pq/M{M}_opq"
    act_dir = f"{ART}/{name}/pq/{fam}"

    R = np.load(f"{opq_dir}/opq_rotation.npy").astype(np.float32)
    Rt = np.ascontiguousarray(R.T)

    with open(base, "rb") as f:
        n, d = np.fromfile(f, np.int32, 2)
    assert int(d) == dim
    mm = np.memmap(base, dtype=dt, mode="r", offset=8, shape=(int(n), int(d)))

    # ---- (3a) rotation sanity -------------------------------------------------
    orth = float(np.abs(R @ R.T - np.eye(dim, dtype=np.float32)).max())
    probe = np.ascontiguousarray(mm[GID_LO:GID_LO + 256], dtype=np.float32)
    roundtrip = float(np.abs((probe @ Rt) @ R - probe).max())
    scale = float(np.abs(probe).max())
    print(f"[rot] orth_err={orth:.3e}  |(x@R.T)@R - x|max={roundtrip:.3e} "
          f"(data |x|max={scale:.1f})", flush=True)

    # ---- samples --------------------------------------------------------------
    tr_idx = np.arange(GID_LO, GID_LO + GID_N, TRAIN_STRIDE)[:TRAIN_SAMPLE]
    hv_idx = np.arange(GID_LO + 1, GID_LO + GID_N, HELD_STRIDE)[:HELD_N]
    assert len(np.intersect1d(tr_idx, hv_idx)) == 0, "held-out overlaps training"
    train = np.ascontiguousarray(mm[tr_idx], dtype=np.float32)
    held = np.ascontiguousarray(mm[hv_idx], dtype=np.float32)
    print(f"[data] train={train.shape} held={held.shape} ({time.time()-t0:.0f}s)",
          flush=True)

    # energy scale for relative numbers
    var = float(((held - held.mean(0)) ** 2).sum(1).mean())

    # ---- (1) reconstruction error --------------------------------------------
    cl_a, co_a = load_codebook(f"{act_dir}/pq_codebook.bin")
    cl_o, co_o = load_codebook(f"{opq_dir}/pq_codebook.bin")
    assert list(co_a) == list(co_o), f"chunk_offsets differ: {co_a} vs {co_o}"

    codes_a = read_codes(f"{act_dir}/pq_all_codes.bin", hv_idx, M)
    codes_o = read_codes(f"{opq_dir}/pq_all_codes.bin", hv_idx, M)
    print(f"[codes] read ({time.time()-t0:.0f}s)", flush=True)

    e_shipped = sq_err(held, decode(codes_a, cl_a, co_a))

    held_rot = np.ascontiguousarray(held @ Rt, dtype=np.float32)
    dec_o = decode(codes_o, cl_o, co_o)                     # in rotated frame
    e_opq_rot = sq_err(held_rot, dec_o)                     # rotated-frame error
    e_opq = sq_err(held, np.ascontiguousarray(dec_o @ R))   # original-frame error

    # re-encode: are the STORED opq codes the optimal assignment?
    codes_o2, _ = encode_and_qe(held_rot, cl_o, co_o)
    e_opq_re = sq_err(held_rot, decode(codes_o2, cl_o, co_o))
    agree = float((codes_o2 == codes_o).mean())

    # control: plain PQ trained on the EXACT sample the OPQ rotation+PQ used
    cl_c = train_codebook(train, np.asarray(co_o), seed=42, niter=20)
    codes_c, _ = encode_and_qe(held, cl_c, co_o)
    e_ctrl = sq_err(held, decode(codes_c, cl_c, co_o))

    # control: OPQ codebook trained on 1M-equivalent is not available; instead
    # report plain PQ on 1M (= shipped) vs plain PQ on 200K to size the train
    # effect, and OPQ-200K vs PQ-200K to size the rotation effect.
    print(f"\n[recon] mean ||x - x_hat||^2 (native units; data variance={var:.4g})")
    for lab, v in [("shipped PQ  (1M, niter15)", e_shipped),
                   ("ctrl PQ     (200K, niter20)", e_ctrl),
                   ("OPQ         (200K, niter20)", e_opq),
                   ("OPQ re-encoded (same cb)", e_opq_re)]:
        print(f"    {lab:30s} {v:14.4f}   ({100.0*v/var:6.3f}% of variance)")
    print(f"    stored-vs-recomputed OPQ code agreement: {100*agree:.2f}%")
    print(f"    [check] OPQ rotated-frame err {e_opq_rot:.4f} vs "
          f"original-frame err {e_opq:.4f}  (must match)")
    print(f"\n    train-size effect  (ctrl200K / shipped1M) = {e_ctrl/e_shipped:6.4f}"
          f"   {'(200K is worse)' if e_ctrl>e_shipped else '(200K is better)'}")
    print(f"    ROTATION effect    (OPQ200K / ctrl200K)   = {e_opq/e_ctrl:6.4f}"
          f"   {'(rotation HURTS)' if e_opq>e_ctrl else '(rotation helps)'}")
    print(f"    net vs shipped     (OPQ200K / shipped1M)  = {e_opq/e_shipped:6.4f}",
          flush=True)

    # ---- (2) distance quality on real queries --------------------------------
    q_raw = np.load(qpath)
    print(f"\n[query] {qpath}")
    print(f"    stored dtype={q_raw.dtype} shape={q_raw.shape} "
          f"min={q_raw.min()} max={q_raw.max()}")
    q_native = q_raw.astype(dt)          # exactly what run_q1.load_bin does
    if not np.array_equal(q_native.astype(np.float64), q_raw.astype(np.float64)):
        print("    !! WARNING: astype(dtype) is LOSSY for this query file")
    q = q_native[:NQ].astype(np.float32)
    print(f"    as float32 -> min={q.min()} max={q.max()} "
          f"(n>127: {int((q>127).sum())})")

    cand = held[:NCAND]
    ca = codes_a[:NCAND]
    co_codes = codes_o[:NCAND]
    exact = ((q * q).sum(1)[:, None] - 2.0 * (q @ cand.T)
             + (cand * cand).sum(1)[None, :]).astype(np.float32)

    d_pq = lut_dists(q, ca, cl_a, co_a)
    q_rot = np.ascontiguousarray(q @ Rt, dtype=np.float32)
    d_opq = lut_dists(q_rot, co_codes, cl_o, co_o)

    # decisive: the OPQ LUT distance must equal the original-frame distance to the
    # reconstructed vector. If the rotation direction or the dtype promotion were
    # wrong, this identity breaks.
    xhat_o = np.ascontiguousarray(decode(co_codes, cl_o, co_o) @ R)
    d_opq_orig = ((q * q).sum(1)[:, None] - 2.0 * (q @ xhat_o.T)
                  + (xhat_o * xhat_o).sum(1)[None, :]).astype(np.float32)
    rel = float(np.abs(d_opq - d_opq_orig).max() / max(1e-9, float(np.abs(d_opq_orig).max())))
    print(f"\n[lut] identity check  max|LUT(R q) - ||q - x_hat||^2| / scale = {rel:.3e}"
          f"   {'OK' if rel < 1e-3 else '*** MISMATCH ***'}")

    print(f"\n[dist] estimate vs exact over {NQ} queries x {NCAND} candidates")
    print(f"    {'arm':6s} {'bias':>12s} {'MAE':>12s} {'rel MAE':>9s} "
          f"{'spearman':>9s} {'top10 ovl':>10s}")
    for lab, est in [("PQ", d_pq), ("OPQ", d_opq)]:
        bias = float((est - exact).mean())
        mae = float(np.abs(est - exact).mean())
        print(f"    {lab:6s} {bias:12.3f} {mae:12.3f} "
              f"{mae/float(exact.mean()):9.4f} {spearman(est, exact):9.4f} "
              f"{topk_overlap(est, exact):10.4f}")
    print(f"    (exact mean distance = {float(exact.mean()):.3f})", flush=True)

    return dict(name=name, e_shipped=e_shipped, e_ctrl=e_ctrl, e_opq=e_opq,
                rot_effect=e_opq / e_ctrl, train_effect=e_ctrl / e_shipped,
                net=e_opq / e_shipped)


def main():
    only = sys.argv[1:] or None
    out = []
    for spec in DS:
        if only and spec[0] not in only:
            continue
        out.append(run(*spec))
    print(f"\n\n{'='*78}\n=== SUMMARY (reconstruction error ratios; >1 = worse)\n{'='*78}")
    print(f"{'dataset':12s} {'train(200K/1M)':>16s} {'ROTATION':>12s} {'net(OPQ/ship)':>15s}")
    for r in out:
        print(f"{r['name']:12s} {r['train_effect']:16.4f} {r['rot_effect']:12.4f} "
              f"{r['net']:15.4f}")


if __name__ == "__main__":
    main()
