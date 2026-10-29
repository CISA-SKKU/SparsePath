#!/usr/bin/env python
"""Diagnostic: decompose PQ-filter false-drops by neighbor RANK.

Question: when the filter false-drops a candidate (exact <= d_worst, i.e. it
would have entered the beam), is that candidate a genuine top-K neighbor
(harmful to recall@K) or a marginal one (rank > K, harmless)?

The harness already records per drop: drop_qidx, drop_gid, drop_exact,
drop_worst (no rebuild needed). We classify each *false* drop by whether its
gid is in the query's ground-truth top-K, and whether it was recovered in the
final result anyway. Run for the shipped codebook AND a degraded one to show
the extra false-drops under degradation stay marginal.

Runs single-node; graph loaded once, PQ swapped per codebook.
"""
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "lib")); sys.path.insert(0, _HERE)
import pq_search_bindings as pqsb                      # noqa: E402
import gen_degraded_codebook as gen                    # noqa: E402
from run_m11 import load_bin, load_gt                  # noqa: E402
import yaml                                            # noqa: E402

_HARNESS = {"float32": pqsb.PQSearchHarnessFloat, "float": pqsb.PQSearchHarnessFloat,
            "uint8": pqsb.PQSearchHarnessUInt8, "int8": pqsb.PQSearchHarnessInt8}
_NPD = {"float32": np.float32, "float": np.float32, "uint8": np.uint8, "int8": np.int8}


def classify(h, queries, gt, coord, partition_of, K, L, cw, ct, id_offset, alpha, max_drops, label, save_npz=None):
    h.set_alpha(alpha)
    r = h.search_batch(queries, K=K, L_search=L, conv_window=cw, conv_threshold_pct=ct,
                       filter_on=True, record_drops=True, max_drops=max_drops,
                       partition_of=partition_of, coordinator=coord)
    qidx = np.asarray(r["drop_qidx"]); gid = np.asarray(r["drop_gid"])
    exact = np.asarray(r["drop_exact"], np.float64); worst = np.asarray(r["drop_worst"], np.float64)
    res = np.asarray(r["result_tags"]).astype(np.int64) - 1 + id_offset      # (Nq,K) final gids
    fm = exact <= worst                                                       # false-drops
    fq, fg, fe, fw = qidx[fm], gid[fm], exact[fm], worst[fm]
    n = len(fg)
    # is each false-dropped gid a true top-K neighbor of its query?
    gtsets = [set(int(x) for x in gt[q, :K]) for q in range(gt.shape[0])]
    ressets = [set(int(x) for x in res[q]) for q in range(res.shape[0])]
    in_gt = np.fromiter((fg[i] in gtsets[fq[i]] for i in range(n)), bool, n)
    in_res = np.fromiter((fg[i] in ressets[fq[i]] for i in range(n)), bool, n)
    harmful = in_gt & (~in_res)                     # true top-K, dropped, and NOT recovered
    recovered = in_gt & in_res                      # true top-K, dropped once, found via other path
    ratio = np.divide(fe, fw, out=np.ones_like(fe), where=fw > 0)
    if save_npz is not None:
        # subsample for a compact, plottable distribution of exact/d_worst
        cap = 600000
        sel = (np.linspace(0, n - 1, cap).astype(np.int64) if n > cap else np.arange(n))
        np.savez_compressed(save_npz, ratio=ratio[sel].astype(np.float32), in_gt=in_gt[sel])
        print(f"  saved distribution -> {save_npz} (n={len(sel)})", flush=True)
    out = {
        "label": label, "alpha": alpha, "L": L,
        "n_drops": int(len(qidx)), "n_false": int(n),
        "false_in_gtK": int(in_gt.sum()), "false_in_gtK_pct": round(100 * in_gt.mean(), 3) if n else 0,
        "false_harmful_lost": int(harmful.sum()), "false_harmful_pct": round(100 * harmful.mean(), 3) if n else 0,
        "false_recovered": int(recovered.sum()),
        "false_marginal_pct": round(100 * (~in_gt).mean(), 3) if n else 0,
        "median_exact_over_worst": round(float(np.median(ratio)), 4) if n else 0,
        "exact_over_worst_ge_0.9_pct": round(100 * (ratio >= 0.9).mean(), 2) if n else 0,
        "median_ratio_marginal": round(float(np.median(ratio[~in_gt])), 4) if (~in_gt).any() else None,
        "median_ratio_in_gtK": round(float(np.median(ratio[in_gt])), 4) if in_gt.any() else None,
    }
    print(f"\n[{label}] α={alpha} L={L}: drops={out['n_drops']} false(exact<=worst)={n}", flush=True)
    print(f"  false-drops that are TRUE top-{K} neighbors : {out['false_in_gtK']} ({out['false_in_gtK_pct']}%)", flush=True)
    print(f"    of those, recovered via other path        : {out['false_recovered']}", flush=True)
    print(f"    -> actually HARMFUL (top-{K}, lost)        : {out['false_harmful_lost']} ({out['false_harmful_pct']}% of false)", flush=True)
    print(f"  false-drops that are MARGINAL (rank>{K})     : {n-out['false_in_gtK']} ({out['false_marginal_pct']}%)", flush=True)
    print(f"  median exact/d_worst  all={out['median_exact_over_worst']} "
          f"marginal={out['median_ratio_marginal']} in-top{K}={out['median_ratio_in_gtK']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--config-name", default=None)
    ap.add_argument("--alpha", type=float, default=0.15); ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--degrade-sigma", type=float, default=0.8, help="also run one noise-degraded codebook")
    ap.add_argument("--max-drops", type=int, default=8_000_000)
    ap.add_argument("--tmp", default="/tmp/tierB_tmp"); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["datasets"][args.dataset]
    dtype = cfg["dtype"]; npd = _NPD[dtype]; Harness = _HARNESS[dtype]
    dim = int(cfg["dim"]); K = int(cfg.get("K", 10))
    cw = int(cfg.get("conv_window", 3)); ct = float(cfg.get("conv_threshold_pct", 1.0))
    id_offset = int(cfg.get("id_offset", 0)); max_cap = int(cfg["max_capacity"])
    max_points = int(cfg.get("max_points", max_cap))
    active_lo = int(cfg.get("active_gid_lo", 0)); active_n = int(cfg.get("active_n", max_points))
    active_hi = active_lo + active_n
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.tmp, exist_ok=True)

    h = Harness(dim=dim, max_points=max_points, R=int(cfg.get("R", 64)),
                L_build=int(cfg.get("L_build", 100)), num_threads=int(cfg.get("num_threads", 0)))
    t0 = time.time(); h.load_index(cfg["graph_file"], cfg["base_file"], cfg["tags_file"], search_L=args.L)
    print(f"graph loaded ({time.time()-t0:.1f}s)", flush=True)
    queries = load_bin(cfg["query_file"], npd); gt = load_gt(cfg["gt_file"])

    parts = cfg.get("partitions", [])
    ps = next((p for p in parts if p["name"] == args.config_name), parts[0])
    pid = np.load(ps["path"]).astype(np.uint32); Mpart = int(ps["M"])
    mm = np.memmap(cfg["base_file"], dtype=npd, mode="r", offset=8).reshape(-1, dim)
    stride = int(cfg.get("centroid_sample_stride", 97))
    idx = np.arange(active_lo, active_hi, stride, dtype=np.int64)
    sub = np.asarray(mm[idx], np.float64); pp = pid[idx]
    C = np.zeros((Mpart, dim));
    for p in range(Mpart):
        msk = pp == p
        if msk.any(): C[p] = sub[msk].mean(0)
    C = C.astype(np.float32); q = queries.astype(np.float32)
    d = (q*q).sum(1)[:, None] - 2.0*(q@C.T) + (C*C).sum(1)[None, :]
    coord = d.argmin(1).astype(np.uint32)
    print(f"config[{ps['name']}] routed-hist={np.bincount(coord, minlength=Mpart).tolist()}", flush=True)

    rows = []
    # pass 1: shipped codebook
    h.load_pq(cfg["pq_codebook"], cfg["pq_codes"], max_capacity=max_cap, alpha=args.alpha)
    rows.append(classify(h, queries, gt, coord, pid, K, args.L, cw, ct, id_offset, args.alpha, args.max_drops,
                         "shipped", save_npz=os.path.join(args.out, "rank_ratios_shipped.npz")))

    # pass 2: degraded (noise sigma) codebook — single encode (qe ratio already known from Tier B)
    if args.degrade_sigma > 0:
        base_slice = np.ascontiguousarray(mm[active_lo:active_hi], dtype=np.float32)
        cl, co = gen.load_codebook(cfg["pq_codebook"])
        cents = gen.degrade_noise(cl, args.degrade_sigma, seed=42)
        codes, qe = gen.encode_and_qe(base_slice, cents, co)
        cb, cf = os.path.join(args.tmp, "rd_cb.bin"), os.path.join(args.tmp, "rd_codes.bin")
        gen.write_codebook(cb, cents, co); gen.write_codes(cf, codes, active_lo, active_hi)
        h.load_pq(cb, cf, max_capacity=max_cap, alpha=args.alpha)
        print(f"\n(degraded noise σ={args.degrade_sigma}: QE={qe:.5f})", flush=True)
        rows.append(classify(h, queries, gt, coord, pid, K, args.L, cw, ct, id_offset, args.alpha, args.max_drops,
                             f"noise_{args.degrade_sigma}",
                             save_npz=os.path.join(args.out, f"rank_ratios_noise_{args.degrade_sigma}.npz")))
        for p in (cb, cf):
            if os.path.exists(p): os.remove(p)

    with open(os.path.join(args.out, f"rank_decomp_{args.dataset}.json"), "w") as f:
        json.dump({"dataset": args.dataset, "config": ps["name"], "K": K, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}/rank_decomp_{args.dataset}.json", flush=True)


if __name__ == "__main__":
    main()
