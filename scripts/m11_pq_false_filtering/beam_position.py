#!/usr/bin/env python
"""Summarize false-drop position within the beam from M11 drop_samples.npz.

For each dropped candidate the M11 harness recorded exact_dist/d_worst
(exact_over_worst). A *false* drop has ratio <= 1.0 (it would have entered the
beam). Where in [0,1] it sits is a proxy for its rank: ~1.0 = back of the beam
(rank ~L, marginal); << 1.0 = much closer than the worst beam element
(potentially a true top neighbour, harmful). Complements the GT-rank
decomposition in rank_decomp.py.
"""
import argparse, json, os, numpy as np

def summarize(path):
    z = np.load(path); d = {}
    for a in ["000", "005", "010", "015", "020", "025"]:
        k = f"a{a}_exact_over_worst"
        if k not in z:
            continue
        e = z[k]; e = e[np.isfinite(e)]; f = e[e <= 1.0]
        if not len(f):
            continue
        buckets = {f"{lo:.2f}-{hi:.2f}": int(((f > lo) & (f <= hi)).sum())
                   for lo, hi in [(0.99, 1.0), (0.95, 0.99), (0.9, 0.95), (0.7, 0.9), (0.0, 0.7)]}
        d[f"alpha_{a}"] = {
            "n_drops_sampled": int(len(e)), "n_false_drops": int(len(f)),
            "false_frac_of_drops_pct": round(100 * len(f) / len(e), 3),
            "median_exact_over_worst": round(float(np.median(f)), 4),
            "frac_ratio_gt_0.70_pct(marginal)": round(100 * (f > 0.70).mean(), 2),
            "frac_ratio_le_0.70_pct(genuinely_close)": round(100 * (f <= 0.70).mean(), 2),
            "buckets_exact_over_worst": buckets,
        }
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="tag=path/to/drop_samples.npz ...")
    args = ap.parse_args()
    out = {}
    for pair in args.pairs:
        tag, path = pair.split("=", 1)
        if os.path.exists(path):
            out[tag] = summarize(path)
    res = {"note": "false-drop position within beam = exact_dist/d_worst (M11 filter_all drop_samples). "
                   "~1.0 = back of beam (rank~L, marginal); <=0.70 = much closer than worst (potentially "
                   "top-rank, harmful). See rank_decomp.py for GT-rank (top-K membership).",
           "datasets": out}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    for tag, d in out.items():
        a = d.get("alpha_015", {})
        if a:
            print(f"  {tag} α0.15: false={a['n_false_drops']} median_ratio={a['median_exact_over_worst']} "
                  f"marginal(>0.70)={a['frac_ratio_gt_0.70_pct(marginal)']}%")

if __name__ == "__main__":
    main()
