#!/usr/bin/env python
"""Plot the false-drop distribution shift: shipped vs degraded codebook.

Reads rank_ratios_{shipped,noise_*}.npz (ratio=exact/d_worst per false-drop,
in_gt=is-true-top-K) written by rank_decomp.py, and shows that degrading the
codebook shifts false-drops toward CLOSER candidates (lower exact/d_worst) and
grows the harmful (true top-K) subset — directly answering "does a worse
codebook cause more false-drops of near candidates?".
"""
import argparse, os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--degraded", required=True)
    ap.add_argument("--summary", default=None, help="rank_decomp_*.json for true counts")
    ap.add_argument("--label-deg", default="degraded (σ0.8, QE≈1.31)")
    ap.add_argument("--density", action="store_true",
                    help="normalize to area=1 (hides count growth). default = raw COUNTS")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    dens = args.density
    ylab = "density (area=1)" if dens else "false-drop COUNT (per bin)"

    s = np.load(args.shipped); d = np.load(args.degraded)
    sr, sg = s["ratio"], s["in_gt"].astype(bool)
    dr, dg = d["ratio"], d["in_gt"].astype(bool)
    counts = {}
    if args.summary and os.path.exists(args.summary):
        j = json.load(open(args.summary))
        for r in j["rows"]:
            counts[r["label"]] = r

    bins = np.linspace(0.0, 1.0, 101)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: all false-drops, exact/d_worst — shipped vs degraded (density)
    ax[0].hist(sr, bins=bins, density=dens, alpha=0.55, color="#2c7fb8", label=f"shipped/vanilla  (n={len(sr):,}, median {np.median(sr):.3f})")
    ax[0].hist(dr, bins=bins, density=dens, alpha=0.55, color="#d95f0e", label=f"{args.label_deg}  (n={len(dr):,}, median {np.median(dr):.3f})")
    ax[0].axvline(np.median(sr), color="#2c7fb8", ls="--", lw=1)
    ax[0].axvline(np.median(dr), color="#d95f0e", ls="--", lw=1)
    ax[0].set_xlabel("exact_dist / d_worst  (1.0 = beam back = rank~L = marginal;  low = close = top-rank)")
    ax[0].set_ylabel(ylab); ax[0].set_xlim(0.4, 1.02); ax[0].legend(fontsize=9)
    ax[0].set_title("All false-drops: degraded shifts toward CLOSER (lower ratio)")

    # Panel B: true top-K (harmful) subset only — shipped vs degraded (density) + true counts
    ax[1].hist(sr[sg], bins=bins, density=dens, alpha=0.55, color="#2c7fb8", label=f"shipped/vanilla: true top-10 (n={int(sg.sum()):,})")
    ax[1].hist(dr[dg], bins=bins, density=dens, alpha=0.55, color="#d95f0e", label=f"{args.label_deg}: true top-10 (n={int(dg.sum()):,})")
    ax[1].set_xlabel("exact_dist / d_worst (harmful = true top-10 false-drops)")
    ax[1].set_ylabel(ylab); ax[1].set_xlim(0.3, 1.02); ax[1].legend(fontsize=9)
    txt = ""
    if counts:
        sh = counts.get("shipped", {}); de = counts.get("noise_0.8", {})
        if sh and de:
            txt = (f"shipped: false={sh['n_false']:,}  top-10={sh['false_in_gtK']:,} ({sh['false_in_gtK_pct']}%)\n"
                   f"degraded: false={de['n_false']:,}  top-10={de['false_in_gtK']:,} ({de['false_in_gtK_pct']}%)\n"
                   f"harmful count {sh['false_in_gtK']:,}→{de['false_in_gtK']:,}  = more errors × skew-to-close")
    ax[1].set_title("Harmful (top-10) false-drops: more, and closer, under degradation")
    if txt:
        ax[1].text(0.32, ax[1].get_ylim()[1] * 0.62, txt, fontsize=8,
                   bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    fig.suptitle("PQ-filter false-drop distribution: shipped vs degraded codebook (deep100M/LITS, α=0.15, L=100)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(args.out_dir, f"{ts}_rank_shift_deep100M_lits150_{'density' if dens else 'count'}.png")
    fig.savefig(p, dpi=130)
    print("wrote", p)


if __name__ == "__main__":
    main()
