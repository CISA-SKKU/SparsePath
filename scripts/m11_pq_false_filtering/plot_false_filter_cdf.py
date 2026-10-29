#!/usr/bin/env python
"""Clearer false-filtering CDF: overlay exact_dist/d_worst so false-filters are
read directly off the figure.

A candidate is DROPPED when pq_dist > (1+alpha)*d_worst. It is a *false* filter
iff its EXACT distance <= d_worst (it would have entered the beam). So among the
dropped candidates, the fraction with exact/d_worst <= 1.0 IS the false-filter
rate — the height of the exact-CDF at x=1.0.

Reads M11 drop_samples.npz (a{aaa}_pq_over_worst, a{aaa}_exact_over_worst).
"""
import argparse, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cdf(x):
    x = np.sort(x[np.isfinite(x)])
    return x, np.linspace(0, 1, len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--amain", default="015")
    args = ap.parse_args()
    z = np.load(args.npz)
    ts = time.strftime("%Y%m%d_%H%M%S")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Panel A: exact/d_worst CDF for dropped candidates, per alpha ----
    alphas = [("000", 0.00), ("010", 0.10), ("015", 0.15), ("020", 0.20), ("025", 0.25)]
    ff_main = None
    for tag, a in alphas:
        k = f"a{tag}_exact_over_worst"
        if k not in z:
            continue
        e = z[k]
        x, y = cdf(e)
        lw = 3.0 if tag == args.amain else 1.3
        al = 1.0 if tag == args.amain else 0.5
        ax[0].plot(x, y, lw=lw, alpha=al, label=f"α={a:.2f}" + ("  ← headline" if tag == args.amain else ""))
        if tag == args.amain:
            ff_main = float((e[np.isfinite(e)] <= 1.0).mean())
    ax[0].axvline(1.0, color="black", lw=1.5)
    ax[0].text(1.005, 0.5, "useful boundary\n(exact = d_worst)", rotation=90, va="center", fontsize=10)
    if ff_main is not None:
        # mark the false-filter fraction on the headline curve at x=1.0
        ax[0].plot([1.0], [ff_main], "o", color="crimson", ms=9, zorder=5)
        ax[0].annotate(f"false-filter = {ff_main*100:.1f}% of drops\n(exact ≤ d_worst @ α=0.15)",
                       xy=(1.0, ff_main), xytext=(1.15, ff_main + 0.22),
                       fontsize=11, color="crimson",
                       arrowprops=dict(arrowstyle="->", color="crimson"))
    ax[0].fill_between([0, 1.0], [1, 1], color="crimson", alpha=0.06)
    ax[0].set_xlim(0.55, 2.0); ax[0].set_ylim(0, 1.0)
    ax[0].set_xlabel("exact_dist / d_worst   (dropped candidates)")
    ax[0].set_ylabel("CDF of dropped candidates")
    ax[0].set_title(f"{args.name}: were the dropped candidates actually useful?\n"
                    f"(curve height at x=1.0 = false-filter rate)")
    ax[0].legend(loc="lower right", fontsize=10); ax[0].grid(alpha=0.3)

    # ---- Panel B: pq vs exact CDF for the headline alpha ----
    a = float(int(args.amain)) / 100.0
    pe = z[f"a{args.amain}_pq_over_worst"]; ee = z[f"a{args.amain}_exact_over_worst"]
    xp, yp = cdf(pe); xe, ye = cdf(ee)
    ax[1].plot(xp, yp, lw=2.5, color="#1f77b4", ls="--", label="PQ distance / d_worst (what the filter saw)")
    ax[1].plot(xe, ye, lw=2.5, color="#d62728", label="EXACT distance / d_worst (ground truth)")
    ax[1].axvline(1.0, color="black", lw=1.5)
    ax[1].axvline(1.0 + a, color="gray", ls=":", lw=1.8)
    ax[1].text(1.0 + a + 0.01, 0.05, f"drop threshold\n1+α = {1+a:.2f}", fontsize=10, color="gray")
    ax[1].fill_between([0, 1.0], [1, 1], color="crimson", alpha=0.08)
    ax[1].text(0.62, 0.9, "← false-filters\n(dropped but exact ≤ d_worst)", color="crimson", fontsize=10)
    ax[1].set_xlim(0.55, 2.0); ax[1].set_ylim(0, 1.0)
    ax[1].set_xlabel("distance / d_worst   (dropped candidates)")
    ax[1].set_ylabel("CDF")
    ax[1].set_title(f"{args.name}: PQ overestimates → drop; exact reveals few were useful (α={a:.2f})")
    ax[1].legend(loc="lower right", fontsize=10); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(args.out_dir, f"{ts}_false_filter_cdf_{args.name}.png")
    fig.savefig(p, dpi=140)
    print("wrote", p, "| false-filter@0.15 =", round((ff_main or 0)*100, 2), "% of drops")


if __name__ == "__main__":
    main()
