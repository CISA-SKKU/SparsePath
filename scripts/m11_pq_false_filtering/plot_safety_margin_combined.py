#!/usr/bin/env python
"""Combined Tier B safety-margin figure: deep/spacev/LAION overlaid.

recall Δ (OFF−ON) @α=0.15 vs codebook QE ratio, one panel per L. The shipped
codebook (QE=1.0) marks the real-drift level; biased 1-cluster codebooks (the
realistic worst-case drift) are starred. Shows the safety margin holds across
96/100/512 dimensions.
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"deep": "#1f77b4", "spacev": "#2ca02c", "laion": "#d62728"}


def load(path):
    j = json.load(open(path))
    return j["rows"]


def series(rows, L, a=0.15):
    noise = sorted([r for r in rows if r["L"] == L and abs(r["alpha"] - a) < 1e-9
                    and r.get("lever") == "noise" and np.isfinite(r["qe_ratio"])],
                   key=lambda r: r["qe_ratio"])
    biased = [r for r in rows if r["L"] == L and abs(r["alpha"] - a) < 1e-9 and r.get("lever") == "biased"]
    return noise, (biased[0] if biased else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="label=path/to/rows.json (label in deep/spacev/laion)")
    ap.add_argument("--dims", nargs="+", default=["deep=96", "spacev=100(int8)", "laion=512"])
    ap.add_argument("--Ls", nargs="+", type=int, default=[100, 400])
    ap.add_argument("--xaxis", choices=["qe", "corr"], default="qe",
                    help="qe = codebook QE ratio (reconstruction); corr = pq-vs-exact rank preservation")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    XKEY = "qe_ratio" if args.xaxis == "qe" else "corr"

    data = {}
    for spec in args.inputs:
        lab, path = spec.split("=", 1)
        data[lab] = load(path)
    dimlab = dict(s.split("=", 1) for s in args.dims)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, len(args.Ls), figsize=(7 * len(args.Ls), 5.4), sharey=True)
    if len(args.Ls) == 1:
        ax = [ax]

    for i, L in enumerate(args.Ls):
        for lab, rows in data.items():
            c = COLORS.get(lab, None)
            noise, biased = series(rows, L)
            if noise:
                x = [r[XKEY] for r in noise]; y = [r["recall_delta"] for r in noise]
                ax[i].plot(x, y, "o-", color=c, lw=2, ms=6, label=f"{lab} ({dimlab.get(lab,'')}-d)")
                if args.xaxis == "corr":   # highlight shipped (noise σ0 = highest corr = real operating point)
                    ax[i].plot([noise[0][XKEY]], [noise[0]["recall_delta"]], "o", mfc="none",
                               mec=c, mew=2.2, ms=15, zorder=6)
            if biased is not None:
                ax[i].plot([biased[XKEY]], [biased["recall_delta"]], "*", color=c, ms=17,
                           markeredgecolor="black", markeredgewidth=0.6, zorder=6)
        if args.xaxis == "qe":
            ax[i].axvline(1.0, color="green", ls=":", lw=2)
            ax[i].axvspan(1.0, 1.30, color="green", alpha=0.06)
            ax[i].set_xlabel("codebook QE ratio (degraded / shipped)")
        else:
            ax[i].set_xlabel("pq-vs-exact distance corr  (higher ← = better rank fidelity;  degradation → )")
            ax[i].invert_xaxis()   # higher corr (shipped) on the left; degradation increases rightward
        ax[i].set_title(f"L = {L}")
        ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("recall Δ  (no-filter − filter)  @α=0.15")
    ymax = ax[0].get_ylim()[1]
    if args.xaxis == "qe":
        ax[0].text(1.02, ymax * 0.93, "real drift\n(QE≈1.0)", color="green", fontsize=11, va="top")
        ax[0].text(1.32, ymax * 0.55, "realistic\nrange", color="green", fontsize=10, alpha=0.8)
        ax[0].plot([], [], "k*", ms=13, label="biased 1-cluster\n(realistic worst-case drift)")
    else:
        ax[0].plot([], [], "o", mfc="none", mec="black", mew=2, ms=12, label="shipped (real operating corr)")
        ax[0].plot([], [], "k*", ms=13, label="biased 1-cluster")
    ax[0].legend(loc="upper left", fontsize=10)

    sub = ("recall Δ vs codebook QE (reconstruction error)" if args.xaxis == "qe"
           else "recall Δ vs pq-vs-exact corr (what the filter actually needs: rank preservation)")
    fig.suptitle(f"Tier B stale-codebook safety margin — {sub}  (deep/spacev/LAION × LITS)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = os.path.join(args.out_dir, f"{ts}_tierB_safety_margin_combined_{args.xaxis}.png")
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    main()
