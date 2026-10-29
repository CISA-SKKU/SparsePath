#!/usr/bin/env python
"""M11 paper figure v7 — Recall@10 with the filter on, anchored at the oracle's operating point.

The anchor is chosen by the *no-filter oracle*, not by us: for each cell we take the
smallest search-list length L at which the oracle reaches Recall@10 = 0.95, and report
what the filter does at that same L. That answers the reviewer's question directly --
"at the configuration a no-filter system would deploy, how much recall does the filter
give up?" -- and the anchor cannot be accused of being chosen to flatter the filter.

Bars are Recall@10 with the filter ON at four alpha values; the red tick per group is
the oracle at the same L. The gap between the darkest bar (default alpha=0.15) and the
tick is the number quoted in the text. alpha is an ordered variable, so it gets a
single-hue ordinal ramp (validate_palette.js --ordinal: ALL CHECKS PASS).

Cells are the budgets SparsePath deploys: DEEP 32 B, SIFT 16 B, SPACEV 10 B, LAION 64 B.

Usage:
  python plot_figure_recall_v7.py --out <dir> [--base <results root>]
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_2 = "#5c5b55"
GRID = "#e3e2dd"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]   # ordinal, light->dark
ORACLE_BAR = "#6f6e66"   # neutral: the baseline is a reference, not another series
ALPHAS = [0.00, 0.05, 0.10, 0.15]
DEFAULT_A = 0.15
TARGET = 0.95

BASE_DEFAULT = ("/experiments"
                "/results/m11_pq_false_filtering")

# (label, rows-file, mode, bytes/vector)
CELLS = [
    ("DEEP\n100M/4p",   "M32_rerun_20260803/deep100M_M32_lits150_rows.json", "nc_LITS_4p", 32),
    ("SIFT\n100M/4p",   "matrix7_20260804/sift100M_4p_rows.json",            "nc_LITS_4p", 16),
    ("SPACEV\n100M/4p", "spacev100M/spacev100M_lits150_rows.json",           "nc_LITS_4p", 10),
    ("LAION\n100M/4p",  "matrix7_20260804/laion100M_4p_rows.json",           "nc_LITS_4p", 64),
    ("DEEP\n200M/8p",   "msweep_20260804/deep200M_8p_M32/deep200M_8p_M32_rows.json",
                                                                             "nc_LITS_8p", 32),
    ("SIFT\n200M/8p",   "matrix7_20260804/sift200M_8p_rows.json",            "nc_LITS_8p", 16),
]


def load(base, rel, mode):
    """Return (anchor L, recall_on per alpha, oracle recall) at the oracle's 0.95 point."""
    path = os.path.join(base, rel)
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))["rows"]
    sub = [r for r in rows if r["mode"] == mode]

    # anchor: smallest L where the OFF curve reaches the target (alpha-independent)
    ref = sorted([r for r in sub if abs(r["alpha"] - DEFAULT_A) < 1e-9],
                 key=lambda r: r["L"])
    anchor = next((r["L"] for r in ref if r["recall_off"] >= TARGET), None)
    if anchor is None:
        return None

    at = {round(r["alpha"], 3): r for r in sub if r["L"] == anchor}
    on = [at[round(a, 3)]["recall_on"] if round(a, 3) in at else np.nan for a in ALPHAS]
    off = at[round(DEFAULT_A, 3)]["recall_off"]
    return anchor, on, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    labels, ons, offs, budgets, anchors = [], [], [], [], []
    for lab, rel, mode, B in CELLS:
        got = load(a.base, rel, mode)
        if got is None:
            print(f"  [skip] {rel}")
            continue
        anchor, on, off = got
        labels.append(lab); ons.append(on); offs.append(off)
        budgets.append(B); anchors.append(anchor)

    # leftmost bar in every group is the no-filter oracle, so each alpha bar is read
    # against a baseline standing right next to it rather than against a floating tick.
    n = len(labels)
    nb = len(ALPHAS) + 1
    x = np.arange(n)
    w = 0.84 / nb

    fig, ax = plt.subplots(figsize=(11.2, 4.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    def slot(j):                       # j = 0 is the oracle, 1..4 are the alphas
        return (j - (nb - 1) / 2) * w

    ax.bar(x + slot(0), offs, width=w * 0.86, color=ORACLE_BAR,
           zorder=3, edgecolor=SURFACE, linewidth=0.6)
    for j, al in enumerate(ALPHAS):
        ax.bar(x + slot(j + 1), [ons[i][j] for i in range(n)], width=w * 0.86,
               color=RAMP[j], zorder=3, edgecolor=SURFACE, linewidth=0.6)

    # carry the oracle's height across its own group, so "does the bar reach it" is
    # answerable without measuring against the left-hand bar by eye
    for i in range(n):
        ax.plot([x[i] + slot(0) - w * 0.43, x[i] + slot(nb - 1) + w * 0.43],
                [offs[i], offs[i]], color=ORACLE_BAR, linewidth=1.0,
                linestyle=(0, (3, 2)), zorder=6)
        gap = 100.0 * (offs[i] - ons[i][ALPHAS.index(DEFAULT_A)])
        ax.annotate(f"{gap:.2f} pp", xy=(x[i], offs[i]), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.6, color=INK, zorder=7, fontweight="bold")

    lo = min(min(min(o) for o in ons), min(offs))
    hi = max(max(max(o) for o in ons), max(offs))
    ax.set_ylim(lo - 0.006, hi + 0.006)

    # reference line, labelled in the right margin so it never collides with a bar
    ax.set_xlim(-0.60, (n - 1) + 1.06)
    ax.axhline(TARGET, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("0.95\ntarget", xy=((n - 1) + 0.56, TARGET), xytext=(0, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=7.2, color=INK_2, linespacing=1.25)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{b} B/vec   L={L}"
                        for l, b, L in zip(labels, budgets, anchors)],
                       fontsize=8.2, color=INK)
    ax.set_ylabel("Recall@10", fontsize=9, color=INK)
    ax.set_title("At the search list where the no-filter oracle reaches 0.95, "
                 "the PQ filter gives up almost no recall",
                 fontsize=10.5, color=INK, loc="left", pad=8)

    handles = [Patch(facecolor=ORACLE_BAR, edgecolor=SURFACE, label="no filter (oracle)")]
    handles += [Patch(facecolor=RAMP[j], edgecolor=SURFACE,
                      label=f"$\\alpha$ = {al:.2f}" + ("  (default)" if al == DEFAULT_A else ""))
                for j, al in enumerate(ALPHAS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, -0.045), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(a.out, f"{ts}_m11_fig_recall_v7_data.csv")
    with open(csv, "w") as f:
        f.write("cell,bytes_per_vector,anchor_L,alpha,recall_on,recall_oracle,gap_pp\n")
        for i, lab in enumerate(labels):
            for j, al in enumerate(ALPHAS):
                f.write(f"{lab.replace(chr(10),' ')},{budgets[i]},{anchors[i]},{al},"
                        f"{ons[i][j]:.4f},{offs[i]:.4f},"
                        f"{100*(offs[i]-ons[i][j]):.3f}\n")
    png = os.path.join(a.out, f"{ts}_m11_fig_recall_v7.png")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=SURFACE)

    print(csv)
    print(png)
    print("\ncell                 L    oracle   ON(a=.15)   gap")
    for i, lab in enumerate(labels):
        g = 100 * (offs[i] - ons[i][ALPHAS.index(DEFAULT_A)])
        flag = "" if ons[i][ALPHAS.index(DEFAULT_A)] >= TARGET else "  <0.95"
        print(f"{lab.replace(chr(10),' '):20s} {anchors[i]:<4} {offs[i]:.4f}  "
              f"{ons[i][ALPHAS.index(DEFAULT_A)]:.4f}   {g:+.2f} pp{flag}")


if __name__ == "__main__":
    main()
