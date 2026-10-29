#!/usr/bin/env python
"""M11 paper figure, v2 — datasets on the x-axis, alpha as an ordinal colour ramp.

Why v2 differs from v1:
  * v1 overlaid a line (recall cost) on bars (false-filter rate) on a shared axis.
    Both are percentages, so it was not formally a dual axis — but the two are
    percentages *of different things* (share of eligible candidates vs. recall
    points), so putting them on one scale invites a comparison that is not
    meaningful. v2 gives each measure its own panel.
  * v2 puts the DATASET on x (the comparison the paper actually makes: LAION is
    smallest, M=32 removes the cost) and encodes alpha — an ordered variable —
    as a single-hue ordinal ramp, light→dark, so the sweep reads as a staircase.

Panels:
  (a) false-filtered candidates (% of eligible)   [top-left]
  (b) recall cost (points vs no-filter oracle)    [bottom-left]
  (c) QPS at Recall@10=0.95                       [right]  -- DUMMY placeholder

Usage:
  python plot_figure_cost_benefit_v2.py --results <results/m11_pq_false_filtering> --out <dir>
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_2 = "#5c5b55"
GRID = "#e3e2dd"
# ordinal ramp for alpha (validated: validate_palette.js --ordinal, light mode)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ALPHAS = [0.00, 0.05, 0.10, 0.15, 0.25]
DEFAULT_A = 0.15
L_FIX = 100
MODE = "nc_LITS_4p"


def load_rows(path):
    rows = []
    files = [path] if path.endswith(".json") else sorted(glob.glob(os.path.join(path, "*_rows.json")))
    for fp in files:
        with open(fp) as f:
            rows += json.load(f).get("rows", [])
    return rows


def series_at(rows, field):
    out = []
    for a in ALPHAS:
        hit = [r for r in rows
               if r["mode"] == MODE and int(r["L"]) == L_FIX and abs(r["alpha"] - a) < 1e-9]
        out.append(float(hit[0][field]) if hit else np.nan)
    return out


def style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def grouped(ax, labels, vals, ylabel, title, show_xticklabels):
    """vals[config][alpha]"""
    ng, na = len(labels), len(ALPHAS)
    x = np.arange(ng)
    w = 0.80 / na
    k = ALPHAS.index(DEFAULT_A)
    for j, a in enumerate(ALPHAS):
        off = (j - (na - 1) / 2) * w
        col = [vals[i][j] for i in range(ng)]
        ax.bar(x + off, col, width=w * 0.86, color=RAMP[j], zorder=3,
               edgecolor=SURFACE, linewidth=0.6)
    # selective direct labels: the shipped margin only
    off_k = (k - (na - 1) / 2) * w
    for i in range(ng):
        v = vals[i][k]
        if np.isnan(v):
            continue
        ax.annotate(f"{v:.2f}", xy=(x[i] + off_k, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.4,
                    color=INK, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6, color=INK)
    if not show_xticklabels:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(ylabel, fontsize=8.8, color=INK)
    ax.set_title(title, fontsize=9.8, color=INK, loc="left", pad=6)
    flat = np.asarray(vals, float)
    ax.set_ylim(0, np.nanmax(flat) * 1.22)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = a.results
    os.makedirs(a.out, exist_ok=True)

    src = [
        ("DEEP-100M\nM=12",   os.path.join(R, "M32_rerun_20260803", "M12_lits_baseline")),
        ("DEEP-100M\nM=32",   os.path.join(R, "M32_rerun_20260803", "deep100M_M32_lits150_rows.json")),
        ("SPACEV-100M\nM=10", os.path.join(R, "spacev100M", "spacev100M_lits150_rows.json")),
        ("LAION-100M\nM=64",  os.path.join(R, "laion100M", "laion100M_iter150_rows.json")),
    ]
    labels, ff, dd = [], [], []
    for label, path in src:
        rows = load_rows(path)
        labels.append(label)
        ff.append([v * 100 for v in series_at(rows, "false_filter_rate_of_eligible")])
        dd.append([v * 100 for v in series_at(rows, "recall_delta")])

    fig = plt.figure(figsize=(11.8, 5.0))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.42, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])
    for ax in (ax_a, ax_b):
        style(ax)
    style(ax_c)

    grouped(ax_a, labels, ff, "false-filtered\n(% of eligible)",
            "(a)  How often the filter is wrong", show_xticklabels=False)
    grouped(ax_b, labels, dd, "recall cost\n(points)",
            "(b)  What that costs in recall", show_xticklabels=True)

    # (c) DUMMY benefit panel — same x (dataset) and same alpha ramp as (a)/(b),
    # normalised to alpha=0 so no absolute QPS is needed and datasets are comparable.
    # A larger margin filters less -> more remote fetches -> lower QPS; "filter OFF"
    # (grey) is the alpha->infinity limit and anchors what the filter buys.
    dummy = {                       # per dataset: QPS(alpha)/QPS(alpha=0), then OFF
        "DEEP-100M":   [1.00, 0.97, 0.93, 0.90, 0.84, 0.58],
        "SPACEV-100M": [1.00, 0.98, 0.95, 0.92, 0.87, 0.65],
        "LAION-100M":  [1.00, 0.96, 0.92, 0.88, 0.81, 0.55],
    }
    xs = np.arange(len(dummy))
    nb = len(ALPHAS) + 1
    wc = 0.80 / nb
    for i, (ds, vals) in enumerate(dummy.items()):
        for j, v in enumerate(vals):
            off = (j - (nb - 1) / 2) * wc
            col = RAMP[j] if j < len(ALPHAS) else "#c9c8c1"
            ax_c.bar(xs[i] + off, v, width=wc * 0.86, color=col, zorder=3,
                     edgecolor=SURFACE, linewidth=0.6)
    ax_c.axhline(1.0, color=GRID, linewidth=1.0, zorder=2)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(list(dummy.keys()), fontsize=8.6, color=INK)
    ax_c.set_ylim(0, 1.18)
    ax_c.set_title("(c)  What the filter buys", fontsize=9.8, color=INK, loc="left", pad=6)
    ax_c.set_ylabel("QPS at Recall@10 = 0.95\n(relative to α = 0)", fontsize=8.8, color=INK)
    ax_c.set_xlabel("rightmost grey bar = filter OFF", fontsize=8.8, color=INK)
    ax_c.text(0.5, 0.55, "DUMMY", transform=ax_c.transAxes, ha="center", va="center",
              fontsize=46, color="#d33", alpha=0.16, fontweight="bold", rotation=18, zorder=6)
    ax_c.text(0.5, 0.985, "PLACEHOLDER — invented numbers, shape only.\n"
                          "Replace with the distributed α-sweep at iso-recall.",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=7.5,
              color="#b02", linespacing=1.5, zorder=7)

    handles = [Patch(facecolor=RAMP[j], edgecolor=SURFACE,
                     label=f"α = {a_:.2f}" + ("  (default)" if a_ == DEFAULT_A else ""))
               for j, a_ in enumerate(ALPHAS)]
    handles.append(Patch(facecolor="#c9c8c1", edgecolor=SURFACE, label="filter OFF  (c only)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, -0.075), labelcolor=INK)
    fig.suptitle("PQ filter: measured cost (a, b) and unmeasured benefit (c)  —  "
                 f"SparsePath (LITS) placement, L={L_FIX}",
                 fontsize=10.6, color=INK, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(a.out, f"{ts}_m11_fig_cost_benefit_v2_data.csv")
    with open(csv, "w") as f:
        f.write("panel,config,alpha,L,mode,false_filter_pct_of_eligible,recall_cost_points\n")
        for i, lab in enumerate(labels):
            for j, al in enumerate(ALPHAS):
                f.write(f"a/b,{lab.replace(chr(10), ' ')},{al},{L_FIX},{MODE},"
                        f"{ff[i][j]:.4f},{dd[i][j]:.4f}\n")
        f.write("c,DUMMY placeholder — no measurement exists; "
                "needs distributed filter-ON/OFF A/B at iso-recall,,,,,\n")
    print(csv)
    p = os.path.join(a.out, f"{ts}_m11_fig_cost_benefit_v2.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(p)


if __name__ == "__main__":
    main()
