#!/usr/bin/env python
"""M11 paper figure, v4 — single panel: false-filtering bars + iso-recall QPS line.

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
LINE_C = "#b02"        # the unmeasured (dummy) series — deliberately off-palette


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


def grouped(ax, labels, vals, ylabel, title, show_xticklabels, line=None, line_label=None):
    """vals[config][alpha]; optional `line[config][alpha]` drawn on a secondary axis
    whose range is FIXED to 0..1.05 because the values are ratios to alpha=0 — the
    scale is set by meaning, not chosen, so the bar/line crossing is not arbitrary."""
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
    if line is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1.55)
        ax2.set_facecolor("none")
        for sp in ("top", "left"):
            ax2.spines[sp].set_visible(False)
        ax2.spines["right"].set_color(GRID)
        ax2.spines["bottom"].set_color(GRID)
        ax2.tick_params(colors=LINE_C, labelsize=8, length=3, width=0.8)
        for i in range(ng):
            xs_ = x[i] + (np.arange(na) - (na - 1) / 2) * w
            ax2.plot(xs_, line[i], color=LINE_C, linewidth=2.0, zorder=6,
                     marker="o", markersize=5.0, markerfacecolor=LINE_C,
                     markeredgecolor=SURFACE, markeredgewidth=1.1)
        ax2.set_ylabel(line_label, fontsize=8.8, color=LINE_C)
        ax2.text(0.985, 0.06, "DUMMY", transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=16, color="#d33", alpha=0.30, fontweight="bold", zorder=7)


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

    # DUMMY normalised QPS at iso-recall (ratio to alpha=0). Replace with the
    # distributed sweep. Shape is UNIMODAL by construction, and that is the point:
    #   * small alpha -> filter is aggressive, recall drops, so L must grow a lot to
    #     still reach 0.95 -> more hops -> lower QPS
    #   * large alpha (-> filter off) -> almost nothing is filtered -> remote fetches
    #     explode -> lower QPS
    #   * in between the two effects trade off and throughput peaks.
    # A monotonically falling curve would wrongly read as "fastest with the filter off".
    qps_dummy = {
        "DEEP-100M\nM=12":   [1.00, 1.18, 1.31, 1.33, 1.19],
        "DEEP-100M\nM=32":   [1.00, 1.16, 1.28, 1.30, 1.17],
        "SPACEV-100M\nM=10": [1.00, 1.14, 1.25, 1.27, 1.15],
        "LAION-100M\nM=64":  [1.00, 1.21, 1.35, 1.37, 1.22],
    }
    qq = [qps_dummy[l] for l in labels]

    fig = plt.figure(figsize=(11.0, 3.5))
    fig.patch.set_facecolor(SURFACE)
    ax_a = fig.add_subplot(111)
    style(ax_a)

    grouped(ax_a, labels, ff, "false-filtered candidates\n(% of eligible)",
            "What the PQ filter costs (bars) and what it buys (line)",
            show_xticklabels=True,
            line=qq, line_label="QPS at Recall@10 = 0.95\n(relative to α = 0)")

    fig.text(0.5, 0.995, "red line = PLACEHOLDER (invented, shape only). Replace with the "
                         "distributed α-sweep at iso-recall.", ha="center", va="top",
             fontsize=7.6, color="#b02")

    handles = [Patch(facecolor=RAMP[j], edgecolor=SURFACE,
                     label=f"α = {a_:.2f}" + ("  (default)" if a_ == DEFAULT_A else ""))
               for j, a_ in enumerate(ALPHAS)]
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color=LINE_C, lw=2, marker="o", markersize=5,
                          label="QPS rel. α=0  (dummy)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, -0.03), labelcolor=INK)
    fig.suptitle(f"SparsePath (LITS) placement, L={L_FIX}",
                 fontsize=9.4, color=INK_2, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(a.out, f"{ts}_m11_fig_cost_benefit_v4_data.csv")
    with open(csv, "w") as f:
        f.write("panel,config,alpha,L,mode,false_filter_pct_of_eligible,recall_cost_points\n")
        for i, lab in enumerate(labels):
            for j, al in enumerate(ALPHAS):
                f.write(f"a/b,{lab.replace(chr(10), ' ')},{al},{L_FIX},{MODE},"
                        f"{ff[i][j]:.4f},{dd[i][j]:.4f}\n")
        f.write("line,DUMMY placeholder — no measurement exists; "
                "needs distributed filter-ON/OFF A/B at iso-recall,,,,,\n")
    print(csv)
    p = os.path.join(a.out, f"{ts}_m11_fig_cost_benefit_v4.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(p)


if __name__ == "__main__":
    main()
