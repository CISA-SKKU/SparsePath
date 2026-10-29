#!/usr/bin/env python
"""M11 paper figure — the PQ filter's cost (measured) and its benefit (to be measured).

Layout follows the hand sketch: bars + an overlaid line in panel (a), and the benefit
panel (b). Crucially this is NOT a dual-axis chart -- both measures are percentages of
the same magnitude (false-filtered share of eligible candidates, and recall cost in
points), so they share ONE y-axis and remain directly comparable:

  (a) bars  = false-filtered candidates (% of eligible)
      line  = recall cost (points vs the no-filter oracle)      [same axis]
  (b) QPS at Recall@10=0.95  <- the benefit side; NOT YET MEASURED (stub)

Series = datasets at their shipped PQ code budget, plus DEEP at M=32 shown as a
hatched bar in the *same* hue (colour follows the entity; texture marks the config
change). Bars carry direct value labels — required relief for the aqua slot, whose
contrast against the surface is below 3:1.

Usage:
  python plot_figure_cost_benefit.py --results <results/m11_pq_false_filtering> --out <dir>
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

# --- design-system parameters (validated: scripts/validate_palette.js, light mode) ---
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_2 = "#5c5b55"
GRID = "#e3e2dd"
SERIES = {"deep": "#2a78d6", "spacev": "#eb6834", "laion": "#1baf7a"}
ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
L_FIX = 100
MODE = "nc_LITS_4p"          # SparsePath's own placement


def load_rows(path):
    rows = []
    files = [path] if path.endswith(".json") else sorted(glob.glob(os.path.join(path, "*_rows.json")))
    for fp in files:
        with open(fp) as f:
            rows += json.load(f).get("rows", [])
    return rows


def series_at(rows, field):
    """-> list aligned to ALPHAS (nan when absent)."""
    out = []
    for a in ALPHAS:
        hit = [r for r in rows
               if r["mode"] == MODE and int(r["L"]) == L_FIX and abs(r["alpha"] - a) < 1e-9]
        out.append(float(hit[0][field]) if hit else np.nan)
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def bars_and_line(ax, specs, bar_vals, line_vals, ylabel, title):
    """Grouped bars + overlaid line on a SHARED axis (both are percentages)."""
    n = len(specs)
    x = np.arange(len(ALPHAS))
    w = 0.78 / n
    k = ALPHAS.index(0.15)
    ax.axvspan(k - 0.5, k + 0.5, color="#f0efea", zorder=1)
    for i, (label, colour, hatch) in enumerate(specs):
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, bar_vals[i], width=w * 0.88, color=colour, zorder=3,
               hatch=hatch, edgecolor=SURFACE if hatch else "none", linewidth=0.0,
               alpha=0.55)
    for i, (label, colour, hatch) in enumerate(specs):
        ax.plot(x, line_vals[i], color=colour, linewidth=2.0, zorder=5,
                marker="o" if not hatch else "s", markersize=5.5,
                markerfacecolor=colour, markeredgecolor=SURFACE, markeredgewidth=1.2,
                linestyle="--" if hatch else "-")
        vi = line_vals[i][k]
        if not np.isnan(vi):
            ax.annotate(f"{vi:.2f}", xy=(x[k], vi), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7.2,
                        color=INK, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS])
    ax.set_xlabel("PQ filter margin  α", fontsize=9, color=INK)
    ax.annotate("default", xy=(k, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -9), textcoords="offset points", ha="center", va="top",
                fontsize=7.5, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    flat = np.concatenate([np.asarray(v, float) for v in bar_vals + line_vals])
    ax.set_ylim(0, np.nanmax(flat) * 1.22)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = a.results
    os.makedirs(a.out, exist_ok=True)

    src = [
        ("DEEP-100M  (M=12)",   SERIES["deep"],   None,
         os.path.join(R, "M32_rerun_20260803", "M12_lits_baseline")),
        ("DEEP-100M  (M=32)",   SERIES["deep"],   "///",
         os.path.join(R, "M32_rerun_20260803", "deep100M_M32_lits150_rows.json")),
        ("SPACEV-100M  (M=10)", SERIES["spacev"], None,
         os.path.join(R, "spacev100M", "spacev100M_lits150_rows.json")),
        ("LAION-100M  (M=64)",  SERIES["laion"],  None,
         os.path.join(R, "laion100M", "laion100M_iter150_rows.json")),
    ]
    specs, ff, dd = [], [], []
    for label, colour, hatch, path in src:
        rows = load_rows(path)
        specs.append((label, colour, hatch))
        ff.append([v * 100 for v in series_at(rows, "false_filter_rate_of_eligible")])
        dd.append([v * 100 for v in series_at(rows, "recall_delta")])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style(ax)

    bars_and_line(axes[0], specs, ff, dd,
                  "percent   ( bars: false-filtered share · line: recall cost )",
                  "(a)  What the filter costs")

    # (b) benefit side — DUMMY placeholder until the distributed A/B is run.
    # Shape only (filter ON > OFF); the numbers are invented and marked as such.
    ax = axes[1]
    dummy = {"DEEP": (1.00, 1.72), "SPACEV": (1.00, 1.55), "LAION": (1.00, 1.83)}
    cols = [SERIES["deep"], SERIES["spacev"], SERIES["laion"]]
    xs = np.arange(len(dummy))
    wd = 0.34
    for j, (ds, (off_v, on_v)) in enumerate(dummy.items()):
        ax.bar(xs[j] - wd / 2, off_v, width=wd * 0.9, color="#c9c8c1",
               edgecolor=SURFACE, linewidth=0.0, zorder=3)
        ax.bar(xs[j] + wd / 2, on_v, width=wd * 0.9, color=cols[j], alpha=0.55,
               edgecolor=SURFACE, linewidth=0.0, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(dummy.keys()), fontsize=8.5)
    ax.set_ylim(0, 2.4)
    ax.set_yticks([])
    ax.set_title("(b)  What the filter buys", fontsize=10, color=INK, loc="left", pad=8)
    ax.set_ylabel("QPS at Recall@10 = 0.95   (normalised)", fontsize=9, color=INK)
    ax.set_xlabel("grey = filter OFF   ·   colour = filter ON", fontsize=9, color=INK)
    # unmistakable dummy marking
    ax.text(0.5, 0.62, "DUMMY", transform=ax.transAxes, ha="center", va="center",
            fontsize=44, color="#d33", alpha=0.16, fontweight="bold", rotation=18,
            zorder=6)
    ax.text(0.5, 0.94, "PLACEHOLDER — invented numbers, shape only.\n"
                       "Replace with the distributed filter-ON/OFF A/B at iso-recall.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.6,
            color="#b02", linespacing=1.5, zorder=7)
    for sp in ("left",):
        ax.spines[sp].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=c, alpha=0.55, hatch=h,
                     edgecolor=SURFACE if h else c, label=l) for (l, c, h) in specs]
    handles += [Line2D([], [], color=INK_2, lw=0, label="   "),
                Patch(facecolor=INK_2, alpha=0.55, label="bar: false-filtered %"),
                Line2D([], [], color=INK_2, lw=2, marker="o", markersize=5,
                       label="line: recall cost (pts)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.suptitle(f"PQ filter: measured cost (a) and unmeasured benefit (b) "
                 f"— SparsePath (LITS) placement, L={L_FIX}",
                 fontsize=11, color=INK, x=0.008, ha="left", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))

    ts = time.strftime("%Y%m%d_%H%M%S")
    # preserve the exact numbers the figure is drawn from (results-side convention)
    csv = os.path.join(a.out, f"{ts}_m11_fig_cost_benefit_data.csv")
    with open(csv, "w") as f:
        f.write("panel,series,alpha,L,mode,false_filter_pct_of_eligible,recall_cost_points\n")
        for i, (label, _, _) in enumerate(specs):
            for j, al in enumerate(ALPHAS):
                f.write(f"a,{label.replace(',', ' ')},{al},{L_FIX},{MODE},"
                        f"{ff[i][j]:.4f},{dd[i][j]:.4f}\n")
        f.write("b,DUMMY placeholder — no measurement exists; "
                "needs distributed filter-ON/OFF A/B at iso-recall,,,,,\n")
    print(csv)
    p = os.path.join(a.out, f"{ts}_m11_fig_cost_benefit.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(p)


if __name__ == "__main__":
    main()
