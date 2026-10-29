#!/usr/bin/env python
"""M11 paper figure v6 — Recall@10 with the filter on, at a fixed search list.

The message is "false filtering happens, but it barely moves recall", so the figure
shows recall itself rather than a derived gap: grouped bars are Recall@10 with the
filter ON at four alpha values, and a marker per group is the no-filter oracle at the
same L. If the bars reach the marker, the filter cost nothing.

Groups are (dataset, scale/partitions) — 4 cells at 100M/4p and 3 at 200M/8p — all on
LITS placement, the one SparsePath actually deploys. alpha is an ordered variable so it
gets a single-hue ordinal ramp (validated with validate_palette.js --ordinal).

Usage:
  python plot_figure_recall_v6.py --rows <dir with *_rows.json> --out <dir>
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
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]      # ordinal, light->dark
ALPHAS = [0.00, 0.05, 0.10, 0.15]
DEFAULT_A = 0.15
ORACLE = "#d1495b"
L_FIX = 100

# (label, file, mode, bytes/vector)
CELLS = [
    ("DEEP\n100M/4p",   "deep100M_M32_rows.json",   "nc_LITS_4p", 32),
    ("SIFT\n100M/4p",   "sift100M_4p_rows.json",    "nc_LITS_4p", 16),
    ("SPACEV\n100M/4p", "spacev100M_rows.json",     "nc_LITS_4p", 10),
    ("LAION\n100M/4p",  "laion100M_4p_rows.json",   "nc_LITS_4p", 64),
    ("DEEP\n200M/8p",   "deep200M_8p_rows.json",    "nc_LITS_8p", 12),
    ("SIFT\n200M/8p",   "sift200M_8p_rows.json",    "nc_LITS_8p", 16),
    ("SPACEV\n200M/8p", "spacev200M_8p_rows.json",  "nc_LITS_8p", 10),
]


def load(rows_dir, fname, mode):
    p = os.path.join(rows_dir, fname)
    if not os.path.exists(p):
        return None, None
    rows = json.load(open(p))["rows"]
    on, off = [], None
    for a in ALPHAS:
        hit = [r for r in rows if r["mode"] == mode and int(r["L"]) == L_FIX
               and abs(r["alpha"] - a) < 1e-9]
        if hit:
            on.append(hit[0]["recall_on"])
            off = hit[0]["recall_off"]
        else:
            on.append(np.nan)
    return on, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    labels, ons, offs, budgets = [], [], [], []
    for lab, f, mode, B in CELLS:
        on, off = load(a.rows, f, mode)
        if on is None:
            print(f"  [skip] {f} 없음")
            continue
        labels.append(lab); ons.append(on); offs.append(off); budgets.append(B)

    n, na = len(labels), len(ALPHAS)
    x = np.arange(n)
    w = 0.80 / na

    fig, ax = plt.subplots(figsize=(11.4, 3.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for j, al in enumerate(ALPHAS):
        off_x = (j - (na - 1) / 2) * w
        ax.bar(x + off_x, [ons[i][j] for i in range(n)], width=w * 0.86,
               color=RAMP[j], zorder=3, edgecolor=SURFACE, linewidth=0.6)

    # no-filter oracle: a wide tick spanning the group
    for i in range(n):
        ax.plot([x[i] - 0.45, x[i] + 0.45], [offs[i], offs[i]],
                color=ORACLE, linewidth=2.0, zorder=6, solid_capstyle="butt")
        ax.annotate(f"{offs[i]:.3f}", xy=(x[i] + 0.45, offs[i]), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.8, color=ORACLE, zorder=7)

    lo = min(min(min(o) for o in ons), min(offs))
    ax.set_ylim(max(0, lo - 0.03), 1.005)
    ax.axhline(0.95, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("Recall@10 = 0.95 target", xy=(n - 0.5, 0.95), xytext=(-2, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7.2, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{b} B/vec" for l, b in zip(labels, budgets)],
                       fontsize=8.2, color=INK)
    ax.set_ylabel("Recall@10", fontsize=9, color=INK)
    ax.set_title(f"PQ filter costs almost no recall — LITS placement, L={L_FIX}",
                 fontsize=10.5, color=INK, loc="left", pad=8)

    handles = [Patch(facecolor=RAMP[j], edgecolor=SURFACE,
                     label=f"α = {al:.2f}" + ("  (default)" if al == DEFAULT_A else ""))
               for j, al in enumerate(ALPHAS)]
    handles.append(Line2D([], [], color=ORACLE, lw=2, label="no-filter oracle"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, -0.04), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(a.out, f"{ts}_m11_fig_recall_v6_data.csv")
    with open(csv, "w") as f:
        f.write("cell,bytes_per_vector,L,alpha,recall_on,recall_off_oracle\n")
        for i, lab in enumerate(labels):
            for j, al in enumerate(ALPHAS):
                f.write(f"{lab.replace(chr(10),' ')},{budgets[i]},{L_FIX},{al},"
                        f"{ons[i][j]:.4f},{offs[i]:.4f}\n")
    print(csv)
    p = os.path.join(a.out, f"{ts}_m11_fig_recall_v6.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(p)


if __name__ == "__main__":
    main()
