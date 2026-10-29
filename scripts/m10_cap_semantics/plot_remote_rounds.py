#!/usr/bin/env python
"""Plot the honest CAP QPS figure: partition-aware remote-fetch-round reduction.

Reads <ds>_remote_rounds.json (from run_m10_remote.py) and draws, at L=100, B=8:
  (left)  remote-round reduction (honest, RDMA round-trips) per placement per
          dataset, with the naive all-round reduction marked as a reference —
          the gap is how much the plain fetch-round count overstates the QPS win.
  (right) absolute remote rounds after CAP (B=8) per placement — LITS minimizes
          the RDMA round-trips regardless of the batching speedup.

Usage: python plot_remote_rounds.py --dir <results/remote_rounds> [--L 100]
"""
import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLACEMENTS = ["balanced", "centroid", "lits"]
COLORS = {"balanced": "#d1495b", "centroid": "#edae49", "lits": "#00798c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--datasets", default="deep100M,sift100M,spacev100M")
    ap.add_argument("--L", type=int, default=100)
    args = ap.parse_args()
    datasets = args.datasets.split(",")

    # cell[ds][pl] = row (L, B=8)
    data = {}
    for ds in datasets:
        p = os.path.join(args.dir, f"{ds}_remote_rounds.json")
        rows = json.load(open(p))["rows"]
        data[ds] = {pl: next(r for r in rows if r["placement"] == pl and r["L"] == args.L and r["B"] == 8)
                    for pl in PLACEMENTS}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(datasets)); w = 0.26
    # left: remote-round reduction (bars) + naive all-round reduction (hatched ref)
    for i, pl in enumerate(PLACEMENTS):
        rr = [data[ds][pl]["remote_round_reduction"] for ds in datasets]
        allr = [data[ds][pl]["all_round_reduction"] for ds in datasets]
        ax[0].bar(x + (i - 1) * w, rr, w, color=COLORS[pl], label=f"{pl} (honest)")
        ax[0].bar(x + (i - 1) * w, allr, w, facecolor="none", edgecolor="gray",
                  hatch="////", linewidth=0.4, zorder=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels([d.replace("100M", "") for d in datasets])
    ax[0].set_ylabel("round-trip reduction (best-first / CAP)")
    ax[0].set_title(f"CAP remote-round↓ (honest) vs naive all-round↓ (hatched)  L={args.L}, B=8")
    ax[0].legend(fontsize=8, ncol=3, loc="upper right")
    ax[0].grid(axis="y", ls=":", alpha=0.5)

    # right: absolute remote rounds after CAP (B=8) per placement
    for i, pl in enumerate(PLACEMENTS):
        rem = [data[ds][pl]["avg_remote_rounds"] for ds in datasets]
        ax[1].bar(x + (i - 1) * w, rem, w, color=COLORS[pl], label=pl)
    ax[1].set_xticks(x); ax[1].set_xticklabels([d.replace("100M", "") for d in datasets])
    ax[1].set_ylabel("avg remote fetch rounds (CAP, B=8)")
    ax[1].set_title("Absolute RDMA round-trips per query (lower = faster) — LITS lowest")
    ax[1].legend(fontsize=8); ax[1].grid(axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.dir, f"{ts}_m10_remote_round_reduction.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
