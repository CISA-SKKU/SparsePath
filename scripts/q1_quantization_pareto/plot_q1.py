#!/usr/bin/env python
"""Plot Q1 (M08) memory-recall Pareto across precision levels (raw / SQ8 / PQ / RaBitQ)
plus rerank recovery. Reads the per-dataset *_q1_pareto.json produced by run_q1.py.

Panel A: bytes/vec (log) vs recall@10 at a fixed L — the memory-recall Pareto.
         base quantized-search points (filled) + full-precision rerank (hollow, same x).
Panel B: rerank recovery vs candidate pool L (recall vs L) for pq / sq8 / rabitq.
"""
import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# base precision -> (marker, label)
BASE = [("raw", "o", "full precision"),
        ("sq8", "P", "SQ8 (1 B/dim)"),
        ("pq", "s", "PQ"),
        ("rabitq", "D", "RaBitQ (1 bit/dim)")]
RR = {"sq8": "sq8_rerank", "pq": "pq_rerank", "rabitq": "rabitq_rerank"}
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]


def load_all(root):
    ds = {}
    for fp in sorted(glob.glob(os.path.join(root, "**", "*_q1_pareto.json"), recursive=True)):
        if any(x in fp for x in ("rbqcheck", "_bak", "check")):
            continue   # skip throwaway verification runs
        with open(fp) as f:
            rows = json.load(f)
        if rows:
            # prefer the row-set with the most modes (full 7-mode over partial)
            key = rows[0]["dataset"]
            if key not in ds or len(rows[0]) > len(ds[key][0]):
                ds[key] = sorted(rows, key=lambda r: r["L"])
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    ds = load_all(a.root)
    if not ds:
        raise SystemExit(f"no *_q1_pareto.json under {a.root}")
    outdir = a.outdir or a.root

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Panel A: memory-recall Pareto at L=operating
    for i, (name, rows) in enumerate(sorted(ds.items())):
        c = COLORS[i % len(COLORS)]
        row = min(rows, key=lambda r: abs(r["L"] - a.L))
        for key, mk, _ in BASE:
            rk, bk = f"recall_{key}", f"bytes_{key}"
            if rk in row and row[rk] is not None:
                axA.scatter(row[bk], row[rk], marker=mk, s=85, color=c,
                            edgecolor="k", linewidth=0.5, zorder=3)
            # rerank variant (hollow, same x, up)
            rr = RR.get(key)
            if rr and f"recall_{rr}" in row and row[f"recall_{rr}"] is not None:
                axA.scatter(row[bk], row[f"recall_{rr}"], marker=mk, s=85,
                            facecolor="none", edgecolor=c, linewidth=1.4, zorder=3)
                axA.plot([row[bk], row[bk]], [row[rk], row[f"recall_{rr}"]],
                         "-", color=c, alpha=0.35, lw=1)
        axA.plot([], [], "-", color=c, label=f"{name} (d={row['dim']})")
    axA.set_xscale("log")
    axA.set_xlabel("bytes / vector  (log)")
    axA.set_ylabel(f"recall@10  (L={a.L})")
    axA.set_title("(A) memory–recall Pareto: raw / SQ8 / PQ / RaBitQ\n(filled = quantized search, hollow = + rerank)")
    axA.grid(True, alpha=0.3)
    mleg = [Line2D([0], [0], marker=mk, color="gray", linestyle="", markersize=9,
                   markeredgecolor="k", label=lab) for (k, mk, lab) in BASE]
    l1 = axA.legend(handles=mleg, loc="lower right", fontsize=7.5, title="precision")
    axA.add_artist(l1)
    axA.legend(loc="center right", fontsize=7.5, title="dataset")

    # Panel B: rerank recovery vs L (pq / sq8 / rabitq) for the first dataset
    name0, rows0 = sorted(ds.items())[0]
    Ls = [r["L"] for r in rows0]
    axB.plot(Ls, [r.get("recall_raw") for r in rows0], "k-", lw=2, label="raw (ceiling)")
    style = {"pq": ("s", "#16a34a"), "sq8": ("P", "#dc2626"), "rabitq": ("D", "#9333ea")}
    for base, rr in RR.items():
        if f"recall_{rr}" in rows0[0]:
            mk, col = style[base]
            axB.plot(Ls, [r.get(f"recall_{rr}") for r in rows0], mk + "--", color=col, label=f"{base}+rerank")
            axB.plot(Ls, [r.get(f"recall_{base}") for r in rows0], mk + ":", color=col, alpha=0.5, label=f"{base} (no rerank)")
    axB.set_xlabel("candidate pool L  (= raw rerank fetches / query)")
    axB.set_ylabel("recall@10")
    axB.set_title(f"(B) rerank recovery vs L — {name0}")
    axB.grid(True, alpha=0.3); axB.legend(fontsize=7, ncol=2)

    fig.suptitle("Q1 (M08 / R5·W2): SparsePath under quantized vectors — memory–recall Pareto "
                 "across precision (mild SQ8 ≈ lossless; aggressive PQ/RaBitQ need rerank)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(outdir, f"{ts}_q1_memory_recall_pareto.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
