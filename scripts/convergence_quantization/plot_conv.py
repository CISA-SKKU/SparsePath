#!/usr/bin/env python
"""Convergence-under-quantization figures (W1/W2/W3/W4, handoff 2026-07-24).

Fig-W1  latch behaviour per mode: latch rate + conv-hop p50 across L (per dataset/graph).
Fig-W2  convergence profile (traced queries): median normalized best-dist vs hop and
        median recall@hop vs hop, per mode, with the production detector's median
        latch hop marked on each curve.  One figure per (dataset, L).
Fig-W3  detector comparison: early-rate vs hops-saved scatter (per mode facet) +
        current-rule theta robustness (early-rate vs theta, lines per mode).
Fig-W4  rerank-pool alpha-margin gate: recall vs alpha and rerank fetch/query vs
        alpha, side-by-side panels (no dual axis), lines per L.

Colors: validated categorical slots (see plot_a1.py); identity fixed per mode:
exact=blue, pq low-M=orange, pq M32=purple, sq8=green, rabitq=yellow.
Direct labels everywhere (contrast WARN relief). Raw copies get a timestamp
prefix in RES; clean-name copies go to DOCS_FIG.
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

RES = os.path.expanduser(
    "/experiments/results/m10_convergence_quantization")
DOCS_FIG = os.path.expanduser(
    "./figures")

SURFACE = "#fcfcfb"
INK, INK2 = "#1a1a19", "#5f5e56"
MODE_COLOR = {
    "exact": "#2a78d6", "pq_M12": "#eb6834", "pq_M10": "#eb6834",
    "pq_M32": "#9a5fd0", "sq8": "#1baf7a", "rabitq": "#eda100",
}
MODE_ORDER = ["exact", "pq_M12", "pq_M10", "pq_M32", "sq8", "rabitq"]
UMAX = np.iinfo(np.uint32).max


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(True, color="#e8e7e0", lw=0.6, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig, name, stamp):
    os.makedirs(RES, exist_ok=True)
    os.makedirs(DOCS_FIG, exist_ok=True)
    raw = os.path.join(RES, f"{stamp}_{name}.png")
    fig.savefig(raw, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(DOCS_FIG, f"{name}.png"), dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {raw} (+docs copy)")


def _modes_of(rows):
    ms = [m for m in MODE_ORDER if any(r["mode"] == m for r in rows)]
    return ms + sorted({r["mode"] for r in rows} - set(ms))


# ---------------- W1: latch rate + conv hop across L ----------------
def fig_w1(res_dir, stamp):
    for path in sorted(glob.glob(os.path.join(res_dir, "*_w1_latch.json"))):
        rows = json.load(open(path))
        ds = rows[0]["dataset"]
        modes = _modes_of(rows)
        Ls = sorted({r["L"] for r in rows})
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        for ax, key, ttl in ((axes[0], "latch_rate", "latch rate (frac. of queries)"),
                             (axes[1], "conv_hop_p50", "latch hop (p50)")):
            style_ax(ax)
            for m in modes:
                pts = {r["L"]: r[key] for r in rows if r["mode"] == m and r[key] is not None}
                if not pts:
                    continue
                xs = [L for L in Ls if L in pts]
                ys = [pts[L] for L in xs]
                ax.plot(xs, ys, "-o", ms=4.5, lw=1.8, color=MODE_COLOR.get(m, INK2),
                        zorder=3)
                ax.annotate(m, (xs[-1], ys[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=7.5,
                            color=INK, va="center")
            ax.set_xlabel("L (search list)", fontsize=9, color=INK)
            ax.set_title(ttl, fontsize=9.5, color=INK)
            ax.set_xticks(Ls)
        axes[0].set_ylim(-0.03, 1.05)
        fig.suptitle(f"{ds} — production ConvergenceDetector under quantized distances "
                     f"(w=3, θ=1.0%)", fontsize=10.5, color=INK)
        save(fig, f"conv_w1_latch_{ds}", stamp)


# ---------------- W2: profile curves from traces ----------------
def _median_curves(tr, id_off):
    """-> (hops_grid, med_best_norm, med_recall) over traced queries."""
    offs = tr["offsets"]
    K = int(tr["K"])
    topk = tr["topk"]
    curves_b, curves_r = [], []
    for qi in range(len(offs) - 1):
        a, b = int(offs[qi]), int(offs[qi + 1])
        if b <= a:
            continue
        best = tr["best"][a:b].astype(np.float64)
        final = best[-1] if best[-1] > 0 else 1.0
        curves_b.append(best / final)
        g = set(int(x) for x in tr["gt"][qi])
        tk = topk[a:b].reshape(b - a, K)
        rec = np.empty(b - a, np.float64)
        for i, row in enumerate(tk):
            f = set(int(t) - 1 + id_off for t in row if t != UMAX)
            rec[i] = len(f & g) / K
        curves_r.append(rec)
    H = max(len(c) for c in curves_b)
    def med(curves):
        M = np.full((len(curves), H), np.nan)
        for i, c in enumerate(curves):
            M[i, :len(c)] = c
            M[i, len(c):] = c[-1]          # hold final value
        return np.nanmedian(M, axis=0)
    return np.arange(1, H + 1), med(curves_b), med(curves_r)


def fig_w2(res_dir, stamp, only_L=None):
    files = sorted(glob.glob(os.path.join(res_dir, "traces", "*.npz")))
    by_dsL = {}
    for f in files:
        tr = np.load(f)
        ds, mode, L = str(tr["dataset"]), str(tr["mode"]), int(tr["L"])
        if only_L and L not in only_L:
            continue
        by_dsL.setdefault((ds, L), []).append((mode, f))
    for (ds, L), entries in sorted(by_dsL.items()):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
        for ax in axes:
            style_ax(ax)
        order = {m: i for i, m in enumerate(MODE_ORDER)}
        for mode, f in sorted(entries, key=lambda e: order.get(e[0], 99)):
            tr = np.load(f)
            hg, mb, mr = _median_curves(tr, int(tr["id_offset"]))
            c = MODE_COLOR.get(mode, INK2)
            latch = tr["conv_hop"]
            lp50 = int(np.percentile(latch[latch > 0], 50)) if (latch > 0).any() else None
            axes[0].plot(hg, mb, lw=1.8, color=c, zorder=3)
            axes[1].plot(hg, mr, lw=1.8, color=c, zorder=3)
            for ax, curve in ((axes[0], mb), (axes[1], mr)):
                if lp50 and lp50 <= len(curve):
                    ax.plot([lp50], [curve[lp50 - 1]], "v", ms=7, color=c,
                            mec="white", mew=0.8, zorder=4)
            axes[1].annotate(mode, (hg[-1], mr[-1]), xytext=(4, 0),
                             textcoords="offset points", fontsize=7.5, color=INK,
                             va="center")
            axes[0].annotate(mode, (hg[-1], mb[-1]), xytext=(4, 0),
                             textcoords="offset points", fontsize=7.5, color=INK,
                             va="center")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("best distance / final (median)", fontsize=9, color=INK)
        axes[1].set_ylabel(f"recall@{int(tr['K'])} vs GT (median)", fontsize=9, color=INK)
        for ax in axes:
            ax.set_xlabel("hop", fontsize=9, color=INK)
        fig.suptitle(f"{ds} L={L} — convergence profile per distance mode "
                     f"(▼ = production-detector latch p50)", fontsize=10.5,
                     color=INK)
        save(fig, f"conv_w2_profile_{ds}_L{L}", stamp)


# ---------------- W3: detector comparison ----------------
def fig_w3(res_dir, stamp):
    p = os.path.join(res_dir, "w3_detector_comparison.json")
    if not os.path.exists(p):
        print(f"skip W3 ({p} missing)")
        return
    data = json.load(open(p))
    rows = data["rows"]
    dss = sorted({r["dataset"] for r in rows})
    for ds in dss:
        sub = [r for r in rows if r["dataset"] == ds and r.get("latch_rate", 0) > 0]
        modes = _modes_of(sub)
        Ls = sorted({r["L"] for r in sub})
        for L in Ls:
            fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
            for ax in axes:
                style_ax(ax)
            # (a) early-rate vs hops-saved, one marker per detector, exact vs quantized
            KIND_MARK = {"current": "o", "ema": "s", "turnover": "D",
                         "jaccard": "^", "worst": "P"}
            for r in sub:
                if r["L"] != L or "early_rate" not in r:
                    continue
                c = MODE_COLOR.get(r["mode"], INK2)
                axes[0].scatter(r["early_rate"], r["hops_saved_mean"],
                                marker=KIND_MARK.get(r["kind"], "x"), s=34,
                                color=c, edgecolors="white", linewidths=0.6,
                                zorder=3)
            axes[0].set_xlabel("early-latch rate (latch < oracle)", fontsize=9, color=INK)
            axes[0].set_ylabel("hops saved after latch (mean)", fontsize=9, color=INK)
            axes[0].set_title("detector operating points (marker = rule family)",
                              fontsize=9.5, color=INK)
            for kind, mk in KIND_MARK.items():
                axes[0].scatter([], [], marker=mk, s=30, color=INK2, label=kind)
            axes[0].legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
            # (b) current-rule theta robustness: early-rate vs theta per mode
            for m in modes:
                pts = [(r["params"]["theta"], r["early_rate"]) for r in sub
                       if r["L"] == L and r["mode"] == m and r["kind"] == "current"
                       and "early_rate" in r]
                if not pts:
                    continue
                pts.sort()
                xs, ys = zip(*pts)
                axes[1].plot(xs, ys, "-o", ms=4, lw=1.6,
                             color=MODE_COLOR.get(m, INK2), zorder=3)
                axes[1].annotate(m, (xs[-1], ys[-1]), xytext=(4, 0),
                                 textcoords="offset points", fontsize=7.5,
                                 color=INK, va="center")
            axes[1].set_xscale("log")
            axes[1].set_xlabel("current-rule θ (%)", fontsize=9, color=INK)
            axes[1].set_ylabel("early-latch rate", fontsize=9, color=INK)
            axes[1].set_title("θ robustness of the production rule",
                              fontsize=9.5, color=INK)
            fig.suptitle(f"{ds} L={L} — offline detector comparison vs oracle "
                         f"(oracle = final top-K stable)", fontsize=10.5, color=INK)
            save(fig, f"conv_w3_detectors_{ds}_L{L}", stamp)


# ---------------- W4: rerank gate ----------------
def fig_w4(res_dir, stamp):
    for path in sorted(glob.glob(os.path.join(res_dir, "*_w4_gate.json"))):
        rows = json.load(open(path))
        ds = rows[0]["dataset"]
        Ls = sorted({r["L"] for r in rows})
        fetch_keys = [k for k in rows[0] if k.startswith("rerank_local_")]
        lab = fetch_keys[0].replace("rerank_local_", "") if fetch_keys else None
        L_COL = {L: c for L, c in zip(Ls, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])}
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        for ax in axes:
            style_ax(ax)
        for L in Ls:
            sub = sorted([r for r in rows if r["L"] == L],
                         key=lambda r: (r["gate_alpha"] < 0, r["gate_alpha"]))
            base = next((r for r in sub if r["gate_alpha"] < 0), None)
            gated = [r for r in sub if r["gate_alpha"] >= 0]
            xs = [r["gate_alpha"] for r in gated]
            axes[0].plot(xs, [r["recall"] for r in gated], "-o", ms=4.5, lw=1.8,
                         color=L_COL[L], zorder=3)
            if lab:
                f = [r[f"rerank_local_{lab}"] + r[f"rerank_remote_{lab}"] for r in gated]
                axes[1].plot(xs, f, "-o", ms=4.5, lw=1.8, color=L_COL[L], zorder=3)
                axes[1].annotate(f"L={L}", (xs[-1], f[-1]), xytext=(4, 0),
                                 textcoords="offset points", fontsize=7.5,
                                 color=INK, va="center")
            axes[0].annotate(f"L={L}", (xs[-1], gated[-1]["recall"]),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=7.5, color=INK, va="center")
            if base:
                axes[0].axhline(base["recall"], color=L_COL[L], lw=0.9, ls=":",
                                zorder=2)
                if lab:
                    axes[1].axhline(base[f"rerank_local_{lab}"] +
                                    base[f"rerank_remote_{lab}"],
                                    color=L_COL[L], lw=0.9, ls=":", zorder=2)
        axes[0].set_xlabel("gate α (margin over d̂_K)", fontsize=9, color=INK)
        axes[0].set_ylabel("recall@10", fontsize=9, color=INK)
        axes[1].set_xlabel("gate α (margin over d̂_K)", fontsize=9, color=INK)
        axes[1].set_ylabel("rerank raw fetches / query", fontsize=9, color=INK)
        fig.suptitle(f"{ds} — rerank-pool α-margin gate (pq_rerank beam; "
                     f"dotted = ungated)", fontsize=10.5, color=INK)
        save(fig, f"conv_w4_gate_{ds}", stamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="results_conv_<ds> dir")
    ap.add_argument("--which", default="all",
                    choices=["w1", "w2", "w3", "w4", "all"])
    ap.add_argument("--w2-L", default=None,
                    help="comma list of L values for W2 figures")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    only_L = {int(x) for x in a.w2_L.split(",")} if a.w2_L else None
    if a.which in ("w1", "all"):
        fig_w1(a.res, stamp)
    if a.which in ("w2", "all"):
        fig_w2(a.res, stamp, only_L)
    if a.which in ("w3", "all"):
        fig_w3(a.res, stamp)
    if a.which in ("w4", "all"):
        fig_w4(a.res, stamp)


if __name__ == "__main__":
    main()
