#!/usr/bin/env python
"""A1 (M08) figures from results_a1_{deep100M,spacev100M}.

Fig-A  memory-recall Pareto (@L400): searchable bytes/vec (log2) vs recall@10,
       quantizer families as series; rerank points get a dual-accounting arrow
       (code-only DRAM position -> code+raw resident position).
Fig-B  bytes/query-recall: remote traffic per query (log10) vs recall@10 over the
       L sweep for the 4 system configs (B-r, B-q owner-side, SP-r, SP-q SQ8-fetch).

Colors: validated categorical slots (blue/orange/aqua/yellow), identity fixed;
direct labels everywhere (contrast WARN relief).
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.expanduser(
    "/experiments/results/m08_quantization_pareto")
DOCS_FIG = os.path.expanduser(
    "./figures")

SURFACE = "#fcfcfb"
INK, INK2 = "#1a1a19", "#5f5e56"
C_RAW, C_PQ, C_SQ8, C_RBQ = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_BR, C_BQ, C_SPR, C_SPQ = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

DS = {
    "deep100M":   dict(raw=384, sq8=96, pq=12, rbq=12, dim=96, esz=4),
    "spacev100M": dict(raw=100, sq8=100, pq=10, rbq=13, dim=100, esz=1),
}
K = 10


def load(ds, graph):
    p = os.path.join(RES, f"results_a1_{ds}", f"{ds}_{graph}_q1_pareto.json")
    return {r["L"]: r for r in json.load(open(p))}


def load_m32(ds="deep100M"):
    d = {"deep100M": "results_a1_m32", "spacev100M": "results_a1_m32_spacev"}[ds]
    p = os.path.join(RES, d, f"{ds}_B_iter150_m32_q1_pareto.json")
    return {r["L"]: r for r in json.load(open(p))}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(True, color="#e8e7e0", lw=0.6, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8)


def fig_a(stamp):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor=SURFACE)
    for ax, ds in zip(axes, DS):
        b = DS[ds]
        r = load(ds, "B_iter150")[400]     # graph axis is recall-neutral (verified)
        style_ax(ax)
        is_int8 = b["raw"] == b["sq8"]
        pts = [  # (family, color, code_bytes, no-rerank key, rerank key, label dy)
            ("raw (= SQ8, int8)" if is_int8 else "raw",
                       C_RAW, b["raw"], "recall_raw", None, 10),
            ("PQ",     C_PQ,  b["pq"],  "recall_pq",  "recall_pq_rerank", 10),
            ("RaBitQ", C_RBQ, b["rbq"], "recall_rabitq", "recall_rabitq_rerank", -18),
        ]
        if not is_int8:   # on int8 sets SQ8 coincides with raw — one point, one label
            pts.insert(1, ("SQ8", C_SQ8, b["sq8"], "recall_sq8", None, -16))
        # A1-M: standard 32B navigation budget (same PQ family, both datasets)
        m32 = load_m32(ds)[400]
        r = dict(r)
        r["recall_pq32"] = m32["recall_pq"]
        r["recall_pq32_rerank"] = m32["recall_pq_rerank"]
        pts.append(("PQ-32B", C_PQ, 32, "recall_pq32", "recall_pq32_rerank", 10))
        for name, c, xb, k0, k1, dy in pts:
            y0 = r[k0]
            ax.scatter([xb], [y0], s=46, facecolor=c, edgecolor=SURFACE,
                       lw=1.2, zorder=3)
            ax.annotate(name, (xb, y0), textcoords="offset points",
                        xytext=(6, dy), ha="left", fontsize=7.5, color=INK)
            if k1:
                y1 = r[k1]
                x1 = xb + b["raw"]          # accounting (1): raw must stay resident
                ax.scatter([xb], [y1], s=46, facecolor=SURFACE, edgecolor=c,
                           lw=1.6, zorder=3)
                ax.scatter([x1], [y1], s=46, facecolor=c, edgecolor=SURFACE,
                           lw=1.2, zorder=3)
                ax.annotate("", (x1, y1), (xb, y1),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.1))
                ax.annotate(f"{name}+rerank", (x1, y1), textcoords="offset points",
                            xytext=(-4, dy), ha="right", fontsize=7.5, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xlim(b["pq"] * 0.5, b["raw"] * 3.2)
        ax.set_ylim(0.08, 1.06)
        ax.set_xlabel("searchable bytes / vector (log2)", fontsize=8.5, color=INK)
        ax.set_title(f"{ds}  (L=400, iteration150)", fontsize=9.5, color=INK)
    axes[0].set_ylabel(f"recall@{K}", fontsize=8.5, color=INK)
    from matplotlib.lines import Line2D
    fam = [("raw", C_RAW), ("PQ", C_PQ), ("SQ8", C_SQ8), ("RaBitQ", C_RBQ)]
    axes[0].legend(handles=[Line2D([], [], ls="", marker="o", ms=6, mfc=c,
                                   mec=SURFACE, label=n) for n, c in fam],
                   fontsize=6.8, loc="center right", frameon=False, labelcolor=INK)
    axes[1].annotate("open marker = code-only DRAM (raw displaced)\n"
                     "arrow -> code + raw resident (in-memory serving)",
                     xy=(0.02, 0.05), xycoords="axes fraction",
                     fontsize=7.2, color=INK2)
    fig.tight_layout()
    out = os.path.join(RES, f"{stamp}_a1_fig_a_memory_recall.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    return out, "a1_memory_recall_pareto.png"


def fig_b(stamp):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor=SURFACE)
    for ax, ds in zip(axes, DS):
        b = DS[ds]
        rawB = b["raw"]
        rb = load(ds, "B_iter150")
        rs = load(ds, "SP_lits150")
        Ls = sorted(rb)
        style_ax(ax)

        handles = []

        def curve(xs, ys, c, label, dashed=False, dy=-2, label_at=-1):
            (ln,) = ax.plot(xs, ys, "-" if not dashed else "--", color=c, lw=2,
                            marker="o", ms=4.5, mec=SURFACE, mew=0.8, zorder=3,
                            label=label)
            handles.append(ln)
            ax.annotate(label.split(" (")[0], (xs[label_at], ys[label_at]),
                        textcoords="offset points", xytext=(5, dy),
                        fontsize=7.5, color=INK)

        # B-r: balanced placement, exact traversal -> remote raw fetches
        curve([rb[L]["scored_remote_balanced_raw"] * rawB for L in Ls],
              [rb[L]["recall_raw"] for L in Ls], C_BR, "B-r (Balanced, raw)", dy=6)
        # SP-r: lits placement, exact traversal
        curve([rs[L]["scored_remote_lits_raw"] * rawB for L in Ls],
              [rs[L]["recall_raw"] for L in Ls], C_SPR, "SP-r (lits, raw)",
              dy=-12, label_at=0)
        # B-q: local codes + owner-side rerank (computation shipping, generous):
        # query broadcast to M owners + candidate gids (8B each) + K distances back
        curve([4 * (b["dim"] * b["esz"] + 8 * K) + 8 * L for L in Ls],
              [rb[L]["recall_pq_rerank"] for L in Ls], C_BQ,
              "B-q (owner-side rerank)", dy=6, label_at=0)
        # B-q naive fetch-raw rerank (dashed reference)
        curve([rb[L]["rerank_remote_balanced_pq_rerank"] * rawB for L in Ls],
              [rb[L]["recall_pq_rerank"] for L in Ls], C_BQ, "B-q (naive fetch)",
              dashed=True)
        # SP-q: lits + SQ8 codes fetched remotely — Phase B target. On int8 sets
        # SQ8 = raw, so the curve coincides with SP-r: annotate instead of plotting.
        if b["sq8"] < b["raw"]:
            curve([rs[L]["scored_remote_lits_sq8"] * b["dim"] for L in Ls],
                  [rs[L]["recall_sq8"] for L in Ls], C_SPQ,
                  "SP-q (lits, SQ8 fetch)", dy=6, label_at=0)
        else:
            ax.annotate("SP-q ≡ SP-r (int8: SQ8 = raw)", xy=(0.98, 0.32),
                        xycoords="axes fraction", ha="right",
                        fontsize=7.2, color=INK2)

        ax.legend(handles=handles, fontsize=6.8, loc="lower right",
                  frameon=False, labelcolor=INK)
        ax.set_xscale("log")
        ax.set_ylim(0.35, 1.05)
        ax.set_xlabel("remote bytes / query (accounting)", fontsize=8.5, color=INK)
        ax.set_title(ds, fontsize=9.5, color=INK)
    axes[0].set_ylabel(f"recall@{K}", fontsize=8.5, color=INK)
    axes[0].annotate("anchor (Q4, 4-node measured @L100):\n"
                     "Balanced 73 Gbps saturated / lits 45 Gbps",
                     xy=(0.02, 0.05), xycoords="axes fraction",
                     fontsize=7.2, color=INK2)
    fig.tight_layout()
    out = os.path.join(RES, f"{stamp}_a1_fig_b_bytes_query_recall.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    return out, "a1_bytes_query_recall.png"


def fig_c(stamp):
    """QPS-recall Pareto, deep100M only: QPS = Q4 4-node MEASURED (iter150, exact,
    placement rr vs lits), recall = A1 measured at the same L on the same
    checkpoints. B-q has no distributed QPS measurement -> shown as its recall
    ceiling (vertical line): no QPS can move it right of that line."""
    q4_qps = {  # measured, Q4 doc (deep100M 4p iter150, T=64)
        "B-r":  {50: 23474, 100: 14184, 200: 7770, 400: 4288},
        "SP-r": {50: 29674, 100: 25063, 200: 14815, 400: 8251},
    }
    rb = load("deep100M", "B_iter150")
    rs = load("deep100M", "SP_lits150")
    rec = {"B-r": {L: rb[L]["recall_raw"] for L in q4_qps["B-r"]},
           "SP-r": {L: rs[L]["recall_raw"] for L in q4_qps["SP-r"]}}
    ceil_bq = rb[400]["recall_pq_rerank"]      # navigation-bound ceiling
    ceil_sq8 = rb[400]["recall_sq8"]           # SP-q (SQ8) potential ceiling

    fig, ax = plt.subplots(figsize=(5.8, 4.0), facecolor=SURFACE)
    style_ax(ax)
    Ls = sorted(q4_qps["B-r"])
    for name, c in (("B-r (Balanced, raw)", C_BR), ("SP-r (lits, raw)", C_SPR)):
        key = name.split(" ")[0]
        xs = [rec[key][L] for L in Ls]
        ys = [q4_qps[key][L] / 1000.0 for L in Ls]
        ax.plot(xs, ys, "-", color=c, lw=2, marker="o", ms=5, mec=SURFACE,
                mew=0.8, zorder=3, label=name + " — measured")
        for L, x, y in zip(Ls, xs, ys):
            ax.annotate(f"L{L}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=6.8, color=INK2)
        ax.annotate(key, (xs[1], ys[1]), textcoords="offset points",
                    xytext=(-6, -12 if key == "B-r" else 8), ha="right",
                    fontsize=8, color=INK)
    # Estimated curves anchored on MEASURED compute ratios (A1-T, worker-host, warm rows):
    #   pq_rerank(M12)/exact = 1.38-1.47x ; pq_rerank(M32)/exact = 0.98-0.99x ;
    #   SQ8 kept at parity (harness SQ8 is a 3-pass simulator, not representative).
    # Estimated QPS = SP-r measured QPS x ratio, at the config's own recall.
    m32 = load_m32()
    spr_qps = [q4_qps["SP-r"][L] / 1000.0 for L in Ls]
    C_BQ_LIGHT = "#f5a582"
    for key, src, c, rkey, ratio in (
            ("B-q 12B", rb, C_BQ_LIGHT, "recall_pq_rerank", 1.4),
            ("B-q 32B", m32, C_BQ, "recall_pq_rerank", 1.0),
            ("SP-q SQ8", rs, C_SPQ, "recall_sq8", 1.0)):
        xs = [src[L][rkey] for L in Ls]
        ys = [q * ratio for q in spr_qps]
        ax.plot(xs, ys, "--", color=c, lw=1.8, marker="o", ms=5,
                mfc=SURFACE, mec=c, mew=1.4, zorder=3,
                label=f"{key} — est. (measured {ratio}x compute x Q4 anchor)")
        ax.annotate(key, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(4, -11), fontsize=7.5, color=INK)
    # Full SparsePath (CAP: PQ filter + batching) — Q4b measured point @L100.
    # recall shown = A1 exact - M11 filter delta (0.014 @L100, alpha 0.15).
    x_full, y_full = rec["SP-r"][100] - 0.014, 34.0
    ax.scatter([x_full], [y_full], s=90, marker="*", facecolor=C_SPR,
               edgecolor=INK, lw=0.6, zorder=4,
               label="SP-full (+PQ filter+batching) — Q4b measured")
    ax.annotate("SP-full L100\n(Q4b, 34.0K)", (x_full, y_full),
                textcoords="offset points", xytext=(-8, 2), ha="right",
                fontsize=7.2, color=INK)
    ax.axvline(ceil_bq, color=C_BQ_LIGHT, ls="--", lw=1.0, alpha=0.6)
    ax.annotate(f"12B ceiling {ceil_bq:.3f}", (ceil_bq, 42.5), ha="right",
                fontsize=7.0, color=C_BQ_LIGHT, xytext=(-4, 0),
                textcoords="offset points")
    ax.set_xlim(0.38, 1.005)
    ax.set_ylim(0, 44)
    ax.set_xlabel("recall@10  (A1 measured, iteration150)", fontsize=8.5, color=INK)
    ax.set_ylabel("QPS x1000  (Q4 measured, 4 nodes)", fontsize=8.5, color=INK)
    ax.set_title("deep100M — QPS-recall Pareto (measured)", fontsize=9.5, color=INK)
    ax.legend(fontsize=6.8, loc="lower left", frameon=False, labelcolor=INK)
    fig.tight_layout()
    out = os.path.join(RES, f"{stamp}_a1_fig_c_qps_recall.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    return out, "a1_qps_recall_pareto.png"


def fig_d(stamp):
    """Single-node MEASURED QPS-recall (worker-host, 64T, deep100M iter150, B graph).
    Everything is local (no NIC, no placement) -> pure compute comparison of
    search modes. L50 / warmup rows excluded (cold-start). SQ8 timing excluded
    (harness simulator, not a representative kernel)."""
    t12 = {r["L"]: r for r in json.load(open(os.path.join(
        RES, "results_a1_timing", "deep100M_B_iter150_q1_pareto.json")))}
    t32 = {r["L"]: r for r in json.load(open(os.path.join(          # dup L100:
        RES, "results_a1_timing_m32", "deep100M_B_iter150_q1_pareto.json")))}
    # dict-comp keeps the LAST duplicate row = the warm L100 run.

    fig, ax = plt.subplots(figsize=(5.6, 3.9), facecolor=SURFACE)
    style_ax(ax)
    C_BQ_LIGHT = "#f5a582"
    series = [
        ("exact (raw)",     C_RAW,      t12, "recall_raw", "qps_1node_raw", [100, 200, 400]),
        ("PQ-12B + rerank", C_BQ_LIGHT, t12, "recall_pq_rerank", "qps_1node_pq_rerank", [100, 200, 400]),
        ("PQ-32B + rerank", C_PQ,       t32, "recall_pq_rerank", "qps_1node_pq_rerank", [100, 400]),
    ]
    for name, c, src, rk, qk, Ls in series:
        xs = [src[L][rk] for L in Ls]
        ys = [src[L][qk] / 1000.0 for L in Ls]
        ax.plot(xs, ys, "-", color=c, lw=2, marker="o", ms=5, mec=SURFACE,
                mew=0.8, zorder=3, label=name)
        for L, x, y in zip(Ls, xs, ys):
            ax.annotate(f"L{L}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=6.8, color=INK2)
        ax.annotate(name.split(" (")[0], (xs[0], ys[0]),
                    textcoords="offset points", xytext=(-6, 6), ha="right",
                    fontsize=7.5, color=INK)
    ax.set_xlim(0.48, 1.005)
    ax.set_ylim(0, 40)
    ax.set_xlabel("recall@10", fontsize=8.5, color=INK)
    ax.set_ylabel("QPS x1000  (single node, worker-host 64T, measured)",
                  fontsize=8.5, color=INK)
    ax.set_title("deep100M — single-node QPS-recall (all-local: compute only)",
                 fontsize=9.5, color=INK)
    ax.legend(fontsize=6.8, loc="lower left", frameon=False, labelcolor=INK)
    ax.annotate("no NIC / no placement here — quantization's bandwidth\n"
                "benefit does not appear on a single node by construction",
                xy=(0.98, 0.72), xycoords="axes fraction", ha="right",
                fontsize=7.0, color=INK2)
    fig.tight_layout()
    out = os.path.join(RES, f"{stamp}_a1_fig_d_qps_recall_1node.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    return out, "a1_qps_recall_1node.png"


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(DOCS_FIG, exist_ok=True)
    for fn in (fig_a, fig_b, fig_c, fig_d):
        raw_path, clean = fn(stamp)
        clean_path = os.path.join(DOCS_FIG, clean)
        import shutil
        shutil.copy(raw_path, clean_path)
        print(f"{raw_path}\n  -> {clean_path}")


if __name__ == "__main__":
    main()
