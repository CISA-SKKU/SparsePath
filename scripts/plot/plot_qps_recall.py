#!/usr/bin/env python3
"""Plot QPS vs. recall@10 curves from one or more distributed-search results.csv.

results.csv is written per run by run_experiment.sh (experiments/analysis/stats_to_csv.py):
one row per (seq, L, node) with columns incl. `seq, T, L, recall_10, qps`.
QPS is summed across the 4 workers within each (seq, L) — the paper's "Σ QPS"
convention — and plotted against recall@10; one curve per CSV (and per seq).

Usage:
  scripts/plot/plot_qps_recall.py RUN/results.csv                         # single arm
  scripts/plot/plot_qps_recall.py --label Balanced BAL/results.csv \
                                  --label SparsePath SP/results.csv -o fig13_deep100M_4p.pdf
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path


def load_results(path):
    """{seq: [(L, recall, sum_qps), ...]} sorted by L (recall = first non-empty per (seq,L))."""
    cells = defaultdict(lambda: {"qps": 0.0, "recall": None})
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                key = (int(r["seq"] or 0), int(r["L"]))
                qps = float(r.get("qps") or 0.0)
            except (KeyError, ValueError):
                continue
            c = cells[key]
            c["qps"] += qps
            rec = r.get("recall_10", "")
            if rec not in ("", None) and c["recall"] is None:
                c["recall"] = float(rec)
    out = defaultdict(list)
    for (seq, L), c in sorted(cells.items()):
        if c["recall"] is not None:
            out[seq].append((L, c["recall"], c["qps"]))
    return dict(out)


def iso_qps(rows, target=0.95):
    """Highest Σ QPS among points with recall >= target (paper's iso@0.95), or None."""
    ok = [q for _, rec, q in rows if rec >= target]
    return max(ok) if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="results.csv path(s)")
    ap.add_argument("--label", action="append", default=[], help="curve label, one per CSV (in order)")
    ap.add_argument("-o", "--output", help="output figure (default: next to the first CSV, qps_recall.pdf)")
    ap.add_argument("--title", default="deep100M / 4 workers — QPS vs recall@10")
    ap.add_argument("--iso", type=float, default=0.95, help="recall target for the iso-QPS printout")
    a = ap.parse_intermixed_args()   # allows --label A csvA --label B csvB
    if a.label and len(a.label) != len(a.csv):
        sys.exit("--label count must match the number of CSVs")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, p in enumerate(a.csv):
        p = Path(p)
        if not p.exists():
            sys.exit(f"not found: {p}")
        by_seq = load_results(p)
        if not by_seq:
            sys.exit(f"no (seq, L, recall_10, qps) tuples in {p}")
        base = a.label[i] if a.label else p.parent.name
        for seq, rows in sorted(by_seq.items()):
            lab = base if len(by_seq) == 1 else f"{base} seq={seq}"
            ax.plot([r[1] for r in rows], [r[2] for r in rows], marker="o", ms=3.5, label=lab)
            iso = iso_qps(rows, a.iso)
            print(f"{lab:>24}: {len(rows)} points, iso@{a.iso:.2f} ΣQPS = "
                  f"{iso:,.0f}" if iso is not None else f"{lab:>24}: {len(rows)} points, no point reaches recall {a.iso}")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("QPS (Σ over workers)")
    ax.set_title(a.title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = a.output or str(Path(a.csv[0]).parent / "qps_recall.pdf")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
