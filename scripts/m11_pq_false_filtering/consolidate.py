#!/usr/bin/env python
"""Aggregate all M11 per-run rows.json into a master CSV + a headline summary.

Usage: python consolidate.py [results_dir]
  results_dir default: 02_SparsePath_experiments/results/m11_pq_false_filtering
Outputs (in results_dir):
  MASTER.csv      — every (dataset, run, mode, L, alpha) row, all metrics
  SUMMARY.md      — headline: recall Δ / false-filter% at L=100, alpha=0.15, per (dataset, mode)
"""
import json, glob, csv, os, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    "/experiments/results/m11_pq_false_filtering"

FIELDS = ["run", "dataset", "mode", "dim", "L", "alpha", "recall_on", "recall_off",
          "recall_delta", "false_filter_rate_of_eligible", "false_filter_rate_of_dropped",
          "drop_rate", "n_filter_eligible", "n_pq_filtered", "n_false_local", "avg_hops"]

rows = []
for rj in sorted(glob.glob(os.path.join(BASE, "**", "*_rows.json"), recursive=True)):
    run = os.path.basename(os.path.dirname(rj))
    d = json.load(open(rj))
    for r in d.get("rows", []):
        r = dict(r); r["run"] = run
        rows.append(r)

os.makedirs(BASE, exist_ok=True)
with open(os.path.join(BASE, "MASTER.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"MASTER.csv: {len(rows)} rows from {len(set(r['run'] for r in rows))} runs")

# ---- headline summary: L=100, alpha=0.15, per (dataset, mode) ----
def sel(L, a):
    return [r for r in rows if r.get("L") == L and abs(r.get("alpha", -9) - a) < 1e-6]

with open(os.path.join(BASE, "SUMMARY.md"), "w") as f:
    f.write("# M11 PQ False-Filtering — headline summary\n\n")
    f.write("recall Δ = recall(no-filter oracle) − recall(PQ filter on), at **L=100, α=0.15**, "
            "nearest-centroid routing. Small Δ = false filtering negligible.\n\n")
    f.write("| dataset | mode (placement/Np) | recall(OFF) | recall(ON) | **recall Δ** | false-filter%(elig) |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in sorted(sel(100, 0.15), key=lambda r: (r["run"], r["mode"])):
        f.write(f"| {r['run']} | {r['mode']} | {r['recall_off']:.4f} | {r['recall_on']:.4f} | "
                f"{r['recall_delta']:+.4f} | {r['false_filter_rate_of_eligible']*100:.3f} |\n")
    # L-dependence for the SparsePath-relevant modes (Centroid/LITS)
    f.write("\n## L-dependence (Centroid/LITS modes, α=0.15) — Δ shrinks as L grows\n\n")
    f.write("| run | mode | " + " | ".join(f"L{L}" for L in [50,100,200,400]) + " |\n")
    f.write("|---|---|" + "---|" * 4 + "\n")
    seen = set()
    for r in rows:
        if abs(r.get("alpha", -9) - 0.15) > 1e-6: continue
        if not any(k in r["mode"] for k in ("Centroid", "LITS")): continue
        key = (r["run"], r["mode"])
        if key in seen: continue
        seen.add(key)
        deltas = {}
        for L in [50, 100, 200, 400]:
            m = [x for x in rows if x["run"] == r["run"] and x["mode"] == r["mode"]
                 and x.get("L") == L and abs(x.get("alpha", -9) - 0.15) < 1e-6]
            deltas[L] = f"{m[0]['recall_delta']:+.4f}" if m else "-"
        f.write(f"| {r['run']} | {r['mode']} | " + " | ".join(deltas[L] for L in [50,100,200,400]) + " |\n")
print("SUMMARY.md written")
