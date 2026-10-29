#!/usr/bin/env python
"""Compare M11 false-filtering across PQ codebooks (M12 / M32_stale / M32).

Builds the 2x2 that separates the two variables the raw M12-vs-M32 delta confounds:

              trained [0,100M) = STALE      trained [150M,250M) = FRESH
  M=12        pq/M12                        --
  M=32        pq/M32_stale                  pq/M32

  M effect       = M32_stale - M12          (vintage fixed at stale)
  vintage effect = M32       - M32_stale    (M fixed at 32)

Reads the per-run *_rows.json the harness writes and prints markdown tables at
alpha=0.15 for L=100 / L=400, plus the operating point (smallest L with
recall(ON) >= --target).

Usage:
  python compare_codebooks.py \
     --run M12=/experiments/results/m11_pq_false_filtering/deep100M_iter150 \
     --run M32=/experiments/results/m11_pq_false_filtering/M32_rerun_20260803 \
     [--alpha 0.15] [--target 0.95]
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load_rows(path):
    """path may be a directory (all *_rows.json inside) or a single json file."""
    out = []
    files = [path] if path.endswith(".json") else sorted(glob.glob(os.path.join(path, "*_rows.json")))
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for r in d.get("rows", []):
            r = dict(r)
            r["_sanity_corr"] = d.get("sanity_corr")
            r["_file"] = os.path.basename(fp)
            out.append(r)
    return out


def norm_mode(m):
    """strip run-specific prefixes so modes line up across codebooks."""
    return m.replace("nc_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=path (dir or *_rows.json). Repeat; order = column order.")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--Ls", default="100,400")
    a = ap.parse_args()

    labels, data = [], {}
    for spec in a.run:
        lab, path = spec.split("=", 1)
        labels.append(lab)
        data[lab] = load_rows(path)
        corrs = {r["_sanity_corr"] for r in data[lab] if r.get("_sanity_corr")}
        print(f"# {lab}: {len(data[lab])} rows from {path}")
        print(f"#   sanity pq-vs-exact corr = {sorted(round(c,4) for c in corrs)}")
    print()

    def pick(lab, mode, L):
        for r in data[lab]:
            if norm_mode(r["mode"]) == mode and int(r["L"]) == L and abs(r["alpha"] - a.alpha) < 1e-9:
                return r
        return None

    modes = []
    for lab in labels:
        for r in data[lab]:
            m = norm_mode(r["mode"])
            if m not in modes:
                modes.append(m)

    for L in [int(x) for x in a.Ls.split(",")]:
        print(f"## recall Delta @ L={L}, alpha={a.alpha}\n")
        print("| mode | " + " | ".join(f"{l} Δ" for l in labels) + " | " +
              " | ".join(f"{l} ff%" for l in labels) + " |")
        print("|---" * (1 + 2 * len(labels)) + "|")
        for m in modes:
            rs = [pick(l, m, L) for l in labels]
            if not any(rs):
                continue
            d = " | ".join(f"{r['recall_delta']:.4f}" if r else "-" for r in rs)
            ff = " | ".join(f"{100*r['false_filter_rate_of_eligible']:.3f}" if r else "-" for r in rs)
            print(f"| {m} | {d} | {ff} |")
        print()

    print(f"## operating point (smallest L with recall(ON) >= {a.target}), alpha={a.alpha}\n")
    print("| mode | " + " | ".join(f"{l} L* / Δ@L* / ON" for l in labels) + " |")
    print("|---" * (1 + len(labels)) + "|")
    for m in modes:
        cells = []
        any_ = False
        for lab in labels:
            rs = sorted([r for r in data[lab]
                         if norm_mode(r["mode"]) == m and abs(r["alpha"] - a.alpha) < 1e-9],
                        key=lambda r: int(r["L"]))
            ok = [r for r in rs if r["recall_on"] >= a.target]
            if ok:
                r = ok[0]; any_ = True
                cells.append(f"{int(r['L'])} / {r['recall_delta']:.4f} / {r['recall_on']:.4f}")
            else:
                cells.append("-" if rs else "n/a")
        if any_:
            print(f"| {m} | " + " | ".join(cells) + " |")
    print()

    if len(labels) >= 2:
        print("## variable separation (needs M12, M32_stale, M32 labels)\n")
        need = ("M12", "M32_stale", "M32")
        if all(n in labels for n in need):
            for L in [int(x) for x in a.Ls.split(",")]:
                print(f"**L={L}**\n")
                print("| mode | M12 | M32_stale | M32 | M effect (stale→M32s) | vintage effect (M32s→M32) |")
                print("|---|---|---|---|---|---|")
                for m in modes:
                    r12, rs32, rf32 = (pick("M12", m, L), pick("M32_stale", m, L), pick("M32", m, L))
                    if not (r12 and rs32 and rf32):
                        continue
                    print(f"| {m} | {r12['recall_delta']:.4f} | {rs32['recall_delta']:.4f} | "
                          f"{rf32['recall_delta']:.4f} | {rs32['recall_delta']-r12['recall_delta']:+.4f} | "
                          f"{rf32['recall_delta']-rs32['recall_delta']:+.4f} |")
                print()
        else:
            print(f"(skipped — need labels {need}, have {labels})\n")


if __name__ == "__main__":
    main()
