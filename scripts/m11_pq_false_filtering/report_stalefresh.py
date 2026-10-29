#!/usr/bin/env python
"""Summarise one M11 stale/fresh PQ pair.

HEADLINE MODE = the PARTITION-AWARE row (`nc_LITS_4p` at 100M, `nc_LITS_8p` at 200M).
That mode filters a candidate only when it is actually remote
(partition_of[gid] != coordinator[qi]) and computes exact distances for local ones --
i.e. what the deployed system does.  It is auto-detected as the unique `nc_*` mode.

`filter_all` is NOT the headline: it pretends every candidate is remote (the M->infinity
limit), so it filters far more aggressively than the real system ever does and reports a
much larger recall cost.  It is carried only as a clearly-labelled secondary column.

Reports:
  * recall(OFF) agreement between the two arms (must be identical at every L --
    recall(OFF) is codebook-independent, so a mismatch means the setup is wrong),
  * Delta = recall(OFF) - recall(ON) and false-filter% at a chosen alpha for chosen Ls,
  * the operating point (smallest L with recall(ON) >= 0.95) and Delta there,
  * the headline vintage effect = Delta(stale) - Delta(fresh).

Usage:
  python report_stalefresh.py --dir <results_dir> --stale <key> --fresh <key> \
      [--alpha 0.15] [--Ls 100 400] [--target 0.95] [--mode nc_LITS_4p]
"""
import argparse
import json
import os


def load(d, key, mode=None):
    """Returns (sanity_corr, headline_rows, upper_bound_rows, mode_name)."""
    with open(os.path.join(d, f"{key}_rows.json")) as f:
        blob = json.load(f)
    modes = {r["mode"] for r in blob["rows"]}
    if mode is None:
        nc = sorted(m for m in modes if m.startswith("nc_"))
        if len(nc) != 1:
            raise SystemExit(
                f"{key}: cannot auto-pick the partition-aware mode from {sorted(modes)}; "
                f"pass --mode explicitly")
        mode = nc[0]
    elif mode not in modes:
        raise SystemExit(f"{key}: mode {mode!r} not present; have {sorted(modes)}")
    head = [r for r in blob["rows"] if r["mode"] == mode]
    ub = [r for r in blob["rows"] if r["mode"] == "filter_all"]
    return blob["sanity_corr"], head, ub, mode


def pick(rows, L, alpha):
    for r in rows:
        if r["L"] == L and abs(r["alpha"] - alpha) < 1e-9:
            return r
    return None


def op_point(rows, alpha, target):
    """smallest L whose recall(ON) >= target, at the given alpha"""
    cands = sorted({r["L"] for r in rows})
    for L in cands:
        r = pick(rows, L, alpha)
        if r and r["recall_on"] >= target:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stale-dir", default=None, help="defaults to --dir")
    ap.add_argument("--fresh-dir", default=None, help="defaults to --dir")
    ap.add_argument("--stale", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--Ls", type=int, nargs="+", default=[100, 400])
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--mode", default=None,
                    help="partition-aware mode; auto-detected as the unique nc_* mode")
    a = ap.parse_args()

    corr_s, rs, ubs, mode_s = load(a.stale_dir or a.dir, a.stale, a.mode)
    corr_f, rf, ubf, mode_f = load(a.fresh_dir or a.dir, a.fresh, a.mode)
    if mode_s != mode_f:
        raise SystemExit(f"arms disagree on partition-aware mode: {mode_s} vs {mode_f}")

    print(f"# pair: {a.stale}  vs  {a.fresh}   (alpha={a.alpha})")
    print(f"headline mode = {mode_s}  (partition-aware: filters only genuinely remote "
          f"candidates -- what the deployed system does)")
    print(f"sanity pq-vs-exact corr: stale={corr_s:.3f}  fresh={corr_f:.3f}  "
          f"({'OK' if min(corr_s, corr_f) > 0.5 else 'SUSPECT'})")

    # --- recall(OFF) must match arm-to-arm, at every L ---
    off_s = {r["L"]: r["recall_off"] for r in rs}
    off_f = {r["L"]: r["recall_off"] for r in rf}
    common = sorted(set(off_s) & set(off_f))
    bad = [L for L in common if abs(off_s[L] - off_f[L]) > 1e-9]
    if bad:
        print(f"!! recall(OFF) MISMATCH at L={bad} -- setup is wrong, do not report this pair")
        for L in bad:
            print(f"     L={L}: stale={off_s[L]:.4f} fresh={off_f[L]:.4f}")
    else:
        print(f"recall(OFF) identical across all {len(common)} L values: OK")

    # --- per-L table (headline = partition-aware; filter_all carried as a labelled UB) ---
    print(f"\n| L | arm | recall(OFF) | recall(ON) | Delta | false-filter% | "
          f"[UB] filter_all Delta |")
    print(f"|---|-----|-------------|------------|-------|---------------|"
          f"----------------------|")
    for L in a.Ls:
        for tag, rows, ub in (("stale", rs, ubs), ("fresh", rf, ubf)):
            r = pick(rows, L, a.alpha)
            if not r:
                print(f"| {L} | {tag} | (missing) | | | | |")
                continue
            u = pick(ub, L, a.alpha)
            ustr = f"{u['recall_delta']:+.4f}" if u else "-"
            print(f"| {L} | {tag} | {r['recall_off']:.4f} | {r['recall_on']:.4f} | "
                  f"{r['recall_delta']:+.4f} | {r['false_filter_rate_of_eligible']*100:.3f} | "
                  f"{ustr} |")
        a_s, a_f = pick(rs, L, a.alpha), pick(rf, L, a.alpha)
        if a_s and a_f:
            print(f"| {L} | **vintage effect** | | | "
                  f"**{a_s['recall_delta'] - a_f['recall_delta']:+.4f}** | | |")

    # --- operating point ---
    print(f"\noperating point (smallest L with recall(ON) >= {a.target}, alpha={a.alpha}):")
    ops = {}
    for tag, rows in (("stale", rs), ("fresh", rf)):
        r = op_point(rows, a.alpha, a.target)
        ops[tag] = r
        if r:
            print(f"  {tag}: L={r['L']}  recall(ON)={r['recall_on']:.4f}  "
                  f"Delta={r['recall_delta']:+.4f}  ff%={r['false_filter_rate_of_eligible']*100:.3f}")
        else:
            print(f"  {tag}: never reaches {a.target} in the swept L range")
    if ops["stale"] and ops["fresh"]:
        print(f"  vintage effect at each arm's own operating point: "
              f"{ops['stale']['recall_delta'] - ops['fresh']['recall_delta']:+.4f}")
        Lc = max(ops["stale"]["L"], ops["fresh"]["L"])
        cs, cf = pick(rs, Lc, a.alpha), pick(rf, Lc, a.alpha)
        if cs and cf:
            print(f"  vintage effect at the common operating L={Lc}: "
                  f"{cs['recall_delta'] - cf['recall_delta']:+.4f} "
                  f"(stale {cs['recall_delta']:+.4f} vs fresh {cf['recall_delta']:+.4f})")


if __name__ == "__main__":
    main()
