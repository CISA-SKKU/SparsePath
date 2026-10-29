#!/usr/bin/env python
"""W3 — offline convergence-detector comparison on W2 traces (no re-search).

Input: trace npz files produced by run_conv.py (per dataset/mode/L): per-hop
{best, worst, n_inserted, beam_size, topk} streams + recorded conv_hop (C++
latch) + final result_tags + gt.

Oracles (per query, hops are 1-based):
  oracle_stable : first hop from which the beam top-K SET never changes again.
  oracle_recall : first hop from which recall@K vs GT stays at its final value.

Detectors (latch = first hop where the rule fires; None = never):
  current(theta, w)   : rel. best-dist improvement < theta% for w consecutive hops
                        (verbatim ConvergenceDetector semantics; theta=1.0,w=3
                        must reproduce the C++ conv_hop -> built-in validation).
  ema(theta, lam)     : EMA of rel. improvement < theta%.
  turnover(theta, w)  : beam insertions / beam_size < theta for w hops.
  jaccard(w)          : top-K set identical (J=1) to previous hop, w consecutive.
  worst(theta, w)     : current-rule applied to the WORST (d_L) stream.

Metrics per (detector, param): latch hop stats, delta vs oracle, early rate
(latch < oracle_stable — the CAP risk direction), miss rate (never latched),
hops saved after latch, and recall_at_latch - final_recall (what acting at the
latch point would cost).

Usage:
  python analyze_w3.py --traces results_conv_deep100M/traces --out results_conv_deep100M
"""
import argparse
import glob
import json
import os
import numpy as np

UMAX = np.iinfo(np.uint32).max


def per_query(tr):
    """Yield (qi, best, worst, n_ins, beam, topk[hops,K]) slices."""
    offs = tr["offsets"]
    K = int(tr["K"])
    topk = tr["topk"]
    for qi in range(len(offs) - 1):
        a, b = int(offs[qi]), int(offs[qi + 1])
        yield qi, tr["best"][a:b], tr["worst"][a:b], tr["n_inserted"][a:b], \
            tr["beam_size"][a:b], topk[a:b].reshape(b - a, K)


def topk_sets(tk):
    return [frozenset(int(t) for t in row if t != UMAX) for row in tk]


def oracle_stable(sets):
    """First 1-based hop from which the top-K set never changes again."""
    final = sets[-1]
    h = len(sets)
    for i in range(len(sets) - 1, -1, -1):
        if sets[i] == final:
            h = i + 1
        else:
            break
    return h


def recall_curve(tk, gt_row, id_offset):
    g = set(int(x) for x in gt_row)
    K = len(gt_row)
    out = np.empty(len(tk), np.float32)
    for i, row in enumerate(tk):
        f = set(int(t) - 1 + id_offset for t in row if t != UMAX)
        out[i] = len(f & g) / K
    return out


def oracle_from_curve(curve):
    """First 1-based hop from which the curve stays at its final value."""
    final = curve[-1]
    h = len(curve)
    for i in range(len(curve) - 1, -1, -1):
        if curve[i] >= final:
            h = i + 1
        else:
            break
    return h


def rel_impr_pct(stream):
    """delta_pct[h] for h>=1 (vs prev hop); delta_pct[0] = inf."""
    s = np.asarray(stream, np.float64)
    out = np.full(len(s), np.inf)
    prev = s[:-1]
    ok = prev > 0
    out[1:][ok] = np.abs(s[1:][ok] - prev[ok]) / prev[ok] * 100.0
    return out


def latch_consec(cond, w, start=1):
    """First 1-based hop where cond held for w consecutive hops (cond indexed
    per hop; hops < start never count). Mirrors stable_count semantics."""
    c = 0
    for h in range(start, len(cond)):
        if cond[h]:
            c += 1
            if c >= w:
                return h + 1   # 1-based hop at which the detector fires
        else:
            c = 0
    return None


def det_current(best, theta, w):
    return latch_consec(rel_impr_pct(best) < theta, w)


def det_worst(worst, theta, w):
    return latch_consec(rel_impr_pct(worst) < theta, w)


def det_ema(best, theta, lam):
    d = rel_impr_pct(best)
    ema = None
    for h in range(1, len(d)):
        ema = d[h] if ema is None else lam * d[h] + (1 - lam) * ema
        if ema < theta:
            return h + 1
    return None


def det_turnover(n_ins, beam, theta, w):
    frac = n_ins.astype(np.float64) / np.maximum(1, beam.astype(np.float64))
    return latch_consec(frac < theta, w)


def det_jaccard(sets, w):
    same = [False] + [sets[i] == sets[i - 1] for i in range(1, len(sets))]
    return latch_consec(np.asarray(same), w)


DET_GRID = (
    [(f"current_t{t}", "current", dict(theta=t, w=3)) for t in (0.1, 0.5, 1.0, 2.0, 5.0)] +
    [(f"ema_t{t}", "ema", dict(theta=t, lam=0.3)) for t in (0.1, 0.5, 1.0, 2.0)] +
    [(f"turnover_t{t}", "turnover", dict(theta=t, w=3)) for t in (0.02, 0.05, 0.1, 0.2)] +
    [("jaccard_w3", "jaccard", dict(w=3)), ("jaccard_w5", "jaccard", dict(w=5))] +
    [(f"worst_t{t}", "worst", dict(theta=t, w=3)) for t in (0.1, 0.5, 1.0)]
)


def analyze_trace(path):
    tr = np.load(path, allow_pickle=False)
    Nt = len(tr["offsets"]) - 1
    id_off = int(tr["id_offset"])
    meta = dict(dataset=str(tr["dataset"]), mode=str(tr["mode"]),
                L=int(tr["L"]), K=int(tr["K"]), n_traced=Nt)

    names = [n for n, _, _ in DET_GRID]
    latch = {n: np.full(Nt, -1, np.int32) for n in names}
    cpp_latch = tr["conv_hop"].astype(np.int32)          # 0 = never
    o_stab = np.empty(Nt, np.int32)
    o_rec = np.empty(Nt, np.int32)
    hops = tr["hops"].astype(np.int32)
    final_rec = np.empty(Nt, np.float32)
    rec_at = {n: np.full(Nt, np.nan, np.float32) for n in names}

    for qi, best, worst, n_ins, beam, tk in per_query(tr):
        sets = topk_sets(tk)
        o_stab[qi] = oracle_stable(sets)
        curve = recall_curve(tk, tr["gt"][qi], id_off)
        o_rec[qi] = oracle_from_curve(curve)
        final_rec[qi] = curve[-1]
        for name, kind, kw in DET_GRID:
            if kind == "current":
                h = det_current(best, kw["theta"], kw["w"])
            elif kind == "ema":
                h = det_ema(best, kw["theta"], kw["lam"])
            elif kind == "turnover":
                h = det_turnover(n_ins, beam, kw["theta"], kw["w"])
            elif kind == "jaccard":
                h = det_jaccard(sets, kw["w"])
            else:
                h = det_worst(worst, kw["theta"], kw["w"])
            if h is not None:
                latch[name][qi] = h
                rec_at[name][qi] = curve[min(h, len(curve)) - 1]

    # validation: python current_t1.0 must reproduce the C++ latch
    v = latch["current_t1.0"].copy()
    v[v < 0] = 0
    n_mismatch = int((v != cpp_latch).sum())

    rows = []
    for name, kind, kw in DET_GRID:
        lh = latch[name]
        ok = lh > 0
        row = dict(**meta, detector=name, kind=kind, params=kw,
                   latch_rate=round(float(ok.mean()), 4))
        if ok.any():
            d_or = (lh[ok] - o_stab[ok]).astype(np.float64)
            row.update(
                latch_p50=int(np.percentile(lh[ok], 50)),
                latch_mean=round(float(lh[ok].mean()), 1),
                delta_oracle_p50=int(np.percentile(d_or, 50)),
                delta_oracle_mean=round(float(d_or.mean()), 1),
                early_rate=round(float((lh[ok] < o_stab[ok]).mean()), 4),
                hops_saved_mean=round(float((hops[ok] - lh[ok]).mean()), 1),
                recall_at_latch_delta=round(
                    float(np.nanmean(rec_at[name][ok] - final_rec[ok])), 4),
            )
        rows.append(row)
    summary = dict(**meta,
                   cpp_replication_mismatch=n_mismatch,
                   oracle_stable_p50=int(np.percentile(o_stab, 50)),
                   oracle_recall_p50=int(np.percentile(o_rec, 50)),
                   hops_p50=int(np.percentile(hops, 50)),
                   final_recall_mean=round(float(final_rec.mean()), 4))
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="dir with trace npz files")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.traces, "*.npz")))
    if not files:
        raise SystemExit(f"no trace npz under {a.traces}")
    summaries, all_rows = [], []
    for f in files:
        s, rows = analyze_trace(f)
        summaries.append(s)
        all_rows += rows
        tag = f"{s['dataset']} {s['mode']} L={s['L']}"
        print(f"[{tag}] oracle_stable_p50={s['oracle_stable_p50']} "
              f"oracle_recall_p50={s['oracle_recall_p50']} hops_p50={s['hops_p50']} "
              f"cpp_mismatch={s['cpp_replication_mismatch']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "w3_detector_comparison.json"), "w") as f:
        json.dump(dict(summaries=summaries, rows=all_rows), f, indent=2)
    print(f"-> {os.path.join(a.out, 'w3_detector_comparison.json')}", flush=True)


if __name__ == "__main__":
    main()
