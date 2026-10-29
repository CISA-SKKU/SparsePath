#!/usr/bin/env python
"""W1/W2/W4 — convergence detection under quantized distances (handoff 2026-07-24).

Phases (single index load per dataset+graph, phases share the harness):
  w1  : latch telemetry sweep — for each search_dist mode x L, run search_batch
        (filter OFF -> pure measurement, detector still updates) and record
        per-query {conv_hop, hops} + recall.  At trace_Ls, additionally dump the
        per-hop stream (best/worst dist, insertions, beam top-K tags) for the
        first trace_n_queries queries -> offline W2 profiles + W3 detector
        comparison without re-searching.
  w4  : rerank-pool alpha-margin gate sweep — pq_rerank (beam scope) with
        rerank_gate_alpha in w4_gate_alphas; exports {recall, rerank fetches
        per query (local/remote per label), gated count}.

Modes syntax in conv_modes: "exact", "sq8", "rabitq", or "pq@<variant>" where
<variant> names an entry of pq_variants (e.g. pq@M12, pq@M32). pq ranking is
identical between pq and pq_rerank — W1 uses plain pq (no rerank fetches).

Usage (inside worker-host danns container):
  python run_conv.py --config conv_deep100M.yaml --out results_conv_deep100M
  python run_conv.py --config conv_deep100M.yaml --phase w4 --dataset deep100M_SP_lits150
"""
import argparse
import json
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "q1_quantization_pareto"))
from run_q1 import load_bin, load_gt, recall_at_k, load_partition, _HARNESS, _NP  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("pyyaml not available")


def _pq_variant(cfg, name):
    v = next((v for v in cfg.get("pq_variants", []) if v["name"] == name), None)
    if v is None:
        sys.exit(f"pq variant '{name}' not in pq_variants")
    return v


def _parse_modes(cfg):
    """-> list of (label, search_dist, variant_name_or_None)."""
    out = []
    for m in [x.strip() for x in str(cfg.get("conv_modes", "exact")).split(",") if x.strip()]:
        if m.startswith("pq@"):
            out.append((m.replace("@", "_"), "pq", m.split("@", 1)[1]))
        else:
            out.append((m, m, None))
    return out


def _latch_stats(r, dt):
    """Summarise conv_hop/hops arrays from one search_batch result."""
    ch = np.asarray(r["conv_hop_per_query"])
    hp = np.asarray(r["hops_per_query"])
    nq = max(1, int(r["n_queries"]))
    latched = ch > 0
    s = {
        "latch_rate": round(float(latched.mean()), 4),
        "hops_mean": round(float(hp.mean()), 1),
        "qps_1node": round(nq / dt, 1),
        "search_s": round(dt, 2),
    }
    if latched.any():
        lc = ch[latched].astype(np.float64)
        frac = lc / np.maximum(1, hp[latched].astype(np.float64))
        s.update(conv_hop_mean=round(float(lc.mean()), 1),
                 conv_hop_p50=int(np.percentile(lc, 50)),
                 conv_hop_p90=int(np.percentile(lc, 90)),
                 conv_frac_of_hops=round(float(frac.mean()), 3))
    else:
        s.update(conv_hop_mean=None, conv_hop_p50=None, conv_hop_p90=None,
                 conv_frac_of_hops=None)
    return s


def _load_harness(name, cfg):
    dtype = cfg["dtype"]
    H = _HARNESS[dtype]
    dim = int(cfg["dim"])
    L_all = [int(x) for x in cfg.get("w1_L_sweep", [100])] + \
            [int(x) for x in cfg.get("w4_Ls", [])]
    h = H(dim=dim, max_points=int(cfg.get("max_points", cfg["max_capacity"])),
          R=int(cfg.get("R", 64)), L_build=int(cfg.get("L_build", 100)),
          num_threads=int(cfg.get("num_threads", 0)))
    t0 = time.time()
    h.load_index(cfg["graph_file"], cfg["base_file"], cfg["tags_file"],
                 search_L=max(L_all))
    print(f"  [{name}] loaded num_points={h.num_points()} ({time.time()-t0:.1f}s)",
          flush=True)
    return h


def run_w1(name, cfg, h, queries, gt, out_dir):
    K = int(cfg.get("K", 10))
    cw = int(cfg.get("conv_window", 3))
    ct = float(cfg.get("conv_threshold_pct", 1.0))
    id_off = int(cfg.get("id_offset", 0))
    L_sweep = [int(x) for x in cfg.get("w1_L_sweep", [100])]
    trace_Ls = {int(x) for x in cfg.get("trace_Ls", [])}
    trace_n = int(cfg.get("trace_n_queries", 0))
    max_cap = int(cfg["max_capacity"])

    conv_dir = os.path.join(out_dir, "conv")
    trace_dir = os.path.join(out_dir, "traces")
    os.makedirs(conv_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    rows = []
    cur_variant = None
    for label, sd, variant in _parse_modes(cfg):
        if variant is not None and variant != cur_variant:
            v = _pq_variant(cfg, variant)
            h.load_pq(v["codebook"], v["codes"], max_capacity=max_cap, alpha=0.0)
            print(f"  [{name}] pq variant {variant}: M={h.M()}", flush=True)
            cur_variant = variant
        for L in L_sweep:
            tn = trace_n if L in trace_Ls else 0
            t0 = time.time()
            r = h.search_batch(queries, K=K, L_search=L, conv_window=cw,
                               conv_threshold_pct=ct, filter_on=False,
                               record_drops=False, search_dist=sd,
                               rerank_scope="beam", trace_max_queries=tn)
            dt = time.time() - t0
            rec = recall_at_k(r["result_tags"], gt, K, id_off)
            row = dict(dataset=name, mode=label, search_dist=sd, L=L, K=K,
                       conv_window=cw, conv_threshold_pct=ct,
                       recall=round(rec, 4), **_latch_stats(r, dt))
            rows.append(row)
            print(f"  [{name}] {label} L={L}: recall={rec:.4f} "
                  f"latch={row['latch_rate']:.2%} conv_hop_p50={row['conv_hop_p50']} "
                  f"hops={row['hops_mean']}", flush=True)
            np.savez(os.path.join(conv_dir, f"{name}_{label}_L{L}.npz"),
                     conv_hop=np.asarray(r["conv_hop_per_query"]),
                     hops=np.asarray(r["hops_per_query"]))
            if tn > 0:
                np.savez_compressed(
                    os.path.join(trace_dir, f"{name}_{label}_L{L}.npz"),
                    offsets=np.asarray(r["trace_offsets"]),
                    best=np.asarray(r["trace_best"]),
                    worst=np.asarray(r["trace_worst"]),
                    beam_size=np.asarray(r["trace_beam_size"]),
                    n_inserted=np.asarray(r["trace_n_inserted"]),
                    expand_tag=np.asarray(r["trace_expand_tag"]),
                    topk=np.asarray(r["trace_topk"]),
                    conv_hop=np.asarray(r["conv_hop_per_query"])[:tn],
                    hops=np.asarray(r["hops_per_query"])[:tn],
                    result_tags=np.asarray(r["result_tags"])[:tn],
                    gt=gt[:tn], K=K, L=L, id_offset=id_off,
                    conv_window=cw, conv_threshold_pct=ct,
                    mode=label, dataset=name)
    return rows


def run_w4(name, cfg, h, queries, gt, out_dir):
    K = int(cfg.get("K", 10))
    cw = int(cfg.get("conv_window", 3))
    ct = float(cfg.get("conv_threshold_pct", 1.0))
    id_off = int(cfg.get("id_offset", 0))
    max_cap = int(cfg["max_capacity"])
    Ls = [int(x) for x in cfg.get("w4_Ls", [100, 200, 400])]
    alphas = [float(a) for a in cfg.get("w4_gate_alphas",
                                        [-1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])]
    variant = str(cfg.get("w4_variant", "M32"))
    v = _pq_variant(cfg, variant)
    h.load_pq(v["codebook"], v["codes"], max_capacity=max_cap, alpha=0.0)
    print(f"  [{name}] W4 pq variant {variant}: M={h.M()}", flush=True)

    pnames = cfg.get("w4_partitions", [])
    pids, coords = [], []
    for pn in pnames:
        pid, coord = load_partition(cfg, pn, queries)
        pids.append(pid)
        coords.append(coord)

    rows = []
    for L in Ls:
        for ag in alphas:
            t0 = time.time()
            r = h.search_batch(queries, K=K, L_search=L, conv_window=cw,
                               conv_threshold_pct=ct, filter_on=False,
                               record_drops=False, search_dist="pq_rerank",
                               rerank_scope="beam",
                               partition_of=(pids if pids else None),
                               coordinator=(coords if coords else None),
                               rerank_gate_alpha=ag)
            dt = time.time() - t0
            rec = recall_at_k(r["result_tags"], gt, K, id_off)
            nq = max(1, int(r["n_queries"]))
            row = dict(dataset=name, pq_variant=variant, L=L, K=K,
                       gate_alpha=ag, recall=round(rec, 4),
                       gated_per_q=round(int(r["n_rerank_gated"]) / nq, 2),
                       qps_1node=round(nq / dt, 1), search_s=round(dt, 2))
            for i, pn in enumerate(pnames):
                lc = r["label_counters"][i]
                row[f"rerank_local_{pn}"] = round(int(lc["rerank_local"]) / nq, 2)
                row[f"rerank_remote_{pn}"] = round(int(lc["rerank_remote"]) / nq, 2)
            rows.append(row)
            ftot = (row.get(f"rerank_local_{pnames[0]}", 0) +
                    row.get(f"rerank_remote_{pnames[0]}", 0)) if pnames else None
            print(f"  [{name}] W4 L={L} alpha={ag}: recall={rec:.4f} "
                  f"fetch/q={ftot} gated/q={row['gated_per_q']}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="run only this dataset key")
    ap.add_argument("--phase", default="all", choices=["w1", "w4", "all"])
    ap.add_argument("--out", default="results_conv")
    a = ap.parse_args()
    with open(a.config) as f:
        docs = yaml.safe_load(f)
    ds = docs["datasets"]
    keys = [a.dataset] if a.dataset else list(ds.keys())
    os.makedirs(a.out, exist_ok=True)
    for k in keys:
        cfg = ds[k]
        npd = _NP[cfg["dtype"]]
        queries = load_bin(cfg["query_file"], npd)
        gt = load_gt(cfg["gt_file"])
        nq = cfg.get("n_query")
        if nq:
            queries = queries[:int(nq)]
            gt = gt[:int(nq)]
        print(f"\n=== conv {k}: queries={queries.shape} ===", flush=True)
        h = _load_harness(k, cfg)
        if a.phase in ("w1", "all"):
            rows = run_w1(k, cfg, h, queries, gt, a.out)
            p = os.path.join(a.out, f"{k}_w1_latch.json")
            with open(p, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  -> {p}", flush=True)
        if a.phase in ("w4", "all"):
            rows = run_w4(k, cfg, h, queries, gt, a.out)
            p = os.path.join(a.out, f"{k}_w4_gate.json")
            with open(p, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  -> {p}", flush=True)
        del h


if __name__ == "__main__":
    main()
