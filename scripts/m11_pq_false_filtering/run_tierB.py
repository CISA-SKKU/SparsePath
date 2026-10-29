#!/usr/bin/env python
"""Tier B — stale-codebook drift-stress driver (single node, no RDMA).

Answers M05/R3·W3·D7 reinforcement: "what if the distribution drifts so the
codebook is stale?". We inject *controlled* codebook degradation (a proxy for
drift the i.i.d. benchmark lacks) and trace filter recall vs codebook quality
(QE). Story: recall stays flat until the codebook's distance error crosses the
alpha margin — a quantified safety margin; observed real drift (QE ratio ~1.00)
sits far inside it.

Efficiency: the Vamana graph (iter150, ~10-15 min to load) is loaded ONCE; only
the PQ codebook/codes are swapped per variant (harness `load_pq` is re-callable
because every variant writes the SAME N so the flat code array is fully
overwritten). recall(OFF) is codebook-independent -> computed once per L.

Levers (see gen_degraded_codebook.py):
  noise:<s1,s2,...>       centroid Gaussian noise, sigma_rel ladder (0 == shipped control)
  biased:<p1,p2,...>      retrain on ONE partition's active vectors (narrow init dist)
  undertrain:<n1,n2,...>  retrain on a starved sample

Raw output -> --out (put under 02_SparsePath_experiments/results/... per convention).

Usage (inside danns container on worker-host/8):
  python run_tierB.py --config datasets_deep100M_variants.yaml \
     --dataset deep100M_lits150 --config-name LITS_4p \
     --levers 'noise:0,0.05,0.1,0.2,0.4,0.8,1.5;biased:0' \
     --alphas 0.10,0.15,0.20 --Ls 100,400 \
     --tmp /experiments/tierB_tmp --out /experiments/results/m11_pq_false_filtering/tierB_deep100M
"""
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "lib"))
sys.path.insert(0, _HERE)

import pq_search_bindings as pqsb                      # noqa: E402
import gen_degraded_codebook as gen                    # noqa: E402
from run_m11 import load_bin, load_gt, _recall_mapped  # noqa: E402
import yaml                                            # noqa: E402

_HARNESS = {"float32": pqsb.PQSearchHarnessFloat, "float": pqsb.PQSearchHarnessFloat,
            "uint8": pqsb.PQSearchHarnessUInt8, "int8": pqsb.PQSearchHarnessInt8}
_NPD = {"float32": np.float32, "float": np.float32, "uint8": np.uint8, "int8": np.int8}


def parse_levers(spec):
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        lever, vals = part.split(":")
        for v in vals.split(","):
            out.append((lever.strip(), float(v) if lever.strip() == "noise" else int(float(v))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config-name", default=None, help="which partition config (e.g. LITS_4p); default=first")
    ap.add_argument("--levers", default="noise:0,0.05,0.1,0.2,0.4,0.8,1.5;biased:0")
    ap.add_argument("--alphas", default="0.10,0.15,0.20")
    ap.add_argument("--Ls", default="100,400")
    ap.add_argument("--tmp", required=True, help="local dir for per-variant codebook/codes (big, overwritten)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--biased-sample", type=int, default=100000)
    ap.add_argument("--undertrain-from", type=int, default=100000, help="pool the undertrain sample draws from")
    ap.add_argument("--lowmem-base", action="store_true",
                    help="stream base from memmap per batch instead of a resident float32 copy "
                         "(for tight-RAM nodes, e.g. LAION 512-d where the harness alone ~247GB)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["datasets"][args.dataset]
    dtype = cfg["dtype"]; npd = _NPD[dtype]; Harness = _HARNESS[dtype]
    dim = int(cfg["dim"]); K = int(cfg.get("K", 10))
    conv_window = int(cfg.get("conv_window", 3)); conv_thr = float(cfg.get("conv_threshold_pct", 1.0))
    id_offset = int(cfg.get("id_offset", 0))
    max_cap = int(cfg["max_capacity"]); max_points = int(cfg.get("max_points", max_cap))
    active_lo = int(cfg.get("active_gid_lo", 0)); active_n = int(cfg.get("active_n", max_points))
    active_hi = active_lo + active_n
    N_codes = active_hi                                 # constant across variants (clean re-load)
    alphas = [float(a) for a in args.alphas.split(",")]
    Ls = [int(x) for x in args.Ls.split(",")]
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.tmp, exist_ok=True)

    # -------- load graph ONCE --------
    print(f"=== Tier B {args.dataset} (dim={dim}) ===", flush=True)
    h = Harness(dim=dim, max_points=max_points, R=int(cfg.get("R", 64)),
                L_build=int(cfg.get("L_build", 100)), num_threads=int(cfg.get("num_threads", 0)))
    t0 = time.time()
    h.load_index(cfg["graph_file"], cfg["base_file"], cfg["tags_file"], search_L=max(Ls))
    print(f"  graph loaded: num_points={h.num_points()} ({time.time()-t0:.1f}s)", flush=True)

    queries = load_bin(cfg["query_file"], npd); gt = load_gt(cfg["gt_file"])
    if cfg.get("n_query"):
        queries = queries[:int(cfg["n_query"])]; gt = gt[:int(cfg["n_query"])]
    print(f"  queries={queries.shape} gt={gt.shape}", flush=True)

    # -------- LITS (or chosen) partition config: nearest-centroid coordinator --------
    parts = cfg.get("partitions", [])
    ps = next((p for p in parts if p["name"] == args.config_name), parts[0]) if parts else None
    if ps is None:
        sys.exit("no partitions in config")
    pid = np.load(ps["path"]).astype(np.uint32); Mpart = int(ps["M"])
    stride = int(cfg.get("centroid_sample_stride", 97))
    mm = np.memmap(cfg["base_file"], dtype=npd, mode="r", offset=8).reshape(-1, dim)
    idx = np.arange(active_lo, active_hi, stride, dtype=np.int64)
    sub = np.asarray(mm[idx], dtype=np.float64); pp = pid[idx]
    C = np.zeros((Mpart, dim), dtype=np.float64)
    for p in range(Mpart):
        msk = pp == p
        if msk.any():
            C[p] = sub[msk].mean(axis=0)
    C = C.astype(np.float32)
    q = queries.astype(np.float32)
    d = (q * q).sum(1)[:, None] - 2.0 * (q @ C.T) + (C * C).sum(1)[None, :]
    coord = d.argmin(axis=1).astype(np.uint32)
    print(f"  config[{ps['name']}] M={Mpart} routed-hist={np.bincount(coord, minlength=Mpart).tolist()}", flush=True)
    partition_of = pid                                  # gid-indexed

    # -------- active base slice (reused for every encode) --------
    # lowmem: keep it a lazy memmap VIEW (encode reads per-batch, materializing only
    # each batch/chunk in float32) so we never hold a 2nd ~200GB copy next to the
    # harness. Otherwise materialize a resident float32 copy (faster re-use).
    t0 = time.time()
    if args.lowmem_base:
        base_slice = mm[active_lo:active_hi]           # memmap view; dtype = base dtype
        print(f"  base slice = memmap VIEW [{active_lo}:{active_hi}] {base_slice.shape} "
              f"(low-mem streaming, no resident copy)", flush=True)
    else:
        base_slice = np.ascontiguousarray(mm[active_lo:active_hi], dtype=np.float32)
        print(f"  active base slice {base_slice.shape} in RAM "
              f"({base_slice.nbytes/1e9:.1f} GB, {time.time()-t0:.1f}s)", flush=True)
    part_active = np.ascontiguousarray(pid[active_lo:active_hi])   # partition of each active gid (biased lever)

    ship_cl, chunk_offsets = gen.load_codebook(cfg["pq_codebook"])

    # -------- recall(OFF) once per L (codebook-independent) --------
    recall_off = {}
    for L in Ls:
        off = h.search_batch(queries, K=K, L_search=L, conv_window=conv_window,
                             conv_threshold_pct=conv_thr, filter_on=False)
        recall_off[L] = _recall_mapped(off["result_tags"], gt, K, id_offset)
        print(f"  recall(OFF) L={L} = {recall_off[L]:.4f}", flush=True)

    cb_tmp = os.path.join(args.tmp, "tierB_codebook.bin")
    codes_tmp = os.path.join(args.tmp, "tierB_codes.bin")

    rows = []; qe_ship = None
    for (lever, param) in parse_levers(args.levers):
        vname = f"{lever}_{param}"
        t0 = time.time()
        if lever == "noise":
            cents = gen.degrade_noise(ship_cl, param, seed=42)
        elif lever == "biased":
            cents = gen.degrade_biased(base_slice, chunk_offsets, part_active, param,
                                       sample_size=args.biased_sample, seed=42)
        elif lever == "undertrain":
            cents = gen.degrade_undertrain(base_slice, chunk_offsets, param, seed=42)
        else:
            print(f"  [skip] unknown lever {lever}"); continue
        codes, qe = gen.encode_and_qe(base_slice, cents, chunk_offsets)
        if lever == "noise" and param == 0.0:
            qe_ship = qe
        qe_ratio = (qe / qe_ship) if qe_ship else float("nan")
        gen.write_codebook(cb_tmp, cents, chunk_offsets)
        gen.write_codes(codes_tmp, codes, active_lo, N_codes)
        h.load_pq(cb_tmp, codes_tmp, max_capacity=max_cap, alpha=alphas[-1])
        # gid-mapping / pq-quality sanity
        probe = h.search_batch(queries[:min(1000, queries.shape[0])], K=K, L_search=min(100, max(Ls)),
                               conv_window=conv_window, conv_threshold_pct=conv_thr,
                               filter_on=True, record_drops=True, max_drops=200000)
        pqd = np.asarray(probe["drop_pqdist"], np.float64); exd = np.asarray(probe["drop_exact"], np.float64)
        corr = float(np.corrcoef(pqd, exd)[0, 1]) if len(pqd) > 3 else float("nan")
        print(f"  [{vname}] qe={qe:.5f} qe_ratio={qe_ratio:.4f} corr={corr:.3f} (gen+encode {time.time()-t0:.1f}s)", flush=True)

        for L in Ls:
            for a in alphas:
                h.set_alpha(a)
                r = h.search_batch(queries, K=K, L_search=L, conv_window=conv_window,
                                   conv_threshold_pct=conv_thr, filter_on=True,
                                   record_drops=False, partition_of=partition_of, coordinator=coord)
                r_on = _recall_mapped(r["result_tags"], gt, K, id_offset)
                elig = int(r["n_filter_eligible"]); drop = int(r["n_pq_filtered"]); false_ = int(r["n_false_local"])
                rows.append(dict(dataset=args.dataset, config=ps["name"], lever=lever, param=param,
                                 variant=vname, qe=round(qe, 6), qe_ratio=round(qe_ratio, 5),
                                 corr=round(corr, 4), L=L, alpha=a,
                                 recall_on=round(r_on, 4), recall_off=round(recall_off[L], 4),
                                 recall_delta=round(recall_off[L] - r_on, 4),
                                 false_filter_rate_of_eligible=round((false_/elig) if elig else 0.0, 6),
                                 drop_rate=round((drop/elig) if elig else 0.0, 4)))
                print(f"    L={L} a={a:.2f} ON={r_on:.4f} Δ={recall_off[L]-r_on:+.4f} "
                      f"ff={100*(false_/elig if elig else 0):.3f}%", flush=True)

    # -------- persist + plot --------
    with open(os.path.join(args.out, f"tierB_{args.dataset}_rows.json"), "w") as f:
        json.dump({"dataset": args.dataset, "config": ps["name"], "qe_shipped": qe_ship,
                   "recall_off": recall_off, "rows": rows}, f, indent=2)
    _plot(args.dataset, ps["name"], rows, args.out)
    for p in (cb_tmp, codes_tmp):
        if os.path.exists(p):
            os.remove(p)
    print(f"\nwrote {args.out}/tierB_{args.dataset}_rows.json  ({len(rows)} rows)", flush=True)


def _plot(name, cfgname, rows, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    a0 = 0.15
    Ls = sorted({r["L"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for L in Ls:
        rs = sorted([r for r in rows if r["L"] == L and abs(r["alpha"] - a0) < 1e-9 and np.isfinite(r["qe_ratio"])],
                    key=lambda r: r["qe_ratio"])
        if rs:
            ax[0].plot([r["qe_ratio"] for r in rs], [r["recall_delta"] for r in rs], "o-", label=f"L={L}")
    ax[0].axvline(1.0, color="green", ls=":", label="shipped (real drift ~1.0)")
    ax[0].set_xlabel("codebook QE ratio (degraded / shipped)")
    ax[0].set_ylabel(f"recall Δ (OFF−ON) @α={a0}")
    ax[0].set_title(f"{name}/{cfgname}: safety margin — recall Δ vs codebook degradation")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    # panel 2: recall Δ vs alpha for a few QE levels (L=min)
    Lmin = min(Ls)
    qrs = sorted({r["qe_ratio"] for r in rows if np.isfinite(r["qe_ratio"])})
    pick = qrs[:: max(1, len(qrs)//4)] if qrs else []
    for qr in pick:
        rs = sorted([r for r in rows if r["L"] == Lmin and abs(r["qe_ratio"]-qr) < 1e-9], key=lambda r: r["alpha"])
        if rs:
            ax[1].plot([r["alpha"] for r in rs], [r["recall_delta"] for r in rs], "s--", label=f"QE×{qr:.2f}")
    ax[1].axvline(0.15, color="gray", ls=":")
    ax[1].set_xlabel("alpha"); ax[1].set_ylabel(f"recall Δ (L={Lmin})")
    ax[1].set_title(f"{name}/{cfgname}: recall Δ vs α per codebook quality")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{ts}_tierB_{name}_{cfgname}.png")
    fig.savefig(p, dpi=130); print(f"  plot -> {p}", flush=True)


if __name__ == "__main__":
    main()
