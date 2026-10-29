#!/usr/bin/env python3
"""LITS update-cycle graph builder — build_updatecycle.py + diskann-lits port.

Same update-cycle scheme as build_updatecycle.py (initial build + insert/delete/
consolidate loop, DATA-LESS checkpoints: graph + .tags + .frozen, vectors
recoverable via base[tag-1]) with the LITS build features from the diskann-lits
fork enabled through the extended diskann_bindings:

  - capacity-balanced placement (place_vector_balanced: visit-histogram + theta
    eligibility + precharge/redirect counters, K clusters)
  - LITS edge filter at insert AND delete-consolidation pruning
    (filter_mode off|full|notostart|long_inter_only, tau = c * mean)
  - tag-indexed cluster labels: preloaded for the initial window (tag 0 = frozen
    start), assigned at insert for new tags, dumped as <ckpt>.labels every save

Inputs come from prep_lits_cluster.py (--lits-labels/.centroids, K = target Np).
Kept as a SEPARATE file so the live 400M chain's build_updatecycle.py is never
touched. Requires a diskann_bindings.so with enable_lits (worktree
lits-updatecycle build); pass its lib dir via --bindings-lib until the mainline
.so is updated.

Canonical LITS defaults (diskann-lits example.yaml == paper #354):
  filter_mode=long_inter_only, tau_scale=1.0, threshold(theta)=0.7, K=Np.
Checkpoint names default to lits_iter{N} (100M embed convention) so they can sit
next to vamana iteration{N} checkpoints without collision.
"""
import argparse, os, sys, time
import numpy as np

_DT = {"float32": np.float32, "uint8": np.uint8, "int8": np.int8}
_CLS = {"float32": "DiskANNIndex", "uint8": "DiskANNIndexUInt8", "int8": "DiskANNIndexInt8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base pool .bin (may have >600M vecs)")
    ap.add_argument("--dtype", required=True, choices=list(_DT))
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--out-ckpt-dir", required=True)
    ap.add_argument("--initial-size", type=int, required=True)
    ap.add_argument("--max-capacity", type=int, required=True)
    ap.add_argument("--points-insert", type=int, required=True)
    ap.add_argument("--points-delete", type=int, required=True)
    ap.add_argument("--consolidate-every", type=int, required=True)
    ap.add_argument("--max-iterations", type=int, required=True)
    ap.add_argument("--save-every", type=int, required=True)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--num-threads", type=int, default=0)
    # --- LITS ---
    ap.add_argument("--lits-labels", required=True,
                    help="<prefix>.labels from prep_lits_cluster.py (aw+1 entries)")
    ap.add_argument("--lits-centroids", required=True,
                    help="<prefix>.centroids from prep_lits_cluster.py (K x dim f32)")
    ap.add_argument("--lits-clusters", type=int, required=True,
                    help="K > 0 (balanced placement; = target partition count Np)")
    ap.add_argument("--lits-threshold", type=float, default=0.7, help="theta (default 0.7)")
    ap.add_argument("--lits-filter-mode", default="long_inter_only",
                    choices=["off", "full", "notostart", "long_inter_only"])
    ap.add_argument("--lits-tau-scale", type=float, default=1.0, help="long iff dist > c*mean")
    ap.add_argument("--lits-tau-median", action="store_true", help="tau base = median")
    ap.add_argument("--ckpt-name-prefix", default=None,
                    help="checkpoint basename prefix; default lits_k{K}_iter -> "
                         "lits_k{K}_iter{N} (flat name-separation from vamana "
                         "iteration{N}; legacy 100M embeds were K-less lits_iter{N})")
    ap.add_argument("--bindings-lib", default="/workspace/third_party/lits_build_tree/lib",
                    help="dir containing diskann_bindings.so WITH enable_lits "
                         "(use the lits-updatecycle worktree lib until mainline is rebuilt)")
    args = ap.parse_args()

    if args.lits_clusters <= 0:
        sys.exit("--lits-clusters must be > 0 (filter-only streaming needs full-size labels; "
                 "not supported by this builder)")
    if args.ckpt_name_prefix is None:
        args.ckpt_name_prefix = f"lits_k{args.lits_clusters}_iter"

    sys.path.insert(0, args.bindings_lib)
    import diskann_bindings as diskann
    dt = _DT[args.dtype]
    IndexCls = getattr(diskann, _CLS[args.dtype])
    os.makedirs(args.out_ckpt_dir, exist_ok=True)

    consume = args.initial_size + args.max_iterations * args.points_insert

    # As-run build config 자동 보존 (규칙 2026-07-14: artifacts/<ds>/build_config.yaml).
    try:
        ds_root = os.path.dirname(os.path.dirname(os.path.abspath(args.out_ckpt_dir)))
        rec = {
            "generator": "build_updatecycle_lits.py (as-run auto dump)",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.uname().nodename,
            "argv": sys.argv,
            "args": {k: getattr(args, k) for k in vars(args)},
            "index_class": _CLS[args.dtype],
            "bindings": os.path.join(args.bindings_lib, "diskann_bindings.so"),
            "convention": {
                "update": "insert 1% + delete 1%(oldest window) per iter",
                "consolidate": f"every {args.consolidate_every} iters",
                "checkpoints": f"data-less (graph+tags+.frozen+.labels), every {args.save_every} iters + iter0",
                "reload": "load_graph_only(graph, base, tags); vectors = base[tag-1]",
                "lits": {
                    "source": "diskann-lits port (worktree lits-updatecycle)",
                    "filter_mode": args.lits_filter_mode,
                    "tau_scale": args.lits_tau_scale,
                    "tau_median": bool(args.lits_tau_median),
                    "balanced": {"K": args.lits_clusters, "threshold": args.lits_threshold},
                    "labels": args.lits_labels, "centroids": args.lits_centroids,
                },
            },
        }
        import yaml as _yaml
        with open(os.path.join(ds_root, "build_config.yaml"), "w") as f:
            _yaml.safe_dump(rec, f, sort_keys=False, allow_unicode=True)
        print(f"[build] as-run config -> {ds_root}/build_config.yaml", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[build] WARN: build_config dump failed: {e}", flush=True)

    # base header + mmap
    hdr = np.fromfile(args.base, dtype=np.uint32, count=2)
    n_total, d = int(hdr[0]), int(hdr[1])
    assert d == args.dim, f"dim mismatch: file {d} != --dim {args.dim}"
    assert consume <= n_total, f"pool too small: need {consume:,}, have {n_total:,}"
    vectors = np.memmap(args.base, dtype=dt, mode="r", offset=8, shape=(n_total, d))
    print(f"[build] {_CLS[args.dtype]} base={args.base} n={n_total:,} d={d} dtype={args.dtype} "
          f"consume={consume:,} R={args.R} L={args.L} a={args.alpha} nthreads={args.num_threads}", flush=True)
    print(f"[build] LITS: mode={args.lits_filter_mode} c={args.lits_tau_scale} "
          f"K={args.lits_clusters} theta={args.lits_threshold} labels={args.lits_labels}", flush=True)

    index = IndexCls(d, args.max_capacity, args.R, args.L, args.alpha, args.num_threads, True)

    # LITS on BEFORE the initial build (diskann-lits scenario order): iteration0
    # is already LITS-filtered; window labels preload + counter seeding happen here.
    index.enable_lits(
        labels_file=args.lits_labels,
        centroids_file=args.lits_centroids,
        num_clusters=args.lits_clusters,
        threshold=args.lits_threshold,
        active_window=args.initial_size,
        max_points_to_insert=consume,
        filter_mode=args.lits_filter_mode,
        tau_scale=args.lits_tau_scale,
        tau_median=bool(args.lits_tau_median),
    )

    def save_dataless(it):
        ckpt = os.path.join(args.out_ckpt_dir, f"{args.ckpt_name_prefix}{it}")
        t = time.time()
        index.save(ckpt, True, False)   # compact + graph/tags/.frozen/.labels, NO .data
        if os.path.exists(ckpt + ".frozen") and os.path.exists(ckpt + ".data"):
            os.remove(ckpt + ".data")
        print(f"  saved {args.ckpt_name_prefix}{it} data-less ({time.time()-t:.1f}s)", flush=True)

    # --- initial build ---
    t0 = time.time()
    print(f"[build] initial build {args.initial_size:,} points...", flush=True)
    initial = np.ascontiguousarray(vectors[:args.initial_size])
    index.build(initial)
    del initial
    st = index.lits_stats()
    print(f"[build] built {index.num_points:,} in {time.time()-t0:.1f}s  "
          f"lits filtered_total={st['filtered_total']} counts_sum={st.get('counts_sum')}", flush=True)
    save_dataless(0)

    # counts_sum invariant: initial+1(frozen), then +inserted-deleted per iter
    expected_sum = args.initial_size + 1

    # --- update loop (identical semantics to build_updatecycle.py) ---
    cursor = args.initial_size
    for it in range(1, args.max_iterations + 1):
        t_iter = time.time()
        insert_end = min(cursor + args.points_insert, n_total)
        n_ins = 0
        if cursor < insert_end:
            newv = np.ascontiguousarray(vectors[cursor:insert_end])
            t_i = time.time()
            tags = index.insert_batch(newv)   # binding precharges homes internally
            n_ins = len(tags)
            print(f"  Iter {it}: inserted {n_ins} [{cursor},{insert_end}) "
                  f"tags[{tags[0]},{tags[-1]}] ({time.time()-t_i:.1f}s)", flush=True)
            cursor = insert_end
        n_del = 0
        if args.points_delete > 0 and index.num_points > args.points_delete:
            off = (it - 1) * args.points_delete + 1
            index.delete_batch(list(range(off, off + args.points_delete)))  # binding releases counters
            n_del = args.points_delete
            print(f"  Iter {it}: deleted {n_del} tags[{off},{off+n_del})", flush=True)
        expected_sum += n_ins - n_del

        st = index.lits_stats(reset_redirect=True)
        got_sum = st.get("counts_sum")
        ok = (got_sum == expected_sum)
        print(f"  Iter {it}: lits filtered_total={st['filtered_total']} "
              f"redirected={st.get('redirected')}/{st.get('placed')} "
              f"counts_sum={got_sum} (expect {expected_sum}) {'OK' if ok else '** MISMATCH **'}",
              flush=True)
        if not ok:
            print(f"  Iter {it}: WARNING counts_sum invariant broken", flush=True)

        if it % args.consolidate_every == 0:
            t_c = time.time()
            rem = index.consolidate()
            print(f"  Iter {it}: consolidate -> {rem} ({time.time()-t_c:.1f}s)", flush=True)
        if it % args.save_every == 0 or it == args.max_iterations:
            save_dataless(it)
            print(f"  Iter {it}: iter total {time.time()-t_iter:.1f}s", flush=True)

    st = index.lits_stats()
    print(f"[build] DONE total {time.time()-t0:.1f}s  lits filtered_total={st['filtered_total']} "
          f"counts_sum={st.get('counts_sum')}", flush=True)


if __name__ == "__main__":
    main()
