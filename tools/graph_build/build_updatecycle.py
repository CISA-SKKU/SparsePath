#!/usr/bin/env python3
"""Unified update-cycle graph builder (dtype-aware) using the PARALLEL SparsePath
diskann_bindings. Builds initial index + runs the insert/delete/consolidate update
loop, saving DATA-LESS checkpoints (graph + tags + .frozen; vectors recoverable from
the base pool via base[tag-1]). Reuses the exact semantics validated for deep200M.

Works for float32 / uint8 / int8 datasets (picks DiskANNIndex / *UInt8 / *Int8).
The base pool may be a 1B+ file; only the first (initial + max_iter*insert) vectors
are consumed, so no pre-slicing is needed.

Checkpoints saved at iteration{0, save_every, 2*save_every, ..., max_iter}.
Reload later with load_graph_only(graph, base, tags) — there is NO .data file.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, "/workspace/lib")
import diskann_bindings as diskann

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
    args = ap.parse_args()

    dt = _DT[args.dtype]
    IndexCls = getattr(diskann, _CLS[args.dtype])
    os.makedirs(args.out_ckpt_dir, exist_ok=True)

    # As-run build config 자동 보존 (규칙 2026-07-14: artifacts/<ds>/build_config.yaml).
    # out-ckpt-dir = <root>/<ds>/graph/checkpoints -> <root>/<ds>/build_config.yaml
    # 산출물 승격(mv) 시 함께 이동한다. 실패해도 빌드는 계속(베스트 에포트).
    try:
        ds_root = os.path.dirname(os.path.dirname(os.path.abspath(args.out_ckpt_dir)))
        rec = {
            "generator": "build_updatecycle.py (as-run auto dump)",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.uname().nodename,
            "argv": sys.argv,
            "args": {k: getattr(args, k) for k in vars(args)},
            "index_class": _CLS[args.dtype],
            "bindings": "/workspace/lib/diskann_bindings.so (parallel insert)",
            "convention": {
                "update": "insert 1% + delete 1%(oldest window) per iter",
                "consolidate": f"every {args.consolidate_every} iters (5% accumulated)",
                "checkpoints": f"data-less (graph+tags+.frozen), every {args.save_every} iters + iter0",
                "reload": "load_graph_only(graph, base, tags); vectors = base[tag-1]",
            },
        }
        import yaml as _yaml
        with open(os.path.join(ds_root, "build_config.yaml"), "w") as f:
            _yaml.safe_dump(rec, f, sort_keys=False, allow_unicode=True)
        print(f"[build] as-run config -> {ds_root}/build_config.yaml", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[build] WARN: build_config dump failed: {e}", flush=True)

    # base header + mmap (no RAM allocation for the pool)
    hdr = np.fromfile(args.base, dtype=np.uint32, count=2)
    n_total, d = int(hdr[0]), int(hdr[1])
    assert d == args.dim, f"dim mismatch: file {d} != --dim {args.dim}"
    consume = args.initial_size + args.max_iterations * args.points_insert
    assert consume <= n_total, f"pool too small: need {consume:,}, have {n_total:,}"
    vectors = np.memmap(args.base, dtype=dt, mode="r", offset=8, shape=(n_total, d))
    print(f"[build] {_CLS[args.dtype]} base={args.base} n={n_total:,} d={d} dtype={args.dtype} "
          f"consume={consume:,} R={args.R} L={args.L} a={args.alpha} nthreads={args.num_threads}", flush=True)

    index = IndexCls(d, args.max_capacity, args.R, args.L, args.alpha, args.num_threads, True)

    def save_dataless(it):
        ckpt = os.path.join(args.out_ckpt_dir, f"iteration{it}")
        t = time.time()
        index.save(ckpt, True, False)            # compact + graph/tags/.frozen, NO .data
        if os.path.exists(ckpt + ".frozen") and os.path.exists(ckpt + ".data"):
            os.remove(ckpt + ".data")
        print(f"  saved iteration{it} data-less ({time.time()-t:.1f}s)", flush=True)

    # --- initial build ---
    t0 = time.time()
    print(f"[build] initial build {args.initial_size:,} points...", flush=True)
    initial = np.ascontiguousarray(vectors[:args.initial_size])   # RAM copy of active slice
    index.build(initial)
    del initial
    print(f"[build] built {index.num_points:,} in {time.time()-t0:.1f}s", flush=True)
    save_dataless(0)

    # --- update loop (mirrors verified deep200M semantics) ---
    cursor = args.initial_size
    for it in range(1, args.max_iterations + 1):
        t_iter = time.time()
        insert_end = min(cursor + args.points_insert, n_total)
        if cursor < insert_end:
            newv = np.ascontiguousarray(vectors[cursor:insert_end])
            t_i = time.time()
            tags = index.insert_batch(newv)
            print(f"  Iter {it}: inserted {len(tags)} [{cursor},{insert_end}) "
                  f"tags[{tags[0]},{tags[-1]}] ({time.time()-t_i:.1f}s)", flush=True)
            cursor = insert_end
        if args.points_delete > 0 and index.num_points > args.points_delete:
            off = (it - 1) * args.points_delete + 1
            index.delete_batch(list(range(off, off + args.points_delete)))
            print(f"  Iter {it}: deleted {args.points_delete} tags[{off},{off+args.points_delete})", flush=True)
        if it % args.consolidate_every == 0:
            t_c = time.time()
            rem = index.consolidate()
            print(f"  Iter {it}: consolidate -> {rem} ({time.time()-t_c:.1f}s)", flush=True)
        if it % args.save_every == 0 or it == args.max_iterations:
            save_dataless(it)
            print(f"  Iter {it}: iter total {time.time()-t_iter:.1f}s", flush=True)

    print(f"[build] DONE total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
