#!/usr/bin/env python3
"""P1 hot-path byte-match harness — single-node, single-engine, no cluster.

Loads ONE shard-graph colocation partition (self-contained: every edge VID
stays inside the partition, so no peer/RDMA is needed) and runs
search_colocation_pure_batch. Run it once against the LIVE lib/ .so and once
against the worktree .so, then byte-compare the dumped indices/distances.

Coverage note: with everything local, this exercises the sequential path
(visited/resolve/nbr_loop/Pass2 local distance/queue/map/top-K extraction) and
the CAP batch path's candidate-resolve + Phase-2 collection + local distance
branch — i.e. all scratch-reuse and timer changes on local branches. The
RDMA remote branches (Pass1/Pass3, batch remote fetch, interleaved completion)
compile but are not executed here; they are validated in the distributed A/B.

Example (inside danns container, worker-host):
  python3 scripts/p1_hotpath_verify/single_node_bytematch.py \
    --lib /workspace/lib \
    --data-dir /workspace/datasets_local/p1_verify \
    --queries "/experiments/artifacts/deep100M/gt/gt_iteration150_shard_K10_Q10000_queries.npy" \
    --nq 1000 --L 100 --K 10 --threads 8 --batch-size 8 \
    --out /tmp/p1_live_cap.npz
Flags: --skip-timing, --batch-interleave (worktree .so only), --batch-size 0
for the pure sequential path.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="dir containing distributed_bindings*.so")
    ap.add_argument("--data-dir", required=True,
                    help="dir with partition0.coloc / global_to_vectorid.bin / "
                         "tag_to_local_p0.bin / local_to_tag_p0.bin / colocation_meta.json")
    ap.add_argument("--queries", required=True, help=".npy (N, dim) float32 queries")
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="CAP batch size (0 = sequential only)")
    ap.add_argument("--conv-window", type=int, default=3)
    ap.add_argument("--conv-threshold-pct", type=float, default=0.7)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--batch-interleave", action="store_true",
                    help="pass batch_interleave_completion=True (new .so only)")
    ap.add_argument("--out", required=True, help="output .npz (indices, distances)")
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    import distributed_bindings as db
    print(f"[harness] distributed_bindings from: {db.__file__}")

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "colocation_meta.json").read_text())
    p0 = meta["partitions"]["0"]

    # The iteration150_shard meta (regenerated 07-16) is stale by one entry
    # vs the May-01 .coloc payload; derive num_entries from the file itself
    # (ColocationStore::load checks exact size). Identical for both .so runs.
    coloc_bytes = (data_dir / p0["file"]).stat().st_size
    file_entries = coloc_bytes // meta["entry_size"]
    if file_entries != p0["num_entries"]:
        print(f"[harness] WARN: meta num_entries={p0['num_entries']} but file has "
              f"{file_entries} entries — using file value")
        p0 = dict(p0, num_entries=int(file_entries))

    eng = db.MultiPeerGraphStore(node_id=0, total_nodes=1,
                                 vector_dim=meta["dim"], rdma_device="", gid_index=1)
    eng.init_distance_only(meta["dim"], meta.get("data_type", "float32"))

    t0 = time.time()
    eng.load_colocation(
        coloc_file=str(data_dir / p0["file"]),
        global_to_vectorid_file=str(data_dir / "global_to_vectorid.bin"),
        tag_to_local_files={0: str(data_dir / "tag_to_local_p0.bin")},
        local_to_tag_files={0: str(data_dir / "local_to_tag_p0.bin")},
        partition_id=0,
        dim=meta["dim"], R=meta["R"], slack_R=meta["slack_R"],
        entry_size=meta["entry_size"], alignment=meta["alignment"],
        num_entries=p0["num_entries"], total_vectors=meta["total_vectors"],
        element_size=meta.get("element_size", 4),
    )
    print(f"[harness] colocation loaded in {time.time()-t0:.1f}s "
          f"({p0['num_entries']} entries)")

    cloc = p0.get("centroid_local_loc", -1)
    if cloc >= 0:
        eng.set_centroid_local_locs({0: cloc}, p0["num_entries"])
        print(f"[harness] centroid_local_loc={cloc}")
    floc = p0.get("frozen_local_loc", -1)
    if floc >= 0:
        eng.set_frozen_local_loc(floc)

    q = np.load(args.queries)[: args.nq].astype(np.float32, copy=False)
    qids = np.arange(q.shape[0], dtype=np.uint32)
    print(f"[harness] queries: {q.shape} {q.dtype}")

    kwargs = dict(
        num_threads=args.threads,
        batch_size=args.batch_size,
        conv_window=args.conv_window,
        conv_threshold_pct=args.conv_threshold_pct,
        entry_mode=0,
        skip_timing=args.skip_timing,
        compute_skip=False,
        interleave_factor=1,
    )
    if args.batch_interleave:
        kwargs["batch_interleave_completion"] = True

    t0 = time.time()
    res = eng.search_colocation_pure_batch(q, qids, args.K, args.L, 0, **kwargs)
    dt = time.time() - t0

    idx = np.ascontiguousarray(np.asarray(res["indices"], dtype=np.uint32))
    dist = np.ascontiguousarray(np.asarray(res["distances"], dtype=np.float32))
    stats = res.get("stats", {})
    print(f"[harness] search done in {dt:.2f}s | hops={stats.get('total_n_hops')} "
          f"cmps={stats.get('total_n_cmps')} "
          f"batch_rounds={stats.get('total_batch_rounds')} "
          f"conv_hop={stats.get('total_convergence_hop')}")

    np.savez(args.out, indices=idx, distances=dist)
    h_idx = hashlib.md5(idx.tobytes()).hexdigest()
    h_dist = hashlib.md5(dist.tobytes()).hexdigest()
    print(f"[harness] md5 indices={h_idx} distances={h_dist}")
    print(f"[harness] wrote {args.out}")


if __name__ == "__main__":
    main()
