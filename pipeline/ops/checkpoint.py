"""Checkpoint operation for the distributed engine."""

from __future__ import annotations

import json
import time
from typing import Any

from pipeline.operation_base import Operation
from pipeline.ops.checkpoint_utils import save_deleted_tags, save_tag_to_local


class CheckpointOp(Operation):
    """Dump the exact in-memory colocation state to run_dir/checkpoint/<label>/.

    Each node writes its own partition's files (no broadcast needed); the
    op-level barriers in OperationEngine guarantee the buffers are quiescent.

        partition{p}.coloc     byte-identical colocation buffer dump
        tag_to_local_p{p}.bin  [max_tag][table] uint32 snapshot
        deleted_tags_p{p}.bin  sorted uint32 lazy-deleted (pre-consolidate) tags
        colocation_meta.json   layout meta snapshot, written by the
                               coordinator — or by the partition-0 worker when
                               the coordinator is orchestrator-only and never
                               executes ops (engine.save() report-writer
                               idiom). Only the writer's own partition's
                               num_active is refreshed.

    File naming mirrors build_colocation output so a checkpoint dir can later
    serve as a colocation dir (reload wiring is out of scope; see the
    2026-06-12 design doc). Workload spec: {op: checkpoint, label: my_label};
    label is optional and defaults to ckpt00, ckpt01, ... per run.

    All nodes share the label dir on NFS, so stale-file cleanup on label
    re-use is strictly per-node: a node unlinks only its own partition's
    files (and the meta writer the meta) before writing.
    """
    name = "checkpoint"
    needs_broadcast = False

    def execute(self, ctx: UpdateContext, payload: Any):
        t0 = time.time()

        run_dir = getattr(ctx, 'run_dir', None)
        if run_dir is None:
            ctx.log("[WARN] No run_dir set, skipping checkpoint")
            return

        seq = getattr(ctx, '_checkpoint_seq', 0)
        ctx._checkpoint_seq = seq + 1
        label = str(self.params.get('label') or f"ckpt{seq:02d}")
        ckpt_dir = run_dir / "checkpoint" / label
        p = ctx.my_partition

        if not hasattr(ctx.rdma_store, 'dump_colocation'):
            raise RuntimeError(
                "rdma_store has no dump_colocation — rebuild the C++ bindings "
                "(./build_all.sh inside the container)")

        # Meta writer: the coordinator, or — when the coordinator is an
        # orchestrator-only node that never executes ops — the partition-0
        # worker (same idiom as engine.save()'s report writer).
        writes_meta = ctx.is_coordinator or (
            not getattr(ctx, 'coordinator_is_worker', True)
            and ctx.my_partition == 0)

        my_files = [f"partition{p}.coloc", f"tag_to_local_p{p}.bin",
                    f"deleted_tags_p{p}.bin"]
        if writes_meta:
            my_files.append("colocation_meta.json")
            my_files.append("inter_partition_edges.json")
        if ckpt_dir.exists():
            stale = [name for name in my_files if (ckpt_dir / name).exists()]
            if stale:
                ctx.log(f"[WARN] checkpoint label re-used, replacing own "
                        f"files in {ckpt_dir}: {stale}")
                for name in stale:
                    (ckpt_dir / name).unlink(missing_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ctx.log(f"CheckpointOp[{label}]: dumping to {ckpt_dir}")

        coloc_path = ckpt_dir / f"partition{p}.coloc"
        ctx.rdma_store.dump_colocation(str(coloc_path))

        n_mapped = 0
        if ctx.slot_manager is not None:
            n_mapped = save_tag_to_local(
                ctx.slot_manager.tag_to_local,
                ckpt_dir / f"tag_to_local_p{p}.bin")
        n_deleted = save_deleted_tags(
            getattr(ctx, 'delete_tags', set()) or set(),
            ckpt_dir / f"deleted_tags_p{p}.bin")

        if writes_meta and ctx.colocation_meta:
            parts = ctx.colocation_meta.get('partitions', {})
            if str(p) in parts and ctx.slot_manager is not None:
                parts[str(p)]['num_active'] = ctx.slot_manager.active_count
            with open(ckpt_dir / "colocation_meta.json", 'w') as f:
                json.dump(ctx.colocation_meta, f, indent=2)

        # Legacy DiskANN-format save. Default OFF: its VID->tag conversion is
        # broken (local_loc used as tag; ~97% of edges lost) and nothing
        # consumes the output. Kept behind the flag for reference only.
        save_ckpt = ctx.config.get('distributed_engine', {}).get(
            'save_checkpoint', False)
        if save_ckpt:
            ctx._save_checkpoint(run_dir)

        elapsed = time.time() - t0
        self.stats.custom['checkpoint_time_s'] = round(elapsed, 3)
        self.stats.custom['checkpoint_label'] = label
        self.stats.custom['mapped_tags'] = n_mapped
        self.stats.custom['deleted_tags'] = n_deleted
        ctx.log(f"CheckpointOp[{label}] done in {elapsed:.2f}s: "
                f"{coloc_path.name}, mapped={n_mapped}, deleted={n_deleted}")
