"""Insert operation for the distributed engine."""

from __future__ import annotations

import pickle
import time

import numpy as np

from pipeline.danns_types import encode_vector_id
from pipeline.config import get_pq_filter_config
from pipeline.operation_base import Operation
from pipeline.ops.redirect_utils import (adopt_redirected_in,
                                         settle_redirected_out)
from pipeline.slot_manager import INVALID_LOC


def _pack_insert_vectors(vecs_list):
    """Prepare the insert vector buffer for the C++ ``insert_batch`` binding.

    The binding signature is ``py::array_t<float>`` and C++ ``reinterpret_cast``s
    the raw bytes back to int8/uint8 (``distributed_bindings.cpp`` insert cases),
    striding rows by ``aligned_dim = round_up(dim, 8)``. The old code passed
    ``np.stack(...).astype(np.float32)``, which *value-converts* each integer to
    its float32 encoding — so the reinterpret read IEEE-754 bytes as the vector
    (garbage). That silently corrupted every engine-inserted int8/uint8 vector,
    decaying recall under churn (float32 was unaffected: the cast is identity).

    Fix: for integer dtypes, pack the native bytes into an ``aligned_dim``-wide
    zero-padded row buffer and ``view`` it as float32 so the C++ reinterpret
    recovers the exact integer vector. The aligned_dim padding also fixes the
    per-row stride mismatch for dims not divisible by 8 (e.g. spacev dim 100).
    Float32 keeps the original (correct) path.
    """
    stacked = np.ascontiguousarray(np.stack(vecs_list))
    if stacked.dtype == np.float32:
        return stacked
    n, vdim = stacked.shape
    adim = ((vdim + 7) // 8) * 8                      # round_up(dim, 8)
    buf = np.zeros((n, adim), dtype=stacked.dtype)    # int8 / uint8
    buf[:, :vdim] = stacked
    # adim is a multiple of 8 → byte width divisible by 4, view is exact.
    return np.ascontiguousarray(buf).view(np.float32)


class InsertOp(Operation):
    """Insert points into the distributed graph.

    params:
        count: int — number of points to insert
        seed: Optional[int] — RNG seed for reproducibility
    """
    name = "insert"
    needs_broadcast = True

    def _num_threads(self, ctx) -> int:
        t = self.params.get('num_threads', None)
        if t is not None:
            return int(t)
        return int(ctx.config.get('insert', {}).get('num_threads', 1))

    def prepare(self, ctx: UpdateContext) -> dict:
        count = self.params.get('count', 0)
        seed = self.params.get('seed', None)
        return ctx._prepare_insert_assignments(count, seed=seed)

    def execute(self, ctx: UpdateContext, assignments: dict):
        my_gids = assignments.get(ctx.my_partition, [])
        meta = assignments.get('_meta', {})
        ctx.log(f"InsertOp: {len(my_gids)} points for partition {ctx.my_partition}")

        # Track new (partition, local_loc, tag) mappings for cross-node sync
        ctx._new_local_to_tag = []
        my_inserts = []
        tags_list, locs_list, vecs_list = [], [], []
        skipped = 0

        # The pre-insert work below (NFS vector read, slot allocation) can
        # fail. With the LoadBalancer enabled, the lb-counters-pushed barrier
        # further down MUST still fire on this node: engine.py catches op
        # exceptions and proceeds straight to the op-end barrier, and the
        # sync protocol's frames are positional — one node skipping a mid-op
        # barrier while peers wait desyncs the whole cluster. Capture and
        # re-raise after the barrier.
        pre_barrier_error = None
        try:
            # Load vectors for my gids from base file (each node reads its own)
            if my_gids:
                start_gid = meta.get('start_gid', min(my_gids))
                end_gid = meta.get('end_gid', max(my_gids) + 1)
                chunk = ctx._read_vectors_range(start_gid, end_gid - start_gid)
                for gid in my_gids:
                    vec = chunk[gid - start_gid]
                    my_inserts.append((gid, vec))
                del chunk
                ctx.log(f"InsertOp: loaded {len(my_inserts)} vectors from base file "
                        f"(range [{start_gid}, {end_gid}))")

            # --- Slot allocation (Python, fast local-only) ---
            # Pre-extend tag_to_local arrays once for the whole batch. Otherwise
            # each per-insert extend triggers an O(L) np.concatenate on a multi-
            # hundred-MB array, making the loop O(N*L). Worker's slot_manager.t2l
            # and ctx.tag_to_local[my_partition] are both indexed by tag and need
            # the same coverage.
            if my_inserts and ctx.slot_manager is not None:
                batch_max_tag = max(gid + 1 for gid, _ in my_inserts)
                sm_t2l = ctx.slot_manager.tag_to_local
                if batch_max_tag >= len(sm_t2l):
                    ext = np.full(batch_max_tag + 1 - len(sm_t2l),
                                  INVALID_LOC, dtype=np.uint32)
                    ctx.slot_manager.tag_to_local = np.concatenate([sm_t2l, ext])
                ctx_t2l = ctx.tag_to_local.get(ctx.my_partition)
                if ctx_t2l is None:
                    ctx.tag_to_local[ctx.my_partition] = np.full(
                        batch_max_tag + 1, INVALID_LOC, dtype=np.uint32)
                elif batch_max_tag >= len(ctx_t2l):
                    ext = np.full(batch_max_tag + 1 - len(ctx_t2l),
                                  INVALID_LOC, dtype=np.uint32)
                    ctx.tag_to_local[ctx.my_partition] = np.concatenate([ctx_t2l, ext])

            for global_id, point in my_inserts:
                tag = global_id + 1  # DiskANN 1-based tag
                # Check duplicate tag
                if ctx.slot_manager is not None and ctx.slot_manager.get_local_loc(tag) is not None:
                    skipped += 1
                    continue
                if ctx.slot_manager is None:
                    ctx.log("[WARN] No slot manager — cannot insert")
                    break
                local_loc = ctx.slot_manager.allocate(tag)

                # Update tag_to_local array (Python side). Pre-extension above means
                # the index is in-bounds for all batch tags; this is just a write.
                t2l = ctx.tag_to_local[ctx.my_partition]
                t2l[tag] = np.uint32(local_loc)

                # Update C++ PartitionMapper tag/loc mappings
                ctx.rdma_store.update_tag_to_local_entry(ctx.my_partition, tag, local_loc)
                ctx.rdma_store.update_local_to_tag_entry(ctx.my_partition, local_loc, tag)

                # Record for cross-node sync
                ctx._new_local_to_tag.append((ctx.my_partition, local_loc, tag))

                tags_list.append(tag)
                locs_list.append(local_loc)
                vecs_list.append(point)

            if skipped > 0:
                ctx.log(f"InsertOp: {skipped} points skipped (duplicate tag)")
        except Exception as e:
            pre_barrier_error = e

        # --- Stage B LoadBalancer (measure-only): publish my live vector
        # count to the RDMA counter MR, then barrier so every node's counter
        # is current before any node snapshots them in insert_batch.
        # active_count already includes this batch's allocations (allocate()
        # ran above) — symmetric on all nodes. The barrier fires on every
        # worker regardless of batch size or pre-insert failure and is
        # relayed by the orchestrator (_relay_mid_op_barriers), both gated
        # on load_balancer.enabled — gating drift = deadlock.
        lb_cfg = ctx.config.get('load_balancer', {}) or {}
        lb_enabled = bool(lb_cfg.get('enabled', False))
        # Live redirect: in-batch redirect via the alloc server
        # (docs/features/live_redirect_design.md). All nodes and the
        # orchestrator relay must agree on this flag — it gates the
        # redirect window + the redirect-batches-done barrier below.
        redirect_enabled = lb_enabled and bool(lb_cfg.get('redirect', False))
        # Parallel insert (redirect-on supported: alloc client + counter
        # reads are serialized per-peer in C++). Workload spec:
        #   {op: insert, count: N, num_threads: T, use_multi_qp: true}
        # GATING CONTRACT: pipeline/orchestrator.py _relay_mid_op_barriers
        # mirrors ins_multi_qp EXACTLY — drift = deadlock.
        ins_threads = self._num_threads(ctx)
        ins_use_mqp = bool(self.params.get('use_multi_qp', True))
        num_peers = sum(1 for nid in ctx.partition_to_node.values()
                        if nid != ctx.node_id)
        ins_multi_qp = ins_threads > 1 and ins_use_mqp and num_peers > 0
        window_open = False
        if lb_enabled:
            if ctx.slot_manager is not None and \
                    hasattr(ctx.rdma_store, 'set_local_vector_count'):
                ctx.rdma_store.set_local_vector_count(
                    int(ctx.slot_manager.active_count))
            else:
                ctx.log("[WARN] LoadBalancer enabled but counter MR binding "
                        "or slot manager unavailable — counter not pushed")
            if redirect_enabled and ctx.slot_manager is not None and \
                    hasattr(ctx.rdma_store, 'open_redirect_window'):
                # Lend the remaining freelist to C++ as the redirect reserve
                # (Python freelist is frozen until close). max_tag bounds the
                # pre-sized tables: tags are gid+1, batch gids < end_gid.
                # Captured: a throw here would skip the barriers below and
                # hang the cluster (pre_barrier_error discipline).
                try:
                    reserve = np.array(ctx.slot_manager.free_slots,
                                       dtype=np.uint32)
                    ctx.slot_manager.free_slots = []
                    max_tag = int(meta.get('end_gid', max(tags_list or [0])))
                    ctx.rdma_store.open_redirect_window(
                        reserve, max_tag, int(ctx.num_partitions))
                    window_open = True
                except Exception as e:
                    if pre_barrier_error is None:
                        pre_barrier_error = e
            ctx.sync.barrier("lb-counters-pushed")

        # With redirect on, all peers are committed to the
        # redirect-batches-done barrier below — raising before it would
        # leave them waiting forever (the same failure class the lb-counters
        # capture exists for). Carry the error to after the window close.
        if pre_barrier_error is not None and not redirect_enabled \
                and not ins_multi_qp:
            raise pre_barrier_error
        mid_op_error = pre_barrier_error

        # --- C++ batch insert (search + prune + write + reverse edges) ---
        t0 = time.time()

        # Build special_vids set (frozen + centroid entries to exclude).
        # Hoisted out of the phase-1 guard: a node with an empty phase-1
        # batch can still receive phase-2 redirects.
        special_vids = []
        for p_str, p_meta in ctx.colocation_meta.get('partitions', {}).items():
            p = int(p_str)
            fl = p_meta.get('frozen_local_loc', -1)
            cl = p_meta.get('centroid_local_loc', -1)
            if fl >= 0:
                special_vids.append(encode_vector_id(p, fl))
            if cl >= 0:
                special_vids.append(encode_vector_id(p, cl))

        # LITS (long∧inter) filter on the insert forward pool. Reuses the
        # same `lits.enabled` switch as consolidation (engine_context.py).
        # inter is judged against the inserting partition (= the final owner;
        # phase-2 re-inserts pass m_assign via my_partition).
        lits_enabled = bool(ctx.config.get('lits', {}).get('enabled', False))

        def _insert_batch(vectors_arr, tags_arr, locs_arr,
                          load_balance=False, redirect=False):
            return ctx.rdma_store.insert_batch(
                vectors=vectors_arr,
                tags=tags_arr,
                local_locs=locs_arr,
                my_partition=ctx.my_partition,
                R=ctx.R, L=ctx.L, alpha=ctx.alpha, maxc=ctx.maxc,
                slack_R=ctx.slack_R,
                special_vids_list=special_vids,
                num_threads=ins_threads,
                entry_mode=1,  # FROZEN_ONLY
                lits_filter=lits_enabled,
                lits_my_partition=ctx.my_partition,
                load_balance=load_balance,
                lb_threshold=float(lb_cfg.get('candidate_threshold', 0.7)),
                redirect=redirect,
                redirect_timeout_ms=int(lb_cfg.get('alloc_timeout_ms', 500)))

        # Per-thread QPs for parallel insert's beam_search (without this,
        # threads fall back to the shared fetcher serialized by
        # fetcher_mutexes — correct but contended). Same bracket as the
        # search op (pipeline/ops/search/op.py:132-157).
        if ins_multi_qp:
            my_node = ctx.cluster.get_node(ctx.node_id)
            ctx.log(f"InsertOp: setting up {ins_threads} per-thread QPs...")
            ctx.rdma_store.setup_multi_qp(ins_threads, my_node.rdma_port)
            ctx.sync.barrier("insert-multi-qp-setup")

        redirected_out = []
        if vecs_list and mid_op_error is None:
            try:
                insert_stats = _insert_batch(
                    _pack_insert_vectors(vecs_list),
                    np.array(tags_list, dtype=np.uint32),
                    np.array(locs_list, dtype=np.uint32),
                    load_balance=lb_enabled,
                    redirect=redirect_enabled)
            except Exception as e:
                if redirect_enabled or ins_multi_qp:
                    mid_op_error = e
                else:
                    raise
        if vecs_list and mid_op_error is None:
            self.stats.items_processed = insert_stats.get('inserted', 0)
            self.stats.items_failed = insert_stats.get('failed', 0)
            self.stats.custom['latency_breakdown'] = {
                'search_sec': insert_stats.get('search_sec', 0),
                'prune_sec': insert_stats.get('prune_sec', 0),
                'write_sec': insert_stats.get('write_sec', 0),
                'reverse_sec': insert_stats.get('reverse_sec', 0),
            }
            self.stats.custom['rdma_stats'] = {
                'local_vec_reads': insert_stats.get('local_vec_reads', 0),
                'remote_vec_reads': insert_stats.get('remote_vec_reads', 0),
                'local_nbr_reads': insert_stats.get('local_nbr_reads', 0),
                'remote_nbr_reads': insert_stats.get('remote_nbr_reads', 0),
                'local_nbr_writes': insert_stats.get('local_nbr_writes', 0),
                'remote_nbr_writes': insert_stats.get('remote_nbr_writes', 0),
            }
            self.stats.custom['lits'] = {
                'checked': insert_stats.get('lits_checked', 0),
                'filtered': insert_stats.get('lits_filtered', 0),
            }
            if lb_enabled:
                self.stats.custom['load_balancer'] = {
                    'considered': insert_stats.get('redirect_considered', 0),
                    'would_redirect': insert_stats.get('redirect_would_happen', 0),
                    # Live redirect counts redirected points as inserted
                    # (they complete fully, just on m_assign) — denominator
                    # already matches Stage B's definition.
                    'inserted': insert_stats.get('inserted', 0),
                    'counters': {int(k): int(v) for k, v in
                                 insert_stats.get('lb_counters', {}).items()},
                }
            # Reflect C++ insert RDMA stats into Python-side counters for report
            remote_vec = insert_stats.get('remote_vec_reads', 0)
            remote_nbr = insert_stats.get('remote_nbr_reads', 0)
            remote_wrt = insert_stats.get('remote_nbr_writes', 0)
            ctx.rdma_counters.vector_access_local += insert_stats.get('local_vec_reads', 0)
            ctx.rdma_counters.vector_access_remote += remote_vec
            ctx.rdma_counters.neighbor_access_local += insert_stats.get('local_nbr_reads', 0)
            ctx.rdma_counters.neighbor_access_remote += remote_nbr
            ctx.rdma_counters.write_neighbors_local += insert_stats.get('local_nbr_writes', 0)
            ctx.rdma_counters.write_neighbors_remote += remote_wrt
            ctx.rdma_counters.fetch_entry_calls += remote_vec + remote_nbr + remote_wrt
            ctx.rdma_counters.fetch_entry_items += remote_vec + remote_nbr + remote_wrt
        else:
            self.stats.items_processed = 0

        if ins_multi_qp:
            # Barrier FIRST so every peer has stopped RDMA-reading before
            # QPs/MRs are destroyed. Fires even when the batch errored
            # (mid_op_error) — peers are committed to this barrier; the
            # deferred raise below the redirect section handles the error.
            ctx.sync.barrier("insert-multi-qp-teardown")
            ctx.rdma_store.teardown_multi_qp()

        # =================================================================
        # Live redirect settlement (docs/features/live_redirect_design.md).
        # Grants can arrive until the LAST node finishes its batch, so the
        # window closes only after the redirect-batches-done barrier. Errors
        # captured above are carried so the fence + close always run.
        # =================================================================
        if redirect_enabled:
            ctx.sync.barrier("redirect-batches-done")
            ledger = {'granted': [], 'reserve_unused': []}
            if window_open:
                ledger = ctx.rdma_store.close_redirect_window()

            # Origin side: reclaim the provisional slots of moved points and
            # retract their broadcast entries. C++ cleared its own runtime
            # tables live at redirect time; this is Python's bookkeeping.
            redirected_out = []
            if mid_op_error is None and vecs_list:
                redirected_out = [
                    (int(t), int(o), int(fp), int(fl))
                    for t, o, fp, fl in insert_stats.get('redirected_out', [])]
            released, ctx._new_local_to_tag = settle_redirected_out(
                [(t, fp) for t, _, fp, _ in redirected_out],
                ctx.slot_manager, ctx.tag_to_local[ctx.my_partition],
                ctx._new_local_to_tag, ctx.my_partition)

            # Target side: adopt the slots my alloc server granted mid-batch
            # (they came from the lent reserve, not allocate()); the owner
            # publishes the final mapping via _new_local_to_tag.
            # A mapping is published ONLY for a slot whose entry proves the
            # claim (entry_vid == encode(my_partition, tag)) — a grant whose
            # client timed out before writing (late/lost reply) is reclaimed
            # instead, neutralizing every stale-grant race in this family.
            granted = [(int(t), int(f)) for t, f in ledger.get('granted', [])]
            claimed, orphaned = [], []
            if granted and hasattr(ctx.rdma_store, 'local_get_entry_vid'):
                for t, f in granted:
                    if ctx.rdma_store.local_get_entry_vid(f) == \
                            encode_vector_id(ctx.my_partition, t):
                        claimed.append((t, f))
                    else:
                        orphaned.append((t, f))
            else:
                claimed = granted
            for t, f in orphaned:
                ctx.rdma_store.update_tag_to_local_entry(
                    ctx.my_partition, t, 0xFFFFFFFF)
                ctx.rdma_store.update_local_to_tag_entry(
                    ctx.my_partition, f, 0)
                ctx.slot_manager.free_slots.append(f)
                ctx.log(f"[WARN] redirect: orphaned grant reclaimed "
                        f"(tag={t}, loc={f}) — origin fell back")
            adopted = adopt_redirected_in(
                claimed, ctx.slot_manager,
                ctx.tag_to_local[ctx.my_partition],
                ctx._new_local_to_tag, ctx.my_partition)

            # Freelist = released provisionals (appended by release() inside
            # settle) + unused reserve — the spec's (b)+(c) both-or-corrupt
            # rule.
            ctx.slot_manager.free_slots.extend(
                int(x) for x in ledger.get('reserve_unused', []))

            fallback = 0
            if mid_op_error is None and vecs_list:
                fallback = int(insert_stats.get('redirect_fallback', 0))
            lb_custom = self.stats.custom.setdefault('load_balancer', {})
            lb_custom['redirected_out'] = len(redirected_out)
            lb_custom['redirected_in'] = adopted
            lb_custom['redirect_fallback'] = fallback
            ctx.log(f"InsertOp live redirect: out={len(redirected_out)} "
                    f"in={adopted} fallback={fallback}")

        # The barrier-fenced window is closed — now it is safe to fail.
        if mid_op_error is not None:
            raise mid_op_error

        elapsed = time.time() - t0
        if skipped > 0:
            self.stats.custom['items_skipped'] = skipped
        self.stats.custom['num_threads'] = ins_threads

        # Update PQ codes for ALL newly inserted vectors (all partitions).
        # Each node independently encodes using the C++ codebook — no TCP needed.
        # assignments contains all partitions' inserts (broadcast from coordinator).
        pq_cfg = get_pq_filter_config(ctx.config)
        if pq_cfg.get('enabled', False) and hasattr(ctx.rdma_store, 'encode_and_update_pq'):
            try:
                # Partition keys are ints; skip '_meta'. Values are plain gid
                # lists. (Was dead code: the old loop tuple-unpacked gids and
                # tripped on '_meta', so the except below ate every encode.)
                all_insert_gids = []
                for partition_id in sorted(k for k in assignments
                                           if isinstance(k, int)):
                    all_insert_gids.extend(
                        int(g) for g in assignments[partition_id])
                if all_insert_gids:
                    gids_arr = np.array(all_insert_gids, dtype=np.uint32)
                    vecs_arr = np.ascontiguousarray(ctx.vectors[gids_arr], dtype=np.float32)
                    ctx.rdma_store.encode_and_update_pq(gids_arr, vecs_arr)
                    ctx.log(f"InsertOp: PQ codes encoded locally for {len(gids_arr)} vectors")
            except Exception as e:
                ctx.log(f"[WARN] PQ encode failed: {e}")

        ctx.log(f"InsertOp done: {self.stats.items_processed} inserted, "
                f"{skipped} skipped (dup tag), "
                f"{self.stats.items_failed} failed in {elapsed:.2f}s")
