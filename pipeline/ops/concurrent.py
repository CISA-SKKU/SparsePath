"""Concurrent consolidate + search operation.

Runs the C++ consolidation and a real-query search loop simultaneously on
each worker (two Python threads; both bindings release the GIL). Replaces
the ib_read_bw co-run proxy with actual search QPS/latency measured while
consolidation is active.

Safety contract (22_Features/Consolidation_cache/concurrent_search_design.md):
- One shared multi-QP pool sized consol_threads + search_threads is set up
  serially before the window; consolidation uses tids [0, Cn) and search
  passes qp_tid_offset=Cn, so the per-thread QP slices are disjoint.
- ctx._consolidate_local(defer_cleanup=True) skips its own QP setup/teardown
  barriers (ctx._external_multi_qp) and leaves the deleted-set clears /
  local_to_tag zeroing pending; finish_deferred_consol_cleanup() runs them
  only after the search thread joins (R2/R3 fences).
- The search-side cache warmer is disabled for the whole op
  (DANNS_SEARCH_WARMER_OFF) because consolidation concurrently consumes and
  compacts the same cache (R4).

Search phases per node: `baseline_rounds` solo rounds (consolidation waits),
then co-run rounds until consolidation finishes, then `tail_rounds` solo
rounds — one op yields before/during/after QPS from the same query set.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import Any

import numpy as np

from pipeline import diagnostics
from pipeline.operation_base import Operation
from pipeline.ops.search.options import resolve_search_runtime_options

_NP_DTYPE = {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}


def _numa_core_split(rdma_device: str, logical_nic: bool = False):
    """(nic, far) core sets for the NIC socket and the other socket, or
    (None, None) when undeterminable (single socket, no sysfs). Both sides
    are physical-core-filtered (mirrors node_worker._maybe_pin_nic_numa);
    logical_nic=True returns the NIC side unfiltered (SMT siblings included)
    so N threads can be placed 1:1 on N logical cpus instead of 2:1 on
    physical cores."""
    try:
        nic = int(open(f'/sys/class/infiniband/{rdma_device}/device/numa_node')
                  .read())
        if nic < 0:
            return None, None
        nodes = sorted(int(d[4:]) for d in os.listdir('/sys/devices/system/node')
                       if d.startswith('node') and d[4:].isdigit())
        far = next((n for n in nodes if n != nic), None)
        if far is None:
            return None, None

        def cpulist(numa):
            cpus = set()
            cl = open(f'/sys/devices/system/node/node{numa}/cpulist').read().strip()
            for part in cl.split(','):
                a, _, b = part.partition('-')
                cpus.update(range(int(a), int(b or a) + 1))
            return cpus

        def phys(numa):
            out, seen = set(), set()
            for c in sorted(cpulist(numa)):
                core = open(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read().strip()
                pkg = open(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read().strip()
                if (pkg, core) not in seen:
                    seen.add((pkg, core))
                    out.add(c)
            return out

        return (cpulist(nic) if logical_nic else phys(nic)), phys(far)
    except Exception:
        return None, None


class ConcurrentOp(Operation):
    """Workload spec:

    - op: concurrent
      search:
        K: 10
        L: 100
        num_queries: 10000      # per round
        query_offset: 0
        num_threads: 32
        baseline_rounds: 2      # solo rounds before consolidation starts
        tail_rounds: 2          # solo rounds after consolidation ends
        max_corun_rounds: 0     # 0 = until consolidation finishes
      pin: numa                 # numa = consol far socket / search NIC socket
    """
    name = "concurrent"
    needs_broadcast = False

    def execute(self, ctx, payload: Any):
        consol_cfg = ctx.config.get('consolidation', {}) or {}
        cn = int(consol_cfg.get('num_threads', 1))
        sp = dict(self.params.get('search', {}) or {})
        options = resolve_search_runtime_options(ctx, sp)
        sn = int(sp.get('num_threads', options.num_threads) or 16)
        K = int(sp.get('K', 10))
        L = int(sp.get('L', 100))
        nq = int(sp.get('num_queries', 10000))
        qoff = int(sp.get('query_offset', 0))
        baseline_rounds = int(sp.get('baseline_rounds', 2))
        tail_rounds = int(sp.get('tail_rounds', 2))
        max_corun_rounds = int(sp.get('max_corun_rounds', 0))
        pin_mode = str(self.params.get('pin', 'numa')).lower()
        entry_mode = ctx.config.get('distributed_search', {}).get('entry_mode', 0)

        # ---- Query prep (mirrors SearchOp's local routing) ----
        ds_dtype = ctx.config.get('dataset', {}).get('data_type', 'float32')
        np_dtype = _NP_DTYPE.get(ds_dtype, np.float32)
        pool = ctx.queries
        if pool is None or len(pool) == 0:
            raise RuntimeError("ConcurrentOp: no query pool loaded (ctx.queries)")
        end = min(qoff + nq, len(pool))
        query_vectors = np.asarray(pool[qoff:end], dtype=np_dtype)
        query_partitions = ctx.partition_policy.assign_batch(
            query_vectors, ctx.centroids)
        my_ids = np.where(query_partitions == ctx.my_partition)[0]
        queries_array = np.ascontiguousarray(query_vectors[my_ids])
        qids_array = (qoff + my_ids).astype(np.uint32)
        ctx.log(f"ConcurrentOp: consol {cn}T + search {sn}T, "
                f"{len(qids_array)} queries/round (K={K} L={L}), "
                f"baseline={baseline_rounds} tail={tail_rounds} pin={pin_mode}")

        # ---- NUMA core split ----
        # DANNS_CONCURRENT_QUIET=1: suppress per-round logging + the 1ms NIC
        # burst monitor — diagnostic knob for the mgmt-NIC/NFS interference
        # hypothesis (per-round data still lands in the result JSON).
        quiet = os.environ.get('DANNS_CONCURRENT_QUIET', '') not in ('', '0')
        nic_cpus = far_cpus = None
        if pin_mode in ('numa', 'numa_logical'):
            dev = getattr(getattr(ctx.nw, 'rdma_store_wrapper', None),
                          'rdma_device', '') or ''
            nic_cpus, far_cpus = _numa_core_split(
                dev, logical_nic=(pin_mode == 'numa_logical'))
            if nic_cpus:
                ctx.log(f"ConcurrentOp: pin search->nic({sorted(nic_cpus)[:4]}...x{len(nic_cpus)}) "
                        f"consol->far({sorted(far_cpus)[:4]}...x{len(far_cpus)})")
            else:
                ctx.log("ConcurrentOp: [WARN] NUMA split unavailable — unpinned")

        # ---- R4 fence: no search-side cache warming during the window ----
        prev_warmer_env = os.environ.get('DANNS_SEARCH_WARMER_OFF')
        os.environ['DANNS_SEARCH_WARMER_OFF'] = '1'

        # ---- Shared multi-QP pool: [0, cn) consolidate | [cn, cn+sn) search ----
        num_peers = sum(1 for nid in ctx.partition_to_node.values()
                        if nid != ctx.node_id)
        use_mqp = (consol_cfg.get('use_multi_qp', False)
                   and cn > 1 and num_peers > 0)
        if use_mqp:
            my_node = ctx.cluster.get_node(ctx.node_id)
            total_qps = cn + sn
            # DANNS_CONCURRENT_QP_TOTAL: inflate the shared pool beyond cn+sn
            # (the extra QPs are connected but never used) — isolates "pool
            # size" as an interference variable in the thread-count cliff.
            qp_total_env = int(os.environ.get('DANNS_CONCURRENT_QP_TOTAL', '0') or 0)
            if qp_total_env > total_qps:
                total_qps = qp_total_env
            ctx.log(f"ConcurrentOp: setting up {total_qps} shared per-thread QPs...")
            ctx.setup_multi_qp_barriered(
                total_qps, my_node.rdma_port, "concurrent-multiqp")
            ctx.sync.barrier("concurrent-multi-qp-setup")
            ctx._external_multi_qp = True
            ctx.log("ConcurrentOp: shared multi-QP ready")
        qp_offset = cn if use_mqp else 0

        # ---- NIC monitor across the whole window (same infra as ConsolidateOp) ----
        _burst_dir = (ctx.run_dir / "nic_bw") if getattr(ctx, 'run_dir', None) else None
        _burst_proc = _burst_mono = None
        if quiet:
            _burst_dir = None
        if _burst_dir is not None:
            try:
                _burst_dir.mkdir(parents=True, exist_ok=True)
                _burst_csv = str(_burst_dir /
                                 f"nic_bw_concurrent_N{ctx.node_id}_S{self.stats.seq}.csv")
                _burst_proc, _burst_mono = diagnostics.nic_burst_monitor_start(_burst_csv)
            except Exception:
                _burst_proc = None
        _nic_before = diagnostics.rdma_snapshot()

        # ---- Threads ----
        T0 = time.time()
        baseline_done = threading.Event()
        consol_done = threading.Event()
        shared: dict = {'consol_stats': None, 'consol_error': None,
                        'search_error': None,
                        'consol_t_start': None, 'consol_t_end': None}
        rounds_log: list = []

        def run_consol():
            try:
                if far_cpus:
                    os.sched_setaffinity(0, far_cpus)
                baseline_done.wait()
                shared['consol_t_start'] = time.time() - T0
                shared['consol_stats'] = ctx._consolidate_local(defer_cleanup=True)
                shared['consol_t_end'] = time.time() - T0
            except Exception:
                shared['consol_error'] = traceback.format_exc()
            finally:
                consol_done.set()

        def one_round(phase, idx):
            t0 = time.time()
            br = ctx.rdma_store.search_colocation_pure_batch(
                queries_array, qids_array, K, L, ctx.my_partition,
                num_threads=sn,
                batch_size=options.batch_size,
                conv_window=options.conv_window,
                conv_threshold_pct=options.conv_threshold_pct,
                entry_mode=entry_mode,
                compute_skip=False,
                interleave_factor=1,
                qp_tid_offset=qp_offset,
                # FrozenSeedCache is pinned OFF here. Its staleness containment
                # is an op boundary (SearchOp drops the cache before each op),
                # and this op has none: consolidation runs concurrently with
                # these search rounds and can reassign a cached local_loc
                # mid-flight. reset_frozen_seed_cache only exists on a patched
                # .so, so it doubles as the probe for whether the kwarg exists.
                **({'frozen_seed_cache': False}
                   if hasattr(ctx.rdma_store, 'reset_frozen_seed_cache')
                   else {}))
            dt = time.time() - t0
            lat = np.asarray(br.get("per_query_latency_us", []), dtype=np.float64)
            rec = {'phase': phase, 'round': idx,
                   't_start': round(t0 - T0, 3), 'dur_s': round(dt, 3),
                   'queries': int(len(qids_array)),
                   'qps': round(len(qids_array) / dt, 1) if dt > 0 else 0.0}
            if len(lat) > 0:
                rec['lat_p50_us'] = round(float(np.percentile(lat, 50)), 1)
                rec['lat_p99_us'] = round(float(np.percentile(lat, 99)), 1)
            rounds_log.append(rec)
            if not quiet or idx % 100 == 0:
                ctx.log(f"ConcurrentOp: search[{phase}#{idx}] "
                        f"{rec['qps']} qps p50={rec.get('lat_p50_us', 0)}us "
                        f"p99={rec.get('lat_p99_us', 0)}us ({dt:.1f}s)")
            return br

        def run_search():
            last = None
            try:
                if nic_cpus:
                    os.sched_setaffinity(0, nic_cpus)
                for i in range(baseline_rounds):
                    last = one_round('baseline', i)
                baseline_done.set()
                i = 0
                while not consol_done.is_set():
                    last = one_round('corun', i)
                    i += 1
                    if max_corun_rounds and i >= max_corun_rounds:
                        break
                consol_done.wait()  # consolidation still owns the QPs until done
                for i in range(tail_rounds):
                    last = one_round('tail', i)
                # Keep the final round's results for post-hoc recall.
                if last is not None and getattr(ctx, 'run_dir', None):
                    np.savez_compressed(
                        ctx.run_dir / f"concurrent_last_round_N{ctx.node_id}.npz",
                        qids=qids_array,
                        indices=np.asarray(last["indices"]))
            except Exception:
                shared['search_error'] = traceback.format_exc()
                baseline_done.set()  # never leave the consol thread waiting

        th_c = threading.Thread(target=run_consol, name="concurrent-consol")
        th_s = threading.Thread(target=run_search, name="concurrent-search")
        th_s.start()
        th_c.start()
        th_c.join()
        th_s.join()

        # ---- R2/R3 fence: cleanup only after the search thread stopped ----
        ctx.finish_deferred_consol_cleanup()

        _nic_after = diagnostics.rdma_snapshot()
        if _burst_proc is not None:
            diagnostics.nic_burst_monitor_stop(_burst_proc)

        if use_mqp:
            ctx.sync.barrier("concurrent-multi-qp-teardown")
            ctx.rdma_store.teardown_multi_qp()
            ctx._external_multi_qp = False
            ctx.log("ConcurrentOp: shared multi-QP torn down")

        if prev_warmer_env is None:
            os.environ.pop('DANNS_SEARCH_WARMER_OFF', None)
        else:
            os.environ['DANNS_SEARCH_WARMER_OFF'] = prev_warmer_env

        if shared['consol_error']:
            raise RuntimeError(
                f"ConcurrentOp: consolidation failed:\n{shared['consol_error']}")
        if shared['search_error']:
            ctx.log(f"[WARN] ConcurrentOp: search thread failed:\n"
                    f"{shared['search_error']}")

        stats = dict(shared['consol_stats'] or {})
        elapsed = (shared['consol_t_end'] or 0) - (shared['consol_t_start'] or 0)
        nic_bw = diagnostics.rdma_compute_bw(
            _nic_before, _nic_after, max(time.time() - T0, 1e-9))
        if nic_bw:
            stats.update(nic_bw)

        def _phase_qps(phase):
            vals = [r['qps'] for r in rounds_log if r['phase'] == phase]
            return round(float(np.mean(vals)), 1) if vals else 0.0

        summary = {'baseline_qps': _phase_qps('baseline'),
                   'corun_qps': _phase_qps('corun'),
                   'tail_qps': _phase_qps('tail'),
                   'corun_rounds': sum(1 for r in rounds_log if r['phase'] == 'corun'),
                   'consol_t_start': shared['consol_t_start'],
                   'consol_t_end': shared['consol_t_end']}
        stats['concurrent_search'] = summary
        if getattr(ctx, 'run_dir', None):
            tl_file = ctx.run_dir / f"concurrent_search_N{ctx.node_id}_S{self.stats.seq}.json"
            with open(tl_file, 'w') as f:
                json.dump({'summary': summary, 'rounds': rounds_log,
                           'search_error': shared['search_error']}, f, indent=2)

        modified = stats.get('entries_modified', stats.get('modified', 0))
        self.stats.items_processed = modified
        self.stats.custom = stats
        ctx.log(f"ConcurrentOp done: consolidate {modified} modified in "
                f"{elapsed:.2f}s | search qps baseline={summary['baseline_qps']} "
                f"corun={summary['corun_qps']} tail={summary['tail_qps']}")
