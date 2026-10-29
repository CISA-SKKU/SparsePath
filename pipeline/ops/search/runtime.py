"""Search execution runtime helpers.

This module owns the Python boundary around C++ search calls. The hot
optimization for convergence-aware batched fetching (phase-aware batching plus
PQ filtering) lives in the C++ search path. compute_skip is a benchmark/probe
knob;
Python prepares arrays, invokes bindings, and normalizes returned stats.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from .diagnostics import (
    NicBurstSession, merge_rdmabw_stats, pmu_start_for_l,
    pmu_stop_and_merge, rdma_snapshot,
)
from .options import SearchRuntimeOptions


def run_l_sweep(ctx, my_queries: list, L_values: list, K: int, broadcast: bool,
                options: SearchRuntimeOptions, trace_config, stats_seq: int,
                entry_parts=None, work_stealing: bool = False):
    batch_size = options.batch_size
    conv_window = options.conv_window
    conv_threshold_pct = options.conv_threshold_pct
    compute_skip = options.compute_skip
    search_num_threads = options.num_threads
    search_use_multi_qp = options.use_multi_qp
    interleave_factor = options.interleave_factor
    skip_timing = options.skip_timing
    batch_interleave_completion = options.batch_interleave_completion
    batch_remote_first = options.batch_remote_first
    frozen_seed_cache = options.frozen_seed_cache

    # Determine numpy dtype from dataset config
    _NP_DTYPE = {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}
    ds_dtype_str = ctx.config.get('dataset', {}).get('data_type', 'float32')
    np_dtype = _NP_DTYPE.get(ds_dtype_str, np.float32)

    # Pre-build batch arrays (shared across L values)
    queries_array = np.array([qv for _, qv in my_queries], dtype=np_dtype) if my_queries else np.empty((0, 0), dtype=np_dtype)
    qids_array = np.array([qid for qid, _ in my_queries], dtype=np.uint32) if my_queries else np.empty(0, dtype=np.uint32)

    # Run search for each L value
    results_by_L: Dict[int, list] = {}
    pure_stat_keys = ['n_hops', 'n_cmps',
                      'local_vector_accesses', 'remote_vector_accesses',
                      'local_neighbor_accesses', 'remote_neighbor_accesses',
                      'resolve_fail', 'visited_count',
                      'pq_filtered', 'unresolved_vids',
                      # FrozenSeedCache. Follows this list's contract: the C++
                      # batch dict emits total_<key>, the single-query dict
                      # emits <key>.
                      'frozen_seed_cache_hits']
    rdma_entry_keys = ['rdma_entry_calls', 'rdma_entry_items']
    rdma_timing_keys = ['rdma_distance_us', 'rdma_fetch_us', 'rdma_total_us']
    _per_L_stats: Dict[int, dict] = {}
    _per_L_latency: Dict[int, dict] = {}
    last_agg_stats: Dict[str, int] = {}

    # Fine-grained NIC BW monitor around the L-sweep. CSV and per-L windows
    # are managed by the diagnostics session.
    burst_session = NicBurstSession.start(ctx)

    for L in L_values:
        trace_config.start_l(ctx, L)

        results = []
        agg_stats: Dict[str, int] = {k: 0 for k in pure_stat_keys + rdma_entry_keys}
        agg_rdma_timing: Dict[str, float] = {k: 0.0 for k in rdma_timing_keys}
        per_query_total_us: list = []  # per-query E2E latency (both paths)
        last_batch_result = None

        # NIC snapshot BEFORE t0 so subprocess overhead (~100-300ms) is excluded
        _nic_snap_before = rdma_snapshot()
        _pmu_path = pmu_start_for_l(ctx, stats_seq, L)
        # Burst monitor alignment: monotonic elapsed relative to monitor start
        _burst_l_start = burst_session.mark_l_start()
        _search_elapsed = None
        t0 = time.time()
        if len(my_queries) > 0:
            # ---- Batch API: all queries at once (OpenMP if threads > 1) ----
            search_entry_mode = ctx.config.get('distributed_search', {}).get('entry_mode', 0)
            # In broadcast (scatter-gather) mode, return top-L per shard so the
            # coordinator's distance-based merge has enough candidates to recover
            # global top-K. Returning only K per shard caps recall well below the
            # full-graph baseline (e.g. deep100M sharded broadcast saturates at
            # 0.9495 because each shard's local top-10 misses some of its own
            # contribution to the global top-10). Setting K_arg=L makes the
            # binding allocate (N, L) output and the algorithm pulls all L
            # candidates from the search pool — same compute, more bytes back.
            batch_K = L if broadcast else K
            batch_result = ctx.rdma_store.search_colocation_pure_batch(
                queries_array, qids_array, batch_K, L, ctx.my_partition,
                num_threads=search_num_threads,
                batch_size=batch_size,
                conv_window=conv_window,
                conv_threshold_pct=conv_threshold_pct,
                entry_mode=search_entry_mode,
                skip_timing=skip_timing,
                compute_skip=compute_skip,
                interleave_factor=interleave_factor,
                batch_interleave_completion=batch_interleave_completion,
                # P1b: only passed when enabled so a pre-P1b .so (which lacks the
                # kwarg) keeps working; the compiled default is false anyway.
                **({'batch_remote_first': True} if batch_remote_first else {}),
                # FrozenSeedCache: only passed when DISABLED, so a pre-patch
                # .so (which lacks the kwarg) keeps loading; the compiled
                # default is true.
                **({'frozen_seed_cache': False} if not frozen_seed_cache else {}),
                # Per-query entry partition: same conditional-pass rule, so an
                # older .so without the kwarg still loads.
                **({'query_entry_parts': entry_parts}
                   if entry_parts is not None else {}),
                **({'work_stealing': True} if work_stealing else {}))

            # Stop the throughput clock here: everything below is result
            # marshalling (numpy -> python ints/floats per query per K), not
            # search. Leaving it inside cost ~60ms per L point -- 16% of the
            # measured time at L=100 with 2500 queries/node -- which silently
            # deflated every QPS number we have ever reported and every
            # comparison against a baseline that times only its query loop.
            _search_elapsed = time.time() - t0
            result_indices = np.asarray(batch_result["indices"])    # (N, batch_K)
            # Distances are needed for post-hoc top-K merge under broadcast
            # (scatter-gather) mode. Fall back to an inf-filled array if the
            # binding does not return them — recall-merge becomes order-blind
            # but counts still work for sanity checks.
            _raw_dists = batch_result.get("distances")
            if _raw_dists is not None:
                result_dists = np.asarray(_raw_dists, dtype=np.float32)
            else:
                result_dists = np.full(result_indices.shape, np.inf, dtype=np.float32)
            # Work stealing: every node was handed the whole query array but
            # only ran the ids it claimed. per_query_latency_us stays 0.0 for
            # rows this node never touched, so it doubles as the claim mask --
            # emitting unclaimed rows would inject garbage into the gather.
            _lat = np.asarray(batch_result.get("per_query_latency_us", []))
            _claimed = (_lat > 0) if (work_stealing and _lat.size == len(my_queries)) else None
            for i, (qid, _) in enumerate(my_queries):
                if _claimed is not None and not _claimed[i]:
                    continue
                global_ids = []
                for gid in result_indices[i]:
                    gid = int(gid)
                    global_ids.append(gid if gid != 0xFFFFFFFF else -1)
                dist_list = [float(d) for d in result_dists[i]] \
                            if result_dists.ndim == 2 else [float('inf')] * len(global_ids)
                results.append((qid, global_ids, dist_list))

            batch_stats = batch_result.get("stats", {})
            for k in pure_stat_keys:
                agg_stats[k] = int(batch_stats.get(f"total_{k}", 0))
            for k in rdma_entry_keys:
                agg_stats[k] = int(batch_stats.get(f"total_{k}", 0))
            # Batch API uses different naming: total_rdma_vector_us, total_rdma_neighbor_us
            agg_rdma_timing['rdma_distance_us'] = float(batch_stats.get('total_rdma_vector_us', 0.0))
            agg_rdma_timing['rdma_fetch_us'] = float(batch_stats.get('total_rdma_neighbor_us', 0.0))
            agg_rdma_timing['rdma_total_us'] = agg_rdma_timing['rdma_distance_us'] + agg_rdma_timing['rdma_fetch_us']

            # Batch/PQ extra stats (conditionally present in C++ output)
            for ek in ['total_convergence_hop', 'total_batch_rounds',
                       'total_batch_candidates', 'total_fallback_count',
                       # Convergence exposure + FrozenSeedCache warm size.
                       # Unconditional in C++, so always present on a patched
                       # .so and silently absent on an older one.
                       'sum_convergence_hop', 'n_converged_queries',
                       'sum_batch_rounds', 'frozen_seed_cache_entries',
                       'pq_cache_hits', 'pq_cache_misses', 'pq_cache_evictions',
                       'pq_cache_size', 'pq_cache_capacity']:
                val = batch_stats.get(ek)
                if val is not None:
                    agg_stats[ek] = int(val)
            last_batch_result = batch_result
            # CPU breakdown from batch API
            _cpu_bd = {}
            for _ck in ['total_cpu_visited_us', 'total_cpu_resolve_us',
                        'total_cpu_distance_us', 'total_cpu_queue_us',
                        'total_cpu_nbr_loop_us', 'total_cpu_map_us',
                        'total_cpu_pq_us', 'total_lut_build_us',
                        'total_candidate_fetch_phase_us',
                        'total_pass1_us', 'total_pass2_us', 'total_pass3_us',
                        'total_pass3_poll_us', 'total_pass3_compute_us',
                        'total_batch_nbr_poll_us', 'total_batch_vec_poll_us']:
                _cpu_bd[_ck] = float(batch_stats.get(_ck, 0))
            agg_stats['cpu_breakdown'] = _cpu_bd
            # Collect per-query E2E latency from batch result
            lat_arr = np.asarray(batch_result.get("per_query_latency_us", []))
            if len(lat_arr) > 0:
                per_query_total_us = lat_arr.tolist()
        else:
            # ---- Single-query API: one at a time (original path) ----
            for qid, query_vec in my_queries:
                query = np.asarray(query_vec)
                try:
                    search_entry_mode = ctx.config.get('distributed_search', {}).get('entry_mode', 0)
                    indices, distances, search_stats = ctx.rdma_store.search_colocation_pure(
                        query, K, L, search_entry_mode, ctx.my_partition,
                        trace_query_id=qid,
                        batch_size=batch_size,
                        conv_window=conv_window,
                        conv_threshold_pct=conv_threshold_pct,
                        **({'frozen_seed_cache': False}
                           if not frozen_seed_cache else {}))

                    for k in pure_stat_keys:
                        agg_stats[k] += int(search_stats.get(k, 0))
                    for k in rdma_entry_keys:
                        agg_stats[k] += int(search_stats.get(k, 0))
                    # C++ returns rdma_vector_us/rdma_neighbor_us; map to internal keys
                    agg_rdma_timing['rdma_distance_us'] += float(search_stats.get('rdma_vector_us', 0.0))
                    agg_rdma_timing['rdma_fetch_us'] += float(search_stats.get('rdma_neighbor_us', 0.0))
                    agg_rdma_timing['rdma_total_us'] = (
                        agg_rdma_timing['rdma_distance_us'] + agg_rdma_timing['rdma_fetch_us'])

                    # Convergence exposure: the single-query dict carries the
                    # per-query values under the same names the batch dict uses
                    # for its sums, so they accumulate directly.
                    for _ck in ('sum_convergence_hop', 'n_converged_queries',
                                'sum_batch_rounds'):
                        agg_stats[_ck] = agg_stats.get(_ck, 0) + int(
                            search_stats.get(_ck, 0))

                    # Collect per-query E2E latency
                    per_query_total_us.append(float(search_stats.get('total_us', 0.0)))

                    global_ids = []
                    for gid in indices:
                        gid = int(gid)
                        global_ids.append(gid if gid != 0xFFFFFFFF else -1)

                    try:
                        dist_list = [float(d) for d in distances]
                    except TypeError:
                        dist_list = [float('inf')] * len(global_ids)
                    results.append((qid, global_ids, dist_list))
                except Exception as e:
                    ctx.log(f"[WARN] Search query {qid} L={L} failed: {e}")

        trace_config.stop_l(ctx)

        # Batch path stops the clock right after the C++ call returns; the
        # sequential path has no separate marshalling stage, so it falls back
        # to the end-of-block timestamp.
        elapsed = _search_elapsed if _search_elapsed is not None else time.time() - t0
        # Capture burst window end BEFORE _rdma_snapshot subprocess
        # (~100-300ms) to keep the window tight around real search traffic.
        burst_session.mark_l_end(L, _burst_l_start)
        _nic_snap_after = rdma_snapshot()
        results_by_L[L] = results

        # Log aggregate pure-colocation search stats per L value
        total_cmps = agg_stats['n_cmps'] or 1
        local_vec = agg_stats['local_vector_accesses']
        remote_vec = agg_stats['remote_vector_accesses']
        total_vec = local_vec + remote_vec or 1
        pq_filt = agg_stats.get('pq_filtered', 0)
        pq_hits = agg_stats.get('pq_cache_hits', 0)
        pq_misses = agg_stats.get('pq_cache_misses', 0)
        pq_evict = agg_stats.get('pq_cache_evictions', 0)
        pq_csize = agg_stats.get('pq_cache_size', 0)
        pq_info = f" pq_filtered={pq_filt}" if pq_filt > 0 else ""
        if pq_hits + pq_misses > 0:
            pq_hr = pq_hits * 100.0 / (pq_hits + pq_misses)
            pq_info += f" pq_cache(hit={pq_hr:.1f}% evict={pq_evict} size={pq_csize})"
        fsc_hits = agg_stats.get('frozen_seed_cache_hits', 0)
        fsc_info = (f" frozen_seed_cache(hits={fsc_hits} "
                    f"entries={agg_stats.get('frozen_seed_cache_entries', 0)})"
                    if fsc_hits > 0 else "")
        n_conv = agg_stats.get('n_converged_queries', 0)
        conv_info = (f" conv(n={n_conv} "
                     f"mean_hop={agg_stats.get('sum_convergence_hop', 0) / n_conv:.1f})"
                     if n_conv > 0 else "")
        unresolved = agg_stats.get('unresolved_vids', 0)
        unresolved_info = f" unresolved_vids={unresolved}" if unresolved > 0 else ""
        ctx.log(f"  L={L}: {len(results)} queries in {elapsed:.2f}s | "
                f"hops={agg_stats['n_hops']} cmps={agg_stats['n_cmps']} "
                f"local_vec={local_vec} remote_vec={remote_vec} "
                f"resolve_fail={agg_stats['resolve_fail']}{pq_info}{fsc_info}"
                f"{conv_info}{unresolved_info} | "
                f"local={local_vec*100.0/total_vec:.1f}% "
                f"remote={remote_vec*100.0/total_vec:.1f}%")
        if unresolved > 0:
            ctx.log(f"  [WARN] {unresolved} search results had unresolvable VIDs (stale local_to_tag)")

        # Reflect search RDMA stats into Python-side counters for report
        local_nbr = agg_stats['local_neighbor_accesses']
        remote_nbr = agg_stats['remote_neighbor_accesses']
        ctx.rdma_counters.vector_access_local += local_vec
        ctx.rdma_counters.vector_access_remote += remote_vec
        ctx.rdma_counters.neighbor_access_local += local_nbr
        ctx.rdma_counters.neighbor_access_remote += remote_nbr
        ctx.rdma_counters.fetch_entry_calls += agg_stats['rdma_entry_calls']
        ctx.rdma_counters.fetch_entry_items += agg_stats['rdma_entry_items']
        ctx.rdma_counters.search_colocation_calls += len(results)

        # Capture per-L stats
        sum_total_us = sum(per_query_total_us) if per_query_total_us else 0.0
        rdma_total = agg_rdma_timing['rdma_total_us']
        _per_L_stats[L] = {
            'num_queries': len(results),
            'elapsed_s': round(elapsed, 3),
            'n_hops': agg_stats['n_hops'],
            'n_cmps': agg_stats['n_cmps'],
            'local_vector_accesses': local_vec,
            'remote_vector_accesses': remote_vec,
            'local_neighbor_accesses': local_nbr,
            'remote_neighbor_accesses': remote_nbr,
            'fetch_entry_calls': agg_stats['rdma_entry_calls'],
            'fetch_entry_items': agg_stats['rdma_entry_items'],
            'resolve_fail': agg_stats['resolve_fail'],
            'pq_filtered': agg_stats.get('pq_filtered', 0),
            'pq_cache_hits': agg_stats.get('pq_cache_hits', 0),
            'pq_cache_misses': agg_stats.get('pq_cache_misses', 0),
            # FrozenSeedCache: seeds served from the local copy. These are
            # exactly the reads remote_vector_accesses / fetch_entry_calls no
            # longer contain, so (baseline remote - this remote) == this.
            'frozen_seed_cache_hits': agg_stats.get('frozen_seed_cache_hits', 0),
            'frozen_seed_cache_entries': agg_stats.get('frozen_seed_cache_entries', 0),
            # Convergence: mean converged hop = sum_convergence_hop /
            # n_converged_queries (NOT / num_queries -- queries that never
            # converged contribute 0 to the sum).
            'sum_convergence_hop': agg_stats.get('sum_convergence_hop', 0),
            'n_converged_queries': agg_stats.get('n_converged_queries', 0),
            'sum_batch_rounds': agg_stats.get('sum_batch_rounds', 0),
            'rdma_distance_us': round(agg_rdma_timing['rdma_distance_us'], 1),
            'rdma_fetch_us': round(agg_rdma_timing['rdma_fetch_us'], 1),
            'rdma_total_us': round(rdma_total, 1),
            'sum_query_total_us': round(sum_total_us, 1),
            'cpu_total_us': round(max(sum_total_us - rdma_total, 0), 1),
            'qps': round(len(results) / elapsed, 1) if elapsed > 0 else 0,
        }
        # Log CPU breakdown (batch path only)
        cpu_bd = agg_stats.get('cpu_breakdown')
        if cpu_bd and sum_total_us > 0:
            vis = cpu_bd.get('total_cpu_visited_us', 0)
            res = cpu_bd.get('total_cpu_resolve_us', 0)
            dist_us = cpu_bd.get('total_cpu_distance_us', 0)
            que = cpu_bd.get('total_cpu_queue_us', 0)
            map_us = cpu_bd.get('total_cpu_map_us', 0)
            pq_us = cpu_bd.get('total_cpu_pq_us', 0)
            lut_us = cpu_bd.get('total_lut_build_us', 0)
            # Batch (CAP) CQ-drain spin is real RDMA wait: it is measured as
            # batch_*_poll but excluded from rdma_vector/neighbor in C++, so
            # fold it into the RDMA share here (previously leaked into other).
            batch_poll_us = (cpu_bd.get('total_batch_nbr_poll_us', 0)
                             + cpu_bd.get('total_batch_vec_poll_us', 0))
            rdma_t = rdma_total + batch_poll_us
            nbr_raw_us = cpu_bd.get('total_cpu_nbr_loop_us', 0)
            # nbr_loop raw time wraps visited+resolve (beam_search.cpp:611-641).
            # Pure nbr_loop = push_back + partition decode + control flow only.
            nbr_us = max(nbr_raw_us - vis - res, 0)
            # "other" = truly untimed (result extraction, trace, reductions).
            other = max(sum_total_us - vis - res - dist_us - que - map_us
                        - nbr_us - pq_us - lut_us - rdma_t, 0)
            fetch_phase_us = cpu_bd.get('total_candidate_fetch_phase_us', 0)
            pass1_us = cpu_bd.get('total_pass1_us', 0)
            pass2_us = cpu_bd.get('total_pass2_us', 0)
            pass3_us = cpu_bd.get('total_pass3_us', 0)
            pq_str = f"pq={pq_us/sum_total_us*100:.1f}%  " if pq_us > 0 else ""
            if lut_us > 0:
                pq_str += f"lut={lut_us/sum_total_us*100:.1f}%  "
            pass_str = ""
            if fetch_phase_us > 0:
                pass_str = (f"fetch_phase={fetch_phase_us/sum_total_us*100:.1f}% "
                            f"(P1={pass1_us/sum_total_us*100:.1f}% "
                            f"P2={pass2_us/sum_total_us*100:.1f}% "
                            f"P3={pass3_us/sum_total_us*100:.1f}%)  ")
            ctx.log(f"  [CPU breakdown] visited={vis/sum_total_us*100:.1f}%  "
                    f"resolve={res/sum_total_us*100:.1f}%  "
                    f"distance={dist_us/sum_total_us*100:.1f}%  "
                    f"queue={que/sum_total_us*100:.1f}%  "
                    f"map={map_us/sum_total_us*100:.1f}%  "
                    f"nbr_loop={nbr_us/sum_total_us*100:.1f}%  "
                    f"{pq_str}"
                    f"RDMA={rdma_t/sum_total_us*100:.1f}%  "
                    f"{pass_str}"
                    f"other={other/sum_total_us*100:.1f}%")
        if cpu_bd:
            _per_L_stats[L]['cpu_breakdown'] = {
                'visited_us': round(cpu_bd.get('total_cpu_visited_us', 0), 1),
                'resolve_us': round(cpu_bd.get('total_cpu_resolve_us', 0), 1),
                'distance_us': round(cpu_bd.get('total_cpu_distance_us', 0), 1),
                'queue_us': round(cpu_bd.get('total_cpu_queue_us', 0), 1),
                'nbr_loop_us': round(cpu_bd.get('total_cpu_nbr_loop_us', 0), 1),
                'map_us': round(cpu_bd.get('total_cpu_map_us', 0), 1),
                'pq_us': round(cpu_bd.get('total_cpu_pq_us', 0), 1),
                'lut_build_us': round(cpu_bd.get('total_lut_build_us', 0), 1),
                'candidate_fetch_phase_us': round(cpu_bd.get('total_candidate_fetch_phase_us', 0), 1),
                'pass1_us': round(cpu_bd.get('total_pass1_us', 0), 1),
                'pass2_us': round(cpu_bd.get('total_pass2_us', 0), 1),
                'pass3_us': round(cpu_bd.get('total_pass3_us', 0), 1),
                'pass3_poll_us': round(cpu_bd.get('total_pass3_poll_us', 0), 1),
                'pass3_compute_us': round(cpu_bd.get('total_pass3_compute_us', 0), 1),
                'batch_nbr_poll_us': round(cpu_bd.get('total_batch_nbr_poll_us', 0), 1),
                'batch_vec_poll_us': round(cpu_bd.get('total_batch_vec_poll_us', 0), 1),
            }
        # Capture per-L latency (both single-thread and batch)
        if per_query_total_us:
            lat_arr = np.array(per_query_total_us)
            _per_L_latency[L] = {
                'mean_us': round(float(np.mean(lat_arr)), 1),
                'p50_us': round(float(np.percentile(lat_arr, 50)), 1),
                'p95_us': round(float(np.percentile(lat_arr, 95)), 1),
                'p99_us': round(float(np.percentile(lat_arr, 99)), 1),
                'min_us': round(float(np.min(lat_arr)), 1),
                'max_us': round(float(np.max(lat_arr)), 1),
                # Raw per-query latencies needed to compute true SYSTEM
                # percentiles by merging across nodes downstream
                # (10_query_latency_p50-p99/make_csv.py).
                # Size: ~25 L × ~2500 q × 8B ≈ 500 KB/op — negligible.
                'per_query_us': [round(float(x), 1) for x in lat_arr],
            }

        merge_rdmabw_stats(_per_L_stats, L, _nic_snap_before, _nic_snap_after, elapsed)
        pmu_stop_and_merge(_per_L_stats, L, _pmu_path)

        last_agg_stats = dict(agg_stats)

        # Barrier between L values to prevent cross-L RDMA interference
        if search_use_multi_qp:
            ctx.sync.barrier(f"search-L{L}-done")

    burst_session.stop_and_merge(ctx, _per_L_stats)

    return results_by_L, _per_L_stats, _per_L_latency, last_agg_stats
