#!/usr/bin/env python3
"""
compute_recall_posthoc.py - Post-hoc recall computation for distributed engine runs.

Computes recall@K for each search op in a completed distributed engine run by:
1. Replaying the workload config to determine the active set at each search op
2. Computing (or loading cached) brute-force GT for each active range
3. Merging per-node search results and computing recall@K

GT is cached at the dataset level so it can be reused across experiments with
different graph builds but the same workload/dataset.

Usage:
    python compute_recall_posthoc.py <run_dir>
    python compute_recall_posthoc.py <run_dir> --gt-cache-dir /path/to/gt_cache
    python compute_recall_posthoc.py <run_dir> --force-recompute-gt
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Path setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
_scripts_root = os.path.join(_project_root, "scripts")
for _p in [_project_root, _scripts_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.generate_gt import compute_or_load_gt


# =============================================================================
# Config loading
# =============================================================================

def load_run_config(run_dir: Path) -> dict:
    """Load the config yaml from a run directory."""
    import yaml
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml config found in {run_dir}")
    with open(yaml_files[0]) as f:
        return yaml.safe_load(f)


def resolve_data_path(config: dict, run_dir: Path) -> Tuple[str, str]:
    """Resolve base_file and query_file paths from config.

    Config paths are relative to project root and assume container-internal
    mounts (e.g. datasets_nfs/). Run this script inside the container.
    """
    from pipeline.config import resolve_config_paths
    project_root = Path(_project_root)
    resolve_config_paths(config, project_root)

    base_file = config['dataset']['base_file']
    query_file = config['dataset']['query_file']

    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file not found: {base_file}")
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")

    return base_file, query_file


def query_pool_size(query_file: str) -> Optional[int]:
    """Number of query vectors in the file, from its 8-byte header."""
    try:
        with open(query_file, 'rb') as f:
            npts = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        return npts if npts > 0 else None
    except Exception:
        return None


# =============================================================================
# Workload replay
# =============================================================================

def replay_workload(config: dict) -> List[Tuple[int, int, int, dict]]:
    """
    Replay workload to determine active range at each search op.

    Returns:
        List of (seq, active_start, active_end, search_params) for each search op.
        seq is the overall operation index (not search-only index).

    SLIDING-WINDOW ASSUMPTION (limitation):
        This function models the active set as a contiguous gid range
        [start, end) and slides it as inserts/deletes occur:
          - insert  → end += count   (new vids appended)
          - delete  → start += count (lowest gids retired)
        Initial state: [0, build.initial_size).

        This holds ONLY when:
          (a) Build's update phase is disabled OR uses the same sliding
              window discipline (no random/policy-based deletes during build).
          (b) Distributed-engine deletes target the lowest-gid block first
              (vs. e.g. random/policy/oldest-by-tag deletes).
          (c) Inserts append vids in monotonic gid order.

        It BREAKS when:
          1. Build ran update iterations that drift active set
             (sanity_deep100K_consol with 50 update iters → active set
             becomes ~[50K, 150K] after iter50, but this function still
             starts at [0, 100K] → recall against wrong gid range,
             observed 0.43 vs expected 0.95+).
          2. Random deletion: the deleted gid set is scattered, not the
             lowest contiguous block. Then [start, end) overlaps the actual
             active set only partially.
          3. Re-using a colocation built from a checkpoint past iteration0.

        For correctness across these cases, replace this with
        active-set discovery from the colocation tag files
        (local_to_tag_p*.bin) at run time, or by reading op_history's
        recorded gid-set deltas.
    """
    initial_size = config['build']['initial_size']
    workload = config['distributed_engine']['workload']

    start, end = 0, initial_size
    results = []

    for seq, op_def in enumerate(workload):
        op = op_def['op']
        if op == 'insert':
            end += op_def['count']
        elif op == 'delete':
            start += op_def['count']
        elif op == 'search':
            results.append((seq, start, end, op_def))

    return results


# =============================================================================
# Search results merging
# =============================================================================

def load_search_results(run_dir: Path, seq: int) -> Optional[Dict]:
    """
    Load and merge per-node search results for a given search op.

    Tries new format (search_results_seq{seq}_p*.npz) first,
    falls back to legacy single file (search_results.npz).
    """
    # New format: search_results/seq{seq}_p*.npz
    results_dir = run_dir / "search_results"
    if results_dir.is_dir():
        files = sorted(results_dir.glob(f"seq{seq}_p*.npz"))
    else:
        files = []

    if not files:
        # Legacy: single file in run_dir
        legacy = run_dir / "search_results.npz"
        if legacy.exists():
            return _load_single_npz(legacy)
        return None

    # Merge per-node results.
    # Shared-nothing: each qid appears in exactly one node's npz → concat + sort.
    # Broadcast (scatter-gather): each qid appears in N nodes' npz → union the
    # (gid, dist) candidates per qid, sort by dist, keep top-K.
    # Legacy npz without dists falls back to plain concat (broadcast unsupported).
    merged = {}
    K = None
    L_values = None

    for f in files:
        data = np.load(str(f))
        if K is None:
            K = int(data['K'][0])
            L_values = data['L_values']
        for L in L_values:
            qids_key = f'qids_L{L}'
            gids_key = f'gids_L{L}'
            dists_key = f'dists_L{L}'
            if qids_key not in data:
                continue
            qids = data[qids_key]
            gids = data[gids_key]
            dists = data[dists_key] if dists_key in data else None
            if L not in merged:
                merged[L] = {'qids': [], 'gids': [], 'dists': []}
            merged[L]['qids'].append(qids)
            merged[L]['gids'].append(gids)
            if dists is not None:
                merged[L]['dists'].append(dists)

    if not merged:
        return None

    result = {'K': K, 'L_values': L_values}
    for L in merged:
        all_qids = np.concatenate(merged[L]['qids'])
        all_gids = np.concatenate(merged[L]['gids'], axis=0)
        has_dists = (len(merged[L]['dists']) == len(merged[L]['qids']))
        all_dists = (np.concatenate(merged[L]['dists'], axis=0)
                     if has_dists else None)

        unique_qids, qid_counts = np.unique(all_qids, return_counts=True)
        is_broadcast = bool(qid_counts.max() > 1) if qid_counts.size > 0 else False

        if is_broadcast and has_dists:
            # Top-K by distance across union of per-node candidates.
            order_qid = np.argsort(all_qids, kind='stable')
            sorted_qids = all_qids[order_qid]
            sorted_gids = all_gids[order_qid]
            sorted_dists = all_dists[order_qid]
            best_K = int(K)
            final_gids = np.full((len(unique_qids), best_K), -1, dtype=np.int64)
            start = 0
            for i, qid in enumerate(unique_qids):
                cnt = int(qid_counts[i])
                end = start + cnt
                seg_gids = sorted_gids[start:end].reshape(-1)
                seg_dists = sorted_dists[start:end].reshape(-1)
                mask = (seg_gids >= 0) & np.isfinite(seg_dists)
                seg_gids = seg_gids[mask]
                seg_dists = seg_dists[mask]
                start = end
                if seg_gids.size == 0:
                    continue
                ord_d = np.argsort(seg_dists, kind='stable')
                seg_gids = seg_gids[ord_d]
                seen = set()
                kept = []
                for g in seg_gids.tolist():
                    if g in seen:
                        continue
                    seen.add(g)
                    kept.append(g)
                    if len(kept) >= best_K:
                        break
                final_gids[i, :len(kept)] = kept
            result[L] = {'qids': unique_qids, 'gids': final_gids}
        else:
            order = np.argsort(all_qids, kind='stable')
            result[L] = {'qids': all_qids[order], 'gids': all_gids[order]}

    return result


def _load_single_npz(path: Path) -> Optional[Dict]:
    """Load a single search_results.npz (legacy or merged)."""
    data = np.load(str(path))
    K = int(data['K'][0])
    L_values = data['L_values']
    result = {'K': K, 'L_values': L_values}
    for L in L_values:
        qids_key = f'qids_L{L}'
        gids_key = f'gids_L{L}'
        if qids_key in data:
            result[int(L)] = {'qids': data[qids_key], 'gids': data[gids_key]}
    return result


# =============================================================================
# Recall computation
# =============================================================================

def _gt_query_count(params: dict, pool_size: Optional[int]) -> int:
    """How many queries GT is needed for.

    search.num_queries above the pool size means the search op tiled the pool
    to lengthen the measurement window (op.py). Tiled repeats get fresh qids,
    and compute_recall() skips every qid past the end of GT, so recall is still
    computed over exactly one pass of the pool -- but GT itself must stay at
    pool size. Without this clamp a 5x-looped run asks for GT over 50k queries
    that nothing will ever read, which means a multi-hour brute-force scan.
    """
    n = params.get('num_queries', 1000)
    if pool_size and n > pool_size:
        return pool_size
    return n


def compute_recall(merged: Dict, gt_gids: np.ndarray, K: int) -> Dict[int, float]:
    """
    Compute recall@K for each L value.

    Args:
        merged: {L: {'qids': (Q,), 'gids': (Q, max_k)}, ...}
        gt_gids: (num_queries, K) int64 — GT neighbors for queries[0:num_queries]
        K: number of neighbors for recall

    Returns:
        {L: recall@K}
    """
    recall_by_L = {}

    for L in sorted(k for k in merged if isinstance(k, (int, np.integer))):
        data = merged[L]
        qids = data['qids']
        result_gids = data['gids']

        hits = 0
        total = 0
        for i in range(len(qids)):
            qid = int(qids[i])
            if qid >= len(gt_gids):
                continue
            gt_set = set(gt_gids[qid, :K].tolist())
            result_set = set(g for g in result_gids[i, :K].tolist() if g >= 0)
            hits += len(result_set & gt_set)
            total += 1

        recall = hits / (total * K) if total > 0 else 0.0
        recall_by_L[int(L)] = round(recall, 4)

    return recall_by_L


# =============================================================================
# Main pipeline
# =============================================================================

def compute_recall_for_run(
    run_dir: str,
    gt_cache_dir: Optional[str] = None,
    force_recompute: bool = False,
    lowest_t_only: bool = False,
) -> dict:
    """
    Compute post-hoc recall for all search ops in a run.

    Returns dict suitable for JSON serialization.
    """
    run_path = Path(run_dir)
    config = load_run_config(run_path)
    base_file, query_file = resolve_data_path(config, run_path)
    data_type = config['dataset'].get('data_type', 'float32')
    if data_type.startswith('float'):
        data_type = 'float'

    # GT cache at experiment level (sibling of run dirs) for reuse across runs.
    # e.g. experiments/deep100K_4node/distributed_engine/gt_cache/
    if gt_cache_dir is None:
        gt_cache_dir = str(run_path.parent / "gt_cache")

    print(f"Run dir:      {run_path}")
    print(f"Base file:    {base_file}")
    print(f"Query file:   {query_file}")
    pool_size = query_pool_size(query_file)
    if pool_size:
        print(f"Query pool:   {pool_size} (GT is capped at this; looped repeats reuse it)")
    print(f"GT cache dir: {gt_cache_dir}")

    # Replay workload to get active ranges per search op
    search_ops = replay_workload(config)
    if lowest_t_only and search_ops:
        # Keep only the first op at the minimum num_threads — recall is
        # deterministic across T, so measuring once is sufficient.
        min_T = min(p.get('num_threads', 1) for _, _, _, p in search_ops)
        keep_seq = next((s for s, _, _, p in search_ops
                         if p.get('num_threads', 1) == min_T), None)
        search_ops = [op for op in search_ops if op[0] == keep_seq]
        print(f"\n[lowest_t_only] keeping seq={keep_seq} (T={min_T})")
    print(f"\nFound {len(search_ops)} search ops to score:")
    for seq, start, end, params in search_ops:
        print(f"  seq={seq}: active=[{start}, {end}) ({end - start} points), "
              f"K={params.get('K', 10)}, L={params.get('L_values', [])}")

    if force_recompute:
        # Delete cached GT files that match our ranges
        for seq, start, end, params in search_ops:
            K = params.get('K', 10)
            num_queries = _gt_query_count(params, pool_size)
            cache_name = f"gt_K{K}_Q{num_queries}_R{start}-{end}"
            for ext in ['.npy', '_meta.json']:
                cached = Path(gt_cache_dir) / f"{cache_name}{ext}"
                if cached.exists():
                    cached.unlink()
                    print(f"  Removed cached: {cached.name}")

    # Process each search op
    results = []
    # GT cache within a run — same active_range across alphas/batches/threads
    gt_in_run = {}
    for seq, start, end, params in search_ops:
        K = params.get('K', 10)
        num_queries = _gt_query_count(params, pool_size)
        L_values = params.get('L_values', [100])
        num_threads = params.get('num_threads')
        pq_alpha = params.get('pq_alpha')
        batch_size = params.get('batch_size')

        print(f"\n{'='*60}")
        print(f"Search op seq={seq}: active=[{start}, {end}), K={K}, Q={num_queries}, "
              f"T={num_threads}, alpha={pq_alpha}, batch={batch_size}")
        print(f"{'='*60}")

        # 1. Load search results
        merged = load_search_results(run_path, seq)
        if merged is None:
            print(f"  [WARN] No search results found for seq={seq}, skipping")
            results.append({
                'seq': seq,
                'active_range': [start, end],
                'num_active': end - start,
                'K': K,
                'num_queries': num_queries,
                'num_threads': num_threads,
                'pq_alpha': pq_alpha,
                'batch_size': batch_size,
                'status': 'no_results',
            })
            continue

        actual_K = merged.get('K', K)
        actual_queries = max(
            len(merged[L]['qids'])
            for L in merged if isinstance(L, (int, np.integer))
        )
        print(f"  Loaded results: {actual_queries} queries, K={actual_K}, "
              f"L_values={sorted(L for L in merged if isinstance(L, (int, np.integer)))}")

        # 2. Load GT keyed by the active window [start, end). compute_or_load_gt
        # caches per-range (gt_K{K}_Q{Q}_R{start}-{end}.npy), so each search op
        # is scored against GT over its OWN active set.
        #
        # REMOVED (bug): a `if graph_variant:` shortcut here used to load the
        # build-pipeline GT file (gt_{graph_variant}_K{K}_Q{Q}.npy) whenever
        # graph_variant was set. That filename keys on (variant, K, Q) but NOT
        # the active range, so it returned the BUILD-time GT (initial_size
        # active set) for every search op — correct for the pre-insert search,
        # but stale for any search after insert/delete changed the active set.
        # Symptom: post-insert recall under-reported by ~inserted/total (e.g.
        # 10k/110k ≈ 9% at deep1M). The shortcut was a pure speed cache and
        # bought nothing here — for the pre-insert range it produces byte-
        # identical results to the range-keyed path (verified 0.9170/0.9933/
        # 0.9980 both ways), and compute_or_load_gt caches per-range anyway.
        # [2026-07-26] Prebuilt-variant SEARCH-ONLY runs: replay cannot know a
        # checkpoint's active window (no insert/delete ops in the workload), so
        # the range here is the INITIAL window [0, initial_size) — the WRONG
        # active set for iterationN-style variants (actual window is shifted by
        # N iterations; e.g. iteration150 -> [150M, 250M)). Scoring against
        # that GT yields recall≈0. For this case the build-pipeline GT
        # (gt_{graph_variant}_K{K}_Q{Q}.npy) is exactly the right GT: use it.
        # Update-runs (any insert/delete op present) keep the range-keyed path,
        # which is what the earlier shortcut-removal correctly protects.
        _wl = (config.get('distributed_engine') or {}).get('workload') or []
        _search_only = bool(_wl) and all(op.get('op') == 'search' for op in _wl)
        _variant = (config.get('build') or {}).get('graph_variant')
        variant_gt_path = None
        if _search_only and _variant:
            _art = (config.get('artifacts') or {}).get('base_dir', '/experiments/artifacts')
            _ds = (config.get('dataset') or {}).get('id')
            _cand = Path(_art) / str(_ds) / 'gt' / f'gt_{_variant}_K{K}_Q{num_queries}.npy'
            if _cand.exists():
                variant_gt_path = _cand

        cache_key = (start, end, K, num_queries)
        if variant_gt_path is not None:
            gt_gids = np.load(variant_gt_path)[:num_queries, :K].astype(np.int64)
            gt_time = 0.0
            gt_cache_hit = True
            print(f"  [GT] variant shortcut (search-only run): {variant_gt_path}")
        elif cache_key in gt_in_run:
            gt_gids = gt_in_run[cache_key]
            gt_time = 0.0
            gt_cache_hit = True
        else:
            t0 = time.time()
            gt_gids = compute_or_load_gt(
                base_file=base_file,
                query_file=query_file,
                active_start=start,
                active_end=end,
                K=K,
                num_queries=num_queries,
                gt_cache_dir=gt_cache_dir,
                data_type=data_type,
            )
            gt_time = time.time() - t0
            gt_cache_hit = gt_time < 1.0
            gt_in_run[cache_key] = gt_gids

        # 3. Compute recall
        recall_by_L = compute_recall(merged, gt_gids, K)

        print(f"  Recall@{K}:")
        for L, recall in sorted(recall_by_L.items()):
            print(f"    L={L}: {recall:.4f}")

        results.append({
            'seq': seq,
            'active_range': [start, end],
            'num_active': end - start,
            'K': K,
            'num_queries': actual_queries,
            'num_threads': num_threads,
            'pq_alpha': pq_alpha,
            'batch_size': batch_size,
            'recall_by_L': {str(L): r for L, r in recall_by_L.items()},
            'gt_cache_hit': gt_cache_hit,
            'gt_time_s': round(gt_time, 2),
        })

    return {
        'run_dir': str(run_path),
        'experiment_name': config.get('experiment_name', ''),
        'dataset_base': os.path.basename(base_file),
        'gt_cache_dir': gt_cache_dir,
        'search_ops': results,
        'timestamp': datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc recall computation for distributed engine runs")
    parser.add_argument("run_dir",
                        help="Experiment run directory")
    parser.add_argument("--gt-cache-dir", default=None,
                        help="GT cache directory (default: next to base_file)")
    parser.add_argument("--force-recompute-gt", action="store_true",
                        help="Recompute GT even if cached")
    parser.add_argument("--lowest-t-only", action="store_true",
                        help="Only measure recall on the lowest num_threads op "
                             "(deterministic across T — saves time)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: run_dir/recall_posthoc.json)")
    args = parser.parse_args()

    result = compute_recall_for_run(
        run_dir=args.run_dir,
        gt_cache_dir=args.gt_cache_dir,
        force_recompute=args.force_recompute_gt,
        lowest_t_only=args.lowest_t_only,
    )

    # Save results
    output_path = args.output or os.path.join(args.run_dir, "recall_posthoc.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("RECALL SUMMARY")
    print(f"{'='*60}")
    for op in result['search_ops']:
        if 'recall_by_L' in op:
            recalls = ", ".join(
                f"L={L}: {r:.4f}" for L, r in sorted(op['recall_by_L'].items()))
            print(f"  seq={op['seq']} (N={op['num_active']}): {recalls}")
        else:
            print(f"  seq={op['seq']}: {op.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
