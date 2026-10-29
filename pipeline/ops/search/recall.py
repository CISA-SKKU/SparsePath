"""In-run search recall helpers."""

from __future__ import annotations

import hashlib
import json as _json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def compute_recall(ctx, all_results_by_L: Dict[int, list], active_locs: np.ndarray,
                   K: int, op_seq: int) -> Optional[dict]:
    """Compute recall@K for each L value, returning stats for OperationStats."""
    if len(active_locs) == 0:
        ctx.log("[WARN] SearchOp: no active points, skipping recall")
        return None

    largest_L = max(all_results_by_L.keys())
    all_results_sorted = sorted(all_results_by_L[largest_L], key=lambda x: x[0])
    if len(all_results_sorted) == 0:
        return None

    num_queries = len(all_results_sorted)
    gt_locs = load_precomputed_gt(ctx, active_locs, K, num_queries, op_seq)
    t_gt = 0.0

    if gt_locs is None:
        if ctx.vectors is None:
            ctx.log("[WARN] SearchOp: no precomputed GT and vectors not loaded "
                    "(colocation-only mode), skipping recall")
            return None
        try:
            from generate_gt import compute_gt_python
        except ImportError:
            ctx.log("[WARN] SearchOp: generate_gt not available, skipping recall")
            return None

        query_vectors = np.array(
            [ctx.queries[qid].astype(np.float32) for qid, _ in all_results_sorted],
            dtype=np.float32)

        ctx.log(f"SearchOp: computing GT online ({num_queries} queries × "
                f"{len(active_locs)} active)...")
        t0 = time.time()
        gt_locs = compute_gt_python(ctx.vectors, query_vectors, active_locs, K)
        t_gt = time.time() - t0
        ctx.log(f"SearchOp: GT computed in {t_gt:.1f}s")

    gt_sets: Dict[int, set] = {}
    for idx, r in enumerate(all_results_sorted):
        qid = r[0]
        gt_row = gt_locs[idx]
        gt_global_ids = set(int(gt_row[j]) for j in range(min(K, gt_locs.shape[1])))
        gt_sets[qid] = gt_global_ids

    recall_by_L = {}
    for L in sorted(all_results_by_L.keys()):
        results_sorted = sorted(all_results_by_L[L], key=lambda x: x[0])
        recall_hits = 0
        total = 0
        for r in results_sorted:
            qid = r[0]
            result_gids = r[1]
            if qid in gt_sets:
                result_set = set(gid for gid in result_gids[:K] if gid >= 0)
                recall_hits += len(gt_sets[qid] & result_set)
                total += K

        recall_at_K = recall_hits / total if total > 0 else 0.0
        recall_by_L[L] = round(recall_at_K, 4)
        ctx.log(f"  recall@{K} L={L}: {recall_at_K:.4f}")

    stats = {
        'recall_by_L': recall_by_L,
        'K': K,
        'num_queries': num_queries,
        'num_active_points': len(active_locs),
        'gt_time_s': round(t_gt, 3),
    }
    ctx.log(f"SearchOp recall summary: {recall_by_L} "
            f"({num_queries} queries, "
            f"{len(active_locs)} active points, "
            f"GT {'precomputed' if t_gt == 0 else f'online {t_gt:.1f}s'})")
    return stats


def load_precomputed_gt(ctx, active_locs: np.ndarray, K: int, num_queries: int,
                        op_seq: int) -> Optional[np.ndarray]:
    """Try to load precomputed GT from gt_cache/."""
    output_cfg = ctx.config.get('output', {})
    experiment_name = ctx.config.get('experiment_name', 'unnamed')
    base_dir = Path(output_cfg.get('base_dir', '.'))
    gt_cache_dir = base_dir / experiment_name / "gt_cache"

    if not gt_cache_dir.exists():
        ctx.log("[INFO] No gt_cache/ dir, will compute online")
        return None

    gt_path = None
    h = hashlib.sha256(active_locs.tobytes()).hexdigest()[:8]

    graph_variant = ctx.config.get('build', {}).get('graph_variant', None)
    if graph_variant:
        cp = None
        if "cp4" in graph_variant:
            cp = "cp4"
        elif "cp0" in graph_variant:
            cp = "cp0"
        if cp:
            ds_name = experiment_name.replace('_micro26', '')
            candidate = gt_cache_dir / f"gt_{ds_name}_{cp}.npy"
            if candidate.exists():
                gt_path = candidate
                ctx.log(f"SearchOp: matched checkpoint GT for {graph_variant}: {candidate.name}")

    if gt_path is None:
        manifest_names = []
        if graph_variant:
            manifest_names.append(f"search_gt_manifest_{graph_variant}.json")
        manifest_names.append("search_gt_manifest.json")

        for manifest_name in manifest_names:
            manifest_path = gt_cache_dir / manifest_name
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = _json.load(f)
                seq_key = str(op_seq)
                if seq_key in manifest:
                    candidate = gt_cache_dir / manifest[seq_key]
                    if candidate.exists():
                        gt_path = candidate
                        break

    if gt_path is None:
        candidate = gt_cache_dir / f"gt_K{K}_Q{num_queries}_N{len(active_locs)}_h{h}.npy"
        if candidate.exists():
            gt_path = candidate

    if gt_path is None:
        active_min, active_max = int(active_locs.min()), int(active_locs.max())
        for npy_file in sorted(gt_cache_dir.glob("gt_*.npy")):
            try:
                arr = np.load(str(npy_file))
                arr_min, arr_max = int(arr.min()), int(arr.max())
                if arr_min >= active_min - 1 and arr_max <= active_max + 1:
                    gt_path = npy_file
                    break
            except Exception:
                continue

    if gt_path is None:
        ctx.log(f"[INFO] No precomputed GT found (hash={h}, "
                f"active=[{int(active_locs.min())}..{int(active_locs.max())}])")
        return None

    gt_locs = np.load(str(gt_path))
    ctx.log(f"SearchOp: loaded precomputed GT from {gt_path.name} "
            f"(shape={gt_locs.shape})")

    if gt_locs.shape[0] < num_queries:
        ctx.log(f"[WARN] GT has {gt_locs.shape[0]} queries but need {num_queries}")
        return None

    return gt_locs[:num_queries]
