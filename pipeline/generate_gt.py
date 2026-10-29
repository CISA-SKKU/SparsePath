#!/usr/bin/env python3
"""
generate_gt.py - Ground Truth Generation for Graph Checkpoints

Generates brute-force k-NN ground truth for a given graph checkpoint.
Uses C++ compute_groundtruth_with_tags binary (MKL-accelerated) if available,
otherwise falls back to vectorized NumPy computation.

GT is saved alongside the graph checkpoint in a gt_cache/ directory.

Usage:
    # For a specific checkpoint (uses tags to determine active points):
    python generate_gt.py \
        --data /path/to/vectors.fbin \
        --tags /path/to/checkpoint.tags \
        --queries /path/to/queries.fbin \
        --K 10 --num-queries 500

    # With partition directory (auto-resolves data):
    python generate_gt.py \
        --partition-dir /path/to/4_partitions/ \
        --tags /path/to/checkpoint.tags \
        --K 10 --num-queries 500

    # Custom active set (for ablation experiments):
    python generate_gt.py \
        --data /path/to/vectors.fbin \
        --active-locs active_locations.npy \
        --K 10 --num-queries 500

Output goes to gt_cache/ next to the tags file, or --output-dir if specified.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Path setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
_scripts_root = os.path.join(_project_root, "scripts")
for _p in [_project_root, _scripts_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_loader import load_bin_data, load_tags


# =============================================================================
# GT Binary Interface
# =============================================================================

def find_gt_binary() -> Optional[str]:
    """Find compute_groundtruth_with_tags binary."""
    search_paths = [
        # DANNS project build (inside container)
        "/workspace/build/third_party/diskann/apps/utils/compute_groundtruth_with_tags",
        # Legacy locations (host)
        # Older Docker workspace paths
        "/workspace/diskann/build/apps/utils/compute_groundtruth_with_tags",
        "/workspace/diskann/build/apps/compute_groundtruth_with_tags",
    ]
    for path in search_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def save_tags_file(tags: np.ndarray, output_file: str):
    """Save tags in DiskANN binary format: [npts (uint32), dim=1 (uint32), tags...]"""
    with open(output_file, 'wb') as f:
        np.array([len(tags)], dtype=np.uint32).tofile(f)
        np.array([1], dtype=np.uint32).tofile(f)
        tags.astype(np.uint32).tofile(f)


def load_gt_bin(gt_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load GT from DiskANN binary format: [npts, K, ids (uint32), dists (float32)]"""
    with open(gt_file, 'rb') as f:
        npts = np.fromfile(f, dtype=np.int32, count=1)[0]
        K = np.fromfile(f, dtype=np.int32, count=1)[0]
        ids = np.fromfile(f, dtype=np.uint32, count=npts * K).reshape(npts, K)
        remaining = os.path.getsize(gt_file) - f.tell()
        if remaining >= npts * K * 4:
            dists = np.fromfile(f, dtype=np.float32, count=npts * K).reshape(npts, K)
        else:
            dists = np.zeros((npts, K), dtype=np.float32)
    return ids, dists


def save_query_fbin(queries: np.ndarray, output_file: str):
    """Save queries in .fbin format: [npts (int32), dim (int32), data (float32)]"""
    npts, dim = queries.shape
    with open(output_file, 'wb') as f:
        f.write(struct.pack('ii', npts, dim))
        queries.astype(np.float32).tofile(f)


def save_query_bin(queries: np.ndarray, output_file: str, data_type: str = "int8"):
    """Save queries in .bin format: [npts (int32), dim (int32), data (int8/uint8)]"""
    dtype = np.int8 if data_type == "int8" else np.uint8
    npts, dim = queries.shape
    with open(output_file, 'wb') as f:
        f.write(struct.pack('ii', npts, dim))
        queries.astype(dtype).tofile(f)


# =============================================================================
# GT Computation
# =============================================================================

def compute_gt_with_binary(gt_binary: str, base_file: str, query_file: str,
                           tags: np.ndarray, K: int, output_dir: str,
                           data_type: str = "float"
                           ) -> Optional[np.ndarray]:
    """Compute GT using C++ binary. Returns (num_queries, K) tag array."""
    os.makedirs(output_dir, exist_ok=True)
    tags_file = os.path.join(output_dir, "_temp_tags.bin")
    gt_bin_file = os.path.join(output_dir, "_temp_gt.bin")

    try:
        save_tags_file(tags, tags_file)
        cmd = [
            gt_binary,
            "--data_type", data_type,
            "--dist_fn", "l2",
            "--base_file", base_file,
            "--query_file", query_file,
            "--tags_file", tags_file,
            "--gt_file", gt_bin_file,
            "--K", str(K),
        ]
        print(f"  Running: {' '.join(cmd)}")
        # No timeout — large query sets (e.g. spacev's 29316 queries against
        # 100M active) regularly take >1h, and a silent TimeoutExpired here
        # (which bubbles up as "exit=0" through the dispatch runner) wastes
        # all the preceding compute.
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Binary error: {result.stderr[:500]}")
            return None
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-3:]:
                print(f"    {line}")

        if os.path.exists(gt_bin_file):
            ids, _ = load_gt_bin(gt_bin_file)
            return ids
        return None

    except Exception as e:
        print(f"  Binary failed: {e}")
        return None
    finally:
        for f in [tags_file, gt_bin_file]:
            if os.path.exists(f):
                os.remove(f)


def compute_gt_python(vectors: np.ndarray, queries: np.ndarray,
                      active_locs: np.ndarray, K: int) -> np.ndarray:
    """
    Vectorized Python brute-force GT using float32 and large BLAS batches.

    Args:
        vectors: (N, D) all vectors
        queries: (Q, D) query vectors
        active_locs: 1D array of active location indices
        K: number of neighbors

    Returns:
        gt_locs: (Q, K) location IDs of nearest neighbors
    """
    active_vecs = vectors[active_locs].astype(np.float32)
    num_queries = len(queries)
    k = min(K, len(active_locs))
    gt_locs = np.zeros((num_queries, K), dtype=np.int64)

    # Precompute norms once (float32, no per-batch recomputation)
    v_norms = np.sum(active_vecs ** 2, axis=1)  # (N_active,)

    batch_size = 1000
    for i in range(0, num_queries, batch_size):
        end = min(i + batch_size, num_queries)
        batch = queries[i:end].astype(np.float32)  # (B, D)
        q_norms = np.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
        dots = batch @ active_vecs.T  # (B, N_active) — BLAS sgemm
        dists = q_norms + v_norms - 2 * dots  # (B, N_active)

        if k < len(active_locs):
            topk_all = np.argpartition(dists, k, axis=1)[:, :k]  # (B, k)
            for j in range(end - i):
                sorted_idx = topk_all[j][np.argsort(dists[j, topk_all[j]])]
                gt_locs[i + j, :k] = active_locs[sorted_idx]
        else:
            for j in range(end - i):
                sorted_idx = np.argsort(dists[j])[:k]
                gt_locs[i + j, :k] = active_locs[sorted_idx]

        if (i + batch_size) % 5000 == 0 or end == num_queries:
            print(f"    GT: {end}/{num_queries} queries processed", flush=True)

    return gt_locs


# =============================================================================
# High-level API (for use by ablation script)
# =============================================================================

def generate_gt_for_active_set(
    vectors: np.ndarray,
    active_locs: np.ndarray,
    num_queries: int,
    K: int,
    output_dir: str,
    cache_key: str = "default",
    seed: int = 42,
    base_file: Optional[str] = None,
    data_type: str = "float",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate GT for a specific active point set. Caches result to disk.

    Args:
        vectors: (N, D) all vectors
        active_locs: sorted array of active location IDs
        num_queries: number of search queries
        K: number of nearest neighbors
        output_dir: directory for gt_cache
        cache_key: unique identifier for this GT (e.g., "pre" or "ratio_0.05")
        seed: random seed for query selection
        base_file: path to .fbin file (for C++ binary)

    Returns:
        (gt_locs, queries, query_locs)
        gt_locs: (num_queries, K) location IDs
        queries: (num_queries, D) query vectors
        query_locs: (num_queries,) location IDs of query points
    """
    cache_dir = os.path.join(output_dir, "gt_cache")
    os.makedirs(cache_dir, exist_ok=True)

    gt_file = os.path.join(cache_dir, f"gt_{cache_key}_K{K}_Q{num_queries}.npy")
    query_locs_file = os.path.join(cache_dir, f"gt_{cache_key}_K{K}_Q{num_queries}_query_locs.npy")
    meta_file = os.path.join(cache_dir, f"gt_{cache_key}_K{K}_Q{num_queries}.json")

    # Check cache
    if os.path.exists(gt_file) and os.path.exists(query_locs_file) and os.path.exists(meta_file):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if (meta.get("K") == K and meta.get("num_queries") == num_queries
                    and meta.get("num_active") == len(active_locs)):
                print(f"  [GT] Cache hit: {cache_key}")
                gt_locs = np.load(gt_file)
                q_locs = np.load(query_locs_file)
                queries = vectors[q_locs]
                return gt_locs, queries, q_locs
        except Exception:
            pass

    # Sample queries from active points
    rng = np.random.RandomState(seed)
    sample_size = min(num_queries, len(active_locs))
    query_locs = rng.choice(active_locs, size=sample_size, replace=False)
    queries = vectors[query_locs].copy()

    print(f"  [GT] Computing: {cache_key} ({sample_size} queries, "
          f"{len(active_locs)} active points, K={K})")
    t0 = time.time()

    # Try C++ binary
    gt_binary = find_gt_binary()
    gt_locs = None

    if gt_binary and base_file:
        # Create temp query file and tags
        if data_type in ("int8", "uint8"):
            tmp_query = os.path.join(cache_dir, f"_tmp_queries_{cache_key}.bin")
            save_query_bin(queries, tmp_query, data_type)
        else:
            tmp_query = os.path.join(cache_dir, f"_tmp_queries_{cache_key}.fbin")
            save_query_fbin(queries, tmp_query)
        tags = (active_locs + 1).astype(np.uint32)  # tag = loc + 1

        gt_tags = compute_gt_with_binary(
            gt_binary, base_file, tmp_query, tags, K, cache_dir,
            data_type=data_type)

        if os.path.exists(tmp_query):
            os.remove(tmp_query)

        if gt_tags is not None:
            # Convert tags back to locations: loc = tag - 1
            gt_locs = (gt_tags.astype(np.int64) - 1)

    if gt_locs is None:
        print(f"  [GT] Using Python fallback")
        gt_locs = compute_gt_python(vectors, queries, active_locs, K)

    elapsed = time.time() - t0
    print(f"  [GT] Done in {elapsed:.1f}s")

    # Save cache
    np.save(gt_file, gt_locs)
    np.save(query_locs_file, query_locs)
    with open(meta_file, 'w') as f:
        json.dump({
            "K": K,
            "num_queries": sample_size,
            "num_active": len(active_locs),
            "cache_key": cache_key,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    return gt_locs, queries, query_locs


# =============================================================================
# Dataset-level GT with range-based caching (for post-hoc recall)
# =============================================================================

def compute_or_load_gt(
    base_file: str,
    query_file: str,
    active_start: int,
    active_end: int,
    K: int,
    num_queries: int,
    gt_cache_dir: str,
    data_type: str = "float",
) -> np.ndarray:
    """
    Compute or load cached GT keyed by (dataset, active_range).

    Active set is always a contiguous range [active_start, active_end) of
    global IDs (0-based).  Cache key uses the range directly — no hash needed.

    Args:
        base_file: path to base vectors .fbin
        query_file: path to query vectors .fbin
        active_start: inclusive start of active global ID range
        active_end: exclusive end of active global ID range
        K: number of nearest neighbors
        num_queries: number of queries to use from query_file
        gt_cache_dir: directory for caching (dataset-level)
        data_type: "float", "int8", or "uint8"

    Returns:
        gt_gids: (num_queries, K) int64 array of global IDs
    """
    os.makedirs(gt_cache_dir, exist_ok=True)

    cache_name = f"gt_K{K}_Q{num_queries}_R{active_start}-{active_end}"
    gt_file = os.path.join(gt_cache_dir, f"{cache_name}.npy")
    meta_file = os.path.join(gt_cache_dir, f"{cache_name}_meta.json")

    # Check cache
    if os.path.exists(gt_file) and os.path.exists(meta_file):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if (meta.get("K") == K and meta.get("num_queries") == num_queries
                    and meta.get("active_start") == active_start
                    and meta.get("active_end") == active_end):
                print(f"  [GT] Cache hit: {cache_name}")
                return np.load(gt_file)
        except Exception:
            pass

    # Cache miss — compute
    num_active = active_end - active_start
    print(f"  [GT] Computing: {num_active} active points [{active_start}, {active_end}), "
          f"Q={num_queries}, K={K}")

    # Map config data_type → numpy dtype for reading .*bin files.
    _DTYPE_MAP = {"float": np.float32, "float32": np.float32,
                  "int8": np.int8, "uint8": np.uint8}
    load_dtype = _DTYPE_MAP.get(data_type, np.float32)
    vectors = load_bin_data(base_file, dtype=load_dtype)
    queries = load_bin_data(query_file, dtype=load_dtype)[:num_queries]
    active_gids = np.arange(active_start, active_end, dtype=np.int64)

    t0 = time.time()

    # Try C++ binary first
    gt_binary = find_gt_binary()
    gt_gids = None

    if gt_binary:
        tags = (active_gids + 1).astype(np.uint32)  # tag = gid + 1
        tmp_query = os.path.join(gt_cache_dir, f"_tmp_queries_{cache_name}.fbin")
        if data_type in ("int8", "uint8"):
            save_query_bin(queries, tmp_query, data_type)
        else:
            save_query_fbin(queries, tmp_query)

        gt_tags = compute_gt_with_binary(
            gt_binary, base_file, tmp_query, tags, K, gt_cache_dir,
            data_type=data_type)

        if os.path.exists(tmp_query):
            os.remove(tmp_query)

        if gt_tags is not None:
            gt_gids = gt_tags.astype(np.int64) - 1  # tag → global ID

    if gt_gids is None:
        print(f"  [GT] Using Python fallback")
        gt_gids = compute_gt_python(vectors, queries, active_gids, K)

    elapsed = time.time() - t0
    print(f"  [GT] Done in {elapsed:.1f}s")

    # Save cache
    np.save(gt_file, gt_gids)
    with open(meta_file, 'w') as f:
        json.dump({
            "K": K,
            "num_queries": num_queries,
            "active_start": active_start,
            "active_end": active_end,
            "num_active": num_active,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    return gt_gids


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate brute-force k-NN ground truth")

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", help="Path to vector data file (.fbin)")
    data_group.add_argument("--partition-dir", help="Path to partition dir (auto-resolves)")

    parser.add_argument("--tags", required=True, help="Path to .tags file (defines active set)")
    parser.add_argument("--queries", help="Path to query file (.fbin). If not given, samples from active set")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--output-dir", help="Output dir (default: gt_cache/ next to tags file)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve data
    if args.partition_dir:
        from common.data_loader import ensure_reconstructed_data
        data_file, _ = ensure_reconstructed_data(partition_dir=args.partition_dir)
    else:
        data_file = args.data

    print(f"Loading vectors from {data_file}...")
    vectors = load_bin_data(data_file, dtype=np.float32)
    print(f"  {vectors.shape[0]} vectors, dim={vectors.shape[1]}")

    print(f"Loading tags from {args.tags}...")
    tags = load_tags(args.tags)
    # Active locations: where tag != 0
    active_locs = np.array([i for i, t in enumerate(tags) if t != 0], dtype=np.int64)
    print(f"  {len(active_locs)} active points")

    # Output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.tags), "gt_cache")

    # Queries
    if args.queries:
        queries = load_bin_data(args.queries, dtype=np.float32)
        queries = queries[:args.num_queries]
        query_locs = None
    else:
        rng = np.random.RandomState(args.seed)
        sample_size = min(args.num_queries, len(active_locs))
        query_locs = rng.choice(active_locs, size=sample_size, replace=False)
        queries = vectors[query_locs]

    # Cache key from tags filename
    tags_name = os.path.splitext(os.path.basename(args.tags))[0]
    cache_key = tags_name

    gt_locs, _, _ = generate_gt_for_active_set(
        vectors=vectors,
        active_locs=active_locs,
        num_queries=args.num_queries,
        K=args.K,
        output_dir=output_dir,
        cache_key=cache_key,
        seed=args.seed,
        base_file=data_file,
    )

    print(f"\nGT shape: {gt_locs.shape}")
    print(f"Saved to: {output_dir}/gt_cache/")


if __name__ == "__main__":
    main()
