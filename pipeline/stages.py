"""
Build-side pipeline stages for the distributed DiskANN experiment pipeline.

Stages (dependency order):
  1. build              — Build initial DiskANN graph (LOCAL)
  2. partition          — K-means partitioning of active vectors (LOCAL)
  3. colocation         — Build RDMA co-location binaries (LOCAL)
  4. pq_build           — Build PQ codebooks/codes for filtering (LOCAL)
  5. gt                 — Brute-force ground truth (LOCAL)
  6. distributed_engine — Engine pipeline handoff (DISTRIBUTED)
  7. search             — Distributed search simulator (LOCAL, deferred)

Each stage function takes a PipelineContext and returns it (mutated).
"""

import json
import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Shared utilities
_project_root = Path(__file__).resolve().parent.parent
_scripts_dir = _project_root / 'scripts'
for _p in [str(_project_root), str(_scripts_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_loader import (
    load_diskann_graph, load_bin_data, load_fbin, save_fbin,
    load_tags, save_tags,
    get_bin_metadata, detect_data_root, get_dataset_info,
    find_all_checkpoints, reconstruct_from_partitions,
    ensure_reconstructed_data,
)
from common.partitioner import KMeansPartitioner, load_partitions, save_partitions
from common.insert_strategies import get_strategy as get_insert_strategy_impl
from common.partition_paths import (
    get_insert_strategy, partition_dir as build_partition_dir,
    colocation_dir as build_colocation_dir,
    pq_dir as build_pq_dir,
)
from pipeline.config import get_pq_filter_config
from pipeline.experiment_paths import (
    artifact_colocation_dir, artifact_experiment_dir, run_dir as build_run_dir,
)

# Heavy legacy modules are lazy-imported only in stages that need them.
# - DistributedSearch, LatencyConfig, etc.: used in stage_search simulator.


# =============================================================================
# PipelineContext
# =============================================================================

@dataclass
class PipelineContext:
    """Shared state threaded through pipeline stages."""
    config: Dict[str, Any]
    base_dir: Path
    experiment_name: str

    # Dataset
    data_root: Optional[str] = None
    base_file: Optional[str] = None
    query_file: Optional[str] = None
    dimension: int = 0
    data_type: str = "float32"

    # Build outputs
    graph_dir: Optional[Path] = None
    graph_file: Optional[str] = None
    tags_file: Optional[str] = None

    # Partition outputs
    partition_dir: Optional[Path] = None
    centroids_file: Optional[str] = None
    partition_ids_file: Optional[str] = None
    medoids: Optional[Dict[int, int]] = None
    num_partitions: int = 0

    # GT outputs
    gt_file: Optional[str] = None

    # Search outputs
    search_results: Dict[str, Any] = field(default_factory=dict)

    # Colocation outputs
    colocation_dir: Optional[str] = None

    def save_checkpoint(self, checkpoint_file: str = None):
        """Save context state (JSON-serializable parts) for resume."""
        if checkpoint_file is None:
            checkpoint_file = str(self.base_dir / "pipeline_state.json")

        state = {
            'experiment_name': self.experiment_name,
            'data_root': self.data_root,
            'base_file': self.base_file,
            'query_file': self.query_file,
            'dimension': self.dimension,
            'data_type': self.data_type,
            'graph_dir': str(self.graph_dir) if self.graph_dir else None,
            'graph_file': self.graph_file,
            'tags_file': self.tags_file,
            'partition_dir': str(self.partition_dir) if self.partition_dir else None,
            'centroids_file': self.centroids_file,
            'partition_ids_file': self.partition_ids_file,
            'medoids': self.medoids,
            'num_partitions': self.num_partitions,
            'gt_file': self.gt_file,
            'search_results': self.search_results,
        }
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_checkpoint(self, checkpoint_file: str = None):
        """Restore context state from checkpoint."""
        if checkpoint_file is None:
            checkpoint_file = str(self.base_dir / "pipeline_state.json")
        if not os.path.exists(checkpoint_file):
            return False
        with open(checkpoint_file) as f:
            state = json.load(f)

        # Path fields that may need /workspace/ → host translation
        _path_fields = {
            'data_root', 'base_file', 'query_file', 'graph_file',
            'tags_file', 'centroids_file', 'partition_ids_file', 'gt_file',
        }
        _path_dir_fields = {'graph_dir', 'partition_dir'}

        for key, val in state.items():
            if hasattr(self, key):
                if key in _path_dir_fields and val is not None:
                    resolved = _resolve_workspace_path(val)
                    setattr(self, key, Path(resolved))
                elif key in _path_fields and val is not None:
                    setattr(self, key, _resolve_workspace_path(val))
                else:
                    setattr(self, key, val)
        return True


# =============================================================================
# Helpers
# =============================================================================

def _resolve_dataset(ctx: PipelineContext):
    """Resolve dataset paths from config."""
    ds_cfg = ctx.config.get('dataset', {})
    ctx.dimension = ds_cfg.get('dimension', 0)
    ctx.data_type = ds_cfg.get('data_type', 'float32')

    # data_root priority: config explicit > auto-detect
    explicit_root = ds_cfg.get('data_root')
    if explicit_root and os.path.isdir(explicit_root):
        ctx.data_root = explicit_root
    else:
        ctx.data_root = detect_data_root(ds_cfg.get('name'))

    # Resolve base_file / query_file
    base_file = ds_cfg.get('base_file', '')
    query_file = ds_cfg.get('query_file', '')

    if ctx.data_root:
        if not os.path.isabs(base_file):
            base_file = os.path.join(ctx.data_root, base_file)
        if not os.path.isabs(query_file):
            query_file = os.path.join(ctx.data_root, query_file)

    ctx.base_file = base_file
    ctx.query_file = query_file

    if ctx.dimension == 0 and os.path.exists(base_file):
        _, dim = get_bin_metadata(base_file)
        ctx.dimension = dim


def _get_dtype(data_type: str):
    """Map string data type to numpy dtype."""
    return {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}.get(
        data_type, np.float32)


def _resolve_workspace_path(path: Optional[str]) -> Optional[str]:
    """Translate between /workspace/ container paths and the project host path.

    Checkpoint paths may be written from inside the container (/workspace/...) and
    later read on the host, or vice versa. This resolves either direction to an
    existing path for the current the earlier DANNS repo project.
    """
    if not path:
        return path
    if os.path.exists(path):
        return path
    host_base = str(Path(__file__).resolve().parent.parent)
    # Container → host: /workspace/X → <host_base>/X
    if path.startswith("/workspace/"):
        host_path = os.path.join(host_base, path[len("/workspace/"):])
        if os.path.exists(host_path):
            return host_path
    # Host → container: <host_base>/X → /workspace/X
    prefix = host_base + "/"
    if path.startswith(prefix):
        container_path = "/workspace/" + path[len(prefix):]
        if os.path.exists(container_path):
            return container_path
    return path


import re as _re

def _extract_iteration_from_graph(graph_file: Optional[str]) -> int:
    """Extract iteration number from graph checkpoint filename.

    E.g. '.../checkpoints/iteration10' → 10, '.../checkpoints/iteration0' → 0.
    Returns 0 if no match.
    """
    if not graph_file:
        return 0
    basename = Path(graph_file).name
    m = _re.match(r'^iteration(\d+)$', basename)
    return int(m.group(1)) if m else 0


def _compute_medoids(
    vectors: np.ndarray,
    partition_ids: np.ndarray,
    centroids: np.ndarray,
    num_partitions: int,
) -> Dict[int, int]:
    """Compute medoid (closest real point to centroid) for each partition.

    Returns:
        {partition_id: location_index}
    """
    medoids = {}
    for p in range(num_partitions):
        mask = partition_ids == p
        locs = np.where(mask)[0]
        if len(locs) == 0:
            continue
        vecs = vectors[locs].astype(np.float64)
        centroid = centroids[p].astype(np.float64)
        dists = np.sum((vecs - centroid) ** 2, axis=1)
        best_local = int(np.argmin(dists))
        medoids[p] = int(locs[best_local])
    return medoids


def _compute_medoids_streaming(
    bin_file: str,
    dtype,
    partition_ids: np.ndarray,
    centroids: np.ndarray,
    num_partitions: int,
    batch_size: int = 1_000_000,
) -> Dict[int, int]:
    """Compute medoids by streaming vectors from file (no full load)."""
    n_points = len(partition_ids)
    # Track best (min distance) per partition
    best_dist = np.full(num_partitions, np.inf, dtype=np.float64)
    best_gid = np.full(num_partitions, -1, dtype=np.int64)

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch = load_bin_data(bin_file, dtype=dtype,
                              offset_points=start, num_points=end - start
                              ).astype(np.float32)
        batch_pids = partition_ids[start:end]

        for p in range(num_partitions):
            mask = batch_pids == p
            if not mask.any():
                continue
            locs = np.where(mask)[0]
            vecs = batch[locs]
            dists = np.sum((vecs - centroids[p]) ** 2, axis=1)
            local_best = int(np.argmin(dists))
            if dists[local_best] < best_dist[p]:
                best_dist[p] = dists[local_best]
                best_gid[p] = start + int(locs[local_best])

    return {p: int(best_gid[p]) for p in range(num_partitions) if best_gid[p] >= 0}


# =============================================================================
# Stage 1: BUILD
# =============================================================================

def stage_build(ctx: PipelineContext) -> PipelineContext:
    """Build the initial DiskANN graph.

    Uses C++ diskann_bindings for building. Falls back to error if unavailable.

    Outputs:
        ctx.graph_dir, ctx.graph_file, ctx.tags_file
    """
    _resolve_dataset(ctx)
    build_cfg = ctx.config.get('build', {})
    output_cfg = ctx.config.get('output', {})

    # Output directory
    graph_dir = artifact_experiment_dir(ctx.config) / "graph"
    os.makedirs(graph_dir, exist_ok=True)
    ctx.graph_dir = graph_dir

    # Skip decision: build_pipeline creates only the initial graph. Dynamic
    # insert/delete/consolidate runs belong to engine_pipeline.
    skip = ctx.config.get('pipeline', {}).get('skip_if_exists', True)
    checkpoints = find_all_checkpoints(str(graph_dir))
    if skip and checkpoints:
        last_iter, last_path = checkpoints[-1]
        print(f"  [SKIP] Found existing checkpoint at iteration {last_iter}: {last_path}")
        ctx.graph_file = last_path
        tags_path = last_path + ".tags"
        if os.path.exists(tags_path):
            ctx.tags_file = tags_path
        return ctx

    # Import DiskANN bindings
    try:
        # /workspace/lib/ (project root lib, not scripts/lib)
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root / 'lib'))
        sys.path.insert(0, str(project_root / 'build'))
        try:
            import single_engine as diskann      # unified single-node engine
        except ImportError:
            import diskann_bindings as diskann   # legacy module (same classes)
    except ImportError:
        raise RuntimeError(
            "single_engine/diskann_bindings not found. Build the C++ bindings first:\n"
            "  cd lib && python setup.py build_ext --inplace"
        )

    # Load dataset. mmap the base file so 100M-scale subsets (deep300M=115GB,
    # laion300M=614GB) never trigger a full in-memory load. Slicing and the
    # C++ buffer-protocol pass still work transparently on np.memmap; only the
    # initial-build and per-iter insert slices actually fault pages in.
    print(f"  Loading dataset: {ctx.base_file}")
    vectors = load_bin_data(ctx.base_file, dtype=_get_dtype(ctx.data_type), mmap=True)
    total_points, dim = vectors.shape
    ctx.dimension = dim
    print(f"  {total_points} vectors, dim={dim}")

    # Build parameters
    initial_size = build_cfg.get('initial_size', total_points)
    max_capacity = build_cfg.get('max_capacity', int(total_points * 1.1))
    R = build_cfg.get('R', 64)
    L = build_cfg.get('L', 100)
    alpha = build_cfg.get('alpha', 1.2)
    num_threads = build_cfg.get('num_threads', 0)

    # Select index class based on data type
    force_float32 = build_cfg.get('force_float32', False)
    if force_float32:
        IndexClass = diskann.DiskANNIndex
        print(f"  [force_float32] Using float32 index (data will be converted)")
    else:
        IndexClass = diskann.DiskANNIndex
        if ctx.data_type == 'uint8':
            IndexClass = getattr(diskann, 'DiskANNIndexUInt8', diskann.DiskANNIndex)
        elif ctx.data_type == 'int8':
            IndexClass = getattr(diskann, 'DiskANNIndexInt8', diskann.DiskANNIndex)

    save_data_file = bool(output_cfg.get('save_data_file', False))

    t0 = time.time()
    index = IndexClass(dim, max_capacity, R, L, alpha, num_threads, dynamic=True)

    print(f"  Building index: initial_size={initial_size}, R={R}, L={L}, alpha={alpha}")
    initial_data = vectors[:initial_size]
    if force_float32 and initial_data.dtype != np.float32:
        initial_data = initial_data.astype(np.float32)
    index.build(initial_data)
    build_time = time.time() - t0
    print(f"  Build completed in {build_time:.1f}s, {index.num_points} points")

    ckpt_path = str(graph_dir / "checkpoints" / "iteration0")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    index.save(ckpt_path, save_data_file=save_data_file)
    print(f"  Saved initial checkpoint to {ckpt_path} (save_data_file={save_data_file})")

    # Find the final graph and tags files
    all_ckpts = find_all_checkpoints(str(graph_dir))
    if all_ckpts:
        last_iter, last_path = all_ckpts[-1]
        ctx.graph_file = last_path
        tags_path = last_path + ".tags"
        if os.path.exists(tags_path):
            ctx.tags_file = tags_path

    return ctx


# =============================================================================
# Stage 2: PARTITION
# =============================================================================

def stage_partition(ctx: PipelineContext) -> PipelineContext:
    """K-means partitioning of active vectors + pluggable insert-pool strategy.

    Indexing contract (written to metadata.json):
      - partition_ids.npy is gid-indexed, length = total_points
      - frozen point is NOT part of this array (it lives in loc-space)
      - active = [0, initial_size) via balanced k-means
      - insert pool = [initial_size, total_points) via configured strategy

    Outputs:
        ctx.partition_dir, ctx.centroids_file, ctx.partition_ids_file, ctx.medoids
    """
    _resolve_dataset(ctx)
    part_cfg = ctx.config.get('partition', {})
    output_cfg = ctx.config.get('output', {})

    num_partitions = part_cfg.get('num_partitions', 4)
    ctx.num_partitions = num_partitions

    strategy_name = get_insert_strategy(ctx.config)

    # Output directory — strategy embedded so runs don't clobber
    partition_dir = build_partition_dir(
        artifact_experiment_dir(ctx.config),
        num_partitions, strategy_name,
    )
    os.makedirs(partition_dir, exist_ok=True)
    ctx.partition_dir = partition_dir

    metadata_file = partition_dir / "metadata.json"

    # Skip check
    skip = ctx.config.get('pipeline', {}).get('skip_if_exists', True)
    force = part_cfg.get('force', False)
    if skip and not force and metadata_file.exists():
        print(f"  [SKIP] Partition metadata exists: {metadata_file}")
        with open(metadata_file) as f:
            meta = json.load(f)
        ctx.centroids_file = str(partition_dir / "centroids.fbin")
        ctx.partition_ids_file = str(partition_dir / "partition_ids.npy")
        ctx.medoids = {int(k): v for k, v in enumerate(meta.get('medoids', []))}
        return ctx

    # Get file metadata without loading vectors into RAM
    dtype = _get_dtype(ctx.data_type)
    total_points, dim = get_bin_metadata(ctx.base_file)
    initial_size = ctx.config.get('build', {}).get('initial_size', total_points)
    print(f"  Base file: {ctx.base_file} ({total_points} points, dim={dim})")
    print(f"  Insert strategy: {strategy_name}")
    print(f"  Partitioning {initial_size} vectors (initial_size={initial_size})")

    # --- Phase 1: active k-means ------------------------------------------
    mode = part_cfg.get('mode', 'auto')
    batch_size = part_cfg.get('batch_size', 1_000_000)
    max_iter = part_cfg.get('max_iter', 300)
    seed = part_cfg.get('seed', 42)
    balanced = part_cfg.get('balanced', False)
    max_imbalance_ratio = part_cfg.get('max_imbalance_ratio', 1.2)

    t0 = time.time()
    partitioner = KMeansPartitioner(
        n_clusters=num_partitions,
        mode='streaming',
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        balanced=balanced,
        max_imbalance_ratio=max_imbalance_ratio,
    )
    active_ids = partitioner._fit_streaming_from_file(
        ctx.base_file, dtype=dtype, start_point=0, n_points=initial_size)
    active_time = time.time() - t0
    print(f"  Active k-means completed in {active_time:.1f}s")

    centroids = partitioner.centroids
    active_sizes = np.bincount(active_ids.astype(np.int64), minlength=num_partitions)

    # --- Phase 2: insert-pool assignment via strategy ---------------------
    # Balanced strategies reuse max_imbalance_ratio from active k-means config.
    strategy_params = part_cfg.get('insert_strategy_params', {}) or {}
    strategy_params.setdefault('max_imbalance_ratio', max_imbalance_ratio)
    strategy = get_insert_strategy_impl(strategy_name, **strategy_params)

    insert_time = 0.0
    if total_points > initial_size:
        insert_count = total_points - initial_size
        print(f"  Assigning {insert_count} insert-pool points via "
              f"strategy='{strategy_name}'...")
        t1 = time.time()
        insert_ids = strategy.assign(
            base_file=ctx.base_file,
            dtype=dtype,
            insert_start=initial_size,
            insert_count=insert_count,
            centroids=centroids,
            num_partitions=num_partitions,
            active_sizes=active_sizes,
            batch_size=batch_size,
            seed=seed,
        )
        insert_time = time.time() - t1
        print(f"  Insert-pool assignment completed in {insert_time:.1f}s")
        partition_ids = np.concatenate([active_ids, insert_ids])
    else:
        partition_ids = active_ids

    # Compute medoids — streaming (one partition at a time, batched)
    medoids_dict = _compute_medoids_streaming(
        ctx.base_file, dtype, partition_ids, centroids, num_partitions, batch_size)
    ctx.medoids = medoids_dict
    print(f"  Medoids: {medoids_dict}")

    # Partition sizes
    partition_sizes = []
    for p in range(num_partitions):
        count = int(np.sum(partition_ids == p))
        partition_sizes.append(count)
        print(f"    Partition {p}: {count} points")

    # Save outputs
    centroids_file = str(partition_dir / "centroids.fbin")
    save_fbin(centroids_file, centroids.astype(np.float32))
    ctx.centroids_file = centroids_file

    partition_ids_file = str(partition_dir / "partition_ids.npy")
    np.save(partition_ids_file, partition_ids.astype(np.uint32))
    ctx.partition_ids_file = partition_ids_file

    # Save per-partition ID lists (data.fbin is not generated — data layout
    # is always colocation, so vectors are accessed from .coloc files via RDMA)
    for p in range(num_partitions):
        p_dir = partition_dir / f"partition{p}"
        os.makedirs(p_dir, exist_ok=True)
        mask = partition_ids == p
        p_indices = np.where(mask)[0].astype(np.uint32)

        ids_path = str(p_dir / "ids.bin")
        header = np.array([len(p_indices), 1], dtype=np.uint32)
        with open(ids_path, 'wb') as f:
            header.tofile(f)
            p_indices.tofile(f)

    # Save metadata — explicit indexing contract
    metadata = {
        'experiment_name': ctx.experiment_name,
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'status': 'completed',
        'source_file': ctx.base_file,
        'indexing': 'gid',
        'frozen_excluded': True,
        'insert_strategy': strategy_name,
        'insert_strategy_params': strategy_params,
        'num_partitions': num_partitions,
        'total_points': int(len(partition_ids)),
        'initial_size': initial_size,
        'dimension': dim,
        'partition_sizes': partition_sizes,
        'active_partition_sizes': active_sizes.tolist(),
        'medoids': [medoids_dict.get(p, -1) for p in range(num_partitions)],
        'partition_time_s': active_time + insert_time,
        'active_time_s': active_time,
        'insert_time_s': insert_time,
        'mode': getattr(partitioner, '_effective_mode', mode),
        'config': {
            'mode': mode,
            'batch_size': batch_size,
            'max_iter': max_iter,
            'seed': seed,
            'balanced': balanced,
            'max_imbalance_ratio': max_imbalance_ratio,
        },
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Partition outputs saved to {partition_dir}")
    return ctx


# =============================================================================
# Stage 4: GT (Ground Truth Generation)
# =============================================================================

def stage_gt(ctx: PipelineContext) -> PipelineContext:
    """Generate brute-force ground truth for recall measurement.

    Uses query vector file by default (configurable via gt.sample_from_active).
    GT is stored per-experiment and reused across search runs.

    Outputs:
        ctx.gt_file (path to .npy GT file)
    """
    _resolve_dataset(ctx)
    # Translate /workspace/ paths for host execution
    ctx.tags_file = _resolve_workspace_path(ctx.tags_file)
    ctx.graph_file = _resolve_workspace_path(ctx.graph_file)

    gt_cfg = ctx.config.get('gt', {})
    search_cfg = ctx.config.get('search', {})
    output_cfg = ctx.config.get('output', {})

    K = gt_cfg.get('K', search_cfg.get('K', 10))
    num_queries = gt_cfg.get('num_queries', search_cfg.get('num_queries', 1000))
    seed = gt_cfg.get('seed', 42)
    sample_from_active = gt_cfg.get('sample_from_active', False)

    # num_queries=0 means "all queries in the query file".
    if num_queries == 0 and not sample_from_active and ctx.query_file:
        q_path = _resolve_workspace_path(ctx.query_file)
        if q_path and os.path.exists(q_path):
            q_npts, _ = get_bin_metadata(q_path)
            num_queries = int(q_npts)
            print(f"  num_queries=0 → using all {num_queries} queries from file")

    # GT directory — artifact-level, reused across search runs
    experiment_dir = artifact_experiment_dir(ctx.config)
    gt_dir = experiment_dir / "gt"
    os.makedirs(gt_dir, exist_ok=True)

    # Cache key based on graph checkpoint (so GT updates if graph changes)
    if ctx.graph_file:
        ckpt_name = os.path.basename(ctx.graph_file)
    else:
        ckpt_name = "unknown"
    cache_key = f"{ckpt_name}_K{K}_Q{num_queries}"

    gt_file = gt_dir / f"gt_{cache_key}.npy"
    meta_file = gt_dir / f"gt_{cache_key}.json"
    queries_file = gt_dir / f"gt_{cache_key}_queries.npy"

    # Skip check
    skip = ctx.config.get('pipeline', {}).get('skip_if_exists', True)
    if skip and gt_file.exists() and meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("K") == K and meta.get("num_queries") == num_queries:
                print(f"  [SKIP] GT cache hit: {gt_file}")
                ctx.gt_file = str(gt_file)
                return ctx
        except Exception:
            pass

    # Load tags to find active set
    if not ctx.tags_file or not os.path.exists(ctx.tags_file):
        raise RuntimeError("No tags file. Run build stage first.")

    # vectors: only loaded into Python when needed (sample_from_active or
    # Python fallback). For the C++ binary path with a query_file, we avoid
    # a ~38GB mmap/read of the base vectors.
    vectors = None

    print(f"  Loading tags: {ctx.tags_file}")
    tags = load_tags(ctx.tags_file)
    # Convert DiskANN locations to global vector IDs
    # tags[loc] = global_id + 1, so global_id = tags[loc] - 1
    active_global_ids = np.array([int(tags[i]) - 1 for i in range(len(tags)) if tags[i] != 0],
                                  dtype=np.int64)
    active_global_ids.sort()
    print(f"  {len(active_global_ids)} active points (global IDs)")

    # Resolve queries
    if sample_from_active:
        # Sample queries from active set — needs vectors in memory.
        print(f"  Loading vectors (sample_from_active): {ctx.base_file}")
        vectors = load_bin_data(ctx.base_file, dtype=_get_dtype(ctx.data_type))
        rng = np.random.RandomState(seed)
        sample_size = min(num_queries, len(active_global_ids))
        query_global_ids = rng.choice(active_global_ids, size=sample_size, replace=False)
        queries = vectors[query_global_ids].copy()
        print(f"  Sampled {sample_size} queries from active set")
    else:
        # Use query file (default)
        if not ctx.query_file or not os.path.exists(ctx.query_file):
            raise RuntimeError(
                "No query file found. Set dataset.query_file in config "
                "or use gt.sample_from_active: true")
        queries = load_bin_data(ctx.query_file, dtype=_get_dtype(ctx.data_type))
        queries = queries[:num_queries].astype(np.float32)
        print(f"  Loaded {len(queries)} queries from {ctx.query_file}")

    # Import GT computation from local module
    import generate_gt as _gt_mod
    compute_gt_python = _gt_mod.compute_gt_python
    find_gt_binary = _gt_mod.find_gt_binary
    compute_gt_with_binary = _gt_mod.compute_gt_with_binary
    save_query_fbin = _gt_mod.save_query_fbin
    save_query_bin = _gt_mod.save_query_bin

    print(f"  Computing GT: {len(queries)} queries × {len(active_global_ids)} active points, K={K}")
    t0 = time.time()

    # Try C++ binary first
    gt_binary = find_gt_binary()
    gt_global_ids = None

    if gt_binary and ctx.base_file:
        # Save queries in appropriate format for C++ binary
        if ctx.data_type in ('int8', 'uint8'):
            tmp_query = str(gt_dir / "_tmp_queries.bin")
            save_query_bin(queries, tmp_query, data_type=ctx.data_type)
            gt_dtype = ctx.data_type
        else:
            tmp_query = str(gt_dir / "_tmp_queries.fbin")
            save_query_fbin(queries, tmp_query)
            gt_dtype = "float"
        # Tags for binary: compute_groundtruth_with_tags treats the tags file
        # as GID-indexed (tags[gid] == 0 means skip). Build a SPARSE array
        # spanning the whole base_file: tags[gid] = gid+1 for active gids,
        # 0 elsewhere. This is critical post-iteration0, where active gids
        # may lie anywhere in [0, total_vectors) — the binary otherwise
        # loads base_file[0:npts_tags] and attaches our labels, silently
        # computing GT on the wrong vectors.
        total_vectors, _ = get_bin_metadata(ctx.base_file)
        tags_for_binary = np.zeros(total_vectors, dtype=np.uint32)
        tags_for_binary[active_global_ids] = (
            active_global_ids + 1).astype(np.uint32)

        gt_tags = compute_gt_with_binary(
            gt_binary, ctx.base_file, tmp_query, tags_for_binary,
            K, str(gt_dir), data_type=gt_dtype)

        if os.path.exists(tmp_query):
            os.remove(tmp_query)

        if gt_tags is not None:
            gt_global_ids = (gt_tags.astype(np.int64) - 1)

    if gt_global_ids is None:
        print(f"  Using Python fallback")
        # compute_gt_python expects active_locs as indices into vectors array
        # active_global_ids ARE the indices into the vectors array
        gt_global_ids = compute_gt_python(vectors, queries, active_global_ids, K)

    elapsed = time.time() - t0
    print(f"  GT computed in {elapsed:.1f}s: shape={gt_global_ids.shape}")

    # Save — GT contains global vector IDs (indices into base vectors array)
    np.save(str(gt_file), gt_global_ids)
    np.save(str(queries_file), queries)
    with open(meta_file, 'w') as f:
        json.dump({
            "K": K,
            "num_queries": len(queries),
            "num_active": len(active_global_ids),
            "graph_checkpoint": ctx.graph_file,
            "sample_from_active": sample_from_active,
            "seed": seed if sample_from_active else None,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime('%Y%m%d_%H%M%S'),
        }, f, indent=2)

    ctx.gt_file = str(gt_file)
    print(f"  GT saved to: {gt_file}")
    return ctx


# =============================================================================
# Stage 5: SEARCH
# =============================================================================

def stage_search(ctx: PipelineContext) -> PipelineContext:
    """Distributed search evaluation with latency simulation.

    Tests multiple L values and reports recall, latency, hops, etc.

    Outputs:
        ctx.search_results
    """
    from distributed_search import (
        DistributedSearch, LatencyConfig, SearchStats,
        compute_ground_truth, compute_recall,
    )

    _resolve_dataset(ctx)
    search_cfg = ctx.config.get('search', {})
    build_cfg = ctx.config.get('build', {})
    dist_cfg = ctx.config.get('distributed_search', {})
    output_cfg = ctx.config.get('output', {})

    K = search_cfg.get('K', 10)
    L_values = search_cfg.get('L_values', [10, 20, 50, 100])
    num_queries = search_cfg.get('num_queries', 1000)

    # Load queries
    if not ctx.query_file or not os.path.exists(ctx.query_file):
        print("  [SKIP] No query file available")
        return ctx
    all_queries = load_bin_data(ctx.query_file, dtype=_get_dtype(ctx.data_type))
    queries = all_queries[:num_queries].astype(np.float32)
    print(f"  Loaded {len(queries)} queries")

    # Need: graph, vectors, partition_ids, centroids, medoids, tags.
    graph = None
    vectors = None
    tags = None
    partition_ids = None
    centroids = None
    medoids = ctx.medoids or {}

    if not ctx.graph_file:
        raise RuntimeError("No graph_file set. Run build stage first.")

    print(f"  Loading graph: {ctx.graph_file}")
    neighbors, metadata = load_diskann_graph(ctx.graph_file)
    graph = [list(nbrs) for nbrs in neighbors]

    print(f"  Loading vectors: {ctx.base_file}")
    vectors = load_bin_data(ctx.base_file, dtype=np.float32)

    if ctx.tags_file and os.path.exists(ctx.tags_file):
        tags = load_tags(ctx.tags_file)

    if ctx.partition_ids_file and os.path.exists(ctx.partition_ids_file):
        partition_ids = np.load(ctx.partition_ids_file)

    if ctx.centroids_file and os.path.exists(ctx.centroids_file):
        centroids = load_bin_data(ctx.centroids_file, dtype=np.float32)

    if partition_ids is None or centroids is None:
        print("  [WARN] No partition data found. Running non-distributed search.")
        # Assign all points to partition 0
        partition_ids = np.zeros(len(vectors), dtype=np.int32)
        centroids = np.mean(vectors[:min(len(vectors), 100000)], axis=0, keepdims=True).astype(np.float32)
        medoids = {0: 0}

    if not medoids:
        medoids = _compute_medoids(vectors, partition_ids, centroids,
                                   centroids.shape[0])

    # Latency config
    lat_cfg = dist_cfg.get('latency', {})
    latency_config = LatencyConfig(
        local_us=lat_cfg.get('local_us', 0.0),
        remote_us=lat_cfg.get('remote_us', 2.0),
        remote_std_us=lat_cfg.get('remote_std_us', 0.5),
        enable_jitter=lat_cfg.get('enable_jitter', True),
        compute_ns_per_dim=lat_cfg.get('compute_ns_per_dim', 0.1),
    )

    # Build distributed search object
    dsearch = DistributedSearch(
        graph=graph,
        vectors=vectors,
        partition_ids=partition_ids,
        centroids=centroids,
        medoids=medoids,
        tags=tags,
        latency_config=latency_config,
    )

    # Compute ground truth
    print("  Computing ground truth...")
    # Determine active locations
    active_locs = None
    if tags is not None:
        active_mask = tags[:len(graph)] > 0
        active_locs = np.where(active_mask)[0]

    t_gt = time.time()
    gt = compute_ground_truth(vectors, queries, K,
                              active_locs=active_locs, tags=tags)
    print(f"  GT computed in {time.time() - t_gt:.1f}s")

    # Sweep L values
    ctx.search_results = {'K': K, 'num_queries': num_queries, 'L_sweep': []}

    for L in L_values:
        print(f"\n  === L = {L} ===")
        t_search = time.time()
        result_ids, result_dists, all_stats = dsearch.search_batch(queries, K=K, L=L)
        search_time = time.time() - t_search

        # Convert locations to tags for recall
        result_tags = np.zeros_like(result_ids)
        for qi in range(len(result_ids)):
            for ki in range(K):
                loc = int(result_ids[qi, ki])
                if tags is not None and 0 <= loc < len(tags) and tags[loc] > 0:
                    result_tags[qi, ki] = tags[loc]
                else:
                    result_tags[qi, ki] = loc

        recall = compute_recall(result_tags, gt, K)

        # Aggregate stats
        avg_hops = np.mean([s.num_hops for s in all_stats])
        avg_dist_comp = np.mean([s.num_distance_computations for s in all_stats])
        avg_latency = np.mean([s.total_latency_us for s in all_stats])
        avg_local_vec = np.mean([s.local_vector_accesses for s in all_stats])
        avg_remote_vec = np.mean([s.remote_vector_accesses for s in all_stats])
        avg_local_nbr = np.mean([s.local_neighbor_accesses for s in all_stats])
        avg_remote_nbr = np.mean([s.remote_neighbor_accesses for s in all_stats])

        entry = {
            'L': L,
            'recall': recall,
            'avg_hops': float(avg_hops),
            'avg_distance_computations': float(avg_dist_comp),
            'avg_latency_us': float(avg_latency),
            'avg_local_vector_accesses': float(avg_local_vec),
            'avg_remote_vector_accesses': float(avg_remote_vec),
            'avg_local_neighbor_accesses': float(avg_local_nbr),
            'avg_remote_neighbor_accesses': float(avg_remote_nbr),
            'search_time_s': search_time,
        }
        ctx.search_results['L_sweep'].append(entry)

        remote_ratio = (float(avg_remote_vec) + float(avg_remote_nbr)) / max(1,
            float(avg_local_vec) + float(avg_remote_vec) +
            float(avg_local_nbr) + float(avg_remote_nbr))

        print(f"  Recall@{K}: {recall:.4f}")
        print(f"  Avg hops: {avg_hops:.1f}, dist_comp: {avg_dist_comp:.0f}")
        print(f"  Avg latency: {avg_latency:.1f} us")
        print(f"  Remote ratio: {remote_ratio:.2%}")
        print(f"  Time: {search_time:.1f}s")

    # Save results as a per-run output. Split-layout configs write under
    # results/<dataset>/<experiment>_<timestamp>; legacy configs keep
    # runs/<experiment>/<stage>/<timestamp>.
    results_dir = build_run_dir(
        ctx.config, "search", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    results_file = results_dir / "search_results.json"
    with open(results_file, 'w') as f:
        json.dump(ctx.search_results, f, indent=2)
    print(f"\n  Search results saved to {results_file}")

    return ctx


# =============================================================================
# Colocation Stage
# =============================================================================

def stage_colocation(ctx: PipelineContext) -> PipelineContext:
    """Build co-location binary from graph checkpoint and partition data.

    Input:  graph checkpoint, partition data, base vectors
    Output: partitionX.coloc files, global_to_vectorid.bin, colocation_meta.json
    """
    coloc_cfg = ctx.config.get('colocation', {})
    if not coloc_cfg.get('enabled', False):
        print("  [SKIP] Colocation stage disabled")
        return ctx

    partition_cfg = ctx.config.get('partition', {})
    num_partitions = partition_cfg.get('num_partitions', 2)
    strategy_name = get_insert_strategy(ctx.config)

    experiment_dir = ctx.base_dir
    colocation_base_dir = artifact_colocation_dir(ctx.config)
    iteration = _extract_iteration_from_graph(ctx.graph_file)
    graph_variant = ctx.config.get('build', {}).get('graph_variant', None)
    coloc_subdir = graph_variant if graph_variant else f"iteration{iteration}"
    coloc_output_dir = build_colocation_dir(
        colocation_base_dir, num_partitions, strategy_name, variant=coloc_subdir,
    )

    # Check if already built
    meta_path = coloc_output_dir / "colocation_meta.json"
    if meta_path.exists() and ctx.config.get('pipeline', {}).get('skip_if_exists', True):
        print(f"  [SKIP] Co-location already built: {meta_path}")
        ctx.colocation_dir = str(coloc_output_dir)
        return ctx

    # Find inputs
    graph_file = ctx.graph_file
    tags_file = ctx.tags_file
    partition_dir = str(build_partition_dir(
        experiment_dir, num_partitions, strategy_name,
    ))

    base_file = ctx.base_file

    if not graph_file or not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    if not tags_file or not os.path.exists(tags_file):
        raise FileNotFoundError(f"Tags file not found: {tags_file}")
    if not os.path.exists(partition_dir):
        raise FileNotFoundError(f"Partition directory not found: {partition_dir}")

    build_cfg = ctx.config.get('build', {})
    R = build_cfg.get('R', 64)
    slack_factor = coloc_cfg.get('slack_factor', 1.3)
    alignment = coloc_cfg.get('alignment', 64)
    max_capacity_factor = coloc_cfg.get('max_capacity_factor', 1.0)

    # Import and run the builder
    builder_path = Path(__file__).parent / "build_colocation.py"
    if not builder_path.exists():
        raise FileNotFoundError(f"Co-location builder not found: {builder_path}")

    # Use subprocess to run the builder script
    import subprocess
    python = sys.executable

    cmd = [
        python, str(builder_path),
        "--graph-file", graph_file,
        "--tags-file", tags_file,
        "--vectors-file", base_file,
        "--partition-dir", partition_dir,
        "--output-dir", str(coloc_output_dir),
        "--R", str(R),
        "--slack-factor", str(slack_factor),
        "--alignment", str(alignment),
        "--max-capacity-factor", str(max_capacity_factor),
        "--data-type", ctx.data_type,
    ]

    print(f"  Running co-location builder...")
    print(f"  Command: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s):")
        print(result.stderr)
        raise RuntimeError("Co-location builder failed")

    print(f"  Co-location built in {elapsed:.1f}s")
    if result.stdout:
        for line in result.stdout.strip().split('\n')[-5:]:
            print(f"    {line}")

    ctx.colocation_dir = str(coloc_output_dir)
    return ctx


# =============================================================================
# Stage 3b: PQ BUILD (offline PQ codes for remote neighbor filtering)
# =============================================================================

def _compute_chunk_offsets(dim: int, num_chunks: int):
    """Compute chunk dimension offsets for (possibly uneven) PQ subdivision.

    Returns chunk_offsets: list of (num_chunks + 1) ints where chunk m spans
    dimensions [chunk_offsets[m], chunk_offsets[m+1]).
    """
    base_size = dim // num_chunks
    remainder = dim % num_chunks
    offsets = [0]
    for m in range(num_chunks):
        cs = base_size + (1 if m < remainder else 0)
        offsets.append(offsets[-1] + cs)
    return offsets


def _encode_all_vectors_numpy(vectors: np.ndarray, centroids_list: list,
                              chunk_offsets: list,
                              batch_size: int = 5_000_000) -> np.ndarray:
    """Encode vectors to PQ codes. Uses faiss if available, else numpy fallback.

    Args:
        vectors: (N, dim) array (or mmap)
        centroids_list: list of M arrays, each (256, chunk_size_m) float32
        chunk_offsets: [M+1] int list
        batch_size: number of vectors per batch

    Returns:
        codes: (N, M) uint8 array
    """
    N = vectors.shape[0]
    dim = vectors.shape[1]
    M = len(centroids_list)

    # Try faiss first (10-50x faster, C++/AVX)
    try:
        import faiss
        return _encode_all_vectors_faiss(vectors, centroids_list, chunk_offsets,
                                         batch_size, faiss)
    except ImportError:
        pass

    print("    [PQ encode] faiss not available, using numpy fallback (slow)")
    codes = np.zeros((N, M), dtype=np.uint8)
    cent_norms = [np.sum(centroids_list[m] ** 2, axis=1) for m in range(M)]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = vectors[start:end]
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        for m in range(M):
            sub_vecs = batch[:, chunk_offsets[m]:chunk_offsets[m + 1]]
            cents = centroids_list[m]
            dists = (np.sum(sub_vecs ** 2, axis=1, keepdims=True)
                     - 2.0 * sub_vecs @ cents.T
                     + cent_norms[m])
            codes[start:end, m] = np.argmin(dists, axis=1).astype(np.uint8)
        if (start // batch_size) % 10 == 0:
            print(f"    [PQ encode numpy] {end:,}/{N:,}")

    return codes


def _encode_all_vectors_faiss(vectors: np.ndarray, centroids_list: list,
                               chunk_offsets: list, batch_size: int,
                               faiss) -> np.ndarray:
    """Encode using faiss ProductQuantizer (C++/AVX accelerated)."""
    N = vectors.shape[0]
    dim = vectors.shape[1]
    M = len(centroids_list)

    # Build faiss PQ with our trained centroids
    # faiss PQ expects uniform chunk sizes; we handle uneven by setting centroids directly
    pq = faiss.ProductQuantizer(dim, M, 8)  # 8 bits per code

    # Set centroids: faiss expects (M, ksub, dsub) flattened as (M * ksub, dsub_max)
    # But faiss PQ with non-uniform dsub isn't directly supported.
    # So we use per-subquantizer encoding manually with faiss distance computation.

    # Check if chunk sizes are uniform
    chunk_sizes = [chunk_offsets[m + 1] - chunk_offsets[m] for m in range(M)]
    uniform = all(cs == chunk_sizes[0] for cs in chunk_sizes)

    if uniform:
        # Fast path: faiss native PQ
        dsub = chunk_sizes[0]
        pq = faiss.ProductQuantizer(dim, M, 8)

        # Set centroids into faiss PQ
        # faiss centroids layout: (M * ksub * dsub,) contiguous
        all_centroids = np.zeros((M * 256, dsub), dtype=np.float32)
        for m in range(M):
            all_centroids[m * 256:(m + 1) * 256] = centroids_list[m]
        faiss.copy_array_to_vector(all_centroids.ravel(), pq.centroids)

        print(f"    [PQ encode faiss] uniform dsub={dsub}, M={M}, encoding {N:,} vectors...")
        codes = np.zeros((N, M), dtype=np.uint8)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = np.ascontiguousarray(vectors[start:end], dtype=np.float32)
            batch_codes = pq.compute_codes(batch)  # (bs, M) uint8
            codes[start:end] = batch_codes
            if (start // batch_size) % 5 == 0:
                print(f"    [PQ encode faiss] {end:,}/{N:,}")
        return codes

    else:
        # Non-uniform chunks: encode each sub-quantizer separately with faiss IndexFlatL2
        print(f"    [PQ encode faiss] non-uniform chunks {chunk_sizes}, encoding per-subquantizer...")
        codes = np.zeros((N, M), dtype=np.uint8)

        for m in range(M):
            dsub = chunk_sizes[m]
            cents = np.ascontiguousarray(centroids_list[m], dtype=np.float32)  # (256, dsub)
            index = faiss.IndexFlatL2(dsub)
            index.add(cents)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                sub_vecs = np.ascontiguousarray(
                    vectors[start:end, chunk_offsets[m]:chunk_offsets[m + 1]],
                    dtype=np.float32)
                _, I = index.search(sub_vecs, 1)  # (bs, 1)
                codes[start:end, m] = I[:, 0].astype(np.uint8)

            print(f"    [PQ encode faiss] chunk {m}/{M} done (dsub={dsub})")

        return codes


def _parse_diskann_pivots(pivots_path: str, num_chunks: int, dim: int):
    """Parse DiskANN generate_pq_pivots output file.

    DiskANN pivot file layout (written by save_bin calls):
      Offset 0: save_bin<size_t>(cumul_bytes, 4, 1)
        → [4: i32, 1: i32, cumul_bytes[0..3]: uint64[4]]  (40 bytes, padded to 4096)
      Offset cumul_bytes[0] (=4096): save_bin<float>(centroids, 256, dim)
        → [256: i32, dim: i32, centroids: float[256*dim]]
      Offset cumul_bytes[1]: save_bin<float>(centroid, dim, 1)
        → [dim: i32, 1: i32, centroid: float[dim]]  (overall mean, unused)
      Offset cumul_bytes[2]: save_bin<uint32_t>(chunk_offsets, M+1, 1)
        → [M+1: i32, 1: i32, chunk_offsets: uint32[M+1]]

    Returns:
        chunk_offsets: list of (num_chunks + 1) ints
        centroids_list: list of M arrays, each (256, chunk_size_m) float32
    """
    import struct

    with open(pivots_path, 'rb') as f:
        # Read metadata header: [4: i32, 1: i32, cumul_bytes: u64[4]]
        hdr_npts, hdr_ndims = struct.unpack('<ii', f.read(8))
        assert hdr_npts == 4 and hdr_ndims == 1, \
            f"Unexpected metadata header: npts={hdr_npts}, ndims={hdr_ndims}"
        cumul_bytes = struct.unpack('<4Q', f.read(32))
        # cumul_bytes[0] = 4096 (METADATA_SIZE)
        # cumul_bytes[1] = after centroids
        # cumul_bytes[2] = after overall centroid
        # cumul_bytes[3] = after chunk_offsets

        # Read full-dimensional centroids at cumul_bytes[0]
        f.seek(cumul_bytes[0])
        c_npts, c_ndims = struct.unpack('<ii', f.read(8))
        assert c_npts == 256, f"Expected 256 centers, got {c_npts}"
        assert c_ndims == dim, f"Pivot dim {c_ndims} != expected {dim}"
        full_centroids = np.frombuffer(f.read(256 * dim * 4), dtype=np.float32).reshape(256, dim)

        # Skip overall centroid, read chunk_offsets at cumul_bytes[2]
        f.seek(cumul_bytes[2])
        co_npts, co_ndims = struct.unpack('<ii', f.read(8))
        assert co_npts == num_chunks + 1, \
            f"Expected {num_chunks + 1} chunk_offsets, got {co_npts}"
        chunk_offsets_raw = np.frombuffer(f.read((num_chunks + 1) * 4), dtype=np.uint32)
        chunk_offsets = chunk_offsets_raw.tolist()

    # Split full centroids into per-chunk centroids
    centroids_list = []
    for m in range(num_chunks):
        cent_m = full_centroids[:, chunk_offsets[m]:chunk_offsets[m + 1]].copy()
        centroids_list.append(np.ascontiguousarray(cent_m, dtype=np.float32))

    return chunk_offsets, centroids_list


def stage_pq_build(ctx: PipelineContext) -> PipelineContext:
    """Build PQ codebook and per-partition PQ codes for remote neighbor filtering.

    Uses DiskANN's native generate_pq_pivots (k-means++/Lloyd's) for codebook
    training, supporting uneven chunk sizes when dim % M != 0.

    Runs after colocation stage. For each partition, scans local nodes' adjacency
    lists to find the neighbor-reachable remote set, then encodes those vectors
    with Product Quantization.

    Input:  colocation files (partitionX.coloc, colocation_meta.json), base vectors
    Output: pq_codebook.bin, pq_all_codes.bin, pq_meta.json
    """
    import struct

    pq_cfg = get_pq_filter_config(ctx.config)
    if not pq_cfg.get('enabled', False):
        print("  [SKIP] PQ filter disabled in config")
        return ctx

    num_chunks = pq_cfg.get('num_chunks', 12)
    training_sample_size = pq_cfg.get('training_sample_size', 100000)

    # ---- Resolve output dir (dataset-level, outside colocation) ----
    # New layout: {experiment_base}/pq/M{num_chunks}/
    # PQ artifacts (codebook, all_codes) are gid-indexed and partition-agnostic,
    # so they live outside colocation and survive colocation rebuilds. This also
    # lets stage_pq_build run concurrently with stage_colocation.
    output_dir = Path(_resolve_workspace_path(
        str(build_pq_dir(ctx.base_dir, num_chunks))
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    pq_meta_path = output_dir / "pq_meta.json"
    pq_all_codes_path = output_dir / "pq_all_codes.bin"
    codebook_path = output_dir / "pq_codebook.bin"
    all_codes_path = pq_all_codes_path

    # Skip check — require both pq_meta.json AND pq_all_codes.bin (unified format)
    skip = ctx.config.get('pipeline', {}).get('skip_if_exists', True)
    force = pq_cfg.get('force', False)
    if skip and not force and pq_meta_path.exists() and pq_all_codes_path.exists():
        print(f"  [SKIP] PQ data already built: {pq_meta_path}")
        return ctx

    # ---- Locate colocation directory (optional) ----
    # When colocation_meta.json is present we use its `dim`/`entry_size`/... for
    # the reachability scan. When absent (e.g. pq_build running before or in
    # parallel with stage_colocation), we derive `dim` from dataset config and
    # skip the reachability scan — codebook + encoding do not depend on it.
    coloc_dir = getattr(ctx, 'colocation_dir', None)
    if not coloc_dir:
        partition_cfg = ctx.config.get('partition', {})
        num_partitions_cfg = partition_cfg.get('num_partitions', 4)
        strategy_name = get_insert_strategy(ctx.config)
        iteration = _extract_iteration_from_graph(ctx.graph_file)
        graph_variant = ctx.config.get('build', {}).get('graph_variant', None)
        coloc_subdir = graph_variant if graph_variant else f"iteration{iteration}"
        coloc_dir = str(build_colocation_dir(
            artifact_colocation_dir(ctx.config), num_partitions_cfg, strategy_name,
            variant=coloc_subdir,
        ))
    coloc_dir = Path(_resolve_workspace_path(coloc_dir))
    meta_path = coloc_dir / "colocation_meta.json"
    coloc_meta = None
    if meta_path.exists():
        with open(meta_path) as f:
            coloc_meta = json.load(f)

    if coloc_meta is not None:
        dim = coloc_meta['dim']
        entry_size = coloc_meta['entry_size']
        neighbor_offset = coloc_meta['neighbor_offset']
        degree_offset = coloc_meta['degree_offset']
        slack_R = coloc_meta['slack_R']
        num_partitions = coloc_meta['num_partitions']
        partitions = coloc_meta['partitions']
    else:
        # Derive dim from dataset config — reachability scan will be skipped.
        dataset_cfg = ctx.config.get('dataset', {})
        dim = int(dataset_cfg.get('dimension', 0))
        if dim <= 0:
            raise ValueError(
                "dataset.dimension must be set when colocation_meta.json is absent "
                "(stage_pq_build running before colocation)"
            )
        entry_size = neighbor_offset = degree_offset = slack_R = 0
        partition_cfg = ctx.config.get('partition', {})
        num_partitions = int(partition_cfg.get('num_partitions', 4))
        partitions = {}
        print(f"  [INFO] colocation_meta.json absent — dim={dim} from dataset config, "
              f"reachability scan will be skipped")

    # Compute chunk offsets (supports uneven: dim % num_chunks != 0)
    chunk_offsets = _compute_chunk_offsets(dim, num_chunks)

    # Load base vectors
    _resolve_dataset(ctx)
    base_file = _resolve_workspace_path(ctx.base_file)

    dataset_cfg = ctx.config.get('dataset', {})
    data_type_str = dataset_cfg.get('data_type', 'float32')
    load_dtype = {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}.get(
        data_type_str, np.float32)
    print(f"  Loading base vectors: {base_file} (dtype={data_type_str}) [mmap]")
    vectors = load_bin_data(base_file, dtype=load_dtype, mmap=True)
    N = vectors.shape[0]
    print(f"  {N:,} vectors, dim={dim}")

    # Build global_id → (partition, local_loc) lookup from global_to_vectorid.bin
    # If missing (e.g. deployed to DPUs, cleaned from NFS), fall back to
    # using first initial_size gids as valid — sufficient for train+encode.
    gvid_path = coloc_dir / "global_to_vectorid.bin"
    INVALID_VID = 0xFFFFFFFFFFFFFFFF
    LOCAL_MASK = (1 << 48) - 1
    if gvid_path.exists():
        gvid_raw = np.fromfile(str(gvid_path), dtype=np.uint64)
    else:
        print(f"  [INFO] {gvid_path.name} not present — using initial_size as valid range")
        gvid_raw = None

    # 1. Train PQ codebook via DiskANN's generate_pq_pivots
    chunk_sizes_str = ', '.join(str(chunk_offsets[m+1] - chunk_offsets[m]) for m in range(num_chunks))
    print(f"  Training PQ codebook: M={num_chunks}, chunk_sizes=[{chunk_sizes_str}]")
    t0 = time.time()

    # Sample training vectors from active set (those with valid VIDs)
    if gvid_raw is not None:
        valid_mask = gvid_raw != INVALID_VID
        valid_gids = np.where(valid_mask[:N])[0]
    else:
        initial_size = ctx.config.get('build', {}).get('initial_size', N)
        valid_gids = np.arange(min(initial_size, N), dtype=np.int64)
    sample_size = min(training_sample_size, len(valid_gids))
    rng = np.random.RandomState(42)
    sample_gids = rng.choice(valid_gids, sample_size, replace=False)
    train_vectors = np.ascontiguousarray(vectors[sample_gids], dtype=np.float32)

    # Train PQ codebook with faiss (per-chunk k-means, 256 centroids).
    # Uses chunk_offsets computed above — no DiskANN pivots file intermediate.
    import faiss
    centroids_list = []
    for m in range(num_chunks):
        dsub = chunk_offsets[m + 1] - chunk_offsets[m]
        sub = np.ascontiguousarray(
            train_vectors[:, chunk_offsets[m]:chunk_offsets[m + 1]],
            dtype=np.float32,
        )
        kmeans = faiss.Kmeans(dsub, 256, niter=15, verbose=False, seed=42)
        kmeans.train(sub)
        centroids_list.append(np.ascontiguousarray(kmeans.centroids, dtype=np.float32))

    print(f"  PQ codebook ready ({time.time() - t0:.1f}s)")

    # 3. Encode ALL vectors using numpy (batched, float32 conversion per-batch)
    t0 = time.time()
    all_codes = _encode_all_vectors_numpy(vectors, centroids_list, chunk_offsets)
    print(f"  Encoded {N:,} vectors in {time.time() - t0:.1f}s")

    # 4. Save codebook (new format with chunk_offsets)
    # Format: [M: u32][chunk_offsets: u32[M+1]][ksub: u32][centroids: float[variable]]
    with open(codebook_path, 'wb') as f:
        f.write(struct.pack('<I', num_chunks))
        np.array(chunk_offsets, dtype=np.uint32).tofile(f)
        f.write(struct.pack('<I', 256))  # ksub
        for m in range(num_chunks):
            centroids_list[m].astype(np.float32).tofile(f)
    codebook_size = os.path.getsize(codebook_path)
    print(f"  Saved codebook: {codebook_path} ({codebook_size / 1024:.1f} KB)")

    # 5. Save all_codes (global_id-indexed flat array — stable across insert/delete)
    # Format: [N: u32, M: u32, codes: u8[N * M]]
    with open(all_codes_path, 'wb') as f:
        f.write(struct.pack('<II', N, num_chunks))
        all_codes.astype(np.uint8).tofile(f)
    all_codes_size = all_codes.nbytes + 8
    print(f"  Saved all_codes: {all_codes_path} ({all_codes_size / 1024 / 1024:.1f} MB)")

    # 6. Profile remote neighbor reachability (for preload sizing at runtime)
    #    Vectorized scan: process neighbor VIDs in numpy batches.
    # Optional — skip when colocation is absent or partition.coloc / local_to_tag
    # files have not been deployed. total_remote_unique is a hint only.
    total_remote_unique = 0
    skip_reach = pq_cfg.get('skip_reachability', False)
    _coloc_present = coloc_meta is not None and all(
        (coloc_dir / partitions[str(p)]['file']).exists()
        for p in range(num_partitions)
    )
    if skip_reach or not _coloc_present:
        if coloc_meta is None:
            print("\n  [INFO] colocation absent — skipping reachability scan")
        elif not _coloc_present:
            print("\n  [INFO] partition.coloc files not on NFS — skipping reachability scan")
        else:
            print("\n  [INFO] skip_reachability=true — skipping reachability scan")
        _l2t_cache = {}
    else:
        # Pre-load all local_to_tag files
        _l2t_cache = {}
        for p_str in coloc_meta.get('local_to_tag_files', {}):
            l2t_file = coloc_dir / coloc_meta['local_to_tag_files'][p_str]
            _l2t_cache[int(p_str)] = np.fromfile(str(l2t_file), dtype=np.uint32)

    SCAN_BATCH = 200_000

    _reachability_enabled = bool(_l2t_cache)
    for my_p in range(num_partitions) if _reachability_enabled else []:
        p_meta = partitions[str(my_p)]
        coloc_file = coloc_dir / p_meta['file']
        num_active = p_meta['num_active']

        print(f"\n  Partition {my_p}: scanning {num_active:,} local nodes...")
        t0 = time.time()

        coloc_mm = np.memmap(str(coloc_file), dtype=np.uint8, mode='r')
        remote_seen = np.zeros(N, dtype=np.bool_)
        col_idx = np.arange(slack_R, dtype=np.uint32)[None, :]

        for loc_start in range(0, num_active, SCAN_BATCH):
            loc_end = min(loc_start + SCAN_BATCH, num_active)
            batch_n = loc_end - loc_start

            raw_2d = np.array(
                coloc_mm[loc_start * entry_size : loc_end * entry_size]
            ).reshape(batch_n, entry_size)

            # Degrees (uint32 at degree_offset)
            degrees = raw_2d[:, degree_offset:degree_offset + 4].copy().view(np.uint32).ravel()
            degrees = np.minimum(degrees, slack_R)

            # Neighbor VIDs (uint64 at neighbor_offset, max slack_R per node)
            nbr_vids = raw_2d[:, neighbor_offset:neighbor_offset + slack_R * 8] \
                .copy().view(np.uint64).reshape(batch_n, slack_R)

            # Valid: within degree and not INVALID_VID
            valid = (col_idx < degrees[:, None]) & (nbr_vids != INVALID_VID)

            # Remote: owner partition != my_p
            owners = (nbr_vids >> 48).astype(np.uint32)
            remote = valid & (owners != my_p)

            local_locs = nbr_vids & LOCAL_MASK

            # Resolve global IDs per owner partition
            for owner_p in range(num_partitions):
                if owner_p == my_p:
                    continue
                owner_mask = remote & (owners == owner_p)
                if not np.any(owner_mask):
                    continue
                locs = local_locs[owner_mask].astype(np.int64)
                l2t = _l2t_cache[owner_p]
                in_range = locs < len(l2t)
                tags_val = l2t[locs[in_range]]
                gids = tags_val[tags_val > 0].astype(np.int64) - 1
                if len(gids) > 0:
                    remote_seen[gids] = True

        n_remote = int(np.count_nonzero(remote_seen))
        total_remote_unique += n_remote
        print(f"    {n_remote:,} unique reachable remote vectors "
              f"(scan: {time.time() - t0:.1f}s)")
        del coloc_mm, remote_seen

    del _l2t_cache

    # 7. Save metadata
    build_cfg = ctx.config.get('build', {})
    pq_max_capacity = build_cfg.get('max_capacity', int(N * 1.1))
    try:
        base_stat = os.stat(base_file)
        base_mtime = int(base_stat.st_mtime)
        base_size = int(base_stat.st_size)
    except OSError:
        base_mtime = 0
        base_size = 0
    pq_meta = {
        'num_chunks': num_chunks,
        'chunk_offsets': chunk_offsets,
        'ksub': 256,
        'dim': dim,
        'total_vectors': int(N),
        'max_capacity': int(pq_max_capacity),
        'codebook_file': 'pq_codebook.bin',
        'all_codes_file': 'pq_all_codes.bin',
        'training_sample_size': int(sample_size),
        'training_seed': 42,
        'total_remote_unique': total_remote_unique,
        'base_file': str(base_file),
        'base_file_mtime': base_mtime,
        'base_file_size': base_size,
        'created_at': int(time.time()),
    }
    with open(pq_meta_path, 'w') as f:
        json.dump(pq_meta, f, indent=2)
    print(f"\n  PQ meta saved: {pq_meta_path}")
    if total_remote_unique > 0:
        print(f"  Total unique remote vectors: {total_remote_unique:,}")

    return ctx


# =============================================================================
# Stage Registry
# =============================================================================

from pipeline.engine_stage import stage_distributed_engine

STAGE_ORDER = ["build", "partition", "colocation",
               "pq_build", "gt", "distributed_engine", "search"]

STAGE_MAP = {
    "build": stage_build,
    "partition": stage_partition,
    "colocation": stage_colocation,
    "distributed_engine": stage_distributed_engine,
    "pq_build": stage_pq_build,
    "gt": stage_gt,
    "search": stage_search,
}

# Stage dependency declarations (for validation)
STAGE_DEPENDENCIES = {
    "partition": ["build"],
    "colocation": ["build", "partition"],
    "distributed_engine": ["build", "partition", "colocation"],
    # pq_build only needs base vectors + dim. colocation is optional and only
    # used for the (informational) reachability scan. Listing build as the sole
    # hard dep lets pq_build run in parallel with partition + colocation.
    "pq_build": ["build"],
    "gt": ["build"],
    "search": ["build", "partition", "colocation", "gt"],
}


def validate_stage_order(stages):
    """Validate that stage dependencies are met. Returns list of warnings."""
    warnings = []
    seen = set()
    for s in stages:
        deps = STAGE_DEPENDENCIES.get(s, [])
        missing = [d for d in deps if d not in seen]
        if missing:
            warnings.append(
                f"Stage '{s}' depends on {missing} which are not in prior stages. "
                f"Ensure their outputs exist from a previous run."
            )
        seen.add(s)
    return warnings
