"""Engine-pipeline stage handoff and search GT precompute helpers."""

from __future__ import annotations

import json
import os
import re as _re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_scripts_dir = _project_root / 'scripts'
for _p in [str(_project_root), str(_scripts_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_loader import load_bin_data, load_fbin
from common.partition_paths import (
    get_insert_strategy,
    partition_dir as build_partition_dir,
    colocation_dir as build_colocation_dir,
)


def _get_dtype(data_type: str):
    return {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}.get(
        data_type, np.float32)


def _resolve_workspace_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    if os.path.exists(path):
        return path
    host_base = str(Path(__file__).resolve().parent.parent)
    if path.startswith("/workspace/"):
        host_path = os.path.join(host_base, path[len("/workspace/"):])
        if os.path.exists(host_path):
            return host_path
    prefix = host_base + "/"
    if path.startswith(prefix):
        container_path = "/workspace/" + path[len(prefix):]
        if os.path.exists(container_path):
            return container_path
    return path


def _extract_iteration_from_graph(graph_file: Optional[str]) -> int:
    if not graph_file:
        return 0
    basename = Path(graph_file).name
    m = _re.match(r'^iteration(\d+)$', basename)
    return int(m.group(1)) if m else 0


def stage_distributed_engine(ctx: PipelineContext) -> PipelineContext:
    """Engine pipeline handoff — runs on DPU nodes via run_pipeline.sh.

    This is a DISTRIBUTED stage: the orchestrator (run_pipeline.sh) launches
    node_worker.py with --stage distributed_engine on each DPU node.

    Before dispatching to DPU nodes, this stub precomputes GT for any search
    operations in the workload. The GT files are saved to NFS so DPU nodes
    can read them directly for recall computation.
    """
    print("  distributed_engine is a DISTRIBUTED stage (runs on DPU nodes)")
    print("  → Launched by run_pipeline.sh, not run_pipeline.py")

    # Precompute GT for search ops in workload
    if ctx.config.get('gt', {}).get('skip_precompute', False):
        print("  [GT precompute] Skipped (gt.skip_precompute=true)")
    else:
        try:
            _precompute_search_gt(ctx)
        except Exception as e:
            print(f"  [WARN] GT precompute failed: {e}")
            import traceback
            traceback.print_exc()

    return ctx


def _host_to_container_path(host_path: str) -> str:
    """Convert host path to /workspace/ container path."""
    host_base = str(Path(__file__).resolve().parent.parent)
    if host_path.startswith(host_base):
        return "/workspace/" + host_path[len(host_base) + 1:]
    return host_path


_DATA_TYPE_TO_CPP = {'float32': 'float', 'uint8': 'uint8', 'int8': 'int8'}
_GT_BINARY = "/workspace/diskann/build/apps/compute_groundtruth_batched"
_BUILD_CONTAINER = "vecdb-build"


def _precompute_search_gt(ctx: PipelineContext):
    """Simulate the workload to predict active sets at each search op,
    then precompute GT using C++ binary inside vecdb-build container.

    GT files are saved as:
        <experiment>/gt_cache/gt_K{K}_Q{Q}_N{N}_h{hash8}.npy

    A manifest JSON maps (seq_index -> gt_filename) for the engine to load.
    """
    import hashlib
    from generate_gt import save_tags_file, load_gt_bin
    from pipeline.danns_types import NearestCentroidPolicy

    config = ctx.config
    engine_cfg = config.get('distributed_engine', {})
    workload_specs = engine_cfg.get('workload', [])
    if not workload_specs:
        print("  [GT precompute] No workload defined, skipping")
        return

    # Check if any search ops exist
    search_indices = [i for i, spec in enumerate(workload_specs) if spec.get('op') == 'search']
    if not search_indices:
        print("  [GT precompute] No search ops in workload, skipping")
        return

    search_cfg = config.get('search', {})
    default_K = search_cfg.get('K', 10)
    default_num_queries = search_cfg.get('num_queries', 1000)

    # Dataset paths (container paths — used by C++ binary inside vecdb-build)
    dataset_cfg = config.get('dataset', {})
    data_root = dataset_cfg.get('data_root', '')
    container_base_file = os.path.join(data_root, dataset_cfg.get('base_file', ''))
    container_query_file = os.path.join(data_root, dataset_cfg.get('query_file', ''))
    data_type_cpp = _DATA_TYPE_TO_CPP.get(ctx.data_type, 'float')
    max_capacity = config.get('build', {}).get('max_capacity',
                   config.get('build', {}).get('initial_size', 0))

    # Load colocation metadata to get initial active set
    partition_cfg = config.get('partition', {})
    num_partitions = partition_cfg.get('num_partitions', 4)
    strategy_name = get_insert_strategy(config)
    coloc_dir = build_colocation_dir(ctx.base_dir, num_partitions, strategy_name)

    # Find colocation directory (graph_variant or legacy iteration)
    iteration = _extract_iteration_from_graph(ctx.graph_file) if ctx.graph_file else 0
    graph_variant = config.get('build', {}).get('graph_variant', None)
    coloc_subdir = graph_variant if graph_variant else f"iteration{iteration}"
    coloc_iter_dir = coloc_dir / coloc_subdir

    # Early exit: if manifest already exists with all search ops covered
    # AND was generated from the same colocation directory (graph variant)
    gt_cache_dir = ctx.base_dir / "gt_cache"
    # Use variant-specific manifest so different graph_variants don't collide
    manifest_suffix = f"_{coloc_subdir}" if graph_variant else ""
    manifest_path = gt_cache_dir / f"search_gt_manifest{manifest_suffix}.json"
    if manifest_path.exists():
        with open(manifest_path) as _mf:
            existing_manifest = json.load(_mf)
        manifest_coloc = existing_manifest.get("_coloc_dir", "")
        current_coloc = str(coloc_iter_dir)
        if manifest_coloc and manifest_coloc != current_coloc:
            print(f"  [GT precompute] Manifest coloc mismatch: "
                  f"{manifest_coloc} != {current_coloc}, recomputing")
        else:
            all_covered = all(
                str(i) in existing_manifest and (gt_cache_dir / existing_manifest[str(i)]).exists()
                for i in search_indices
            )
            if all_covered:
                print(f"  [GT precompute] All {len(search_indices)} search ops already have GT cached, skipping")
                return
    # --- Fast path: search-only workload with no insert/delete ---
    # If workload is only search ops (no insert/delete that change active set),
    # try to reuse existing GT files from gt_cache/ by variant name + shape matching.
    # This avoids loading 400M+ tag_to_local entries into Python dicts.
    has_mutations = any(s.get('op') in ('insert', 'delete') for s in workload_specs)
    if not has_mutations and gt_cache_dir.exists():
        manifest: Dict[str, str] = {}
        all_matched = True
        for seq_idx in search_indices:
            spec = workload_specs[seq_idx]
            K = spec.get('K', default_K)
            num_queries = spec.get('num_queries', default_num_queries)

            found = None
            # 1. Try variant-specific GT file first (e.g., gt_deep500M_cp4.npy)
            if graph_variant:
                for candidate in sorted(gt_cache_dir.glob(f"gt_*{graph_variant}*.npy")):
                    try:
                        arr = np.load(str(candidate), mmap_mode='r')
                        if arr.shape == (num_queries, K):
                            found = candidate
                            break
                    except Exception:
                        continue
                # Also try variant without baseline_/lits_ prefix
                # e.g., graph_variant="baseline_cp4" → also try "cp4"
                if found is None and '_' in graph_variant:
                    suffix = graph_variant.split('_', 1)[1]  # "cp4"
                    for candidate in sorted(gt_cache_dir.glob(f"gt_*{suffix}*.npy")):
                        try:
                            arr = np.load(str(candidate), mmap_mode='r')
                            if arr.shape == (num_queries, K):
                                found = candidate
                                break
                        except Exception:
                            continue

            # 2. No variant match — do NOT fall back to arbitrary shape match
            #    (different variants can have same shape but different active sets)

            if found:
                manifest[str(seq_idx)] = found.name
                print(f"  [GT precompute] seq {seq_idx}: reusing {found.name} "
                      f"(variant={graph_variant}, shape {num_queries}×{K})")
            else:
                all_matched = False
                print(f"  [GT precompute] seq {seq_idx}: no variant-matched GT found "
                      f"for {graph_variant}, falling back to full computation")
                break

        if all_matched and manifest:
            manifest["_coloc_dir"] = str(coloc_iter_dir)
            os.makedirs(gt_cache_dir, exist_ok=True)
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"  [GT precompute] Search-only workload — all GT found, manifest saved: "
                  f"{manifest_path}")
            return

    # --- Full path: workload has insert/delete, need active set simulation ---
    meta_path = coloc_iter_dir / "colocation_meta.json"

    if not meta_path.exists():
        print(f"  [GT precompute] Colocation meta not found: {meta_path}")
        return

    with open(meta_path) as f:
        coloc_meta = json.load(f)

    # Load tag_to_local for all partitions → initial active tags
    tag_to_local_all: Dict[int, Dict[int, int]] = {}
    t2l_files = coloc_meta.get('tag_to_local_files', {})
    for p_str, filename in t2l_files.items():
        p = int(p_str)
        filepath = coloc_iter_dir / filename
        if filepath.exists():
            raw = np.fromfile(str(filepath), dtype=np.uint32)
            table = raw[1:]  # skip max_tag header
            tag_to_local_all[p] = {tag: int(table[tag]) for tag in range(len(table))
                                    if table[tag] != 0xFFFFFFFF}

    # Build initial active set as partition -> dict of tags (preserving insertion order).
    # Mirrors tag_to_local dict structure in distributed_engine.py so that
    # _prepare_delete_tags() random selection produces the same results.
    # tag_to_local is loaded with tags in ascending order (range(len(table))).
    tag_to_part_sim: Dict[int, Dict[int, bool]] = {
        p: {tag: True for tag in t2l.keys()} for p, t2l in tag_to_local_all.items()
    }

    # Insert cursor: starts from max tag across all partitions
    all_tags = set()
    for tags_dict in tag_to_part_sim.values():
        all_tags.update(tags_dict.keys())
    max_tag = max(all_tags) if all_tags else 0
    next_insert_gid = max_tag  # gid = max_tag; next tag = max_tag + 1

    # Load centroids for partition assignment (used only for insert simulation)
    centroids_file = build_partition_dir(
        ctx.base_dir, num_partitions, strategy_name,
    ) / "centroids.fbin"
    centroids = load_fbin(str(centroids_file)) if centroids_file.exists() else None

    # For insert simulation, load base vectors via C++ is not needed —
    # we only need to know which partition new points go to.
    # Load a small mmap for centroid assignment if insert ops exist.
    _insert_vectors = None
    has_insert_ops = any(s.get('op') == 'insert' for s in workload_specs)
    if has_insert_ops and centroids is not None:
        # Resolve base file to host path for mmap loading
        host_base_file = _resolve_workspace_path(container_base_file)
        if host_base_file and os.path.exists(host_base_file):
            dtype = _get_dtype(ctx.data_type)
            _insert_vectors = load_bin_data(host_base_file, dtype=dtype)
        else:
            print(f"  [GT precompute] Base file not accessible on host for insert sim, "
                  f"assigning inserts round-robin")

    policy = NearestCentroidPolicy()
    delete_seed_counter = 0

    total_active = sum(len(d) for d in tag_to_part_sim.values())
    print(f"  [GT precompute] Initial active: {total_active} tags, "
          f"cursor at gid={next_insert_gid}, {len(search_indices)} search ops")
    print(f"  [GT precompute] Using C++ binary via docker exec {_BUILD_CONTAINER}")

    # Output directory
    gt_cache_dir = ctx.base_dir / "gt_cache"
    os.makedirs(gt_cache_dir, exist_ok=True)

    # Container path for gt_cache
    container_gt_cache = _host_to_container_path(str(gt_cache_dir))

    # Simulate workload and compute GT at each search op
    manifest: Dict[str, str] = {}  # "seq" -> gt_filename
    computed_hashes: Dict[str, str] = {}  # hash -> gt_filename (dedup)

    for seq, spec in enumerate(workload_specs):
        op = spec.get('op', '')

        if op == 'insert':
            count = spec.get('count', 0)
            start_gid = next_insert_gid
            end_gid = min(start_gid + count, max_capacity)
            if centroids is not None and end_gid > start_gid:
                if _insert_vectors is not None:
                    new_points = _insert_vectors[start_gid:end_gid]
                    partition_ids = policy.assign_batch(new_points, centroids)
                    for idx, gid in enumerate(range(start_gid, end_gid)):
                        tag = gid + 1
                        p = int(partition_ids[idx])
                        tag_to_part_sim[p][tag] = True
                else:
                    # Round-robin fallback when base file not accessible
                    for gid in range(start_gid, end_gid):
                        tag = gid + 1
                        p = gid % num_partitions
                        tag_to_part_sim[p][tag] = True
            next_insert_gid = end_gid

        elif op == 'delete':
            count = spec.get('count', 0)
            # Build all_active list in same order as _prepare_delete_tags():
            # iterate partitions in dict key order, then tags in dict iteration order
            all_active = []
            for p in sorted(tag_to_part_sim.keys()):
                for tag in tag_to_part_sim[p]:
                    all_active.append((p, tag))

            seed = spec.get('seed', 1000 + delete_seed_counter)
            rng = np.random.RandomState(seed)
            if len(all_active) > count:
                indices = rng.choice(len(all_active), size=count, replace=False)
                to_delete = [all_active[i] for i in indices]
            else:
                to_delete = all_active
            for p, tag in to_delete:
                tag_to_part_sim[p].pop(tag, None)
            delete_seed_counter += 1

        elif op == 'search':
            K = spec.get('K', default_K)
            num_queries = spec.get('num_queries', default_num_queries)

            # Active global IDs = tag - 1 for all tags across all partitions
            active_global_ids = sorted(
                tag - 1 for tags_dict in tag_to_part_sim.values() for tag in tags_dict)
            active_locs = np.array(active_global_ids, dtype=np.int64)

            # Hash the active set for dedup + filename
            hash_input = active_locs.tobytes()
            h = hashlib.sha256(hash_input).hexdigest()[:8]
            gt_filename = f"gt_K{K}_Q{num_queries}_N{len(active_locs)}_h{h}.npy"

            if h in computed_hashes:
                gt_filename = computed_hashes[h]
                print(f"  [GT precompute] seq {seq}: reusing {gt_filename}")
            else:
                gt_path = gt_cache_dir / gt_filename
                if gt_path.exists():
                    print(f"  [GT precompute] seq {seq}: already exists {gt_filename}")
                else:
                    # Scan gt_cache/ for any existing .npy with matching shape (Q, K)
                    # This catches manually pre-computed GT files like gt_deep500M_cp4.npy
                    _found_existing = None
                    for _candidate in sorted(gt_cache_dir.glob("gt_*.npy")):
                        try:
                            _arr = np.load(str(_candidate), mmap_mode='r')
                            if _arr.shape == (num_queries, K):
                                _found_existing = _candidate
                                break
                        except Exception:
                            continue
                    if _found_existing is not None:
                        # Symlink to hash-based name for future lookups
                        gt_path.symlink_to(_found_existing.resolve())
                        print(f"  [GT precompute] seq {seq}: found existing "
                              f"{_found_existing.name} → {gt_filename}")
                    else:
                        print(f"  [GT precompute] seq {seq}: computing GT for "
                              f"{num_queries} queries × {len(active_locs)} active, K={K}...")

                        # Write tags file to gt_cache (NFS-shared with container)
                        tags = (active_locs + 1).astype(np.uint32)  # tag = global_id + 1
                        tags_host = gt_cache_dir / f"_temp_tags_{h}.bin"
                        save_tags_file(tags, str(tags_host))

                        # Container paths for temp files
                        c_tags = f"{container_gt_cache}/_temp_tags_{h}.bin"
                        c_gt_bin = f"{container_gt_cache}/_temp_gt_{h}.bin"

                        # Run C++ binary inside vecdb-build container (MKL-accelerated)
                        cmd = [
                            "docker", "exec",
                            "-e", "OMP_NUM_THREADS=64",
                            _BUILD_CONTAINER,
                            _GT_BINARY,
                            "--data_type", data_type_cpp,
                            "--dist_fn", "l2",
                            "--base_file", container_base_file,
                            "--query_file", container_query_file,
                            "--tags_file", c_tags,
                            "--gt_file", c_gt_bin,
                            "--K", str(K),
                        ]
                        t0 = time.time()
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                        elapsed = time.time() - t0

                        if result.returncode != 0:
                            print(f"  [GT precompute] seq {seq}: C++ GT FAILED "
                                  f"(exit {result.returncode})")
                            if result.stderr:
                                for line in result.stderr.strip().split('\n')[-5:]:
                                    print(f"    {line}")
                            if result.stdout:
                                for line in result.stdout.strip().split('\n')[-5:]:
                                    print(f"    {line}")
                            # Cleanup and skip
                            tags_host.unlink(missing_ok=True)
                            continue

                        if result.stdout:
                            for line in result.stdout.strip().split('\n')[-5:]:
                                print(f"    {line}")

                        # Convert .bin → .npy (host side)
                        gt_bin_host = gt_cache_dir / f"_temp_gt_{h}.bin"
                        if gt_bin_host.exists():
                            ids, _ = load_gt_bin(str(gt_bin_host))  # (Q, K) uint32 tag IDs
                            gt_locs = (ids.astype(np.int64) - 1)    # tag → global_id
                            np.save(str(gt_path), gt_locs)
                            print(f"  [GT precompute] seq {seq}: saved {gt_filename} "
                                  f"({elapsed:.1f}s, shape={gt_locs.shape})")
                        else:
                            print(f"  [GT precompute] seq {seq}: GT binary output not found")

                        # Cleanup temp files
                        tags_host.unlink(missing_ok=True)
                        gt_bin_host.unlink(missing_ok=True)

                computed_hashes[h] = gt_filename

            manifest[str(seq)] = gt_filename

        # consolidate, checkpoint: no change to active set

    # Save manifest with colocation source for variant validation
    manifest["_coloc_dir"] = str(coloc_iter_dir)
    manifest_path = gt_cache_dir / f"search_gt_manifest{manifest_suffix}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  [GT precompute] Manifest saved: {manifest_path}")
    print(f"  [GT precompute] {len(manifest) - 1} search ops, "
          f"{len(computed_hashes)} unique GT files")
