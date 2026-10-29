#!/usr/bin/env python3
"""
Offline co-location array builder.

Converts DiskANN variable-length graph + vectors + partition data into
fixed-size co-location binary files suitable for RDMA random access.

Entry layout (per node):
    [VectorID:      8B  uint64   partition[16bit]|tag[48bit]]
    [RawVector:     dim×4B float32]
    [Degree:        4B  uint32]
    [NeighborList:  slack_R×8B uint64  (VectorID format)]
    [Padding:       to 64B alignment]

Usage:
    python3 build_colocation.py \\
        --graph-file  <path-to-graph-checkpoint> \\
        --tags-file   <path-to-tags-file> \\
        --vectors-file <path-to-base.fbin> \\
        --partition-dir <path-to-partition/{N}_partitions/> \\
        --output-dir  <output-directory> \\
        [--R 64] [--slack-factor 1.3] [--alignment 64]
"""

import argparse
import json
import math
import numpy as np
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Path setup
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
_scripts_dir = _project_root / "scripts"
for _p in [str(_this_dir), str(_project_root), str(_scripts_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_loader import (load_diskann_graph, load_bin_data, load_fbin,
                                get_bin_metadata)


# =============================================================================
# VectorID encoding/decoding
# =============================================================================

PARTITION_BITS = 16
LOCAL_BITS = 48
LOCAL_MASK = (1 << LOCAL_BITS) - 1
INVALID_VID = 0xFFFFFFFFFFFFFFFF


def encode_vector_id(partition: int, local_loc: int) -> int:
    """Encode (partition, local_loc) into a uint64 VectorID.

    Graph edges store partition|local_loc for direct resolution.
    """
    return (partition << LOCAL_BITS) | (local_loc & LOCAL_MASK)


def encode_entry_vid(partition: int, tag: int) -> int:
    """Encode (partition, tag) into a uint64 entry identity VID.

    Stored in the entry's VID field (offset 0) for global_id recovery.
    tag is DiskANN's 1-based tag value (global_id + 1).
    """
    return (partition << LOCAL_BITS) | (tag & LOCAL_MASK)


def decode_vector_id(vector_id: int) -> Tuple[int, int]:
    """Decode VectorID into (partition, local_loc)."""
    partition = vector_id >> LOCAL_BITS
    local_loc = vector_id & LOCAL_MASK
    return partition, local_loc


# =============================================================================
# Entry layout calculator
# =============================================================================

class EntryLayout:
    """Calculate byte offsets and sizes for co-location entry."""

    _ELEMENT_SIZE = {'float32': 4, 'int8': 1, 'uint8': 1}

    def __init__(self, dim: int, R: int, slack_factor: float = 1.3,
                 alignment: int = 64, data_type: str = 'float32'):
        self.dim = dim
        self.R = R
        self.slack_factor = slack_factor
        self.slack_R = math.ceil(slack_factor * R)
        self.alignment = alignment
        self.data_type = data_type
        self.element_size = self._ELEMENT_SIZE.get(data_type, 4)

        # Offsets
        self.vector_id_offset = 0
        self.vector_id_size = 8  # uint64

        self.vector_offset = self.vector_id_offset + self.vector_id_size
        self.vector_size = dim * self.element_size  # native dtype

        self.degree_offset = self.vector_offset + self.vector_size
        self.degree_size = 4  # uint32

        self.neighbor_offset = self.degree_offset + self.degree_size
        self.neighbor_size = self.slack_R * 8  # uint64 per neighbor

        self.raw_entry_size = self.neighbor_offset + self.neighbor_size

        # 64B-aligned
        self.entry_size = self._align(self.raw_entry_size, alignment)
        self.padding_size = self.entry_size - self.raw_entry_size

    @staticmethod
    def _align(size: int, alignment: int) -> int:
        return ((size + alignment - 1) // alignment) * alignment

    def __repr__(self):
        return (f"EntryLayout(dim={self.dim}, R={self.R}, slack_R={self.slack_R}, "
                f"raw={self.raw_entry_size}B, aligned={self.entry_size}B)")

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "R": self.R,
            "slack_factor": self.slack_factor,
            "slack_R": self.slack_R,
            "alignment": self.alignment,
            "entry_size": self.entry_size,
            "raw_entry_size": self.raw_entry_size,
            "vector_id_offset": self.vector_id_offset,
            "vector_offset": self.vector_offset,
            "degree_offset": self.degree_offset,
            "neighbor_offset": self.neighbor_offset,
            "element_size": self.element_size,
            "data_type": self.data_type,
        }


# =============================================================================
# Main builder  (memory-optimized: mmap vectors, flat graph, numpy mappings)
# =============================================================================

UNMAPPED = np.uint32(0xFFFFFFFF)


class ColocationBuilder:
    """Build co-location binary files from DiskANN graph + partition data.

    Memory optimizations vs. the original implementation:
      1. Vectors are memory-mapped (mmap) instead of loaded into RAM.
      2. Graph is loaded in flat-array mode (~3x less memory than list mode).
      3. ID mappings use compact numpy arrays instead of Python dicts.
      4. Frozen-point vector is loaded by seek (1 vector) not full file.
      5. Centroid neighbor search is batched to limit peak allocation.
    """

    _DTYPE_MAP = {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}

    def __init__(self, graph_file: str, tags_file: str, vectors_file: str,
                 partition_dir: str, R: int = 64, slack_factor: float = 1.3,
                 alignment: int = 64, max_capacity_factor: float = 1.0,
                 data_type: str = 'float32', tag_indexed: bool = False):
        self.graph_file = graph_file
        self.tags_file = tags_file
        self.vectors_file = vectors_file
        self.partition_dir = Path(partition_dir)
        self.R = R
        self.slack_factor = slack_factor
        self.alignment = alignment
        self.max_capacity_factor = max_capacity_factor
        self.data_type = data_type
        self.tag_indexed = tag_indexed

        # --- Populated during build ---
        # Data (vectors is mmap, graph is flat arrays)
        self.vectors = None           # np.memmap (total_vectors, dim)
        self.adj_flat = None          # np.ndarray uint32 — flat edge array
        self.adj_offsets = None       # np.ndarray int64  — per-node offset into adj_flat
        self.adj_degrees = None       # np.ndarray int32  — per-node degree
        self.num_graph_nodes = 0
        self.graph_meta = None
        self.tags = None              # np.ndarray uint32

        # Partition info
        self.partition_ids = None     # np.ndarray uint32 (total_vectors,)
        self.num_partitions = 0
        self.layout = None

        # Numpy-based mappings (replace Python dicts)
        self.gid_to_loc = None        # np.ndarray uint32 (total_vectors,)  global_id → graph location
        self.gid_to_local_loc = None  # np.ndarray uint32 (total_vectors,)  global_id → local_loc
        self.partition_members = {}   # partition → np.ndarray of sorted global_ids

        self.partition_sizes = {}
        self.partition_capacities = {}
        self.medoids = {}

        # Centroids and frozen point
        self.centroids = None
        self.frozen_vector = None
        self.frozen_neighbors = None
        self.frozen_location = -1
        self.frozen_degree = 0
        self.frozen_local_locs = {}
        self.centroid_local_locs = {}

    # -----------------------------------------------------------------
    # Graph helper (flat-array access)
    # -----------------------------------------------------------------

    def _get_neighbors(self, location: int) -> np.ndarray:
        """Get neighbor array for a graph node."""
        if location >= self.num_graph_nodes:
            return np.array([], dtype=np.uint32)
        deg = int(self.adj_degrees[location])
        if deg == 0:
            return np.array([], dtype=np.uint32)
        off = int(self.adj_offsets[location])
        return self.adj_flat[off:off + deg]

    # -----------------------------------------------------------------
    # Build pipeline
    # -----------------------------------------------------------------

    def build(self, output_dir: str, partitions: list = None):
        """Main build pipeline.

        Args:
            output_dir: Output directory for .coloc files
            partitions: List of partition IDs to build (default: all).
                        Use [0], [1] etc. to build one at a time for memory savings.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        print("=" * 70)
        print("  CO-LOCATION ARRAY BUILDER  (memory-optimized)")
        print("=" * 70)

        # Step 1: Load all data
        self._load_data()

        # Step 2: Build ID mappings
        self._build_mappings()

        # Step 3: Create entry layout
        dim = self.vectors.shape[1]
        self.layout = EntryLayout(dim, self.R, self.slack_factor, self.alignment,
                                  data_type=self.data_type)
        print(f"\n  Layout: {self.layout}")

        # Step 4: Write co-location files
        if partitions is None:
            partitions = list(range(self.num_partitions))
        t_write_start = time.perf_counter()
        for p in partitions:
            self._write_partition(p, output_path)
        print(f"\n  Partitions {partitions} written in "
              f"{time.perf_counter() - t_write_start:.2f}s")

        # Step 5: Write global_to_vectorid lookup table
        self._write_global_to_vectorid(output_path)

        # Step 5b: Write tag_to_local lookup tables
        self._write_tag_to_local(output_path)

        # Step 5c: Write local_to_tag lookup tables (reverse mapping)
        self._write_local_to_tag(output_path)

        # Step 6: Write metadata
        self._write_metadata(output_path)

        elapsed = time.perf_counter() - t0
        print(f"\n  Total build time: {elapsed:.2f}s")
        print("=" * 70)

    # -----------------------------------------------------------------
    # Step 1: Load data  (mmap vectors, flat graph, seek-based frozen)
    # -----------------------------------------------------------------

    def _load_data(self):
        """Load vectors, graph, tags, and partition assignments."""
        print("\n  Loading data...")

        # Vectors — memory-mapped (no RAM allocation)
        t0 = time.perf_counter()
        load_dtype = self._DTYPE_MAP.get(self.data_type, np.float32)
        self.vectors = load_bin_data(self.vectors_file, dtype=load_dtype,
                                     mmap=True)
        if self.tag_indexed:
            # 800m-style files: index 0 = mean/frozen vector, data starts at 1.
            # Shift by 1 so vectors[gid] = original_file[gid+1] = file[tag].
            self.vectors = self.vectors[1:]
            print(f"    [tag-indexed] Shifted vectors by +1 "
                  f"(skipped mean vector at index 0)")
        total_vectors = len(self.vectors)
        print(f"    Vectors: {self.vectors.shape} [mmap] "
              f"({time.perf_counter() - t0:.2f}s)")

        # Graph — flat-array mode (~3x less memory than list mode). mmap=True
        # maps the edge data as reclaimable page cache so 400M-scale graphs
        # (~102GB) don't blow past node RAM during load and swap-thrash.
        t0 = time.perf_counter()
        (self.adj_flat, self.adj_offsets, self.adj_degrees), self.graph_meta = \
            load_diskann_graph(self.graph_file, flat=True, mmap=True)
        self.num_graph_nodes = len(self.adj_degrees)
        print(f"    Graph: {self.num_graph_nodes} nodes, "
              f"max_degree={self.graph_meta['max_degree']} [flat] "
              f"({time.perf_counter() - t0:.2f}s)")

        # Tags
        t0 = time.perf_counter()
        with open(self.tags_file, 'rb') as f:
            header = np.fromfile(f, dtype=np.uint32, count=2)
            num_tags = int(header[0])
            self.tags = np.fromfile(f, dtype=np.uint32, count=num_tags)
        print(f"    Tags: {len(self.tags)} entries "
              f"({time.perf_counter() - t0:.2f}s)")

        # Partition metadata
        meta_file = self.partition_dir / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Partition metadata not found: {meta_file}")

        with open(meta_file) as f:
            part_meta = json.load(f)

        # Indexing contract: partition_ids.npy MUST be gid-indexed and
        # exclude the frozen point. Older metadata may omit these keys
        # (treat missing as 'gid'/True for backward compatibility), but
        # any explicit contradiction is fatal — that's exactly the class
        # of bug this contract is here to catch.
        indexing = part_meta.get('indexing', 'gid')
        if indexing != 'gid':
            raise ValueError(
                f"Partition metadata has indexing='{indexing}', but "
                f"build_colocation requires gid-indexed partition_ids. "
                f"Regenerate partition stage. ({meta_file})"
            )
        frozen_excluded = part_meta.get('frozen_excluded', True)
        if not frozen_excluded:
            raise ValueError(
                f"Partition metadata has frozen_excluded=False. "
                f"Partition stage must leave frozen point out of "
                f"partition_ids.npy. ({meta_file})"
            )
        strategy = part_meta.get('insert_strategy', 'nearest_centroid')
        print(f"    Indexing contract OK: gid, frozen_excluded, "
              f"insert_strategy={strategy}")

        self.num_partitions = part_meta['num_partitions']

        # Load medoids from metadata
        medoid_list = part_meta.get('medoids', [])
        for i, m in enumerate(medoid_list):
            self.medoids[i] = int(m)
        print(f"    Medoids: {self.medoids}")

        # Partition assignments — prefer partition_ids.npy (canonical)
        t0 = time.perf_counter()
        npy_path = self.partition_dir / "partition_ids.npy"
        if npy_path.exists():
            all_partition_ids = np.load(str(npy_path))
            if self.tag_indexed:
                # 800m files: both vectors and partition_ids are position-indexed.
                # pids[i] = partition for base_800m[i].
                # We need partition for base_800m[tag] = pids[tag] = pids[gid+1].
                # Shift by +1 so pids[gid] = original[gid+1] = original[tag].
                self.partition_ids = all_partition_ids[1:total_vectors + 1].copy()
                print(f"    [tag-indexed] Shifted partition_ids by +1")
            else:
                self.partition_ids = all_partition_ids[:total_vectors].copy()
            del all_partition_ids
            print(f"    Partition IDs: loaded from partition_ids.npy "
                  f"({time.perf_counter() - t0:.2f}s)")
        else:
            # Fallback: reconstruct from per-partition ids.bin
            self.partition_ids = np.zeros(total_vectors, dtype=np.uint32)
            for p in range(self.num_partitions):
                ids_path = self.partition_dir / f"partition{p}" / "ids.bin"
                if not ids_path.exists():
                    raise FileNotFoundError(
                        f"Partition IDs not found: {ids_path}")
                with open(ids_path, 'rb') as f:
                    npts, _dim = np.fromfile(f, dtype=np.uint32, count=2)
                    ids = np.fromfile(f, dtype=np.uint32, count=int(npts))
                valid = ids[ids < total_vectors]
                self.partition_ids[valid] = p
            print(f"    Partition IDs: reconstructed from ids.bin "
                  f"({time.perf_counter() - t0:.2f}s)")

        unique, counts = np.unique(self.partition_ids, return_counts=True)
        part_dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"    Partition distribution: {part_dist}")

        # Centroids (for virtual entry point injection into colocation files)
        centroids_path = self.partition_dir / "centroids.fbin"
        if centroids_path.exists():
            self.centroids = load_fbin(str(centroids_path))
            print(f"    Centroids: {self.centroids.shape}")
        else:
            print(f"    [WARN] Centroids not found: {centroids_path} "
                  f"(virtual entry points will not be injected)")

        # Frozen point (DiskANN's global hub node)
        # Load ONLY the single frozen vector by seek, not the entire data file
        self.frozen_vector = None
        self.frozen_neighbors = None
        self.frozen_location = -1

        data_file = str(self.graph_file) + ".data"
        # Search for *mean_vector* file in graph dir and its parent
        mean_vector_file = None
        search_dirs = [
            os.path.dirname(str(self.graph_file)),                  # graph/
            os.path.dirname(os.path.dirname(str(self.graph_file))), # experiment/
        ]
        for graph_dir in search_dirs:
            if not os.path.isdir(graph_dir):
                continue
            for fname in os.listdir(graph_dir):
                if 'mean_vector' in fname and fname.endswith('.bin'):
                    mean_vector_file = os.path.join(graph_dir, fname)
                    break
            if mean_vector_file:
                break
        # .frozen sidecar written by data-less checkpoints (save_data_file=False).
        # Format: [npts=1: uint32, dim: uint32, vector: <dtype> × dim]
        frozen_sidecar = str(self.graph_file) + ".frozen"

        frozen_loc = self.graph_meta.get('start_node', -1)
        if frozen_loc >= 0:
            if os.path.exists(data_file):
                npts_data, _dim_data = get_bin_metadata(data_file)
                if frozen_loc < npts_data:
                    single = load_bin_data(data_file, dtype=load_dtype,
                                           offset_points=frozen_loc, num_points=1)
                    self.frozen_vector = single[0].astype(np.float32)
                else:
                    print(f"    [WARN] Frozen location {frozen_loc} out of range "
                          f"(data has {npts_data} points)")
            elif os.path.exists(frozen_sidecar):
                # Data-less checkpoint path: single-vector sidecar written during save()
                single = load_bin_data(frozen_sidecar, dtype=load_dtype)
                self.frozen_vector = single[0].astype(np.float32)
                print(f"    Frozen point restored from {frozen_sidecar}")
            elif mean_vector_file and os.path.exists(mean_vector_file):
                # Legacy fallback: mean_vector.bin
                single = load_bin_data(mean_vector_file, dtype=load_dtype)
                self.frozen_vector = single[0].astype(np.float32)
                print(f"    Frozen point restored from {mean_vector_file}")
            else:
                print(f"    [WARN] None of .data / .frozen / mean_vector.bin found, "
                      f"frozen point will not be included")

            if self.frozen_vector is not None:
                self.frozen_location = frozen_loc
                self.frozen_neighbors = self._get_neighbors(frozen_loc)
                self.frozen_degree = len(self.frozen_neighbors)
                print(f"    Frozen point: location={frozen_loc}, "
                      f"degree={self.frozen_degree}, "
                      f"norm={np.linalg.norm(self.frozen_vector):.4f}")

    # -----------------------------------------------------------------
    # Step 2: Build mappings  (numpy arrays, not Python dicts)
    # -----------------------------------------------------------------

    def _build_mappings(self):
        """Build location↔global_id and global_id→(partition, local_loc) mappings
        using compact numpy arrays instead of Python dicts."""
        print("\n  Building ID mappings (numpy)...")
        t0 = time.perf_counter()

        total_vectors = len(self.vectors)

        # location → global_id: tags[loc] - 1  (only where tag > 0)
        # global_id → location: inverse
        self.gid_to_loc = np.full(total_vectors, UNMAPPED, dtype=np.uint32)

        active_mask = self.tags > 0
        active_locs = np.where(active_mask)[0].astype(np.uint32)
        active_gids = (self.tags[active_locs].astype(np.int64) - 1)

        # Filter to valid range
        valid = (active_gids >= 0) & (active_gids < total_vectors)
        active_locs = active_locs[valid]
        active_gids = active_gids[valid].astype(np.uint32)

        self.gid_to_loc[active_gids] = active_locs

        num_active = len(active_gids)
        print(f"    Active locations: {num_active}")
        print(f"    Active global_ids: {num_active} "
              f"(out of {total_vectors} total)")

        # Group active global_ids by partition, assign local_loc
        self.gid_to_local_loc = np.full(total_vectors, UNMAPPED, dtype=np.uint32)
        part_assignments = self.partition_ids[active_gids]

        for p in range(self.num_partitions):
            p_mask = part_assignments == p
            members = np.sort(active_gids[p_mask])
            self.partition_members[p] = members
            self.partition_sizes[p] = len(members)
            self.partition_capacities[p] = math.ceil(
                len(members) * self.max_capacity_factor)

            # local_loc = position in sorted order
            local_locs = np.arange(len(members), dtype=np.uint32)
            self.gid_to_local_loc[members] = local_locs

        for p in range(self.num_partitions):
            cap = self.partition_capacities[p]
            active = self.partition_sizes[p]
            free = cap - active
            print(f"    Partition {p}: {active} active, {cap} capacity "
                  f"({free} free slots, {self.max_capacity_factor:.1f}x)")

        # Verify medoids are in correct partitions
        for p, medoid_gid in self.medoids.items():
            ll = int(self.gid_to_local_loc[medoid_gid]) \
                if medoid_gid < total_vectors else UNMAPPED
            if ll != UNMAPPED:
                actual_p = int(self.partition_ids[medoid_gid])
                if actual_p != p:
                    print(f"    [WARN] Medoid {medoid_gid} for partition {p} "
                          f"is actually in partition {actual_p}")
                else:
                    print(f"    Medoid for partition {p}: global_id={medoid_gid}, "
                          f"local_loc={ll}")
            else:
                print(f"    [WARN] Medoid {medoid_gid} for partition {p} "
                      f"not found in mappings")

        elapsed = time.perf_counter() - t0
        # Report memory footprint of mapping arrays
        arr_mb = (self.gid_to_loc.nbytes + self.gid_to_local_loc.nbytes) / (1024**2)
        members_mb = sum(m.nbytes for m in self.partition_members.values()) / (1024**2)
        print(f"    Mapping arrays: {arr_mb + members_mb:.1f} MB "
              f"({elapsed:.2f}s)")

    # -----------------------------------------------------------------
    # Vectorized neighbor → VectorID conversion (batch)
    # -----------------------------------------------------------------

    def _convert_neighbor_to_vectorid(self, neighbor_location: int) -> int:
        """Scalar fallback for single-entry operations (frozen/centroid)."""
        if neighbor_location >= len(self.tags):
            return INVALID_VID
        tag = int(self.tags[neighbor_location])
        if tag == 0:
            if (self.frozen_vector is not None and
                    neighbor_location == self.frozen_location and
                    0 in self.frozen_local_locs):
                return encode_vector_id(0, self.frozen_local_locs[0])
            return INVALID_VID
        global_id = tag - 1
        if global_id >= len(self.gid_to_local_loc):
            return INVALID_VID
        local_loc = int(self.gid_to_local_loc[global_id])
        if local_loc == UNMAPPED:
            return INVALID_VID
        partition = int(self.partition_ids[global_id])
        return encode_vector_id(partition, local_loc)

    def _convert_neighbors_batch(self, nbr_locs: np.ndarray) -> np.ndarray:
        """Vectorized batch conversion: neighbor locations → VectorIDs."""
        n = len(nbr_locs)
        result = np.full(n, INVALID_VID, dtype=np.uint64)

        # Filter valid locations
        valid_loc = nbr_locs < len(self.tags)
        safe_locs = np.where(valid_loc, nbr_locs, 0)

        nbr_tags = self.tags[safe_locs]
        nbr_gids = nbr_tags.astype(np.int64) - 1

        valid = (valid_loc & (nbr_tags > 0) &
                 (nbr_gids >= 0) & (nbr_gids < len(self.partition_ids)))
        safe_gids = np.where(valid, nbr_gids, 0)

        nbr_p = self.partition_ids[safe_gids].astype(np.uint64)
        nbr_ll = self.gid_to_local_loc[safe_gids].astype(np.uint64)

        mapped = valid & (self.gid_to_local_loc[safe_gids] != UNMAPPED)
        result[mapped] = (nbr_p[mapped] << np.uint64(LOCAL_BITS)) | nbr_ll[mapped]

        # Handle frozen point (tag=0 but valid)
        if (self.frozen_vector is not None and
                self.frozen_location >= 0 and 0 in self.frozen_local_locs):
            frozen_mask = (valid_loc &
                           (nbr_locs == self.frozen_location) &
                           (nbr_tags == 0))
            if frozen_mask.any():
                fvid = np.uint64(encode_vector_id(0, self.frozen_local_locs[0]))
                result[frozen_mask] = fvid

        return result

    # -----------------------------------------------------------------
    # Numpy views into the byte buffer
    # -----------------------------------------------------------------

    def _make_buf_views(self, buf: np.ndarray, total_entries: int):
        """Create strided numpy views for each entry field."""
        es = self.layout.entry_size
        vec_np = self._DTYPE_MAP.get(self.data_type, np.float32)

        vid_view = np.ndarray(total_entries, dtype='<u8',
                              buffer=buf, offset=0,
                              strides=(es,))
        vec_view = np.ndarray((total_entries, self.layout.dim), dtype=vec_np,
                              buffer=buf, offset=self.layout.vector_offset,
                              strides=(es, np.dtype(vec_np).itemsize))
        deg_view = np.ndarray(total_entries, dtype='<u4',
                              buffer=buf, offset=self.layout.degree_offset,
                              strides=(es,))
        nbr_view = np.ndarray((total_entries, self.layout.slack_R), dtype='<u8',
                              buffer=buf, offset=self.layout.neighbor_offset,
                              strides=(es, 8))
        return vid_view, vec_view, deg_view, nbr_view

    # -----------------------------------------------------------------
    # Step 4: Write one partition's .coloc file  (vectorized)
    # -----------------------------------------------------------------

    def _write_partition(self, partition: int, output_dir: Path):
        """Write co-location binary for one partition (streaming numpy).

        Optimizations vs. previous version:
          - Always use mmap output (lets OS overlap writes with compute).
          - Init ONLY empty slots (active slots are fully overwritten).
          - Single streaming pass: read input mmap in sequential chunks, and
            write vectors / VIDs / degrees / neighbors / centroid-distances
            all in the same loop. Eliminates the separate pre-read buffer
            and the duplicate full scan for centroid distances.
        """
        self._current_partition = partition
        num_active = self.partition_sizes[partition]
        capacity = self.partition_capacities[partition]
        has_centroid = self.centroids is not None
        has_frozen = self.frozen_vector is not None
        extra_slots = (1 if has_centroid else 0) + (1 if has_frozen else 0)
        total_entries = capacity + extra_slots

        if has_frozen:
            self.frozen_local_locs[partition] = \
                capacity + (1 if has_centroid else 0)
        entry_size = self.layout.entry_size
        total_bytes = total_entries * entry_size
        slack_R = self.layout.slack_R

        print(f"\n  Writing partition {partition}: {num_active} active / "
              f"{capacity} capacity"
              f"{' + 1 centroid' if has_centroid else ''}"
              f"{' + 1 frozen' if has_frozen else ''}"
              f", {total_bytes / (1024**2):.2f} MB...")

        coloc_file = output_dir / f"partition{partition}.coloc"
        buf = np.memmap(str(coloc_file), dtype=np.uint8, mode='w+',
                        shape=(total_bytes,))
        print(f"    Output mmap: {total_bytes / (1024**3):.2f} GB")
        vid_view, vec_view, deg_view, nbr_view = \
            self._make_buf_views(buf, total_entries)

        # Empty slots [num_active:capacity] need INVALID_VID and max vector so
        # they never appear in search top-K results.
        # Centroid/frozen entries [capacity:total_entries] get explicit VID/vec
        # writes below, but their per-entry unused neighbor slots
        # [c_deg:slack_R] must also be INVALID — so init neighbors across
        # [num_active:total_entries] in one shot.
        # Active slots [0:num_active] are fully overwritten in the streaming
        # loop (which also inits their neighbor rows per-chunk), so no
        # upfront init is needed for them.
        vec_dtype = vec_view.dtype
        vec_max = (np.finfo(vec_dtype).max if np.issubdtype(vec_dtype, np.floating)
                   else np.iinfo(vec_dtype).max)
        if capacity > num_active:
            vid_view[num_active:capacity] = INVALID_VID
            vec_view[num_active:capacity] = vec_max
        if total_entries > num_active:
            nbr_view[num_active:total_entries] = INVALID_VID

        t0 = time.perf_counter()
        members = self.partition_members[partition]
        locations = self.gid_to_loc[members]

        # Pre-compute raw and capped degrees for all active nodes
        raw_degrees = self.adj_degrees[locations].astype(np.int64)
        capped_degrees = np.minimum(raw_degrees, slack_R)

        # Centroid setup (distances computed in the streaming loop below)
        if has_centroid:
            centroid_vec = self.centroids[partition].astype(np.float32)
            all_dists = np.empty(num_active, dtype=np.float32)

        # --- Streaming pass: chunks of members, sequential input I/O ---
        # For each chunk, bulk-read input mmap over [cmin, cmax+1] (sequential),
        # then index out the exact member rows. Amplification factor is
        # bounded by partition density (~P = num_partitions).
        CHUNK = 500_000
        part_shifted = np.uint64(partition) << np.uint64(LOCAL_BITS)

        for cs in range(0, num_active, CHUNK):
            ce = min(cs + CHUNK, num_active)
            chunk_size = ce - cs
            chunk_members = members[cs:ce]
            chunk_locs = locations[cs:ce]
            chunk_raw_deg = raw_degrees[cs:ce]
            chunk_cap_deg = capped_degrees[cs:ce]

            # 1. Read vectors for this chunk via sequential bulk range
            cmin = int(chunk_members[0])
            cmax = int(chunk_members[-1])
            vec_bulk = np.array(self.vectors[cmin:cmax + 1])
            chunk_vecs = vec_bulk[chunk_members - cmin]
            del vec_bulk

            # 2. Write vectors
            vec_view[cs:ce] = chunk_vecs

            # 3. Centroid distances (reuse chunk_vecs — no extra I/O)
            if has_centroid:
                if vec_dtype != np.float32:
                    diff = chunk_vecs.astype(np.float32) - centroid_vec
                else:
                    diff = chunk_vecs - centroid_vec
                all_dists[cs:ce] = np.sum(diff * diff, axis=1)

            # 4. Entry VIDs: partition[16] | tag[48]
            tags_u64 = (chunk_members + 1).astype(np.uint64)
            vid_view[cs:ce] = part_shifted | tags_u64

            # 5. Degrees
            deg_view[cs:ce] = chunk_cap_deg.astype(np.uint32)

            # 6. Init neighbor slots to INVALID (slots beyond per-entry degree
            #    would otherwise be zero from mmap; searchers must see a
            #    sentinel if they ever read past `degree`).
            nbr_view[cs:ce] = INVALID_VID

            # 7. Neighbors (vectorized gather + convert + scatter)
            chunk_offsets = self.adj_offsets[chunk_locs]

            cum_raw = np.empty(chunk_size + 1, dtype=np.int64)
            cum_raw[0] = 0
            np.cumsum(chunk_raw_deg, out=cum_raw[1:])
            total_raw = int(cum_raw[-1])

            if total_raw > 0:
                base = np.repeat(chunk_offsets, chunk_raw_deg)
                intra = np.arange(total_raw, dtype=np.int64) - \
                        np.repeat(cum_raw[:-1], chunk_raw_deg)
                all_nbr_locs = self.adj_flat[(base + intra).astype(np.int64)]

                all_nbr_vids = self._convert_neighbors_batch(all_nbr_locs)

                cum_cap = np.empty(chunk_size + 1, dtype=np.int64)
                cum_cap[0] = 0
                np.cumsum(chunk_cap_deg, out=cum_cap[1:])
                total_cap = int(cum_cap[-1])

                if total_cap > 0:
                    row_idx = np.repeat(
                        np.arange(chunk_size, dtype=np.int64), chunk_cap_deg)
                    col_idx = np.arange(total_cap, dtype=np.int64) - \
                              np.repeat(cum_cap[:-1], chunk_cap_deg)
                    src_idx = np.repeat(cum_raw[:-1], chunk_cap_deg) + col_idx

                    nbr_view[cs + row_idx, col_idx] = all_nbr_vids[src_idx]

        # --- Write virtual centroid entry (distances already computed above) ---
        if has_centroid:
            centroid_ll = capacity
            self.centroid_local_locs[partition] = centroid_ll

            vid_view[centroid_ll] = encode_entry_vid(partition, 0)

            # Write in NATIVE dtype: centroid_vec is float32 for the distance math
            # above, but the entry slot is dim*element_size (int8/uint8 = 1 byte);
            # float bytes would store garbage at the centroid entry-point seed.
            _native_dt = self._DTYPE_MAP.get(self.data_type, np.float32)
            centroid_bytes = centroid_vec.astype(_native_dt).tobytes()
            byte_off = centroid_ll * entry_size + self.layout.vector_offset
            buf[byte_off:byte_off + len(centroid_bytes)] = \
                np.frombuffer(centroid_bytes, dtype=np.uint8)

            top_k = min(self.R, num_active)
            if top_k > 0:
                nearest_idx = np.argpartition(all_dists, top_k - 1)[:top_k]
                nearest_gids = members[nearest_idx]

                nbr_lls = self.gid_to_local_loc[nearest_gids]
                valid = nbr_lls != UNMAPPED
                c_vids = (part_shifted | nbr_lls[valid].astype(np.uint64))
                c_deg = len(c_vids)

                deg_view[centroid_ll] = c_deg
                nbr_view[centroid_ll, :c_deg] = c_vids
            else:
                c_deg = 0

            print(f"    Centroid entry at local_loc={centroid_ll} "
                  f"(degree={c_deg})")

        # --- Write frozen point entry ---
        if has_frozen:
            frozen_ll = self.frozen_local_locs[partition]

            vid_view[frozen_ll] = encode_entry_vid(partition, 0)

            # Native dtype for the frozen (search entry-point/medoid) vector too —
            # float bytes in an int8/uint8 slot make every search start from garbage.
            _native_dt = self._DTYPE_MAP.get(self.data_type, np.float32)
            fv_bytes = self.frozen_vector.astype(_native_dt).tobytes()
            byte_off = frozen_ll * entry_size + self.layout.vector_offset
            buf[byte_off:byte_off + len(fv_bytes)] = \
                np.frombuffer(fv_bytes, dtype=np.uint8)

            frozen_nbr_vids = []
            for nbr_loc in self.frozen_neighbors:
                nbr_vid = self._convert_neighbor_to_vectorid(int(nbr_loc))
                if nbr_vid != INVALID_VID:
                    frozen_nbr_vids.append(nbr_vid)

            f_deg = min(len(frozen_nbr_vids), slack_R)
            deg_view[frozen_ll] = f_deg
            for i in range(f_deg):
                nbr_view[frozen_ll, i] = frozen_nbr_vids[i]

            print(f"    Frozen point at local_loc={frozen_ll} "
                  f"(degree={f_deg})")

        # Flush mmap to disk
        buf.flush()
        del buf

        elapsed = time.perf_counter() - t0
        free_slots = capacity - num_active
        print(f"    Written: {num_active} active + {free_slots} empty slots"
              f"{' + 1 centroid' if has_centroid else ''}"
              f"{' + 1 frozen' if has_frozen else ''}"
              f", {elapsed:.2f}s")

        self._validate_partition(partition, output_dir, num_active)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate_partition(self, partition: int, output_dir: Path,
                            num_entries: int, num_samples: int = 5):
        """Spot-check a few entries against original data."""
        coloc_file = output_dir / f"partition{partition}.coloc"
        entry_size = self.layout.entry_size

        rng = np.random.RandomState(42 + partition)
        sample_locs = rng.choice(num_entries, size=min(num_samples, num_entries),
                                 replace=False)

        errors = 0
        members = self.partition_members[partition]
        with open(coloc_file, 'rb') as f:
            for local_loc in sample_locs:
                f.seek(local_loc * entry_size)
                entry = f.read(entry_size)

                vid = struct.unpack_from('<Q', entry, self.layout.vector_id_offset)[0]
                dec_part, dec_tag = decode_vector_id(vid)

                expected_global_id = int(members[local_loc])
                expected_tag = expected_global_id + 1
                if dec_part != partition or dec_tag != expected_tag:
                    print(f"    [ERR] Entry {local_loc}: VectorID mismatch "
                          f"(got p={dec_part},tag={dec_tag}, "
                          f"expected p={partition},tag={expected_tag})")
                    errors += 1
                    continue

                global_id = expected_global_id

                vec_bytes = entry[self.layout.vector_offset:
                                  self.layout.vector_offset + self.layout.vector_size]
                _dtype_map = {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}
                native_np = _dtype_map.get(self.data_type, np.float32)
                vec = np.frombuffer(vec_bytes, dtype=native_np)
                original = self.vectors[global_id]

                if not np.array_equal(vec, original):
                    print(f"    [ERR] Entry {local_loc}: Vector mismatch "
                          f"(max diff={np.max(np.abs(vec.astype(np.float32) - original.astype(np.float32))):.2e})")
                    errors += 1
                    continue

                degree = struct.unpack_from('<I', entry,
                                           self.layout.degree_offset)[0]
                location = int(self.gid_to_loc[global_id])
                if location != UNMAPPED and location < self.num_graph_nodes:
                    expected_degree = min(int(self.adj_degrees[location]),
                                         self.layout.slack_R)
                    if degree != expected_degree:
                        print(f"    [ERR] Entry {local_loc}: Degree mismatch "
                              f"(got {degree}, expected {expected_degree})")
                        errors += 1

        if errors == 0:
            print(f"    Validation OK ({len(sample_locs)} samples)")
        else:
            print(f"    [WARN] {errors}/{len(sample_locs)} validation errors")

    # -----------------------------------------------------------------
    # Step 5: Write lookup tables  (vectorized numpy)
    # -----------------------------------------------------------------

    def _write_global_to_vectorid(self, output_dir: Path):
        """Write global_id → VectorID lookup table (uint64[total_vectors])."""
        total = len(self.vectors)
        table = np.full(total, INVALID_VID, dtype=np.uint64)

        active_mask = self.gid_to_local_loc != UNMAPPED
        active_gids = np.where(active_mask)[0]

        partitions = self.partition_ids[active_gids].astype(np.uint64)
        local_locs = self.gid_to_local_loc[active_gids].astype(np.uint64)
        table[active_gids] = (partitions << LOCAL_BITS) | local_locs

        out_file = output_dir / "global_to_vectorid.bin"
        with open(out_file, 'wb') as f:
            np.array([total], dtype=np.uint32).tofile(f)
            table.tofile(f)

        print(f"\n  global_to_vectorid.bin: {total} entries "
              f"({os.path.getsize(out_file) / (1024*1024):.2f} MB)")

    def _write_tag_to_local(self, output_dir: Path):
        """Write per-partition tag → local_loc lookup tables."""
        for p in range(self.num_partitions):
            members = self.partition_members[p]
            if len(members) == 0:
                continue

            max_tag = int(members[-1] + 1)  # members is sorted
            table = np.full(max_tag + 1, 0xFFFFFFFF, dtype=np.uint32)

            tags_arr = (members + 1).astype(np.uint32)
            local_locs = np.arange(len(members), dtype=np.uint32)
            table[tags_arr] = local_locs

            out_file = output_dir / f"tag_to_local_p{p}.bin"
            with open(out_file, 'wb') as f:
                np.array([max_tag], dtype=np.uint32).tofile(f)
                table.tofile(f)

            mapped = np.count_nonzero(table != 0xFFFFFFFF)
            print(f"    tag_to_local_p{p}.bin: max_tag={max_tag}, "
                  f"{mapped} mappings "
                  f"({os.path.getsize(out_file) / (1024*1024):.2f} MB)")

    def _write_local_to_tag(self, output_dir: Path):
        """Write per-partition local_loc → tag lookup tables."""
        for p in range(self.num_partitions):
            capacity = self.partition_capacities[p]
            members = self.partition_members[p]

            table = np.zeros(capacity, dtype=np.uint32)
            tags_arr = (members + 1).astype(np.uint32)
            table[:len(members)] = tags_arr

            out_file = output_dir / f"local_to_tag_p{p}.bin"
            with open(out_file, 'wb') as f:
                np.array([capacity], dtype=np.uint32).tofile(f)
                table.tofile(f)

            mapped = np.count_nonzero(table)
            print(f"    local_to_tag_p{p}.bin: capacity={capacity}, "
                  f"{mapped} mappings "
                  f"({os.path.getsize(out_file) / (1024*1024):.2f} MB)")

    # -----------------------------------------------------------------
    # Step 6: Metadata
    # -----------------------------------------------------------------

    def _write_metadata(self, output_dir: Path):
        """Write colocation_meta.json.

        When building a single partition (--partition), merge with existing
        metadata to preserve centroid/frozen info from other partitions.
        """
        # Load existing metadata if present (for --partition merge)
        meta_file = output_dir / "colocation_meta.json"
        existing_partitions = {}
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    existing = json.load(f)
                existing_partitions = existing.get("partitions", {})
            except Exception:
                pass

        partitions_meta = {}
        for p in range(self.num_partitions):
            medoid_gid = self.medoids.get(p, -1)
            medoid_local = -1
            if medoid_gid >= 0 and medoid_gid < len(self.gid_to_local_loc):
                ll = int(self.gid_to_local_loc[medoid_gid])
                if ll != UNMAPPED:
                    medoid_local = ll

            if p in self.centroid_local_locs:
                # This partition was built in this run — use live data
                has_centroid = True
                centroid_loc = self.centroid_local_locs[p]
            elif str(p) in existing_partitions:
                # Not built this run — preserve from existing metadata
                has_centroid = existing_partitions[str(p)].get("centroid_local_loc", -1) >= 0
                centroid_loc = existing_partitions[str(p)].get("centroid_local_loc", -1)
            else:
                # Assume centroid exists (all partitions get one during build)
                has_centroid = self.centroids is not None
                centroid_loc = self.partition_capacities[p] if has_centroid else -1

            if p in self.frozen_local_locs:
                has_frozen = True
                frozen_loc = self.frozen_local_locs[p]
            elif str(p) in existing_partitions:
                has_frozen = existing_partitions[str(p)].get("frozen_local_loc", -1) >= 0
                frozen_loc = existing_partitions[str(p)].get("frozen_local_loc", -1)
            else:
                has_frozen = False
                frozen_loc = -1

            extra_slots = (1 if has_centroid else 0) + (1 if has_frozen else 0)
            partitions_meta[str(p)] = {
                "num_entries": self.partition_capacities[p] + extra_slots,
                "num_active": self.partition_sizes[p],
                "data_capacity": self.partition_capacities[p],
                "file": f"partition{p}.coloc",
                "medoid_local": medoid_local,
                "medoid_global_id": medoid_gid,
                "centroid_local_loc": centroid_loc,
                "frozen_local_loc": frozen_loc,
            }

        meta = {
            **self.layout.to_dict(),
            "num_partitions": self.num_partitions,
            "total_vectors": len(self.vectors),
            "max_capacity_factor": self.max_capacity_factor,
            "partitions": partitions_meta,
            "id_format": f"partition[{PARTITION_BITS}bit]|local_loc[{LOCAL_BITS}bit]",
            "global_to_vectorid": "global_to_vectorid.bin",
            "tag_to_local_files": {
                str(p): f"tag_to_local_p{p}.bin"
                for p in range(self.num_partitions)
            },
            "local_to_tag_files": {
                str(p): f"local_to_tag_p{p}.bin"
                for p in range(self.num_partitions)
            },
            "source": {
                "graph": str(self.graph_file),
                "tags": str(self.tags_file),
                "vectors": str(self.vectors_file),
                "partition_dir": str(self.partition_dir),
            },
            "graph_meta": {
                "max_degree": self.graph_meta['max_degree'],
                "start_node": self.graph_meta['start_node'],
                "num_frozen": self.graph_meta['num_frozen'],
                "num_nodes": self.graph_meta['num_nodes'],
            },
        }

        meta_file = output_dir / "colocation_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Metadata written: {meta_file}")

        print(f"\n  Entry layout:")
        print(f"    VectorID:     {self.layout.vector_id_offset:4d}  (8B)")
        print(f"    RawVector:    {self.layout.vector_offset:4d}  "
              f"({self.layout.vector_size}B)")
        print(f"    Degree:       {self.layout.degree_offset:4d}  (4B)")
        print(f"    NeighborList: {self.layout.neighbor_offset:4d}  "
              f"({self.layout.neighbor_size}B, {self.layout.slack_R} slots)")
        print(f"    Total:        {self.layout.entry_size}B "
              f"(raw={self.layout.raw_entry_size}B + "
              f"{self.layout.padding_size}B padding)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build co-location binary files for distributed RDMA search')
    parser.add_argument('--graph-file', required=True,
                        help='Path to DiskANN graph checkpoint')
    parser.add_argument('--tags-file', required=True,
                        help='Path to tags file (.tags)')
    parser.add_argument('--vectors-file', required=True,
                        help='Path to base vectors (.fbin)')
    parser.add_argument('--partition-dir', required=True,
                        help='Path to partition directory (e.g., .../2_partitions/)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for .coloc files')
    parser.add_argument('--R', type=int, default=64,
                        help='Max degree (default: 64)')
    parser.add_argument('--slack-factor', type=float, default=1.3,
                        help='Slack factor for neighbor list (default: 1.3)')
    parser.add_argument('--alignment', type=int, default=64,
                        help='Entry alignment in bytes (default: 64)')
    parser.add_argument('--max-capacity-factor', type=float, default=1.0,
                        help='Pre-allocate factor for empty slots (default: 1.0, '
                             'e.g., 1.2 = 20%% extra capacity per partition)')
    parser.add_argument('--data-type', default='float32',
                        choices=['float32', 'int8', 'uint8'],
                        help='Vector data type (default: float32)')
    parser.add_argument('--partition', type=int, default=None,
                        help='Build only this partition (default: all). '
                             'Use to reduce peak memory for large datasets.')
    parser.add_argument('--tag-indexed', action='store_true',
                        help='Base file uses tag-indexing (index 0 = mean vector, '
                             'data starts at index 1). Used for 800m-style files '
                             'where file[tag] = correct vector for tag T.')

    args = parser.parse_args()

    builder = ColocationBuilder(
        graph_file=args.graph_file,
        tags_file=args.tags_file,
        vectors_file=args.vectors_file,
        partition_dir=args.partition_dir,
        R=args.R,
        slack_factor=args.slack_factor,
        alignment=args.alignment,
        max_capacity_factor=args.max_capacity_factor,
        data_type=args.data_type,
        tag_indexed=args.tag_indexed,
    )
    partitions = [args.partition] if args.partition is not None else None
    builder.build(args.output_dir, partitions=partitions)


if __name__ == '__main__':
    main()
