"""
Common types and utilities for the DANNS distributed pipeline.

Extracted from legacy graph_engine.py to eliminate dependency on
the legacy baseline tree. Only the symbols actually used by engine.py
and pipeline modules are included here.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

# ============================================================================
# Constants
# ============================================================================

GRAPH_SLACK_FACTOR = 1.3

# VectorID encoding (matches C++ core/vector_id.h)
LOCAL_BITS = 48
LOCAL_MASK = (1 << LOCAL_BITS) - 1
INVALID_VID = 0xFFFFFFFFFFFFFFFF


def encode_vector_id(partition: int, local_loc: int) -> int:
    """Encode (partition, local_loc) into a graph edge VID."""
    return (partition << LOCAL_BITS) | (local_loc & LOCAL_MASK)


def encode_entry_vid(partition: int, tag: int) -> int:
    """Encode (partition, tag) into an entry identity VID.
    Stored at offset 0 of colocation entry for global_id recovery."""
    return (partition << LOCAL_BITS) | (tag & LOCAL_MASK)


def decode_vector_id(vector_id: int) -> Tuple[int, int]:
    """Decode VectorID → (partition, local_loc)."""
    partition = vector_id >> LOCAL_BITS
    local_loc = vector_id & LOCAL_MASK
    return partition, local_loc


# ============================================================================
# Neighbor dataclass
# ============================================================================

@dataclass
class Neighbor:
    """A neighbor with ID and distance, used in priority queues."""
    id: int
    distance: float
    expanded: bool = False

    def __lt__(self, other):
        if self.distance != other.distance:
            return self.distance < other.distance
        return self.id < other.id


# ============================================================================
# Partition assignment policies
# ============================================================================

class PartitionAssignmentPolicy:
    """Pluggable policy for assigning new insert points to partitions."""

    def assign(self, point: np.ndarray, centroids: np.ndarray) -> int:
        """Return the partition ID for a single insert point."""
        raise NotImplementedError

    def assign_batch(self, points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return partition IDs for a batch of insert points."""
        return np.array([self.assign(p, centroids) for p in points], dtype=np.int32)


class NearestCentroidPolicy(PartitionAssignmentPolicy):
    """Default: assign to the nearest centroid (L2 distance)."""

    def assign(self, point: np.ndarray, centroids: np.ndarray) -> int:
        dists = np.sum(
            (centroids.astype(np.float64) - point.astype(np.float64)) ** 2,
            axis=1,
        )
        return int(np.argmin(dists))

    def assign_batch(self, points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return nearest-centroid partition IDs using chunked matrix multiply."""
        if points is None or len(points) == 0:
            return np.empty(0, dtype=np.int32)
        if centroids is None or len(centroids) == 0:
            raise ValueError("centroids must be non-empty for nearest-centroid routing")

        points_arr = np.asarray(points)
        centroids_arr = np.asarray(centroids)
        if points_arr.ndim != 2 or centroids_arr.ndim != 2:
            raise ValueError("points and centroids must be 2D arrays")
        if points_arr.shape[1] != centroids_arr.shape[1]:
            raise ValueError(
                f"dimension mismatch: points dim={points_arr.shape[1]}, "
                f"centroids dim={centroids_arr.shape[1]}")

        # ||q-c||^2 = ||q||^2 + ||c||^2 - 2q.c. The ||q||^2 term is identical
        # for every centroid, so routing only needs centroid norms and dot products.
        work_dtype = np.result_type(points_arr.dtype, centroids_arr.dtype, np.float32)
        centroids_work = np.ascontiguousarray(centroids_arr, dtype=work_dtype)
        centroid_norms = np.sum(centroids_work * centroids_work, axis=1)
        centroid_t = centroids_work.T

        n = points_arr.shape[0]
        out = np.empty(n, dtype=np.int32)
        chunk_size = 1_000_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = np.ascontiguousarray(points_arr[start:end], dtype=work_dtype)
            scores = centroid_norms - 2 * (chunk @ centroid_t)
            out[start:end] = np.argmin(scores, axis=1).astype(np.int32)
        return out


# ============================================================================
# Operation hooks
# ============================================================================

class OperationHook:
    """Base class for operation hooks. Override methods you need."""

    def on_search_start(self, engine, query: np.ndarray, K: int, L: int):
        pass

    def on_search_hop(self, engine, expanded_node: 'Neighbor',
                      neighbors: List[int], beam_state) -> Optional[List[int]]:
        return None

    def on_search_end(self, engine, query: np.ndarray,
                      result_ids: List[int], result_dists: List[float]):
        pass

    def on_pre_insert(self, engine, point: np.ndarray, tag: int):
        pass

    def on_post_insert(self, engine, location: int, tag: int,
                       forward_edges: List[int], reverse_targets: List[int]):
        pass

    def on_pre_delete(self, engine, tag: int, location: int):
        pass

    def on_pre_consolidation(self, engine, delete_set: Set[int]):
        pass

    def on_consolidation_node(self, engine, location: int,
                              old_neighbors: List[int], new_neighbors: List[int]):
        pass

    def on_post_consolidation(self, engine, delete_set: Set[int], stats: dict):
        pass

    def on_pre_prune(self, engine, location: int,
                     candidates: List['Neighbor']) -> Optional[List['Neighbor']]:
        return None

    def on_post_prune(self, engine, location: int,
                      candidates: List['Neighbor'], result: List[int]) -> Optional[List[int]]:
        return None
