"""
Insert-pool partitioning strategies.

The initial partitioning (active vectors, [0, initial_size)) is always balanced
KMeans. The INSERT POOL ([initial_size, total_points)) is assigned via a
pluggable strategy. All strategies operate in gid-space:

    partition_ids[gid] = partition for gid

Strategies are "static": they decide partitions up-front from base_file
vectors + centroids. Dynamic strategies that look at graph state belong
elsewhere (future work).

Contract for every strategy:
  Input:
    - base_file, dtype, insert_start, insert_count   (streaming read from file)
    - centroids: (num_partitions, dim) float32       (from active k-means)
    - num_partitions, batch_size, seed
    - active_sizes: (num_partitions,) int64          (partition sizes from active only)
  Output:
    - np.ndarray[uint32], shape (insert_count,)      (partition per insert gid)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np

from .data_loader import load_bin_data


class InsertStrategy(ABC):
    name: str = ""

    def __init__(self, **params):
        # Strategies that want tunable hyper-params pull from params.
        # Strategies that don't just ignore it.
        self.params = params

    @abstractmethod
    def assign(
        self,
        base_file: str,
        dtype,
        insert_start: int,
        insert_count: int,
        centroids: np.ndarray,
        num_partitions: int,
        active_sizes: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> np.ndarray:
        ...


class NearestCentroidStrategy(InsertStrategy):
    """Assign each insert point to its nearest centroid (L2).

    This reproduces the previous hard-coded behavior.
    """
    name = "nearest_centroid"

    def assign(self, base_file, dtype, insert_start, insert_count,
               centroids, num_partitions, active_sizes, batch_size, seed):
        ids = np.empty(insert_count, dtype=np.uint32)
        cent_norms = np.sum(centroids ** 2, axis=1)
        n_batches = (insert_count + batch_size - 1) // batch_size
        for bi in range(n_batches):
            lo = bi * batch_size
            nb = min(batch_size, insert_count - lo)
            batch = load_bin_data(
                base_file, dtype=dtype,
                offset_points=insert_start + lo, num_points=nb,
            ).astype(np.float32)
            batch_norms = np.sum(batch ** 2, axis=1, keepdims=True)
            dists = batch_norms + cent_norms - 2.0 * (batch @ centroids.T)
            ids[lo:lo + nb] = np.argmin(dists, axis=1).astype(np.uint32)
        return ids



class RoundRobinStrategy(InsertStrategy):
    """Deterministic even split by position: partition = gid % num_partitions.

    Indexed by insert position (0 = first insert gid).
    """
    name = "round_robin"

    def assign(self, base_file, dtype, insert_start, insert_count,
               centroids, num_partitions, active_sizes, batch_size, seed):
        return (np.arange(insert_count, dtype=np.uint64)
                % np.uint64(num_partitions)).astype(np.uint32)


class NearestCentroidBalancedStrategy(InsertStrategy):
    """Online-greedy balanced nearest-centroid assignment.

    Simulates the online insert process in arrival order. For each insert
    point, candidates are tried in ascending-distance order; the first
    candidate whose tentative assignment keeps `max(totals) / min(totals)`
    within `max_imbalance_ratio` is taken. If all P candidates violate the
    constraint, the point falls back to `argmin(totals)`.

    `totals[p] = active_sizes[p] + (# inserts assigned to p so far)` — this
    mirrors runtime partition state at the moment each insert arrives.

    Params:
        max_imbalance_ratio: target for max(totals) / min(totals). Default 1.1
            (strict 1.0 degenerates to round-robin because every insert against
            a perfectly balanced active set already violates 1.0).
    """
    name = "nearest_centroid_balanced"

    def assign(self, base_file, dtype, insert_start, insert_count,
               centroids, num_partitions, active_sizes, batch_size, seed):
        max_imbalance_ratio = float(self.params.get("max_imbalance_ratio", 1.1))

        # Phase 1: stream and compute full (insert_count, P) distance matrix.
        all_dists = np.empty((insert_count, num_partitions), dtype=np.float32)
        cent_norms = np.sum(centroids ** 2, axis=1)
        n_batches = (insert_count + batch_size - 1) // batch_size
        for bi in range(n_batches):
            lo = bi * batch_size
            nb = min(batch_size, insert_count - lo)
            batch = load_bin_data(
                base_file, dtype=dtype,
                offset_points=insert_start + lo, num_points=nb,
            ).astype(np.float32)
            batch_norms = np.sum(batch ** 2, axis=1, keepdims=True)
            all_dists[lo:lo + nb] = (
                batch_norms + cent_norms - 2.0 * (batch @ centroids.T)
            )

        # Phase 2: online-greedy assignment (numba-JIT).
        active_sizes = np.asarray(active_sizes, dtype=np.int64)
        ids = _online_greedy_balanced(all_dists, active_sizes,
                                      float(max_imbalance_ratio))
        return ids


try:
    from numba import njit

    @njit(cache=True)
    def _online_greedy_balanced(all_dists, active_sizes, max_ratio):
        N, P = all_dists.shape
        ids = np.empty(N, dtype=np.uint32)
        totals = active_sizes.copy()

        for i in range(N):
            # Pick partitions in ascending distance order (small P → partial sort OK).
            order = np.argsort(all_dists[i])
            assigned = False
            for k in range(P):
                p = order[k]
                totals[p] += 1
                mx = totals[0]
                mn = totals[0]
                for q in range(1, P):
                    v = totals[q]
                    if v > mx:
                        mx = v
                    if v < mn:
                        mn = v
                if mn > 0 and (mx / mn) <= max_ratio:
                    ids[i] = np.uint32(p)
                    assigned = True
                    break
                # Undo the tentative assignment and try the next candidate.
                totals[p] -= 1

            if not assigned:
                # Fallback: place in the smallest partition.
                small_p = 0
                for q in range(1, P):
                    if totals[q] < totals[small_p]:
                        small_p = q
                totals[small_p] += 1
                ids[i] = np.uint32(small_p)

        return ids

except ImportError:
    def _online_greedy_balanced(all_dists, active_sizes, max_ratio):
        N, P = all_dists.shape
        ids = np.empty(N, dtype=np.uint32)
        totals = active_sizes.astype(np.int64).copy()
        for i in range(N):
            order = np.argsort(all_dists[i])
            assigned = False
            for k in range(P):
                p = int(order[k])
                totals[p] += 1
                mx = int(totals.max())
                mn = int(totals.min())
                if mn > 0 and (mx / mn) <= max_ratio:
                    ids[i] = np.uint32(p)
                    assigned = True
                    break
                totals[p] -= 1
            if not assigned:
                small_p = int(totals.argmin())
                totals[small_p] += 1
                ids[i] = np.uint32(small_p)
        return ids


_REGISTRY: Dict[str, Type[InsertStrategy]] = {
    cls.name: cls for cls in (
        NearestCentroidStrategy,
        NearestCentroidBalancedStrategy,
        RoundRobinStrategy,
    )
}

def get_strategy(name: str, **params) -> InsertStrategy:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown insert strategy '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**params)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())
