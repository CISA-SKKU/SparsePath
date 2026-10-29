"""Capacity-balanced query -> partition routing.

Nearest-centroid routing (``NearestCentroidPolicy.assign_batch``) is
data-dependent, so the per-node query counts come out badly skewed: measured
4-node shares are 24.6/25.2/23.8/26.4% on deep100M but 27.6/34.3/17.5/20.7% on
sift100M and 16.5/39.8/24.0/19.7% on spacev100M. Cluster wall-clock throughput
is set by the busiest node, so the skew alone costs 27-37% of wall QPS on
sift/spacev (deep, being nearly balanced, loses ~5%).

This module caps every partition at ceil(Q/P) queries and relocates the
overflow. Which node runs a query only picks its *entry point* (the search
graph is global and any node can pull any vector over RDMA), so relocation is
recall-neutral by construction; it only trades a little entry-point locality
for a flat load.

Determinism matters: every node computes the assignment independently over the
same query pool and the same centroids and then keeps its own slice, so the
routing must be a pure function of (queries, centroids). It is -- no RNG, no
node-local state, and ties broken by query index.
"""

from __future__ import annotations

import numpy as np

__all__ = ["squared_distances_to_centroids", "balance_assignment"]


def squared_distances_to_centroids(
    points: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Return the (Q, P) squared-L2 distance matrix.

    Mirrors ``NearestCentroidPolicy.assign_batch``'s ||q-c||^2 expansion, but
    keeps the full matrix instead of only the argmin -- the balancer needs the
    runner-up distances to pick the cheapest queries to relocate.
    """
    pts = np.asarray(points)
    cents = np.asarray(centroids)
    if pts.ndim != 2 or cents.ndim != 2:
        raise ValueError("points and centroids must be 2D arrays")
    if pts.shape[1] != cents.shape[1]:
        raise ValueError(
            f"dimension mismatch: points dim={pts.shape[1]}, "
            f"centroids dim={cents.shape[1]}")

    work_dtype = np.result_type(pts.dtype, cents.dtype, np.float32)
    pts_w = np.ascontiguousarray(pts, dtype=work_dtype)
    cents_w = np.ascontiguousarray(cents, dtype=work_dtype)
    # ||q||^2 is constant across centroids and cancels out of every comparison
    # the balancer makes (argmin and cost differences are both within-row), so
    # it is omitted exactly as assign_batch does.
    return (np.sum(cents_w * cents_w, axis=1)[None, :]
            - 2.0 * (pts_w @ cents_w.T))


def balance_assignment(
    assignment: np.ndarray,
    dists: np.ndarray,
    num_partitions: int,
) -> np.ndarray:
    """Cap each partition at ceil(Q/P) by relocating its cheapest queries.

    ``assignment`` is the nearest-centroid result, ``dists`` the (Q, P) matrix
    from :func:`squared_distances_to_centroids`. A query is relocated to the
    under-capacity partition whose centroid is nearest to it, and the queries
    moved off an overloaded partition are the ones with the smallest distance
    penalty -- i.e. those sitting near a centroid boundary, whose entry point
    barely changes.

    Returns a new assignment array; the input is not modified.
    """
    out = np.asarray(assignment, dtype=np.int32).copy()
    q = out.shape[0]
    if q == 0 or num_partitions <= 1:
        return out
    if dists.shape != (q, num_partitions):
        raise ValueError(
            f"dists shape {dists.shape} != ({q}, {num_partitions})")

    cap = -(-q // num_partitions)  # ceil
    counts = np.bincount(out, minlength=num_partitions).astype(np.int64)

    # At most one pass per (overloaded, underloaded) pair is needed: every
    # iteration either brings an overloaded partition down to cap or fills an
    # underloaded one up to cap, so the loop runs < 2*P times.
    for _ in range(2 * num_partitions):
        over = np.flatnonzero(counts > cap)
        under = np.flatnonzero(counts < cap)
        if over.size == 0 or under.size == 0:
            break
        # Deterministic pick: most overloaded / most underloaded, lowest id on ties.
        src = int(over[np.argmax(counts[over])])
        dst = int(under[np.argmin(counts[under])])

        idx = np.flatnonzero(out == src)
        # Penalty of moving each query src -> dst. Negative values cannot occur
        # for the *nearest* centroid, but can after an earlier relocation, so
        # the sort handles them naturally.
        cost = dists[idx, dst] - dists[idx, src]
        n_move = int(min(counts[src] - cap, cap - counts[dst]))
        # 'stable' keeps ascending query index as the tie-break.
        picked = idx[np.argsort(cost, kind="stable")[:n_move]]

        out[picked] = dst
        counts[src] -= n_move
        counts[dst] += n_move

    return out


def balanced_assign_batch(
    points: np.ndarray, centroids: np.ndarray, num_partitions: int
) -> np.ndarray:
    """nearest-centroid assignment followed by the capacity cap."""
    dists = squared_distances_to_centroids(points, centroids)
    base = np.argmin(dists, axis=1).astype(np.int32)
    return balance_assignment(base, dists, num_partitions)
