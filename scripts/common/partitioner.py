#!/usr/bin/env python3
"""
Vector Partitioning Module

Supports KMeans clustering with three modes based on dataset size:
1. GPU mode: Small datasets that fit in GPU VRAM (L40S ~48GB)
2. CPU mode: Medium datasets that fit in host DRAM (~300GB)
3. Streaming mode: Large datasets requiring batched processing

Partition format: Same as .tags file
  [num_points (uint64), partition_ids (uint32)...]
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal
import time

from .data_loader import load_bin_data, get_bin_metadata


def save_partitions(partition_ids: np.ndarray, output_file: str):
    """
    Save partition assignments to binary file.
    Format: [num_points (uint64), partition_ids (uint32)...]
    Same format as .tags file.
    """
    num_points = np.array([len(partition_ids)], dtype=np.uint64)
    with open(output_file, 'wb') as f:
        num_points.tofile(f)
        partition_ids.astype(np.uint32).tofile(f)


def load_partitions(partition_file: str) -> np.ndarray:
    """
    Load partition assignments from binary file.
    Returns: np.ndarray of partition IDs (uint32)
    """
    with open(partition_file, 'rb') as f:
        num_points = np.fromfile(f, dtype=np.uint64, count=1)[0]
        partition_ids = np.fromfile(f, dtype=np.uint32, count=int(num_points))
    return partition_ids


class KMeansPartitioner:
    """
    KMeans-based vector partitioner with automatic mode selection.

    Modes:
    - 'gpu': Use cuML GPU KMeans (fastest, requires data to fit in VRAM)
    - 'cpu': Use sklearn KMeans (data fits in host DRAM)
    - 'streaming': Use MiniBatch KMeans with disk streaming (any size)
    - 'auto': Automatically select based on data size
    """

    # Size thresholds (in bytes)
    GPU_VRAM_LIMIT = 40 * 1024**3      # 40GB (conservative for L40S 48GB)
    DRAM_LIMIT = 280 * 1024**3         # 280GB (conservative for ~300GB)

    def __init__(
        self,
        n_clusters: int,
        mode: Literal['auto', 'gpu', 'cpu', 'streaming'] = 'auto',
        batch_size: int = 1_000_000,
        max_iter: int = 300,
        random_state: int = 42,
        verbose: bool = True,
        balanced: bool = False,
        max_imbalance_ratio: float = 1.2
    ):
        """
        Args:
            n_clusters: Number of partitions
            mode: Processing mode ('auto', 'gpu', 'cpu', 'streaming')
            batch_size: Number of points per batch (for streaming mode)
            max_iter: Maximum iterations for KMeans
            random_state: Random seed for reproducibility
            verbose: Print progress messages
            balanced: Enable post-hoc rebalancing after KMeans
            max_imbalance_ratio: Max allowed ratio of largest to smallest partition
                                 (1.0 = exact balance, 1.2 = 20% tolerance)
        """
        self.n_clusters = n_clusters
        self.mode = mode
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.balanced = balanced
        self.max_imbalance_ratio = max_imbalance_ratio

        self.centroids = None
        self._effective_mode = None
        self._gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if cuML GPU is available."""
        try:
            import cupy as cp
            # Quick test - this will fail if no CUDA driver
            _ = cp.zeros(10)
            # Also check cuml is importable
            import cuml
            return True
        except ImportError:
            if self.verbose:
                print("Note: cuML/cupy not installed, GPU mode unavailable")
            return False
        except Exception as e:
            if self.verbose:
                print(f"Note: CUDA not available ({e}), GPU mode unavailable")
            return False

    def _estimate_data_size(self, n_points: int, dim: int, dtype) -> int:
        """Estimate data size in bytes."""
        return n_points * dim * np.dtype(dtype).itemsize

    def _select_mode(self, data_size: int) -> str:
        """Auto-select processing mode based on data size."""
        if self.mode != 'auto':
            # If GPU requested but not available, fallback to CPU
            if self.mode == 'gpu' and not self._gpu_available:
                self._log("Warning: GPU mode requested but CUDA not available, falling back to CPU")
                return 'cpu' if data_size <= self.DRAM_LIMIT else 'streaming'
            return self.mode

        if data_size <= self.GPU_VRAM_LIMIT and self._gpu_available:
            return 'gpu'
        elif data_size <= self.DRAM_LIMIT:
            return 'cpu'
        else:
            return 'streaming'

    def _log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit KMeans and return partition assignments.

        Args:
            data: (N, D) array of vectors

        Returns:
            partition_ids: (N,) array of cluster assignments (uint32)
        """
        data_size = data.nbytes
        self._effective_mode = self._select_mode(data_size)

        self._log(f"Data size: {data_size / 1024**3:.2f} GB")
        self._log(f"Mode: {self._effective_mode}")
        self._log(f"Fitting KMeans with {self.n_clusters} clusters on {len(data)} points")

        start_time = time.time()

        if self._effective_mode == 'gpu':
            partition_ids = self._fit_gpu(data)
        elif self._effective_mode == 'cpu':
            partition_ids = self._fit_cpu(data)
        else:  # streaming
            partition_ids = self._fit_streaming(data)

        elapsed = time.time() - start_time
        self._log(f"KMeans completed in {elapsed:.2f}s")

        if self.balanced:
            self._log("Post-hoc rebalancing...")
            t_rebalance = time.time()
            partition_ids = self.rebalance(data, partition_ids, self.max_imbalance_ratio)
            self._log(f"Rebalancing completed in {time.time() - t_rebalance:.2f}s")

        return partition_ids

    def _fit_gpu(self, data: np.ndarray) -> np.ndarray:
        """GPU mode using cuML KMeans."""
        import cupy as cp
        from cuml.cluster import KMeans as cuKMeans

        self._log("Running GPU KMeans (cuML)...")

        data_gpu = cp.asarray(data.astype(np.float32))

        kmeans = cuKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init='scalable-k-means++',  # Better initialization for balanced clusters
            verbose=self.verbose
        )
        kmeans.fit(data_gpu)

        partition_ids = cp.asnumpy(kmeans.labels_).astype(np.uint32)
        self.centroids = cp.asnumpy(kmeans.cluster_centers_)

        return partition_ids

    def _fit_cpu(self, data: np.ndarray) -> np.ndarray:
        """CPU mode using sklearn KMeans."""
        from sklearn.cluster import KMeans

        self._log("Running CPU KMeans (sklearn)...")

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            n_init=1
        )
        kmeans.fit(data.astype(np.float32))

        partition_ids = kmeans.labels_.astype(np.uint32)
        self.centroids = kmeans.cluster_centers_

        return partition_ids

    def _init_centroids_pp(self, data: np.ndarray) -> np.ndarray:
        """Initialize centroids using kmeans++ algorithm."""
        n_points, dim = data.shape
        centroids = np.zeros((self.n_clusters, dim), dtype=np.float32)

        # First centroid: random point
        idx = np.random.randint(n_points)
        centroids[0] = data[idx]

        # Remaining centroids: weighted by distance squared
        for k in range(1, self.n_clusters):
            # Compute distances to nearest existing centroid
            dists = np.full(n_points, np.inf)
            for j in range(k):
                d = np.sum((data - centroids[j]) ** 2, axis=1)
                dists = np.minimum(dists, d)

            # Sample proportional to distance squared
            probs = dists / dists.sum()
            idx = np.random.choice(n_points, p=probs)
            centroids[k] = data[idx]

        return centroids

    def _assign_labels(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid."""
        n_points = len(data)
        labels = np.zeros(n_points, dtype=np.int32)

        # Process in batches to avoid memory issues
        batch_size = 100000
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch = data[start:end]

            # Compute squared distances to all centroids
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
            batch_norms = np.sum(batch ** 2, axis=1, keepdims=True)
            centroid_norms = np.sum(centroids ** 2, axis=1)
            dots = batch @ centroids.T
            dists = batch_norms + centroid_norms - 2 * dots

            labels[start:end] = np.argmin(dists, axis=1)

        return labels

    def _compute_all_distances(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute squared distances from each point to all centroids.

        Returns: (N, K) array of squared distances.
        Handles non-float32 data by converting per-batch (avoids full copy).
        """
        n_points = len(data)
        K = len(centroids)
        centroids_f32 = centroids.astype(np.float32) if centroids.dtype != np.float32 else centroids
        all_dists = np.empty((n_points, K), dtype=np.float32)
        batch_size = 100_000
        centroid_norms = np.sum(centroids_f32 ** 2, axis=1)

        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch = data[start:end].astype(np.float32) if data.dtype != np.float32 else data[start:end]
            batch_norms = np.sum(batch ** 2, axis=1, keepdims=True)
            dots = batch @ centroids_f32.T
            all_dists[start:end] = batch_norms + centroid_norms - 2 * dots

        return all_dists

    def _log_partition_stats(self, partition_ids: np.ndarray, label: str = ""):
        """Log partition size statistics."""
        sizes = np.bincount(partition_ids.astype(int), minlength=self.n_clusters)
        self._log(f"  {label}Partition sizes: {sizes.tolist()}")
        self._log(f"  {label}min={sizes.min()}, max={sizes.max()}, "
                  f"mean={sizes.mean():.0f}, std={sizes.std():.0f}")
        ratio = sizes.max() / max(sizes.min(), 1)
        self._log(f"  {label}Imbalance ratio (max/min): {ratio:.4f}")

    def rebalance(self, data: np.ndarray, partition_ids: np.ndarray,
                  max_imbalance_ratio: float = 1.2) -> np.ndarray:
        """Post-hoc rebalancing of partition assignments.

        Reassigns boundary points from oversized partitions to nearest
        undersized partitions until max/min ratio is within tolerance.

        Args:
            data: (N, D) float32 vectors
            partition_ids: (N,) uint32 current assignments
            max_imbalance_ratio: Max allowed ratio of largest to smallest
                                 partition size (1.0 = exact, 1.2 = 20%)

        Returns:
            Rebalanced partition_ids (N,) uint32
        """
        if self.n_clusters <= 1 or len(partition_ids) <= self.n_clusters:
            return partition_ids

        all_dists = self._compute_all_distances(data, self.centroids)
        result = self._rebalance_from_dists(partition_ids, all_dists, max_imbalance_ratio)

        # Recompute centroids to reflect moved points
        self._update_centroids(data, result)
        return result

    def _fit_streaming(self, data: np.ndarray) -> np.ndarray:
        """Streaming mode using sklearn MiniBatch KMeans."""
        from sklearn.cluster import MiniBatchKMeans

        self._log(f"Running Streaming KMeans (MiniBatch, batch_size={self.batch_size})...")

        n_points = len(data)
        n_batches = (n_points + self.batch_size - 1) // self.batch_size

        self._log(f"Total batches: {n_batches}")

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            batch_size=self.batch_size,
            verbose=1 if self.verbose else 0,
            max_no_improvement=20,
            reassignment_ratio=0.01
        )

        # Partial fit on all batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_points)
            batch = data[start_idx:end_idx].astype(np.float32)

            kmeans.partial_fit(batch)

            if self.verbose and (batch_idx + 1) % 10 == 0:
                self._log(f"  Trained batch {batch_idx + 1}/{n_batches}")

        self.centroids = kmeans.cluster_centers_

        # Assign all points
        self._log("Assigning all points to clusters...")
        partition_ids = np.zeros(n_points, dtype=np.uint32)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_points)
            batch = data[start_idx:end_idx].astype(np.float32)

            partition_ids[start_idx:end_idx] = kmeans.predict(batch).astype(np.uint32)

            if self.verbose and (batch_idx + 1) % 10 == 0:
                self._log(f"  Assigned batch {batch_idx + 1}/{n_batches}")

        return partition_ids

    def fit_from_file(
        self,
        bin_file: str,
        output_file: str,
        dtype=np.float32,
        start_point: int = 0,
        num_points: int = 0
    ) -> np.ndarray:
        """
        Load data from .bin file, partition, and save results.

        For streaming mode with very large files, this method loads
        data in chunks to avoid OOM.

        Args:
            bin_file: Path to input .bin file
            output_file: Path to output partition file
            dtype: Data type of vectors
            start_point: Starting offset in the file (default 0)
            num_points: Number of points to use (0 = use all from start_point)

        Returns:
            partition_ids: Array of cluster assignments
        """
        file_n_points, dim = get_bin_metadata(bin_file)

        # Calculate actual number of points to use
        if num_points > 0:
            n_points = min(num_points, file_n_points - start_point)
        else:
            n_points = file_n_points - start_point

        data_size = self._estimate_data_size(n_points, dim, dtype)
        self._effective_mode = self._select_mode(data_size)

        self._log(f"File contains: {file_n_points} points, {dim} dimensions")
        self._log(f"Using: {n_points} points from offset {start_point}")
        self._log(f"Estimated size: {data_size / 1024**3:.2f} GB")
        self._log(f"Mode: {self._effective_mode}")

        if self._effective_mode == 'streaming' and data_size > self.DRAM_LIMIT:
            # True streaming: don't load all data at once
            partition_ids = self._fit_streaming_from_file(bin_file, dtype, start_point, n_points)
        else:
            # Load all data into memory
            self._log(f"Loading data from {bin_file}")
            data = load_bin_data(bin_file, dtype=dtype, offset_points=start_point, num_points=n_points)
            partition_ids = self.fit(data)

        self._log(f"Saving partitions to {output_file}")
        save_partitions(partition_ids, output_file)

        return partition_ids

    def _fit_streaming_from_file(self, bin_file: str, dtype, start_point: int, n_points: int) -> np.ndarray:
        """
        Stream data from file for very large datasets.
        Never loads entire dataset into memory.

        Args:
            bin_file: Path to input .bin file
            dtype: Data type of vectors
            start_point: Starting offset in the file
            n_points: Number of points to process
        """
        from sklearn.cluster import MiniBatchKMeans

        n_batches = (n_points + self.batch_size - 1) // self.batch_size

        self._log(f"Streaming from file: {n_batches} batches of {self.batch_size}")

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            batch_size=self.batch_size,
            verbose=0,
            max_no_improvement=20,
            reassignment_ratio=0.01
        )

        # Phase 1: Train on batches
        self._log("Phase 1: Training centroids...")
        for batch_idx in range(n_batches):
            batch_start = start_point + batch_idx * self.batch_size
            num_in_batch = min(self.batch_size, n_points - batch_idx * self.batch_size)

            batch = load_bin_data(
                bin_file,
                dtype=dtype,
                offset_points=batch_start,
                num_points=num_in_batch
            ).astype(np.float32)

            kmeans.partial_fit(batch)

            if self.verbose and (batch_idx + 1) % 10 == 0:
                self._log(f"  Trained batch {batch_idx + 1}/{n_batches}")

        self.centroids = kmeans.cluster_centers_

        # Phase 2: Assign all points
        self._log("Phase 2: Assigning points...")
        partition_ids = np.zeros(n_points, dtype=np.uint32)

        for batch_idx in range(n_batches):
            batch_start = start_point + batch_idx * self.batch_size
            local_start = batch_idx * self.batch_size
            num_in_batch = min(self.batch_size, n_points - local_start)

            batch = load_bin_data(
                bin_file,
                dtype=dtype,
                offset_points=batch_start,
                num_points=num_in_batch
            ).astype(np.float32)

            partition_ids[local_start:local_start + num_in_batch] = kmeans.predict(batch).astype(np.uint32)

            if self.verbose and (batch_idx + 1) % 10 == 0:
                self._log(f"  Assigned batch {batch_idx + 1}/{n_batches}")

        # Phase 3: Rebalance if requested (streaming from file)
        if self.balanced:
            self._log("Phase 3: Post-hoc rebalancing (streaming)...")
            t_rebalance = time.time()

            # Compute all distances in batches (don't load full data)
            all_dists = np.empty((n_points, self.n_clusters), dtype=np.float32)
            centroid_norms = np.sum(self.centroids ** 2, axis=1)

            for batch_idx in range(n_batches):
                batch_start = start_point + batch_idx * self.batch_size
                local_start = batch_idx * self.batch_size
                num_in_batch = min(self.batch_size, n_points - local_start)

                batch = load_bin_data(
                    bin_file, dtype=dtype,
                    offset_points=batch_start, num_points=num_in_batch
                ).astype(np.float32)

                batch_norms = np.sum(batch ** 2, axis=1, keepdims=True)
                dots = batch @ self.centroids.T
                all_dists[local_start:local_start + num_in_batch] = (
                    batch_norms + centroid_norms - 2 * dots
                )

            # Rebalance using precomputed distances
            partition_ids = self._rebalance_from_dists(
                partition_ids, all_dists, self.max_imbalance_ratio
            )

            # Recompute centroids in streaming fashion
            K = self.n_clusters
            dim = self.centroids.shape[1]
            sums = np.zeros((K, dim), dtype=np.float64)
            counts = np.zeros(K, dtype=np.int64)
            for batch_idx in range(n_batches):
                batch_start_s = start_point + batch_idx * self.batch_size
                local_start_s = batch_idx * self.batch_size
                num_in_batch_s = min(self.batch_size, n_points - local_start_s)
                batch_s = load_bin_data(
                    bin_file, dtype=dtype,
                    offset_points=batch_start_s, num_points=num_in_batch_s
                ).astype(np.float64)
                batch_pids = partition_ids[local_start_s:local_start_s + num_in_batch_s]
                for p in range(K):
                    mask = batch_pids == p
                    if mask.any():
                        sums[p] += batch_s[mask].sum(axis=0)
                        counts[p] += mask.sum()
            for p in range(K):
                if counts[p] > 0:
                    self.centroids[p] = (sums[p] / counts[p]).astype(np.float32)
            self._log(f"  [Rebalance] Centroids recomputed (streaming)")

            self._log(f"Rebalancing completed in {time.time() - t_rebalance:.2f}s")

        return partition_ids

    def _update_centroids(self, data: np.ndarray, partition_ids: np.ndarray):
        """Recompute centroids from current partition assignments."""
        K = self.n_clusters
        dim = data.shape[1]
        new_centroids = np.zeros((K, dim), dtype=np.float32)
        for p in range(K):
            mask = partition_ids == p
            if mask.any():
                new_centroids[p] = data[mask].mean(axis=0)
            else:
                new_centroids[p] = self.centroids[p]  # keep old if empty
        self.centroids = new_centroids
        self._log(f"  [Rebalance] Centroids recomputed from new assignments")

    def _rebalance_from_dists(self, partition_ids: np.ndarray,
                              all_dists: np.ndarray,
                              max_imbalance_ratio: float) -> np.ndarray:
        """Rebalance using precomputed distance matrix (no raw vectors needed).

        Strategy: iteratively move boundary points from the largest partition
        to the smallest partition until max/min ratio <= max_imbalance_ratio.
        """
        N = len(partition_ids)
        K = self.n_clusters

        self._log_partition_stats(partition_ids, label="[Before] ")

        sizes = np.bincount(partition_ids.astype(int), minlength=K)
        current_ratio = sizes.max() / max(sizes.min(), 1)
        if current_ratio <= max_imbalance_ratio:
            self._log(f"  [Rebalance] Already balanced (ratio={current_ratio:.4f}), skipping")
            return partition_ids

        # Compute margin per point (small = easy to reassign)
        sorted_top2 = np.partition(all_dists, 1, axis=1)[:, :2]
        margins = sorted_top2[:, 1] - sorted_top2[:, 0]

        result = partition_ids.copy()
        total_moved = 0

        for iteration in range(20):
            sizes = np.bincount(result.astype(int), minlength=K)
            current_ratio = sizes.max() / max(sizes.min(), 1)
            if current_ratio <= max_imbalance_ratio:
                break

            # Find largest and smallest partitions
            largest_p = int(np.argmax(sizes))
            smallest_p = int(np.argmin(sizes))

            # How many to move: bring largest down toward ideal
            ideal = N / K
            # Move enough to reduce the ratio, but not overshoot
            to_move = max(1, int(sizes[largest_p] - ideal))

            members = np.where(result == largest_p)[0]
            # Rank by distance to smallest_p's centroid (prefer closer points)
            member_dists_to_target = all_dists[members, smallest_p]
            transfer_order = members[np.argsort(member_dists_to_target)]

            moved = 0
            for idx in transfer_order:
                if moved >= to_move:
                    break
                # Only move if it improves balance
                if sizes[largest_p] <= sizes[smallest_p] + 1:
                    break
                result[idx] = smallest_p
                sizes[largest_p] -= 1
                sizes[smallest_p] += 1
                moved += 1

            total_moved += moved
            if moved == 0:
                break

        self._log(f"  [Rebalance] Moved {total_moved} points in {iteration + 1} iteration(s)")
        self._log_partition_stats(result, label="[After] ")
        return result.astype(np.uint32)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Assign new points to existing clusters.

        Args:
            data: (N, D) array of vectors

        Returns:
            partition_ids: (N,) array of cluster assignments
        """
        if self.centroids is None:
            raise ValueError("Must call fit() before predict()")

        # Use GPU if available and data fits
        if self._gpu_available and data.nbytes < self.GPU_VRAM_LIMIT:
            import cupy as cp
            data_gpu = cp.asarray(data.astype(np.float32))
            centroids_gpu = cp.asarray(self.centroids.astype(np.float32))

            # Compute distances in batches if needed
            if len(data) > 100000:
                partition_ids = np.zeros(len(data), dtype=np.uint32)
                for start in range(0, len(data), 100000):
                    end = min(start + 100000, len(data))
                    batch_gpu = data_gpu[start:end]
                    distances = cp.linalg.norm(
                        batch_gpu[:, None, :] - centroids_gpu[None, :, :],
                        axis=2
                    )
                    partition_ids[start:end] = cp.asnumpy(cp.argmin(distances, axis=1))
            else:
                distances = cp.linalg.norm(
                    data_gpu[:, None, :] - centroids_gpu[None, :, :],
                    axis=2
                )
                partition_ids = cp.asnumpy(cp.argmin(distances, axis=1)).astype(np.uint32)
        else:
            # CPU fallback using numpy (avoids sklearn tcmalloc issues)
            partition_ids = self._assign_labels(data.astype(np.float32), self.centroids).astype(np.uint32)

        return partition_ids

    def save_centroids(self, output_file: str):
        """Save centroids to .npy file."""
        if self.centroids is None:
            raise ValueError("No centroids to save. Call fit() first.")
        np.save(output_file, self.centroids)

    def load_centroids(self, centroids_file: str):
        """Load centroids from .npy file."""
        self.centroids = np.load(centroids_file)
        self.n_clusters = len(self.centroids)


def partition_dataset(
    bin_file: str,
    output_file: str,
    n_clusters: int,
    mode: Literal['auto', 'gpu', 'cpu', 'streaming'] = 'auto',
    batch_size: int = 1_000_000,
    dtype=np.float32,
    save_centroids: bool = True,
    balanced: bool = False,
    max_imbalance_ratio: float = 1.2
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to partition a dataset.

    Args:
        bin_file: Path to input .bin file
        output_file: Path to output partition file (.partition)
        n_clusters: Number of partitions
        mode: Processing mode ('auto', 'gpu', 'cpu', 'streaming')
        batch_size: Batch size for streaming mode
        dtype: Data type of vectors
        save_centroids: Save centroids to .npy file
        balanced: Enable post-hoc rebalancing
        max_imbalance_ratio: Max allowed ratio of largest to smallest partition

    Returns:
        partition_ids: Array of cluster assignments
        centroids: Cluster centroids (if save_centroids=True)
    """
    partitioner = KMeansPartitioner(
        n_clusters=n_clusters,
        mode=mode,
        batch_size=batch_size,
        balanced=balanced,
        max_imbalance_ratio=max_imbalance_ratio
    )

    partition_ids = partitioner.fit_from_file(bin_file, output_file, dtype)

    if save_centroids:
        centroids_file = str(Path(output_file).with_suffix('.centroids.npy'))
        partitioner.save_centroids(centroids_file)
        return partition_ids, partitioner.centroids

    return partition_ids, None
