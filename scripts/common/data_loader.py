# Data loading utilities
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List, Optional


def get_bin_metadata(bin_file: str) -> Tuple[int, int]:
    """Get number of points and dimensions from .bin file header."""
    arr = np.fromfile(bin_file, dtype=np.uint32, count=2)
    return int(arr[0]), int(arr[1])


def load_bin_data(bin_file: str, dtype=np.float32, offset_points: int = 0,
                  num_points: int = None, mmap: bool = False) -> np.ndarray:
    """
    Load vectors from .bin file.

    Args:
        bin_file: Path to .bin file (format: [npts, dim, data...])
        dtype: Data type (np.float32, np.int8, np.uint8)
        offset_points: Skip first N points
        num_points: Number of points to read (None = all remaining)
        mmap: If True, memory-map the file instead of loading into RAM.
              Useful for large datasets (>100GB). Returns read-only array.

    Returns:
        np.ndarray of shape (num_points, dim)
    """
    npts, dim = get_bin_metadata(bin_file)

    if num_points is None:
        num_points = npts - offset_points

    if offset_points + num_points > npts:
        raise ValueError(f"Requested {offset_points}+{num_points} points but file has {npts}")

    # Calculate byte offset (header + skip points)
    header_size = 2 * 4  # Two uint32
    point_size = dim * np.dtype(dtype).itemsize
    byte_offset = header_size + offset_points * point_size

    if mmap:
        data = np.memmap(bin_file, dtype=dtype, mode='r',
                         offset=byte_offset, shape=(num_points, dim))
        return data

    data = np.fromfile(bin_file, dtype=dtype, count=num_points * dim, offset=byte_offset)
    return data.reshape(num_points, dim)


def load_fbin(filename: str) -> np.ndarray:
    """Load float32 binary file in DiskANN format (alias for load_bin_data)"""
    return load_bin_data(filename, dtype=np.float32)


def save_fbin(filename: str, data: np.ndarray):
    """Save float32 binary file in DiskANN format"""
    npts, dim = data.shape
    with open(filename, 'wb') as f:
        np.array([npts, dim], dtype=np.int32).tofile(f)
        data.astype(np.float32).tofile(f)


def save_bin_data(filename: str, data: np.ndarray, dtype=None):
    """Save binary file in DiskANN format with specified dtype.
    Format: [npts (int32), dim (int32), data (dtype)]
    If dtype is None, uses data's own dtype.
    """
    if dtype is not None:
        data = data.astype(dtype)
    npts, dim = data.shape
    with open(filename, 'wb') as f:
        np.array([npts, dim], dtype=np.int32).tofile(f)
        data.tofile(f)


def save_ibin(filename: str, data: np.ndarray):
    """Save int32 binary file"""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    npts, dim = data.shape
    with open(filename, 'wb') as f:
        np.array([npts, dim], dtype=np.int32).tofile(f)
        data.astype(np.int32).tofile(f)


def save_tags(tags: np.ndarray, output_file: str):
    """
    Save tags to binary file in DiskANN vector format.
    Format: [npts (uint32), dim=1 (uint32), tags (uint32)...]
    This matches the format used by DiskANN C++ and Python bindings.
    """
    npts = len(tags)
    with open(output_file, 'wb') as f:
        np.array([npts, 1], dtype=np.uint32).tofile(f)
        tags.astype(np.uint32).tofile(f)


def load_tags(tags_file: str) -> np.ndarray:
    """
    Load tags from binary file in DiskANN vector format.
    Format: [npts (uint32), dim=1 (uint32), tags (uint32)...]
    """
    with open(tags_file, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        npts = header[0]
        tags = np.fromfile(f, dtype=np.uint32, count=int(npts))
    return tags


# =============================================================================
# Partition Reconstruction
# =============================================================================

def reconstruct_from_partitions(
    partition_dir: str,
    num_partitions: int = None,
    exclude_frozen_id: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct global vectors and partition_ids from partitioned data.

    Each partition directory contains:
        - data.fbin: vectors in DiskANN fbin format
        - ids.bin:   global IDs in DiskANN vector format [npts, 1, id0, id1, ...]

    Args:
        partition_dir: Path to partition root (e.g., .../4_partitions/)
        num_partitions: Number of partitions. Auto-detected if None.
        exclude_frozen_id: If set, exclude this global ID (frozen/start node).

    Returns:
        vectors: np.ndarray of shape (N, dim), ordered by global ID
        partition_ids: np.ndarray of shape (N,), partition assignment per global ID
    """
    partition_dir = Path(partition_dir)

    # Auto-detect partitions
    if num_partitions is None:
        num_partitions = sum(1 for d in partition_dir.iterdir()
                            if d.is_dir() and d.name.startswith("partition"))

    # Collect all (global_id, vector, partition_idx) tuples
    all_ids = []
    all_vectors = []
    all_pids = []

    for p in range(num_partitions):
        p_dir = partition_dir / f"partition{p}"
        ids_file = str(p_dir / "ids.bin")
        data_file = str(p_dir / "data.fbin")

        # ids.bin: [npts, 1, id0, id1, ...]
        ids_raw = np.fromfile(ids_file, dtype=np.uint32)
        npts_ids = int(ids_raw[0])
        ids = ids_raw[2:2 + npts_ids]

        vectors = load_bin_data(data_file, dtype=np.float32)
        if len(ids) != len(vectors):
            raise ValueError(
                f"partition{p}: ids count ({len(ids)}) != data count ({len(vectors)})"
            )

        all_ids.append(ids)
        all_vectors.append(vectors)
        all_pids.append(np.full(len(ids), p, dtype=np.int32))

    all_ids = np.concatenate(all_ids)
    all_vectors = np.concatenate(all_vectors)
    all_pids = np.concatenate(all_pids)

    # Exclude frozen node if specified
    if exclude_frozen_id is not None:
        mask = all_ids != exclude_frozen_id
        all_ids = all_ids[mask]
        all_vectors = all_vectors[mask]
        all_pids = all_pids[mask]

    # Sort by global ID and place into dense array
    max_id = int(all_ids.max())
    dim = all_vectors.shape[1]
    n_out = max_id + 1

    vectors_out = np.zeros((n_out, dim), dtype=np.float32)
    pids_out = np.full(n_out, -1, dtype=np.int32)

    vectors_out[all_ids] = all_vectors
    pids_out[all_ids] = all_pids

    # Remove any holes (IDs not present in any partition)
    valid_mask = pids_out >= 0
    if not valid_mask.all():
        vectors_out = vectors_out[valid_mask]
        pids_out = pids_out[valid_mask]

    return vectors_out, pids_out


def ensure_reconstructed_data(
    partition_dir: str,
    output_dir: str = None,
    num_partitions: int = None,
    graph_file: str = None,
) -> Tuple[str, str]:
    """
    Ensure reconstructed .fbin and partition_ids.npy exist, creating if needed.

    Caches results in output_dir (defaults to partition_dir).
    Returns paths immediately if files already exist.

    Args:
        partition_dir: Path to partition root (e.g., .../4_partitions/)
        output_dir: Where to save. Defaults to partition_dir.
        num_partitions: Number of partitions. Auto-detected if None.
        graph_file: If provided, read frozen node info to exclude it.

    Returns:
        (fbin_path, partition_ids_path): Paths to the reconstructed files.
    """
    if output_dir is None:
        output_dir = partition_dir

    output_dir = Path(output_dir)
    fbin_path = output_dir / "reconstructed.fbin"
    pids_path = output_dir / "partition_ids.npy"

    if fbin_path.exists() and pids_path.exists():
        return str(fbin_path), str(pids_path)

    # Determine frozen node
    exclude_frozen = None
    if graph_file is not None:
        _, meta = load_diskann_graph(graph_file)
        if meta.get("num_frozen", 0) > 0:
            exclude_frozen = meta["start_node"]

    print(f"Reconstructing data from {partition_dir} ...")
    vectors, pids = reconstruct_from_partitions(
        partition_dir,
        num_partitions=num_partitions,
        exclude_frozen_id=exclude_frozen,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_fbin(str(fbin_path), vectors)
    np.save(str(pids_path), pids)
    print(f"  Saved: {fbin_path} ({vectors.shape[0]:,} x {vectors.shape[1]})")
    print(f"  Saved: {pids_path}")

    return str(fbin_path), str(pids_path)


# =============================================================================
# Graph I/O Functions (DiskANN format)
# =============================================================================

def load_diskann_graph(filepath: str, flat: bool = False, mmap: bool = False):
    """
    Load graph from DiskANN checkpoint file (iteration{N}).

    DiskANN graph format (variable-length, packed):
        [file_size: uint64]
        [max_degree: uint32]
        [start_node: uint32]
        [num_frozen: uint64]
        For each node:
            [degree: uint32]
            [neighbors: uint32 * degree]

    Args:
        filepath: Path to graph file (e.g., checkpoints/iteration0)
        flat: If True, return flat arrays (adj_flat, adj_offsets, adj_degrees)
              instead of List[np.ndarray]. Much faster for large graphs and
              uses ~3x less memory.

    Returns:
        If flat=False (default):
            neighbors: List of np.ndarray, one per node (neighbor IDs)
            metadata: dict with {max_degree, start_node, num_frozen, num_nodes}
        If flat=True:
            (adj_flat, adj_offsets, adj_degrees): flat adjacency representation
            metadata: dict
    """
    import struct

    with open(filepath, 'rb') as f:
        # Read header (24 bytes)
        header = f.read(24)
        file_size = struct.unpack('<Q', header[0:8])[0]
        max_degree = struct.unpack('<I', header[8:12])[0]
        start_node = struct.unpack('<I', header[12:16])[0]
        num_frozen = struct.unpack('<Q', header[16:24])[0]

        # Bulk-read entire graph data after header (default path)
        if not mmap:
            raw = np.frombuffer(f.read(), dtype=np.uint32)

    if mmap:
        # Memory-map the edge data as reclaimable page cache instead of one big
        # anonymous read. On huge graphs (e.g. 400M @ ~102GB) the frombuffer path
        # keeps raw + the adj_flat copy + the bool mask resident at once (~230GB
        # peak) and swap-thrashes; mapping raw drops the anonymous peak to
        # mask+adj_flat (~127GB), which fits node RAM. offset=24 skips the header
        # parsed above; dtype uint32 matches the frombuffer path exactly.
        raw = np.memmap(filepath, dtype=np.uint32, mode='r', offset=24)

    metadata = {
        'file_size': int(file_size),
        'max_degree': int(max_degree),
        'start_node': int(start_node),
        'num_frozen': int(num_frozen),
    }

    # --- Parse: find degree positions ---
    n = len(raw)
    import time as _t
    _t0 = _t.perf_counter()

    try:
        import numba

        @numba.njit(cache=True)
        def _scan_deg_positions(raw):
            n = len(raw)
            positions = np.empty(n, dtype=np.int64)
            pos = 0
            cnt = 0
            while pos < n:
                positions[cnt] = pos
                cnt += 1
                pos += 1 + raw[pos]
            return positions[:cnt]

        deg_positions = _scan_deg_positions(raw)
    except Exception:
        pos = 0
        deg_positions = []
        while pos < n:
            deg_positions.append(pos)
            pos += 1 + int(raw[pos])
        deg_positions = np.array(deg_positions, dtype=np.int64)

    node_count = len(deg_positions)
    _elapsed = _t.perf_counter() - _t0
    print(f"    Graph scan: {node_count} nodes in {_elapsed:.1f}s")

    metadata['num_nodes'] = node_count

    if flat:
        adj_degrees = raw[deg_positions].astype(np.int32)

        if mmap:
            # Zero-copy: DON'T materialize a degree-word-stripped adj_flat (a full
            # ~O(edges) anonymous copy). Instead keep `raw` (the memmap) AS adj_flat
            # and point adj_offsets at the first neighbor of each node inside the
            # packed layout: neighbors of node i = raw[deg_positions[i]+1 : +degree].
            # Callers index adj_flat[adj_offsets[i] : +adj_degrees[i]] identically to
            # the non-mmap layout, but now only the *touched* edges fault in — a
            # single-partition colocation build reads only its own members' edges
            # (~partition/P of the graph), not the whole graph resident in RAM.
            adj_offsets = (deg_positions + 1).astype(np.int64)
            adj_flat = raw
            return (adj_flat, adj_offsets, adj_degrees), metadata

        # Vectorized fill using degree positions
        total_edges = int(adj_degrees.sum())
        adj_offsets = np.empty(node_count, dtype=np.int64)
        adj_offsets[0] = 0
        np.cumsum(adj_degrees[:-1], out=adj_offsets[1:])

        # Build adj_flat: gather all neighbor entries (everything except degree slots)
        # Create a boolean mask: True = neighbor, False = degree position
        is_neighbor = np.ones(n, dtype=bool)
        is_neighbor[deg_positions] = False
        adj_flat = raw[is_neighbor].copy()

        return (adj_flat, adj_offsets, adj_degrees), metadata

    else:
        # Pass 2: build list of arrays
        neighbors = [None] * node_count
        pos = 0
        for i in range(node_count):
            deg = int(raw[pos])
            if deg > 0:
                neighbors[i] = raw[pos + 1:pos + 1 + deg].copy()
            else:
                neighbors[i] = np.array([], dtype=np.uint32)
            pos += 1 + deg

    metadata['num_nodes'] = len(neighbors)
    return neighbors, metadata


def find_all_checkpoints(graph_dir: str) -> List[Tuple[int, str]]:
    """
    Find all checkpoint files in graph directory.

    Looks for files matching 'iteration{N}' pattern in:
    - graph_dir/checkpoints/
    - graph_dir/ (direct)

    Args:
        graph_dir: Path to graph output directory

    Returns:
        List of (iteration_number, filepath) sorted by iteration number
    """
    checkpoints = []
    graph_path = Path(graph_dir)

    # Check both direct path and checkpoints subdirectory
    search_dirs = [graph_path]
    checkpoint_subdir = graph_path / 'checkpoints'
    if checkpoint_subdir.exists():
        search_dirs.append(checkpoint_subdir)

    pattern = re.compile(r'^iteration(\d+)$')

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for item in search_dir.iterdir():
            if item.is_file():
                match = pattern.match(item.name)
                if match:
                    iteration = int(match.group(1))
                    checkpoints.append((iteration, str(item)))

    # Sort by iteration number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# =============================================================================
# Dataset Auto-Detection
# =============================================================================

# Default dataset root directories (searched in order)
DEFAULT_DATA_ROOTS = [
    "/workspace/datasets_local",      # Docker container (local SSD)
    "/workspace/datasets_nfs",        # Docker container (NFS mount)
    "/workspace/data",
]


def detect_data_root(dataset_name: Optional[str] = None) -> Optional[str]:
    """
    Auto-detect the dataset root directory.

    Args:
        dataset_name: If provided, find root that contains this dataset.
                      If None, return first existing root.

    Returns:
        Path to data root, or None if not found
    """
    for root in DEFAULT_DATA_ROOTS:
        root_path = Path(root)
        if root_path.exists():
            if dataset_name is None:
                return str(root_path.resolve())
            # Check if this root contains the dataset
            if (root_path / dataset_name).exists():
                return str(root_path.resolve())
    return None


# Dataset name aliases (for convenience)
DATASET_ALIASES = {
    "deep": "deep1M",
    "sift1m": "sift",
    "gist1m": "gist",
}


def get_dataset_info(dataset_name: str, data_root: Optional[str] = None) -> dict:
    """
    Get dataset information by name.
    Auto-detects dimension and datatype from fbin file header.

    Args:
        dataset_name: Name of dataset (e.g., "sift", "deep1M", "gist")
                      Supports aliases: "deep" -> "deep1M", etc.
        data_root: Optional override for data root directory

    Returns:
        dict with keys:
            - base_file: Path to base vectors file
            - query_file: Path to query vectors file
            - dimension: Vector dimension
            - data_type: Data type string ("float32", "int8", "uint8")
            - num_base: Number of base vectors
            - num_query: Number of query vectors

    Raises:
        FileNotFoundError: If dataset files not found
    """
    # Resolve alias
    dataset_name = DATASET_ALIASES.get(dataset_name, dataset_name)

    if data_root is None:
        # Find a data root that contains this specific dataset
        data_root = detect_data_root(dataset_name)
        if data_root is None:
            raise FileNotFoundError(
                f"Could not find dataset '{dataset_name}' in any data root. "
                f"Searched: {DEFAULT_DATA_ROOTS}"
            )

    data_root = Path(data_root)
    dataset_dir = data_root / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Find base and query files
    base_file = _find_data_file(dataset_dir, dataset_name, "base")
    query_file = _find_data_file(dataset_dir, dataset_name, "query")

    if base_file is None:
        raise FileNotFoundError(
            f"Base file not found in {dataset_dir}. "
            f"Expected: {dataset_name}_base.fbin or {dataset_name}_base.bin"
        )

    if query_file is None:
        raise FileNotFoundError(
            f"Query file not found in {dataset_dir}. "
            f"Expected: {dataset_name}_query.fbin or {dataset_name}_query.bin"
        )

    # Read metadata from file headers
    num_base, dim_base = get_bin_metadata(str(base_file))
    num_query, dim_query = get_bin_metadata(str(query_file))

    if dim_base != dim_query:
        raise ValueError(
            f"Dimension mismatch: base has {dim_base}, query has {dim_query}"
        )

    # Detect data type from file size
    data_type = _detect_dtype_from_file(base_file, num_base, dim_base)

    return {
        'base_file': str(base_file),
        'query_file': str(query_file),
        'dimension': dim_base,
        'data_type': data_type,
        'num_base': num_base,
        'num_query': num_query,
        'dataset_name': dataset_name,
        'data_root': str(data_root),
    }


def _find_data_file(dataset_dir: Path, dataset_name: str, file_type: str) -> Optional[Path]:
    """
    Find data file with various naming conventions.

    Args:
        dataset_dir: Dataset directory
        dataset_name: Dataset name (e.g., "sift", "deep1M")
        file_type: "base" or "query"

    Returns:
        Path to file, or None if not found
    """
    # Try different naming patterns
    patterns = [
        f"{dataset_name}_{file_type}.fbin",
        f"{dataset_name}_{file_type}.i8bin",
        f"{dataset_name}_{file_type}.u8bin",
        f"{dataset_name}_{file_type}.bin",
        f"{file_type}.fbin",
        f"{file_type}.i8bin",
        f"{file_type}.u8bin",
        f"{file_type}.bin",
    ]

    for pattern in patterns:
        candidate = dataset_dir / pattern
        if candidate.exists():
            # Follow symlinks to get the real file
            return candidate.resolve()

    return None


def _detect_dtype_from_file(filepath: Path, num_points: int, dim: int) -> str:
    """
    Detect data type from file size.

    fbin header is 8 bytes (2 x int32), then data follows.
    """
    file_size = filepath.stat().st_size
    header_size = 8  # 2 x uint32
    data_size = file_size - header_size
    bytes_per_point = data_size / num_points
    bytes_per_element = bytes_per_point / dim

    if abs(bytes_per_element - 4) < 0.1:
        return "float32"
    elif abs(bytes_per_element - 1) < 0.1:
        # Distinguish int8 vs uint8 by file extension
        ext = filepath.suffix.lower()
        if ext == '.i8bin':
            return "int8"
        elif ext == '.u8bin':
            return "uint8"
        # Fallback: check filename for hints
        name = filepath.stem.lower()
        if 'i8' in name:
            return "int8"
        return "uint8"
    elif abs(bytes_per_element - 2) < 0.1:
        return "float16"
    else:
        # Default to float32
        return "float32"


def load_dataset(dataset_name: str, data_root: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a complete dataset by name.

    Args:
        dataset_name: Name of dataset (e.g., "sift", "deep1M", "gist")
        data_root: Optional override for data root directory

    Returns:
        base_vectors: np.ndarray of base vectors
        query_vectors: np.ndarray of query vectors
        info: dict with dataset metadata
    """
    info = get_dataset_info(dataset_name, data_root)

    # Map data type string to numpy dtype
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'int8': np.int8,
        'uint8': np.uint8,
    }
    dtype = dtype_map.get(info['data_type'], np.float32)

    base_vectors = load_bin_data(info['base_file'], dtype=dtype)
    query_vectors = load_bin_data(info['query_file'], dtype=dtype)

    return base_vectors, query_vectors, info


def find_latest_graph_dir(base_dir: str, experiment_name: str) -> Optional[str]:
    """
    Find the latest graph directory for an experiment.

    Structure: base_dir/experiment_name/graph/{timestamp}/

    Args:
        base_dir: Base output directory
        experiment_name: Name of the experiment

    Returns:
        Path to latest graph directory, or None if not found
    """
    graph_base = Path(base_dir) / experiment_name / 'graph'
    if not graph_base.exists():
        return None

    # Find all timestamp directories
    timestamp_dirs = [d for d in graph_base.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        return None

    # Sort by name (timestamp format ensures chronological order)
    timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
    return str(timestamp_dirs[0])
