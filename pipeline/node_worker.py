#!/usr/bin/env python3
"""
Per-node worker for distributed pipeline stages.

Runs inside Docker container on DPU node.  Reads pipeline output
(graph checkpoints, partitions) from NFS and executes real RDMA
search using distributed_bindings.MultiPeerGraphStore.

Usage (launched by run_pipeline.sh):
    python3 node_worker.py \
        --config configs/deep1M_e2e.yaml \
        --node-id 1 --nodes 1,2,3,4 --coordinator 1 \
        --stage search --timestamp 20260304_120000
"""

import argparse
import csv
import ctypes
import json
import numpy as np
import os
import signal
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import threading
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
_scripts_dir = _project_root / "scripts"

for _p in [str(_this_dir), str(_project_root), str(_scripts_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Add lib/ and build/ for distributed_bindings
for _p in [str(_project_root / "lib"), str(_project_root / "build")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import re as _re
from common.data_loader import load_bin_data, get_bin_metadata, get_dataset_info
from common.partition_paths import (
    get_insert_strategy, partition_dir as build_partition_dir,
    colocation_dir as build_colocation_dir,
    colocation_dir_candidates,
    pq_dir as build_pq_dir,
)
from pipeline.config import get_pq_filter_config
from pipeline.experiment_paths import (
    artifact_colocation_dir, artifact_experiment_dir, artifact_dir_candidates, run_dir as build_run_dir,
)
from pipeline.partition_mapping import mapping_from_config


def _extract_iteration_from_graph(graph_file: str) -> int:
    """Extract iteration number from graph checkpoint filename.

    E.g. '.../checkpoints/iteration10' -> 10.  Returns 0 if no match.
    """
    if not graph_file:
        return 0
    m = _re.match(r'^iteration(\d+)$', Path(graph_file).name)
    return int(m.group(1)) if m else 0

# Import RDMA bindings (only available on DPU nodes inside Docker)
try:
    import distributed_bindings as db
    _HAS_RDMA = True
except ImportError as e:
    _HAS_RDMA = False
    print(f"[WARN] RDMA bindings not available ({e})")


# ---------------------------------------------------------------------------
# LOCAL_MMAP fetch mode helpers
# ---------------------------------------------------------------------------
# Used only when colocation.fetch_mode == "local_mmap" (room-at-the-top
# experiments). We mmap each peer partition's .coloc file with MAP_POPULATE
# to pre-fault all pages into RAM, then hand the address to C++ via
# DANNSEngine::register_local_partition_data so the consolidate fetch sites
# memcpy from local RAM instead of doing RDMA reads.

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc_mmap = _libc.mmap
_libc_mmap.restype = ctypes.c_void_p
_libc_mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_PROT_READ = 0x1
_MAP_PRIVATE = 0x02
_MAP_POPULATE = 0x8000  # Linux-specific; pre-faults all pages during mmap


def _mmap_populate_readonly(path: str) -> Tuple[int, int]:
    """mmap a file read-only with MAP_POPULATE so all pages are pre-faulted.

    Returns (addr, size). Caller must keep this returned info alive for the
    duration the C++ side reads through the pointer. We deliberately do not
    munmap — process exit reclaims, which matches the engine lifetime.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size <= 0:
            raise OSError(f"empty/invalid file: {path}")
        addr = _libc_mmap(None, size, _PROT_READ,
                          _MAP_PRIVATE | _MAP_POPULATE, fd, 0)
        # mmap returns (void*)-1 on failure
        if addr is None or addr == ctypes.c_void_p(-1).value or \
                addr == 2**64 - 1:
            err = ctypes.get_errno()
            raise OSError(err, f"mmap({path}): {os.strerror(err)}")
        return int(addr), size
    finally:
        os.close(fd)


from pipeline.config import (
    ClusterConfig,
    ClusterNode,
    apply_config_overrides,
    resolve_config_paths,
)
from pipeline.sync import TCPSync


# NOTE: TCPSync imported from pipeline.sync above.
# Below is the legacy inline copy — kept as dead code reference.
# TODO: remove after verifying sync.py TCPSync works end-to-end.

class _TCPSyncLegacy:
    """LEGACY — do not use. See pipeline.sync.TCPSync."""

    def __init__(self, cluster: ClusterConfig, my_id: int):
        self.cluster = cluster
        self.my_id = my_id
        self.is_coordinator = (my_id == cluster.coordinator_node)
        self.conns = {}
        self.server_socket = None

    @staticmethod
    def _enable_keepalive(sock):
        """Enable TCP keepalive to prevent connection drop during long ops."""
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 30)

    def setup(self):
        my_node = self.cluster.get_node(self.my_id)
        if not my_node:
            raise ValueError(f"Node {self.my_id} not in cluster config")

        if self.is_coordinator:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', my_node.sync_port))
            self.server_socket.listen(self.cluster.num_nodes - 1)
            print(f"[Sync] Coordinator listening on port {my_node.sync_port}")

            for _ in self.cluster.get_peers(self.my_id):
                conn, addr = self.server_socket.accept()
                peer_id = struct.unpack('!I', conn.recv(4))[0]
                self._enable_keepalive(conn)
                self.conns[peer_id] = conn
                print(f"[Sync] Accepted connection from node {peer_id}")
        else:
            coord = self.cluster.get_node(self.cluster.coordinator_node)
            for attempt in range(120):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((coord.host, coord.sync_port))
                    self._enable_keepalive(sock)
                    sock.send(struct.pack('!I', self.my_id))
                    self.conns[coord.id] = sock
                    print(f"[Sync] Connected to coordinator at {coord.host}:{coord.sync_port}")
                    break
                except (ConnectionRefusedError, OSError):
                    if attempt % 10 == 0:
                        print(f"[Sync] Waiting for coordinator {coord.host}:{coord.sync_port} "
                              f"(attempt {attempt+1}/120)...")
                    time.sleep(1)
            else:
                raise RuntimeError("Failed to connect to coordinator after 120s")

    def barrier(self, name: str = ""):
        if self.is_coordinator:
            print(f"[Sync] barrier({name}): coordinator waiting for {list(self.conns.keys())}", flush=True)
            for peer_id, conn in self.conns.items():
                self._recv_exact(conn, 4)  # RDY\x00
            for peer_id, conn in self.conns.items():
                conn.sendall(b'OK\x00\x00')
            print(f"[Sync] barrier({name}): coordinator done", flush=True)
        else:
            coord_conn = self.conns[self.cluster.coordinator_node]
            print(f"[Sync] barrier({name}): node {self.my_id} sending RDY", flush=True)
            coord_conn.sendall(b'RDY\x00')
            self._recv_exact(coord_conn, 4)  # OK\x00\x00
            print(f"[Sync] barrier({name}): node {self.my_id} passed", flush=True)

    def broadcast_value(self, key: str, value: str):
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can broadcast")
        data = value.encode('utf-8')
        for conn in self.conns.values():
            conn.sendall(struct.pack('!I', len(data)))
            conn.sendall(data)

    def recv_value(self, key: str) -> str:
        if self.is_coordinator:
            raise RuntimeError("Only workers can receive")
        conn = self.conns[self.cluster.coordinator_node]
        length = struct.unpack('!I', self._recv_exact(conn, 4))[0]
        return self._recv_exact(conn, length).decode('utf-8')

    # ---- Binary data exchange (for distributed update) ----

    def _recv_exact(self, conn, length: int) -> bytes:
        """Receive exactly `length` bytes from a socket."""
        buf = b''
        while len(buf) < length:
            chunk = conn.recv(min(length - len(buf), 65536))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            buf += chunk
        return buf

    def broadcast_bytes(self, data: bytes):
        """Coordinator sends raw bytes to all workers."""
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can broadcast_bytes")
        for conn in self.conns.values():
            conn.sendall(struct.pack('!I', len(data)))
            conn.sendall(data)

    def recv_bytes(self) -> bytes:
        """Worker receives raw bytes from coordinator."""
        if self.is_coordinator:
            raise RuntimeError("Only workers can recv_bytes")
        conn = self.conns[self.cluster.coordinator_node]
        length = struct.unpack('!I', self._recv_exact(conn, 4))[0]
        return self._recv_exact(conn, length)

    def gather_bytes(self) -> dict:
        """Coordinator gathers raw bytes from all workers. Returns {peer_id: bytes}."""
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can gather_bytes")
        result = {}
        for peer_id, conn in self.conns.items():
            length = struct.unpack('!I', self._recv_exact(conn, 4))[0]
            result[peer_id] = self._recv_exact(conn, length)
        return result

    def send_to_coordinator(self, data: bytes):
        """Worker sends raw bytes to coordinator."""
        if self.is_coordinator:
            raise RuntimeError("Coordinator should not send_to_coordinator")
        conn = self.conns[self.cluster.coordinator_node]
        conn.sendall(struct.pack('!I', len(data)))
        conn.sendall(data)

    def barrier_with_status(self, name: str, success: bool) -> bool:
        """Barrier that also exchanges success/failure status.

        Each node reports whether it succeeded. Returns True only if ALL
        nodes (including self) succeeded. This prevents deadlocks when one
        node fails mid-operation — other nodes learn about the failure
        instead of blocking forever.
        """
        if self.is_coordinator:
            all_ok = success
            for peer_id, conn in self.conns.items():
                status_byte = self._recv_exact(conn, 1)
                peer_ok = struct.unpack('!?', status_byte)[0]
                all_ok = all_ok and peer_ok
            # Broadcast aggregated status to all workers
            for peer_id, conn in self.conns.items():
                conn.sendall(struct.pack('!?', all_ok))
            if not all_ok:
                print(f"[Sync] barrier_with_status({name}): some nodes FAILED", flush=True)
            return all_ok
        else:
            coord_conn = self.conns[self.cluster.coordinator_node]
            coord_conn.sendall(struct.pack('!?', success))
            all_ok = struct.unpack('!?', self._recv_exact(coord_conn, 1))[0]
            return all_ok

    def close(self):
        for conn in self.conns.values():
            conn.close()
        if self.server_socket:
            self.server_socket.close()


# ============================================================================
# RDMA Vector Store
# ============================================================================

class RDMAVectorStore:
    """RDMA-enabled vector store for distributed search."""

    def __init__(self, node_id: int, cluster: ClusterConfig, vector_dim: int,
                 rdma_device: str = "", gid_index: int = 1):
        self.node_id = node_id
        self.cluster = cluster
        self.vector_dim = vector_dim
        self.rdma_device = rdma_device
        self.gid_index = gid_index

        self.vectors = None
        self.partition_ids = None
        self.partition_ids_by_location = None
        self.tags = None
        self.rdma_store = None
        self.local_vectors = None
        self.local_global_ids = None
        self.my_partition = 0

    def load_data(self, vectors: np.ndarray, partition_ids: np.ndarray,
                  tags: np.ndarray = None):
        self.vectors = vectors  # keep native dtype
        self.partition_ids = partition_ids.astype(np.uint32)
        self.tags = tags

        if tags is not None:
            array_size = max(len(tags), int(len(vectors) * 1.1) + 1)
            self.partition_ids_by_location = np.zeros(array_size, dtype=np.uint32)
            for location in range(len(tags)):
                tag = tags[location]
                if tag > 0:
                    global_id = int(tag - 1)
                    if global_id < len(self.partition_ids):
                        self.partition_ids_by_location[location] = self.partition_ids[global_id]
        else:
            self.partition_ids_by_location = self.partition_ids.copy()

        # Exclude coordinator (no RDMA/data) — only DPU worker nodes
        sorted_nodes = sorted(n for n in self.cluster.nodes.keys()
                              if n != self.cluster.coordinator_node)
        self.my_partition = sorted_nodes.index(self.node_id)
        local_mask = (self.partition_ids == self.my_partition)
        self.local_global_ids = np.where(local_mask)[0].astype(np.uint32)
        self.local_vectors = self.vectors[local_mask].copy()

        unique, counts = np.unique(self.partition_ids, return_counts=True)
        print(f"[Node {self.node_id}] Partition dist: {dict(zip(unique.tolist(), counts.tolist()))}")
        print(f"[Node {self.node_id}] My partition: {self.my_partition} (node_id={self.node_id})")
        print(f"[Node {self.node_id}] Local vectors: {len(self.local_global_ids)}/{len(vectors)}")

    def load_data_lightweight(self, total_vectors: int, partition_ids: np.ndarray,
                              partition_dir: str = None, colocation_only: bool = True):
        """Lightweight data loading — colocation data layout only.

        Data layout is always colocation: vectors+neighbors are co-located in
        .coloc binary files accessed via RDMA.  Per-partition data.fbin files
        (old separated data layout) are no longer generated or loaded.
        """
        self.vectors = None
        self.partition_ids = partition_ids.astype(np.uint32)
        self.partition_ids_by_location = self.partition_ids.copy()

        # Exclude coordinator (no RDMA/data) — only DPU worker nodes
        sorted_nodes = sorted(n for n in self.cluster.nodes.keys()
                              if n != self.cluster.coordinator_node)
        self.my_partition = sorted_nodes.index(self.node_id)
        local_mask = (self.partition_ids == self.my_partition)
        self.local_global_ids = np.where(local_mask)[0].astype(np.uint32)
        self.local_vectors = None  # colocation layout — no separate vector store

        unique, counts = np.unique(self.partition_ids, return_counts=True)
        print(f"[Node {self.node_id}] Partition dist: {dict(zip(unique.tolist(), counts.tolist()))}")
        print(f"[Node {self.node_id}] My partition: {self.my_partition} (node_id={self.node_id})")
        print(f"[Node {self.node_id}] Local vectors: {len(self.local_global_ids)}/{total_vectors}")

    def setup_rdma(self, colocation_only: bool = True):
        if not _HAS_RDMA:
            raise RuntimeError("RDMA bindings required")

        self.rdma_store = db.MultiPeerGraphStore(
            node_id=self.node_id,
            total_nodes=self.cluster.num_nodes,
            vector_dim=self.vector_dim,
            rdma_device=self.rdma_device,
            gid_index=self.gid_index
        )

        total = len(self.partition_ids)
        self.rdma_store.set_total_vectors(total)
        self.rdma_store.set_partition_assignments(self.partition_ids.astype(np.int32))

        # Exclude orchestrator (no RDMA) — only DPU worker nodes
        sorted_node_ids = sorted(n for n in self.cluster.nodes.keys()
                                 if n != self.cluster.coordinator_node)
        self.rdma_store.set_partition_node_mapping(sorted_node_ids)

        # Colocation data layout: no separate vector MR needed
        self.rdma_store.enable_detailed_latency(True)
        self.rdma_store.init_rdma()
        print(f"[Node {self.node_id}] RDMA initialized (colocation layout)")

    def get_latency_stats(self) -> dict:
        if self.rdma_store is not None:
            return self.rdma_store.get_latency_stats()
        return {}


# ============================================================================
# 3-Stage Pipeline Stats
# ============================================================================

@dataclass
class PipelineStats:
    total_us: float = 0.0

    neighbor_fetch_us: float = 0.0
    neighbor_fetch_local_us: float = 0.0
    neighbor_fetch_remote_us: float = 0.0

    distance_calc_us: float = 0.0
    distance_calc_local_us: float = 0.0
    distance_calc_remote_us: float = 0.0

    queue_update_us: float = 0.0
    pipeline_sum_us: float = 0.0

    rdma_latency_us: float = 0.0
    rdma_fetch_us: float = 0.0
    rdma_distance_us: float = 0.0
    rdma_total_us: float = 0.0

    local_vector_accesses: int = 0
    remote_vector_accesses: int = 0
    local_neighbor_accesses: int = 0
    remote_neighbor_accesses: int = 0
    rdma_round_trips: int = 0

    n_hops: int = 0
    n_cmps: int = 0

    rdma_neighbor_ns_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    rdma_batch_ns_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    rdma_batch_peer_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    rdma_batch_sizes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    rdma_per_peer_us: dict = field(default_factory=dict)

    # Fine-grained sub-breakdown (us)
    dist_cache_check_us: float = 0.0
    dist_cache_put_us: float = 0.0
    dist_vid_resolution_us: float = 0.0
    dist_pure_distance_us: float = 0.0
    fetch_cache_check_us: float = 0.0
    fetch_cache_put_us: float = 0.0
    fetch_vid_resolution_us: float = 0.0

    # Cache stats
    cache_vec_hits: int = 0
    cache_vec_misses: int = 0
    cache_nbr_hits: int = 0
    cache_nbr_misses: int = 0
    cache_vec_evictions: int = 0
    cache_nbr_evictions: int = 0

    @property
    def cache_vec_hit_rate(self) -> float:
        total = self.cache_vec_hits + self.cache_vec_misses
        return self.cache_vec_hits / total if total > 0 else 0.0

    @property
    def cache_nbr_hit_rate(self) -> float:
        total = self.cache_nbr_hits + self.cache_nbr_misses
        return self.cache_nbr_hits / total if total > 0 else 0.0

    @property
    def remote_vector_ratio(self) -> float:
        total = self.local_vector_accesses + self.remote_vector_accesses
        return self.remote_vector_accesses / total if total > 0 else 0

    def to_dict(self) -> dict:
        return {
            "total_us": round(self.total_us, 2),
            "pipeline_sum_us": round(self.pipeline_sum_us, 2),
            "neighbor_fetch_us": round(self.neighbor_fetch_us, 2),
            "neighbor_fetch_local_us": round(self.neighbor_fetch_local_us, 2),
            "neighbor_fetch_remote_us": round(self.neighbor_fetch_remote_us, 2),
            "distance_calc_us": round(self.distance_calc_us, 2),
            "distance_calc_local_us": round(self.distance_calc_local_us, 2),
            "distance_calc_remote_us": round(self.distance_calc_remote_us, 2),
            "queue_update_us": round(self.queue_update_us, 2),
            "rdma_latency_us": round(self.rdma_latency_us, 2),
            "rdma_fetch_us": round(self.rdma_fetch_us, 2),
            "rdma_distance_us": round(self.rdma_distance_us, 2),
            "rdma_total_us": round(self.rdma_total_us, 2),
            "local_vector_accesses": self.local_vector_accesses,
            "remote_vector_accesses": self.remote_vector_accesses,
            "local_neighbor_accesses": self.local_neighbor_accesses,
            "remote_neighbor_accesses": self.remote_neighbor_accesses,
            "remote_vector_ratio": round(self.remote_vector_ratio, 4),
            "rdma_round_trips": self.rdma_round_trips,
            "n_hops": self.n_hops,
            "n_cmps": self.n_cmps,
            "rdma_neighbor_ns_samples": self.rdma_neighbor_ns_samples.tolist() if len(self.rdma_neighbor_ns_samples) > 0 else [],
            "rdma_batch_ns_samples": self.rdma_batch_ns_samples.tolist() if len(self.rdma_batch_ns_samples) > 0 else [],
            "rdma_batch_peer_ids": self.rdma_batch_peer_ids.tolist() if len(self.rdma_batch_peer_ids) > 0 else [],
            "rdma_batch_sizes": self.rdma_batch_sizes.tolist() if len(self.rdma_batch_sizes) > 0 else [],
            "rdma_per_peer_us": self.rdma_per_peer_us,
            "dist_cache_check_us": round(self.dist_cache_check_us, 2),
            "dist_cache_put_us": round(self.dist_cache_put_us, 2),
            "dist_vid_resolution_us": round(self.dist_vid_resolution_us, 2),
            "dist_pure_distance_us": round(self.dist_pure_distance_us, 2),
            "fetch_cache_check_us": round(self.fetch_cache_check_us, 2),
            "fetch_cache_put_us": round(self.fetch_cache_put_us, 2),
            "fetch_vid_resolution_us": round(self.fetch_vid_resolution_us, 2),
            "cache_vec_hits": self.cache_vec_hits,
            "cache_vec_misses": self.cache_vec_misses,
            "cache_nbr_hits": self.cache_nbr_hits,
            "cache_nbr_misses": self.cache_nbr_misses,
            "cache_vec_evictions": self.cache_vec_evictions,
            "cache_nbr_evictions": self.cache_nbr_evictions,
            "cache_vec_hit_rate": round(self.cache_vec_hit_rate, 4),
            "cache_nbr_hit_rate": round(self.cache_nbr_hit_rate, 4),
        }


# ============================================================================
# Utility
# ============================================================================

def load_fbin(filepath: str) -> np.ndarray:
    with open(filepath, 'rb') as f:
        npts = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=npts * dim)
    return data.reshape(npts, dim)


def load_tags_file(filepath: str) -> np.ndarray:
    """Load tags from DiskANN binary format: [npts (uint32), dim=1 (uint32), tags...]"""
    with open(filepath, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        num_tags = int(header[0])
        tags = np.fromfile(f, dtype=np.uint32, count=num_tags)
    return tags


# ============================================================================
# Node Worker
# ============================================================================

class NodeWorker:
    """Per-node worker that executes distributed pipeline stages."""

    _rdma_monitor_warned = False

    def __init__(self, config_path: str, node_id: int,
                 active_nodes: List[int], coordinator: int = None,
                 stage: str = "search", timestamp: str = None,
                 orchestrator_host: str = None, orchestrator_port: int = 19500,
                 project_root: str = None, overrides: List[str] = None):
        self.node_id = node_id
        self.stage = stage
        self.forced_timestamp = timestamp
        self.config_path = config_path
        self.active_nodes = list(active_nodes)

        # Project root for resolving relative paths (default: /workspace in container)
        self.project_root = Path(project_root).resolve() if project_root else Path("/workspace")

        # Load config + resolve paths
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        apply_config_overrides(self.config, overrides or [], log=True)
        resolve_config_paths(self.config, self.project_root)

        config_dir = str(Path(config_path).parent)
        cluster_cfg = self.config.get('cluster', {})

        # Cluster setup: build_full injects orchestrator as node 0
        coord = coordinator if coordinator is not None else 0
        self.cluster, rdma_cfg = ClusterConfig.build_full(
            cluster_cfg, config_dir, coordinator_node=coord,
            active_nodes=active_nodes)

        # CLI override for orchestrator host/port
        if orchestrator_host and 0 in self.cluster.nodes:
            self.cluster.nodes[0].host = orchestrator_host
        if orchestrator_port and 0 in self.cluster.nodes:
            self.cluster.nodes[0].sync_port = orchestrator_port

        self.config['rdma'] = {**self.config.get('rdma', {}), **rdma_cfg}
        self.is_coordinator = (node_id == self.cluster.coordinator_node)

        # State
        self.sync = None
        self.rdma_store_wrapper = None
        self.vectors = None
        self.queries = None
        self.partition_ids = None
        self.centroids = None
        self.tags = None
        self.base_file = None
        self.output_dir = None
        self.colocation_mode = False
        self.colocation_meta = None

    def log(self, msg: str):
        print(f"[Node {self.node_id}] {msg}")

    def run(self):
        self.log("=" * 60)
        self.log(f"NODE WORKER — stage: {self.stage}")
        self.log("=" * 60)

        try:
            self._setup_sync()
            # Receive timestamp from orchestrator (broadcast before data-loaded)
            self.forced_timestamp = self.sync.recv_value("timestamp")
            self.log(f"Received timestamp: {self.forced_timestamp}")
            self._load_data()
            self._setup_rdma()
            if self.stage == "search":
                self._run_search()
            elif self.stage == "distributed_engine":
                self._run_distributed_engine()
        except Exception as e:
            self.log(f"FAILED: {e}")
            self._mark_failed()
            raise
        finally:
            # Explicitly release RDMA resources then force-exit.
            # os._exit() MUST be called before any Python object destruction
            # triggers pybind11 C++ destructors — those cause SIGSEGV due to
            # RDMA resource ordering issues at interpreter shutdown.
            if self.rdma_store_wrapper and self.rdma_store_wrapper.rdma_store:
                try:
                    self.rdma_store_wrapper.rdma_store.cleanup()
                except Exception:
                    pass
            if self.sync:
                self.sync.close()
            import os as _os
            _os._exit(0)

    def _mark_failed(self):
        """Rename output dir: {timestamp} -> FAILED_{timestamp}."""
        if self.output_dir and self.output_dir.exists():
            failed_name = "FAILED_" + self.output_dir.name
            failed_dir = self.output_dir.parent / failed_name
            if not failed_dir.exists():
                try:
                    self.output_dir.rename(failed_dir)
                    self.output_dir = failed_dir
                    self.log(f"Output dir renamed to: {failed_dir}")
                except OSError as e:
                    self.log(f"Could not rename output dir: {e}")

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def _setup_sync(self):
        self.log("Setting up TCP sync...")
        self.sync = TCPSync(self.cluster, self.node_id)
        self.sync.setup()
        # No barrier("init") — orchestrator doesn't participate.
        # Connection is verified by setup() itself.
        self.log("TCP sync ready")

    # ------------------------------------------------------------------
    # Data loading (reads pipeline output from NFS)
    # ------------------------------------------------------------------

    def _load_data(self):
        self.log("Loading data...")
        dataset_cfg = self.config.get('dataset', {})
        dataset_name = dataset_cfg.get('name', '')

        dtype_map = {
            'float32': np.float32, 'int8': np.int8,
            'uint8': np.uint8, 'float16': np.float16,
        }

        data_root = dataset_cfg.get('data_root', '')
        cfg_base = dataset_cfg.get('base_file', '')
        cfg_query = dataset_cfg.get('query_file', '')

        if cfg_base and data_root:
            # Config has explicit paths — use them directly
            from pathlib import Path
            self.base_file = str(Path(data_root) / cfg_base)
            query_file = str(Path(data_root) / cfg_query)
            self.data_type = dataset_cfg.get('data_type', 'float32')
            dtype = dtype_map.get(self.data_type, np.float32)
        elif dataset_name:
            dataset_info = get_dataset_info(dataset_name, data_root)
            self.base_file = dataset_info['base_file']
            query_file = dataset_info['query_file']
            self.data_type = dataset_info.get('data_type', 'float32')
            dtype = dtype_map.get(self.data_type, np.float32)
        else:
            self.base_file = cfg_base
            query_file = cfg_query
            self.data_type = dataset_cfg.get('data_type', 'float32')
            dtype = dtype_map.get(self.data_type, np.float32)

        # Only read header (npts, dim) — don't load full base vectors into memory.
        # Colocation files already contain all vectors per partition.
        self.total_vectors, self.vector_dim = get_bin_metadata(self.base_file)
        self.vectors = None  # Not loaded; colocation has the data
        self.queries = load_bin_data(query_file, dtype=dtype)
        self.log(f"Vectors: {self.total_vectors} x {self.vector_dim} (header only), "
                 f"Queries: {self.queries.shape}")

        # Load partitions from pipeline output
        self._load_partitions()

        self.sync.barrier("data-loaded")

    def _load_partitions(self):
        """Load partition data from pipeline output directory."""
        num_partitions = self.config.get('partition', {}).get('num_partitions', 4)
        strategy_name = get_insert_strategy(self.config)

        partition_dir = None
        searched = []
        for artifact_dir in artifact_dir_candidates(self.config):
            candidate = build_partition_dir(
                artifact_dir, num_partitions, strategy_name,
            )
            searched.append(candidate)
            if candidate.exists():
                partition_dir = candidate
                break
        self.log(f"Loading partitions from: {partition_dir or searched[0]}")

        if partition_dir is None:
            joined = ", ".join(str(p) for p in searched)
            raise FileNotFoundError(f"Partition directory not found. Searched: {joined}")

        # Centroids
        centroids_path = partition_dir / "centroids.fbin"
        self.centroids = load_fbin(str(centroids_path))
        self.log(f"  Centroids: {self.centroids.shape}")

        num_p = len(self.centroids)

        # Per-partition IDs → reconstruct global partition_ids array
        partition_ids_list = []
        for p in range(num_p):
            ids_path = partition_dir / f"partition{p}" / "ids.bin"
            if not ids_path.exists():
                raise FileNotFoundError(
                    f"Partition ids file not found: {ids_path}. "
                    f"Cannot fallback to centroid assignment (vectors not loaded for memory optimization)."
                )

            with open(ids_path, 'rb') as f:
                npts = np.fromfile(f, dtype=np.uint32, count=1)[0]
                ids = np.fromfile(f, dtype=np.uint32, count=npts)
            partition_ids_list.append((p, ids))
            self.log(f"  Partition {p}: {len(ids)} vectors")

        total_vectors = self.total_vectors
        self.partition_ids = np.zeros(total_vectors, dtype=np.uint32)

        for partition_id, vector_ids in partition_ids_list:
            valid_mask = vector_ids < total_vectors
            valid_ids = vector_ids[valid_mask]
            self.partition_ids[valid_ids] = partition_id

    def _compute_partition_ids(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        centroid_norms = np.sum(centroids ** 2, axis=1)
        partition_ids = np.empty(len(vectors), dtype=np.uint32)
        batch_size = 100000
        for i in range(0, len(vectors), batch_size):
            end = min(i + batch_size, len(vectors))
            batch = vectors[i:end]
            dots = batch @ centroids.T
            sq_dists = centroid_norms - 2 * dots
            partition_ids[i:end] = np.argmin(sq_dists, axis=1)
        return partition_ids

    # ------------------------------------------------------------------
    # RDMA setup
    # ------------------------------------------------------------------


    def _resolve_store_numa_env(self, rdma_device: str):
        """DANNS_STORE_NUMA_NODE=nic|far -> concrete node id BEFORE the
        colocation store allocates (C++ reads the env at load; the NIC's
        socket differs per host). Numeric values pass through; unresolvable
        symbolic values are unset (legacy first-touch)."""
        val = os.environ.get('DANNS_STORE_NUMA_NODE', '').strip().lower()
        if not val or val.lstrip('-').isdigit():
            return
        try:
            if val not in ('nic', 'far'):
                raise ValueError(f"unrecognized value {val!r}")
            nic = int(open(f'/sys/class/infiniband/{rdma_device}/device/numa_node')
                      .read())
            if nic < 0:
                raise ValueError(f'{rdma_device} reports numa_node=-1')
            if val == 'nic':
                node = nic
            else:
                nodes = sorted(
                    int(d[4:]) for d in os.listdir('/sys/devices/system/node')
                    if d.startswith('node') and d[4:].isdigit())
                node = next((n for n in nodes if n != nic), None)
                if node is None:
                    raise ValueError('single NUMA node')
            os.environ['DANNS_STORE_NUMA_NODE'] = str(node)
            self.log(f"Store NUMA bind: '{val}' -> node {node} (NIC socket={nic})")
        except Exception as exc:
            os.environ.pop('DANNS_STORE_NUMA_NODE', None)
            self.log(f"Store NUMA bind: cannot resolve '{val}' ({exc})"
                     " — unset, legacy first-touch")

    def _maybe_pin_nic_numa(self, rdma_device: str):
        """Opt-in (DANNS_PIN_NIC_NUMA=1): pin this worker to the PHYSICAL
        cores of the NUMA node that owns the RDMA NIC. NUMA experiment
        2026-06-12: NIC socket differs per host (hosts A/B: node1,
        hosts C/D: node0), so resolve it from sysfs instead of config.
        DANNS_PIN_NIC_NUMA=other pins to the REMOTE socket (control arm)."""
        mode = os.environ.get('DANNS_PIN_NIC_NUMA', '').strip().lower()
        if mode not in ('1', 'true', 'nic', 'other', 'allphys', 'all'):
            return
        try:
            cpus = set()
            numa = -1
            if mode in ('allphys', 'all'):
                # All physical cores across BOTH sockets (thread-scaling control).
                for d in os.listdir('/sys/devices/system/node'):
                    if d.startswith('node') and d[4:].isdigit():
                        cl = open(f'/sys/devices/system/node/{d}/cpulist').read().strip()
                        for part in cl.split(','):
                            a, _, b = part.partition('-')
                            cpus.update(range(int(a), int(b or a) + 1))
            else:
                numa = int(open(f'/sys/class/infiniband/{rdma_device}/device/numa_node').read())
                if numa < 0:
                    self.log('[pin] NIC numa_node unknown, skipping')
                    return
                if mode == 'other':
                    nodes = sorted(int(d[4:]) for d in os.listdir('/sys/devices/system/node')
                                   if d.startswith('node') and d[4:].isdigit())
                    numa = next(n for n in nodes if n != numa)
                cpulist = open(f'/sys/devices/system/node/node{numa}/cpulist').read().strip()
                for part in cpulist.split(','):
                    a, _, b = part.partition('-')
                    cpus.update(range(int(a), int(b or a) + 1))
            # physical cores only: keep the first sibling of each core
            phys = set()
            seen_cores = set()
            for c in sorted(cpus):
                sib = open(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read().strip()
                pkg = open(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read().strip()
                key = (pkg, sib)
                if key not in seen_cores:
                    seen_cores.add(key)
                    phys.add(c)
            os.sched_setaffinity(0, phys)
            self.log(f'[pin] mode={mode} numa{numa} physical cpus={sorted(phys)}')
        except Exception as e:
            self.log(f'[pin] failed ({e}), continuing unpinned')

    def _setup_rdma(self):
        self.log("Setting up RDMA...")
        if not _HAS_RDMA:
            raise RuntimeError("RDMA bindings required for distributed stages")

        rdma_cfg = self.config.get('rdma', {})
        my_node = self.cluster.get_node(self.node_id)
        rdma_device = (
            my_node.rdma_device
            if my_node and my_node.rdma_device
            else rdma_cfg.get('device', '')
        )
        gid_index = (
            my_node.gid_index
            if my_node and my_node.gid_index is not None
            else rdma_cfg.get('gid_index', 1)
        )
        self.log(f"RDMA device={rdma_device or '<auto>'}, gid_index={gid_index}")
        # Single source of truth for NIC monitors: rdma_monitor.sh and
        # nic_bw_monitor.py honor RDMA_DEVICE, but the container env may carry
        # a wrong per-host value (e.g. compose sets mlx5_1 on hosts A/B whose
        # NIC is mlx5_0). Export the node's resolved device so every monitor
        # subprocess spawned from this worker samples the NIC the engine
        # actually uses.
        if rdma_device:
            os.environ['RDMA_DEVICE'] = rdma_device
        self._maybe_pin_nic_numa(rdma_device)
        self._resolve_store_numa_env(rdma_device)
        self.rdma_store_wrapper = RDMAVectorStore(
            node_id=self.node_id,
            cluster=self.cluster,
            vector_dim=self.vector_dim,
            rdma_device=rdma_device,
            gid_index=gid_index
        )

        coloc_cfg = self.config.get('colocation', {})
        colocation_only = coloc_cfg.get('enabled', False)

        # Resolve partition directory for loading local vectors
        num_partitions = self.config.get('partition', {}).get('num_partitions', 4)
        strategy_name = get_insert_strategy(self.config)
        partition_dir_path = None
        for artifact_dir in artifact_dir_candidates(self.config):
            candidate = build_partition_dir(
                artifact_dir, num_partitions, strategy_name,
            )
            if candidate.exists():
                partition_dir_path = candidate
                break
        if partition_dir_path is None:
            partition_dir_path = build_partition_dir(
                artifact_dir_candidates(self.config)[0], num_partitions, strategy_name,
            )
        partition_dir = str(partition_dir_path)

        self.rdma_store_wrapper.load_data_lightweight(
            total_vectors=self.total_vectors,
            partition_ids=self.partition_ids,
            partition_dir=partition_dir,
            colocation_only=colocation_only,
        )
        self.rdma_store_wrapper.setup_rdma(colocation_only=colocation_only)

        self.sync.barrier("rdma-init")

        # Phase 1: Ordered pairwise QP connect (barrier-synchronized)
        # Exclude orchestrator (no RDMA) — only DPU worker nodes
        all_node_ids = sorted(n for n in self.cluster.nodes.keys()
                              if n != self.cluster.coordinator_node)
        for i in range(len(all_node_ids)):
            for j in range(i + 1, len(all_node_ids)):
                node_a, node_b = all_node_ids[i], all_node_ids[j]
                if self.node_id == node_a or self.node_id == node_b:
                    peer_id = node_b if self.node_id == node_a else node_a
                    peer = self.cluster.nodes[peer_id]
                    if self.node_id < peer_id:
                        self.log(f"Will listen for peer {peer_id}...")
                    else:
                        self.log(f"Connecting to peer {peer_id}...")
                    self.rdma_store_wrapper.rdma_store.connect_to_peer(
                        peer_id, peer.host, peer.rdma_port)
                    self.log(f"Connected to peer {peer_id}")
                self.sync.barrier(f"qp-connect-{node_a}-{node_b}")

        # Phase 1b: live-redirect alloc channels — two dedicated QPs per
        # pair on rdma_port+1000/+2000 (docs/features/live_redirect_design.md:
        # role-split so the alloc server thread and the main thread never
        # share a CQ). Same ordered bootstrap + per-pair barriers as Phase 1.
        # Gating must match orchestrator._relay_rdma_barriers exactly.
        lb_cfg = self.config.get('load_balancer') or {}
        redirect_on = bool(lb_cfg.get('enabled', False)) and \
            bool(lb_cfg.get('redirect', False))
        if redirect_on:
            rs = self.rdma_store_wrapper.rdma_store
            for i in range(len(all_node_ids)):
                for j in range(i + 1, len(all_node_ids)):
                    node_a, node_b = all_node_ids[i], all_node_ids[j]
                    if self.node_id == node_a or self.node_id == node_b:
                        peer_id = node_b if self.node_id == node_a else node_a
                        peer = self.cluster.nodes[peer_id]
                        self.log(f"Connecting redirect alloc channels to peer {peer_id}...")
                        rs.connect_redirect_peer(
                            peer_id, peer.host,
                            peer.rdma_port + 1000, peer.rdma_port + 2000,
                            rdma_device=self.rdma_store_wrapper.rdma_device,
                            gid_index=getattr(self.rdma_store_wrapper, 'gid_index', 1))
                        self.log(f"Redirect alloc channels up with peer {peer_id}")
                    self.sync.barrier(f"redirect-qp-{node_a}-{node_b}")
            rs.start_redirect_server()
            self.sync.barrier("redirect-server-ready")

        # Phase 2: Colocation MR exchange (done in _setup_colocation)
        # Separated data layout MR exchange is deprecated — data layout is
        # always colocation.  Peer MRs for .coloc files are exchanged below.
        self.sync.barrier("rdma-qp-ready")
        self.sync.barrier("rdma-connected")

        # Phase 3: Co-location setup (always — data layout is colocation)
        self._setup_colocation()

        self.log("RDMA setup complete")

    # ------------------------------------------------------------------
    # Co-location setup
    # ------------------------------------------------------------------

    def _setup_colocation(self):
        """Load co-location binary and exchange co-location MR with peers."""
        rdma_store = self.rdma_store_wrapper.rdma_store
        coloc_cfg = self.config.get('colocation', {})
        fetch_mode = coloc_cfg.get('fetch_mode', 'colocation')

        # Find co-location files from pipeline output
        coloc_cfg = self.config.get('colocation', {})
        experiment_name = self.config.get('experiment_name', 'unnamed')
        num_partitions = self.config.get('partition', {}).get('num_partitions', 2)
        strategy_name = get_insert_strategy(self.config)

        experiment_dir = artifact_dir_candidates(self.config)[0]
        graph_file = None
        for candidate_dir in artifact_dir_candidates(self.config):
            try:
                graph_file, _ = self._find_graph_checkpoint(candidate_dir)
                experiment_dir = candidate_dir
                break
            except FileNotFoundError:
                continue
        if graph_file is None:
            graph_file, _ = self._find_graph_checkpoint(experiment_dir)
        iteration = _extract_iteration_from_graph(graph_file)

        # graph_variant: use as colocation subdirectory name instead of "iteration{N}"
        graph_variant = self.config.get('build', {}).get('graph_variant', None)
        coloc_subdir = graph_variant if graph_variant else f"iteration{iteration}"

        # Local SSD path resolution:
        #   1. colocation.local_dir in config (global override, single path for all)
        #   2. cluster.nodes[self.node_id].local_ssd (per-DPU, from nodes.yaml)
        #   3. fall back to base_dir (NFS)
        # Per-DPU is the right default when worker-host uses /data and others use
        # /localssd — the deploy scripts already mirror that layout.
        my_node = self.cluster.get_node(self.node_id)
        node_local_ssd = getattr(my_node, 'local_ssd', '') or ''
        if coloc_cfg.get('local_dir'):
            coloc_base = Path(coloc_cfg['local_dir'])
        elif node_local_ssd:
            coloc_base = Path(node_local_ssd) / "scripts/experiments"
            self.log(f"  Using per-node local_ssd: {coloc_base}")
        else:
            coloc_base = artifact_experiment_dir(self.config)
        if coloc_cfg.get('local_dir') or node_local_ssd:
            candidates = colocation_dir_candidates(
                coloc_base, experiment_name, num_partitions, strategy_name,
                variant=coloc_subdir,
            )
        else:
            candidates = [build_colocation_dir(
                coloc_base, num_partitions, strategy_name,
                variant=coloc_subdir,
            )]
        coloc_dir = None
        meta_path = None
        for candidate in candidates:
            candidate_meta = candidate / "colocation_meta.json"
            if candidate_meta.exists():
                coloc_dir = candidate
                meta_path = candidate_meta
                break

        if meta_path is None or coloc_dir is None:
            searched = "\n    ".join(str(c / "colocation_meta.json") for c in candidates)
            msg = (
                "colocation_meta.json not found. Searched:\n    "
                f"{searched}"
            )
            if coloc_cfg.get('enabled', False):
                raise FileNotFoundError(msg)
            self.log(f"[WARN] {msg} — skipping co-location")
            return
        self.log(f"  Colocation dir: {coloc_dir}")

        with open(meta_path) as f:
            meta = json.load(f)

        self.colocation_meta = meta

        expected_partitions = {str(p) for p in range(num_partitions)}
        for section in ('partitions', 'tag_to_local_files', 'local_to_tag_files'):
            present = set((meta.get(section, {}) or {}).keys())
            missing = expected_partitions - present
            if missing:
                raise FileNotFoundError(
                    f"colocation_meta.json missing {section} entries for "
                    f"partitions {sorted(missing)} at {meta_path}"
                )

        # Determine this node's partition (exclude coordinator — no RDMA/data).
        # Defaults to sorted worker nodes -> dense partitions, but can be
        # overridden via colocation.node_partition_map for 8-node/future runs.
        all_node_ids = sorted(n for n in self.cluster.nodes.keys()
                              if n != self.cluster.coordinator_node)
        node_partition_map = mapping_from_config(self.config, all_node_ids)
        my_partition = node_partition_map[self.node_id]
        # 검색 전용(colocation-only) 경로에서도 이후 단계(_set_partition_entry_points
        # 의 frozen 조회 등)가 쓸 수 있게 속성으로 저장 — update 경로의 353행과 통일.
        self.my_partition = my_partition
        partition_info = meta['partitions'][str(my_partition)]

        coloc_file = str(coloc_dir / partition_info['file'])
        g2v_file = str(coloc_dir / meta['global_to_vectorid'])

        if not os.path.exists(coloc_file):
            raise FileNotFoundError(
                f"Local co-location payload for node {self.node_id} "
                f"(partition {my_partition}) not found: {coloc_file}"
            )
        if not os.path.exists(g2v_file):
            raise FileNotFoundError(
                f"global_to_vectorid file not found: {g2v_file}"
            )

        # Build tag_to_local_files dict: {partition_id: filepath}
        tag_to_local_files = {}
        for p_str, t2l_filename in meta.get('tag_to_local_files', {}).items():
            t2l_path = str(coloc_dir / t2l_filename)
            if os.path.exists(t2l_path):
                tag_to_local_files[int(p_str)] = t2l_path
            else:
                raise FileNotFoundError(
                    f"Required tag_to_local file not found: {t2l_path}"
                )

        # Build local_to_tag_files dict: {partition_id: filepath}
        local_to_tag_files = {}
        for p_str, l2t_filename in meta.get('local_to_tag_files', {}).items():
            l2t_path = str(coloc_dir / l2t_filename)
            if os.path.exists(l2t_path):
                local_to_tag_files[int(p_str)] = l2t_path
            else:
                raise FileNotFoundError(
                    f"Required local_to_tag file not found: {l2t_path}"
                )

        # Load co-location binary into C++
        self.log(f"Loading co-location: {coloc_file}")
        t0 = time.perf_counter()
        rdma_store.load_colocation(
            coloc_file=coloc_file,
            global_to_vectorid_file=g2v_file,
            tag_to_local_files=tag_to_local_files,
            local_to_tag_files=local_to_tag_files,
            partition_id=my_partition,
            dim=meta['dim'],
            R=meta['R'],
            slack_R=meta['slack_R'],
            entry_size=meta['entry_size'],
            alignment=meta['alignment'],
            num_entries=partition_info['num_entries'],
            total_vectors=meta['total_vectors'],
            element_size=meta.get('element_size', 4),
        )
        rdma_store.set_fetch_mode(fetch_mode)
        # VID sanity bound for the search hot path (resolve_vid): torn VIDs
        # under concurrent consolidation must not index past any node's store.
        max_locs = max(int(info.get('num_entries', 0) or 0)
                       for info in meta['partitions'].values())
        if max_locs > 0 and hasattr(rdma_store, 'set_partition_max_local_locs'):
            rdma_store.set_partition_max_local_locs(max_locs)
            self.log(f"  VID bound: max_local_locs={max_locs}")
        self.log(f"Co-location loaded in {time.perf_counter() - t0:.2f}s "
                 f"(partition {my_partition}, {partition_info['num_entries']} entries, "
                 f"mode={fetch_mode})")

        self.sync.barrier("coloc-loaded")

        # Exchange co-location MR with peers (exclude orchestrator).
        # load_balancer.enabled additionally exchanges the 8-byte vector
        # counter MR inside the same ordered handshake (Stage B). The config
        # is shared, so every pair takes the same branch symmetrically.
        exchange_counter = bool(
            (self.config.get('load_balancer') or {}).get('enabled', False))
        # Claim-counter MR for work stealing. Driven by search.work_stealing so
        # every node evaluates the same value -- the exchange is an ordered
        # send/recv pair and an asymmetric flag deadlocks the handshake.
        exchange_claim = bool(
            (self.config.get('search', {}) or {}).get('work_stealing', False))
        for i in range(len(all_node_ids)):
            for j in range(i + 1, len(all_node_ids)):
                node_a, node_b = all_node_ids[i], all_node_ids[j]
                if self.node_id == node_a or self.node_id == node_b:
                    peer_id = node_b if self.node_id == node_a else node_a
                    self.log(f"Exchanging co-location MR with peer {peer_id}..."
                             f"{' (+counter MR)' if exchange_counter else ''}"
                             f"{' (+claim MR)' if exchange_claim else ''}")
                    rdma_store.exchange_coloc_mr_with_peer(
                        peer_id, exchange_counter=exchange_counter,
                        exchange_claim=exchange_claim)
                    self.log(f"Co-location MR exchanged with peer {peer_id}")
                self.sync.barrier(f"coloc-mr-{node_a}-{node_b}")

        # LOCAL_MMAP: cross-deployed peer partitions read from local RAM.
        # Requires partition{N}.coloc for every peer N to be present in
        # coloc_dir on this node (pre-staged by user). Pages are populated
        # via MAP_POPULATE so the consolidate fetch path is pure RAM read,
        # matching the RDMA path's pinned-MR behavior. Hand the address to
        # C++ via register_local_partition_data; node_store::local_data()
        # then short-circuits the RDMA fetch sites in consolidation.cpp.
        if fetch_mode == "local_mmap":
            self._local_mmap_handles = []  # keep list alive for engine lifetime
            for peer_id in all_node_ids:
                if peer_id == self.node_id:
                    continue
                peer_partition = node_partition_map[peer_id]
                peer_coloc_file = coloc_dir / f"partition{peer_partition}.coloc"
                if not peer_coloc_file.exists():
                    raise FileNotFoundError(
                        f"local_mmap mode: peer {peer_id} (partition "
                        f"{peer_partition}) file not found at {peer_coloc_file} "
                        f"— cross-deploy partition{peer_partition}.coloc to this node"
                    )
                t0 = time.perf_counter()
                addr, size = _mmap_populate_readonly(str(peer_coloc_file))
                elapsed = time.perf_counter() - t0
                gb = size / (1024 ** 3)
                rate = gb / max(elapsed, 1e-9)
                self.log(
                    f"  [LOCAL_MMAP] peer {peer_id} (partition {peer_partition}): "
                    f"mmap+populate {gb:.2f} GB in {elapsed:.2f}s ({rate:.2f} GB/s); "
                    f"addr=0x{addr:x}"
                )
                rdma_store.register_local_partition_data(peer_id, addr)
                self._local_mmap_handles.append((peer_id, addr, size))
            self.log(f"[LOCAL_MMAP] {len(self._local_mmap_handles)} peer "
                     f"partitions registered (RDMA path bypassed in fetch)")

        self.colocation_mode = True

        # Load PQ filter data if configured. Split artifact configs are strict:
        # PQ lives under the canonical artifact pq/M{n}/ directory. A local SSD
        # pq/M{n}/ override is still accepted for explicitly staged local PQ.
        pq_cfg = get_pq_filter_config(self.config)
        if pq_cfg.get('enabled', False):
            num_chunks = int(pq_cfg.get('num_chunks', 12))
            local_pq_dir = build_pq_dir(coloc_base / experiment_name, num_chunks)
            artifact_pq_dirs = [
                build_pq_dir(artifact_dir, num_chunks)
                for artifact_dir in artifact_dir_candidates(self.config)
            ]
            candidate_dirs = [(local_pq_dir, "local pq/")]
            candidate_dirs += [(d, "artifact pq/") for d in artifact_pq_dirs]
            codebook_path = None
            all_codes_path = None
            pq_source = None
            for d, label in candidate_dirs:
                cb = d / "pq_codebook.bin"
                ac = d / "pq_all_codes.bin"
                if cb.exists() and ac.exists():
                    codebook_path = cb
                    all_codes_path = ac
                    pq_source = label
                    break
            if codebook_path is not None:
                alpha = float(pq_cfg.get('alpha', 0.1))
                # max_capacity: from pq_meta.json > pq_filter config > build config > 0 (auto)
                max_capacity = int(pq_cfg.get('max_capacity', 0))
                if max_capacity == 0:
                    pq_meta_path = codebook_path.parent / "pq_meta.json"
                    if pq_meta_path.exists():
                        with open(pq_meta_path) as _f:
                            pq_meta = json.load(_f)
                        file_M = int(pq_meta.get('num_chunks', num_chunks))
                        if file_M != num_chunks:
                            self.log(f"[WARN] PQ num_chunks mismatch at {codebook_path.parent}: "
                                     f"file M={file_M}, config M={num_chunks}")
                        max_capacity = int(pq_meta.get('max_capacity', 0))
                if max_capacity == 0:
                    max_capacity = int(self.config.get('build', {}).get('max_capacity', 0))
                self.log(f"Loading PQ filter data from {pq_source}{codebook_path.parent}: "
                         f"M={num_chunks}, alpha={alpha}, max_capacity={max_capacity}")
                t0 = time.perf_counter()
                rdma_store.load_pq_data(str(codebook_path), str(all_codes_path),
                                        alpha, max_capacity)
                self.log(f"PQ filter loaded in {time.perf_counter() - t0:.3f}s")
            else:
                searched = ", ".join(str(d) for d, _ in candidate_dirs)
                self.log(f"[WARN] PQ files not found in any of: {searched} — PQ filter disabled")

        self.sync.barrier("coloc-ready")
        self.log(f"Co-location setup complete (fetch_mode={fetch_mode})")

    # ------------------------------------------------------------------
    # Distributed Update
    # ------------------------------------------------------------------

    def _run_distributed_engine(self):
        """Execute distributed update on this node."""
        from pipeline import engine as _dengine

        self.log("Starting distributed update stage...")

        # Ensure co-location is set up (needed for RDMA addressing)
        coloc_cfg = self.config.get('colocation', {})
        if not self.colocation_mode and coloc_cfg.get('enabled', False):
            self._setup_colocation()

        # Resolve timestamp-based run directory
        timestamp = self.forced_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_coordinator:
            self.sync.broadcast_value("dupdate_timestamp", timestamp)
        else:
            timestamp = self.sync.recv_value("dupdate_timestamp")

        engine = _dengine.OperationEngine(self, self.config, timestamp=timestamp)
        engine.load()
        engine.execute_workload()
        engine.save()

        self.log("Distributed update stage complete")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _run_search(self):
        self.log("Starting RDMA search...")
        rdma_store = self.rdma_store_wrapper.rdma_store

        # Resolve graph/tags from pipeline output
        experiment_dir = artifact_dir_candidates(self.config)[0]
        graph_file = tags_file = None
        for candidate_dir in artifact_dir_candidates(self.config):
            try:
                graph_file, tags_file = self._find_graph_checkpoint(candidate_dir)
                experiment_dir = candidate_dir
                break
            except FileNotFoundError:
                continue
        if graph_file is None or tags_file is None:
            graph_file, tags_file = self._find_graph_checkpoint(experiment_dir)
        self.log(f"Graph: {graph_file}")
        self.log(f"Tags:  {tags_file}")

        # Colocation data layout: graph is embedded in .coloc files, so we only
        # need the distance function and tags here.  No full graph load needed.
        t0 = time.perf_counter()
        rdma_store.load_tags(tags_file)
        rdma_store.init_distance_only(self.vector_dim, self.data_type)
        self.log(f"Distance fn initialized in {time.perf_counter() - t0:.2f}s "
                 f"(colocation layout, graph not loaded)")

        # Load tags for partition mapping
        self.tags = load_tags_file(tags_file)

        # Inject virtual centroid nodes as per-partition entry points
        self._set_partition_entry_points(rdma_store, experiment_dir)

        # Create partition_ids_by_location
        partition_ids_by_location = self._create_partition_ids_by_location(rdma_store)

        # Load pre-computed GT (from pipeline gt stage)
        gt_locs = self._load_gt(experiment_dir)

        # Query routing
        my_partition = self.rdma_store_wrapper.my_partition
        search_cfg = self.config.get('search', {})
        K = search_cfg.get('K', 10)
        L_values = search_cfg.get('L_values', [50, 100])
        num_queries = search_cfg.get('num_queries', 1000)
        num_warmup = 100

        broadcast_mode = self.config.get('search', {}).get('broadcast', False)
        if broadcast_mode:
            routed_indices = np.arange(num_queries, dtype=np.int64)
            self.log(f"\nBroadcast mode: all {num_queries} queries on this node")
        else:
            query_partitions = self._precompute_query_partitions()
            effective_qp = np.array([query_partitions[i % len(self.queries)]
                                     for i in range(num_queries)])
            routed_indices = np.where(effective_qp == my_partition)[0]

            self.log(f"\nQuery routing: {len(routed_indices)}/{num_queries} queries "
                     f"assigned to this node (partition {my_partition})")
            for p in range(self.cluster.num_nodes):
                count = int(np.sum(effective_qp == p))
                self.log(f"  Partition {p}: {count} queries ({100.0*count/num_queries:.1f}%)")

        self.sync.barrier("search-ready")

        # Setup output directory
        timestamp = self.forced_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_coordinator:
            self.sync.broadcast_value("timestamp", timestamp)
        else:
            timestamp = self.sync.recv_value("timestamp")

        self.output_dir = build_run_dir(self.config, "search_rdma", timestamp)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Output directory: {self.output_dir}")

        # Save config snapshot for reproducibility (coordinator only)
        if self.is_coordinator and self.config_path:
            config_copy = self.output_dir / "config.yaml"
            shutil.copy2(self.config_path, config_copy)
            self.log(f"  Config saved to {config_copy}")

        # L value sweep
        all_results = {}

        for L in L_values:
            if L < K:
                continue

            self.log(f"\n  L={L}:")
            all_stats = []
            all_result_global_ids = []  # For recall computation

            # Select search function based on co-location mode
            use_colocation = self.colocation_mode
            search_fn_name = "search_colocation" if use_colocation else "search_with_partition_stats_rdma"
            self.log(f"    Mode: {search_fn_name}")

            verbose = self.config.get("output", {}).get("log_level", "info") == "debug"

            def do_search(q_vec, K_, L_, part_ids, my_part):
                if use_colocation:
                    return rdma_store.search_colocation(
                        q_vec, K_, L_, part_ids, my_part, verbose=verbose)
                else:
                    return rdma_store.search_with_partition_stats_rdma(
                        q_vec, K_, L_, part_ids, my_part)

            # Warmup
            warmup_count = min(num_warmup, len(routed_indices))
            self.log(f"    Warmup ({warmup_count} queries)...")
            for idx in routed_indices[:warmup_count]:
                q = self.queries[idx % len(self.queries)]
                do_search(q, K, L,
                          partition_ids_by_location.astype(np.uint32), my_partition)

            # Actual search
            self.log(f"    Searching ({len(routed_indices)} queries)...")
            search_t0 = time.perf_counter()

            for idx in routed_indices:
                q = self.queries[idx % len(self.queries)]
                indices, distances, cpp_stats = do_search(
                    q, K, L,
                    partition_ids_by_location.astype(np.uint32), my_partition)
                pipeline_stats = self._convert_to_pipeline_stats(cpp_stats)
                all_stats.append(pipeline_stats.to_dict())

                # Convert DiskANN locations to global vector IDs
                global_ids = np.full(K, -1, dtype=np.int64)
                for ki in range(min(K, len(indices))):
                    loc = int(indices[ki])
                    if 0 <= loc < len(self.tags) and self.tags[loc] > 0:
                        global_ids[ki] = int(self.tags[loc]) - 1
                all_result_global_ids.append((idx, global_ids))

            search_time = time.perf_counter() - search_t0

            agg = self._aggregate_stats(all_stats)
            agg["routing"] = {
                "my_partition": my_partition,
                "routed_queries": int(len(routed_indices)),
                "total_queries": num_queries,
                "routed_fraction": len(routed_indices) / num_queries if num_queries > 0 else 0,
            }
            agg["search_time_s"] = round(search_time, 2)

            # Compute recall
            if gt_locs is not None:
                recall = self._compute_recall(all_result_global_ids, gt_locs, K)
                agg["recall"] = recall
                self.log(f"    Recall@{K}: {recall:.4f}")

            all_results[f"L={L}"] = agg

            self._print_summary(agg, f"L={L}")
            self.sync.barrier(f"search-L{L}")

        # Save results
        self._save_results(all_results, K, num_queries)

        self.sync.barrier("search-done")
        self.log("Search complete")

    # ------------------------------------------------------------------
    # RDMA NIC counter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rdma_snapshot() -> dict:
        """Take RDMA counter snapshot via rdma_monitor.sh."""
        script = Path(__file__).resolve().parent.parent / "scripts" / "common" / "rdma_monitor.sh"
        if not script.exists():
            if not NodeWorker._rdma_monitor_warned:
                print(f"[WARN] rdma_monitor.sh not found at {script}, NIC BW stats unavailable", flush=True)
                NodeWorker._rdma_monitor_warned = True
            return {}
        with tempfile.NamedTemporaryFile(mode='r', suffix='.snap', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run([str(script), "snapshot", tmp_path],
                           capture_output=True, timeout=10)
            snap = {}
            with open(tmp_path) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        k, v = line.split('=', 1)
                        snap[k] = v
            return snap
        except Exception:
            return {}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _rdma_compute_bw(before: dict, after: dict, elapsed: float) -> dict:
        """Compute NIC bandwidth from two RDMA snapshots."""
        if not before or not after or elapsed <= 0:
            return {}

        def _delta(key):
            b = before.get(key, '0')
            a = after.get(key, '0')
            try:
                return int(a) - int(b)
            except (ValueError, TypeError):
                return 0

        # IB counters are in 32-bit words (×4 for bytes)
        tx_bytes = _delta('port_xmit_data') * 4
        rx_bytes = _delta('port_rcv_data') * 4

        result = {
            'nic_ib_tx_gbps': round(tx_bytes * 8 / elapsed / 1e9, 3),
            'nic_ib_tx_MBps': round(tx_bytes / elapsed / 1048576, 3),
            'nic_ib_rx_gbps': round(rx_bytes * 8 / elapsed / 1e9, 3),
            'nic_ib_rx_MBps': round(rx_bytes / elapsed / 1048576, 3),
            'nic_rx_read_requests': _delta('rx_read_requests'),
            'nic_rx_write_requests': _delta('rx_write_requests'),
        }

        # Wire-level (physical layer)
        phy_tx = _delta('phy_tx_bytes_phy')
        phy_rx = _delta('phy_rx_bytes_phy')
        if phy_tx > 0 or phy_rx > 0:
            result['nic_wire_tx_gbps'] = round(phy_tx * 8 / elapsed / 1e9, 3)
            result['nic_wire_rx_gbps'] = round(phy_rx * 8 / elapsed / 1e9, 3)

        # RDMA vport
        vport_tx = _delta('vport_tx_vport_rdma_unicast_bytes')
        vport_rx = _delta('vport_rx_vport_rdma_unicast_bytes')
        if vport_tx > 0 or vport_rx > 0:
            result['nic_vport_tx_gbps'] = round(vport_tx * 8 / elapsed / 1e9, 3)
            result['nic_vport_rx_gbps'] = round(vport_rx * 8 / elapsed / 1e9, 3)

        return result

    # ------------------------------------------------------------------
    # PMU monitor helpers (mirror RDMA pattern, bracket-style)
    # ------------------------------------------------------------------
    _pmu_monitor_warned = False

    @staticmethod
    def _pmu_start(out_path: str) -> bool:
        script = Path(__file__).resolve().parent.parent / "scripts" / "common" / "pmu_monitor.sh"
        if not script.exists():
            if not NodeWorker._pmu_monitor_warned:
                print(f"[WARN] pmu_monitor.sh not found at {script}, PMU stats unavailable",
                      flush=True)
                NodeWorker._pmu_monitor_warned = True
            return False
        try:
            r = subprocess.run([str(script), "start", out_path],
                               capture_output=True, timeout=10)
            if r.returncode != 0:
                if not NodeWorker._pmu_monitor_warned:
                    print(f"[WARN] pmu start failed: {r.stderr.decode(errors='ignore')[:200]}",
                          flush=True)
                    NodeWorker._pmu_monitor_warned = True
                return False
            return True
        except Exception as e:
            if not NodeWorker._pmu_monitor_warned:
                print(f"[WARN] pmu start exception: {e}", flush=True)
                NodeWorker._pmu_monitor_warned = True
            return False

    @staticmethod
    def _pmu_stop(out_path: str) -> dict:
        """Close PMU bracket. Output kv file at out_path is kept for
        later inspection."""
        script = Path(__file__).resolve().parent.parent / "scripts" / "common" / "pmu_monitor.sh"
        if not script.exists():
            return {}
        try:
            subprocess.run([str(script), "stop", out_path],
                           capture_output=True, timeout=15)
            out: Dict[str, float] = {}
            with open(out_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    if not k.startswith('derived_'):
                        continue
                    try:
                        out['pmu_' + k[len('derived_'):]] = float(v)
                    except ValueError:
                        pass
            return out
        except Exception:
            return {}

    def _load_gt(self, experiment_dir: Path) -> Optional[np.ndarray]:
        """Load pre-computed GT from pipeline gt stage.

        Resolution order:
        1. Match checkpoint GT in gt_cache/ (e.g., gt_deep500M_cp4.npy)
           graph_variant baseline_cp4/lits_cp4 → cp4, baseline_cp0/lits_cp0 → cp0
        2. Match graph_variant in gt/ or gt_cache/ (e.g., gt_lits_cp4_K10_Q10000.npy)
        3. pipeline_state.json gt_file entry
        4. Fallback: latest GT file in gt/ or gt_cache/ directory
        """
        gt_file = None
        gt_dir = experiment_dir / "gt"
        gt_cache_dir = experiment_dir / "gt_cache"
        graph_variant = self.config.get('build', {}).get('graph_variant', None)

        # 1. Try checkpoint-based GT in gt_cache/ (rebuild_pipeline.sh naming)
        #    baseline_cp4/lits_cp4 → cp4, baseline_cp0/lits_cp0 → cp0
        if graph_variant and gt_cache_dir.exists():
            cp = None
            if "cp4" in graph_variant:
                cp = "cp4"
            elif "cp0" in graph_variant:
                cp = "cp0"
            if cp:
                ds_name = self.config.get('experiment_name', '').replace('_micro26', '')
                candidate = gt_cache_dir / f"gt_{ds_name}_{cp}.npy"
                if candidate.exists():
                    gt_file = str(candidate)
                    self.log(f"Matched checkpoint GT for {graph_variant}: {gt_file}")

        # 2. Try variant-specific GT file in gt/ and gt_cache/
        if not gt_file or not os.path.exists(gt_file):
            for search_dir in [gt_cache_dir, gt_dir]:
                if graph_variant and search_dir.exists():
                    variant_gts = sorted(search_dir.glob(f"gt_{graph_variant}_K*_Q*.npy"))
                    variant_gts = [f for f in variant_gts if "_queries" not in f.name]
                    if variant_gts:
                        gt_file = str(variant_gts[-1])
                        self.log(f"Matched GT for graph_variant={graph_variant}: {gt_file}")
                        break

        # 3. Try pipeline_state.json
        if not gt_file or not os.path.exists(gt_file):
            state_file = experiment_dir / "pipeline_state.json"
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        state = json.load(f)
                    candidate = state.get("gt_file")
                    if candidate and not os.path.exists(candidate):
                        for prefix in []:  # legacy path rewrite (disabled in the artifact)
                            if candidate.startswith(prefix):
                                candidate = "/workspace/" + candidate[len(prefix):]
                                break
                    if candidate and os.path.exists(candidate):
                        gt_file = candidate
                except Exception:
                    pass

        # 4. Fallback: scan gt/ and gt_cache/ for latest GT file
        if not gt_file or not os.path.exists(gt_file):
            for search_dir in [gt_cache_dir, gt_dir]:
                if search_dir.exists():
                    gt_files = sorted(search_dir.glob("gt_*.npy"))
                    gt_files = [f for f in gt_files if "_queries" not in f.name]
                    if gt_files:
                        gt_file = str(gt_files[-1])
                        break

        if gt_file and os.path.exists(gt_file):
            gt_locs = np.load(gt_file)
            self.log(f"Loaded GT: {gt_file} (shape={gt_locs.shape})")
            return gt_locs

        self.log("[WARN] No GT file found — recall will not be computed")
        return None

    def _compute_recall(self, result_pairs: List[Tuple[int, np.ndarray]],
                        gt_locs: np.ndarray, K: int) -> float:
        """Compute recall@K for this node's routed queries.

        Only evaluates queries whose index falls within the GT range
        (i.e., query_idx < len(gt_locs)). Queries beyond GT are skipped.

        Args:
            result_pairs: list of (query_idx, global_ids[K]) from search
            gt_locs: (num_gt_queries, K) global location IDs from GT
            K: number of neighbors
        """
        if len(result_pairs) == 0:
            return 0.0

        num_gt = len(gt_locs)
        total_hits = 0
        total_possible = 0

        for query_idx, result_global_ids in result_pairs:
            if query_idx >= num_gt:
                continue  # No GT for this query
            gt_set = set(gt_locs[query_idx][:K].tolist())
            result_set = set(r for r in result_global_ids[:K].tolist() if r >= 0)
            total_hits += len(gt_set & result_set)
            total_possible += K

        return total_hits / total_possible if total_possible > 0 else 0.0

    def _find_graph_checkpoint(self, experiment_dir: Path) -> Tuple[str, str]:
        """Find the graph checkpoint from pipeline output.

        If config has build.graph_variant (e.g., 'lits_cp0', 'baseline_cp4'),
        use graph/{variant} and graph/{variant}.tags directly (symlinks).
        Otherwise, fall back to checkpoints/iteration{N} (legacy).
        """
        graph_dir = experiment_dir / "graph"

        # Check for graph_variant in config
        graph_variant = self.config.get('build', {}).get('graph_variant', None)
        if graph_variant:
            # Try graph/{variant} (symlink-based layout used by micro26)
            graph_file = str(graph_dir / graph_variant)
            tags_file = graph_file + ".tags"
            if os.path.exists(graph_file) and os.path.exists(tags_file):
                return graph_file, tags_file
            # Fall back to graph/checkpoints/{variant} (layout produced by the
            # VLDB build stage — no symlinks needed).
            ckpt_file = str(graph_dir / "checkpoints" / graph_variant)
            ckpt_tags = ckpt_file + ".tags"
            if os.path.exists(ckpt_file) and os.path.exists(ckpt_tags):
                return ckpt_file, ckpt_tags
            self.log(f"[WARN] graph_variant={graph_variant} not found at "
                     f"{graph_file} or {ckpt_file}")

        # Legacy: try checkpoints/ under graph/
        checkpoints_dir = graph_dir / "checkpoints"
        if checkpoints_dir.exists():
            files = [f.name for f in checkpoints_dir.iterdir()
                     if f.name.startswith("iteration") and '.' not in f.name]
            if files:
                files.sort(key=lambda x: int(x.replace("iteration", "")))
                latest = files[-1]
                graph_file = str(checkpoints_dir / latest)
                tags_file = graph_file + ".tags"
                if os.path.exists(tags_file):
                    return graph_file, tags_file

        # Try timestamped subdirectories under graph/
        if graph_dir.exists():
            subdirs = sorted([d for d in graph_dir.iterdir() if d.is_dir()
                             and d.name != "checkpoints"], reverse=True)
            for subdir in subdirs:
                cp_dir = subdir / "checkpoints"
                if cp_dir.exists():
                    files = [f.name for f in cp_dir.iterdir()
                             if f.name.startswith("iteration") and '.' not in f.name]
                    if files:
                        files.sort(key=lambda x: int(x.replace("iteration", "")))
                        latest = files[-1]
                        graph_file = str(cp_dir / latest)
                        tags_file = graph_file + ".tags"
                        if os.path.exists(tags_file):
                            return graph_file, tags_file

        raise FileNotFoundError(f"No graph checkpoint found in {experiment_dir}")

    def _create_partition_ids_by_location(self, rdma_store) -> np.ndarray:
        """Create location-indexed partition_ids array."""
        array_size = rdma_store.get_location_count()
        if array_size == 0:
            array_size = len(self.tags)
        partition_ids_by_location = np.zeros(array_size, dtype=np.int32)

        tags = self.tags
        active_locs = np.where(tags > 0)[0]
        gids = tags[active_locs].astype(np.int64) - 1
        valid = gids < len(self.partition_ids)
        partition_ids_by_location[active_locs[valid]] = self.partition_ids[gids[valid]]

        unique, counts = np.unique(partition_ids_by_location[:len(tags)], return_counts=True)
        self.log(f"  partition_ids_by_location dist: {dict(zip(unique.tolist(), counts.tolist()))}")
        self.log(f"  partition_ids_by_location size: {array_size} (tags: {len(tags)})")

        return partition_ids_by_location

    def _set_partition_entry_points(self, rdma_store, experiment_dir: Path):
        """Set up virtual centroid nodes as per-partition search entry points.

        In colocation mode: centroid entries are already in the .coloc files
        (built by build_colocation.py). We read centroid_local_loc from the
        colocation metadata and call set_centroid_local_locs() so that
        graph_access_fn and rdma_fn resolve virtual nodes via colocation
        instead of falling back to DiskANN graph store.

        The centroid vector is still injected into DiskANN's _data_store
        (for local distance computation by the search algorithm).
        """
        num_partitions = self.config.get('partition', {}).get('num_partitions', 4)
        strategy_name = get_insert_strategy(self.config)
        partition_dir = build_partition_dir(
            experiment_dir, num_partitions, strategy_name,
        )

        centroids_path = partition_dir / "centroids.fbin"
        if not centroids_path.exists():
            self.log(f"  [WARN] Centroids not found: {centroids_path}, using default entry point")
            return

        centroids = load_fbin(str(centroids_path))  # (P, D)
        self.log(f"  Centroids loaded: {centroids.shape}")

        max_points = rdma_store.get_max_points()
        R = self.config.get('build', {}).get('R', 64)

        use_colocation = self.colocation_mode and hasattr(self, 'colocation_meta') and self.colocation_meta

        centroid_vectors = {}
        neighbor_locations = {}

        _NP_DTYPE = {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}
        native_dtype = _NP_DTYPE.get(self.data_type, np.float32)

        if use_colocation:
            # Colocation mode: inject centroid vectors into _data_store only.
            # Neighbors come from colocation file, not graph store.
            for p in range(num_partitions):
                centroid_vectors[p] = centroids[p].astype(native_dtype)
                neighbor_locations[p] = np.array([], dtype=np.uint32)
        else:
            # Non-colocation mode: inject centroid vectors AND neighbors into graph store
            if self.vectors is None:
                raise RuntimeError(
                    "Non-colocation entry point injection requires full vectors, "
                    "but vectors were not loaded (colocation-only memory optimization). "
                    "Enable colocation to use lightweight mode."
                )
            tag_to_loc = {}
            for loc in range(len(self.tags)):
                tag = int(self.tags[loc])
                if tag > 0:
                    tag_to_loc[tag] = loc

            for p in range(num_partitions):
                centroid = centroids[p].astype(native_dtype)
                centroid_vectors[p] = centroid

                mask = self.partition_ids == p
                member_gids = np.where(mask)[0]
                member_vecs = self.vectors[member_gids]
                dists = np.sum((member_vecs - centroid) ** 2, axis=1)
                top_k = min(R, len(member_gids))
                nearest_idx = np.argpartition(dists, top_k)[:top_k]
                nearest_gids = member_gids[nearest_idx]

                locations = []
                for gid in nearest_gids:
                    tag = int(gid) + 1
                    loc = tag_to_loc.get(tag)
                    if loc is not None:
                        locations.append(loc)
                neighbor_locations[p] = np.array(locations, dtype=np.uint32)

        if not use_colocation:
            rdma_store.inject_virtual_entry_points(centroid_vectors, neighbor_locations, max_points)
            self.log(f"  Centroid vectors injected for {num_partitions} partitions")
        else:
            self.log(f"  Colocation mode: centroid injection skipped (data in .coloc file)")

        # In colocation mode: set centroid_local_locs from colocation metadata
        # so graph_access_fn/rdma_fn resolve virtual nodes via colocation files
        if use_colocation:
            centroid_locs = {}
            for p in range(num_partitions):
                p_meta = self.colocation_meta.get('partitions', {}).get(str(p), {})
                cloc = p_meta.get('centroid_local_loc', -1)
                if cloc >= 0:
                    centroid_locs[p] = cloc
                    self.log(f"  Partition {p}: centroid_local_loc={cloc} (from colocation)")
                else:
                    self.log(f"  [WARN] Partition {p}: no centroid_local_loc in colocation meta")

            if centroid_locs:
                rdma_store.set_centroid_local_locs(centroid_locs, max_points)
                self.log(f"  Centroid local_locs set for {len(centroid_locs)} partitions "
                         f"(max_points={max_points})")

            # Set frozen point local_loc (replicated in every partition)
            # dense-rank (my_partition), NOT node_id-1: 노드셋이 1로 시작하지 않으면
            # (예: worker hosts 4p) 키가 어긋나 frozen 시딩이 조용히 꺼진다 — 2026-08-23
            # sw5-8 tau 캠페인 오염 사고의 근인.
            my_p_meta = self.colocation_meta.get('partitions', {}).get(str(self.my_partition), {})
            frozen_loc = my_p_meta.get('frozen_local_loc', -1)
            if frozen_loc < 0:
                self.log(f"  [WARN] no frozen_local_loc for partition {self.my_partition} "
                         f"— medoid seeding OFF (검색 경로가 정본과 달라짐)")
            if frozen_loc >= 0:
                rdma_store.set_frozen_local_loc(frozen_loc)
                self.log(f"  Frozen point at local_loc={frozen_loc}")
                # Set canonical frozen VID (always partition 0) so beam search
                # visited set correctly deduplicates frozen across partitions.
                p0_meta = self.colocation_meta.get('partitions', {}).get('0', {})
                p0_frozen = p0_meta.get('frozen_local_loc', -1)
                if p0_frozen >= 0:
                    canonical_vid = (0 << 48) | (p0_frozen & 0xFFFFFFFFFFFF)
                    rdma_store.set_frozen_canonical_vid(canonical_vid)
                    self.log(f"  Frozen canonical VID: 0x{canonical_vid:016x} (partition 0, ll={p0_frozen})")

    def _precompute_query_partitions(self) -> np.ndarray:
        centroid_norms = np.sum(self.centroids ** 2, axis=1)
        dots = self.queries @ self.centroids.T
        sq_dists = centroid_norms - 2 * dots
        return np.argmin(sq_dists, axis=1).astype(np.int32)

    def _convert_to_pipeline_stats(self, cpp_stats: dict) -> PipelineStats:
        stats = PipelineStats()

        stats.local_vector_accesses = cpp_stats.get("local_vector_accesses", 0)
        stats.remote_vector_accesses = cpp_stats.get("remote_vector_accesses", 0)
        stats.local_neighbor_accesses = cpp_stats.get("local_neighbor_accesses", 0)
        stats.remote_neighbor_accesses = cpp_stats.get("remote_neighbor_accesses", 0)
        stats.rdma_round_trips = cpp_stats.get("rdma_round_trips", 0)
        stats.n_hops = cpp_stats.get("n_hops", 0)
        stats.n_cmps = cpp_stats.get("n_cmps", 0)

        stats.rdma_distance_us = cpp_stats.get("rdma_distance_us", 0)
        stats.rdma_fetch_us = cpp_stats.get("rdma_fetch_us", 0)
        stats.rdma_total_us = cpp_stats.get("rdma_total_us", 0)
        stats.rdma_latency_us = stats.rdma_total_us

        stats.neighbor_fetch_local_us = cpp_stats.get("fetch_local_us", 0)
        stats.neighbor_fetch_remote_us = cpp_stats.get("fetch_remote_us", 0)
        stats.neighbor_fetch_us = stats.neighbor_fetch_local_us + stats.neighbor_fetch_remote_us

        stats.distance_calc_local_us = cpp_stats.get("dist_local_us", 0)
        stats.distance_calc_remote_us = cpp_stats.get("dist_remote_us", 0)
        stats.distance_calc_us = stats.distance_calc_local_us + stats.distance_calc_remote_us

        queue_us = cpp_stats.get("queue_us", 0)
        visited_us = cpp_stats.get("visited_us", 0)
        stats.queue_update_us = queue_us + visited_us

        stats.total_us = cpp_stats.get("total_us", 0)
        stats.pipeline_sum_us = stats.neighbor_fetch_us + stats.distance_calc_us + stats.queue_update_us

        # Per-operation histogram data
        neighbor_ns = cpp_stats.get("rdma_neighbor_ns_samples", None)
        if neighbor_ns is not None and hasattr(neighbor_ns, '__len__') and len(neighbor_ns) > 0:
            stats.rdma_neighbor_ns_samples = np.asarray(neighbor_ns, dtype=np.int64)
        batch_ns = cpp_stats.get("rdma_batch_ns_samples", None)
        if batch_ns is not None and hasattr(batch_ns, '__len__') and len(batch_ns) > 0:
            stats.rdma_batch_ns_samples = np.asarray(batch_ns, dtype=np.int64)
        batch_peers = cpp_stats.get("rdma_batch_peer_ids", None)
        if batch_peers is not None and hasattr(batch_peers, '__len__') and len(batch_peers) > 0:
            stats.rdma_batch_peer_ids = np.asarray(batch_peers, dtype=np.uint32)
        batch_sizes = cpp_stats.get("rdma_batch_sizes", None)
        if batch_sizes is not None and hasattr(batch_sizes, '__len__') and len(batch_sizes) > 0:
            stats.rdma_batch_sizes = np.asarray(batch_sizes, dtype=np.uint32)
        per_peer = cpp_stats.get("rdma_per_peer_us", None)
        if per_peer is not None and isinstance(per_peer, dict):
            stats.rdma_per_peer_us = dict(per_peer)

        # Fine-grained sub-breakdown
        stats.dist_cache_check_us = cpp_stats.get("dist_cache_check_us", 0)
        stats.dist_cache_put_us = cpp_stats.get("dist_cache_put_us", 0)
        stats.dist_vid_resolution_us = cpp_stats.get("dist_vid_resolution_us", 0)
        stats.dist_pure_distance_us = cpp_stats.get("dist_pure_distance_us", 0)
        stats.fetch_cache_check_us = cpp_stats.get("fetch_cache_check_us", 0)
        stats.fetch_cache_put_us = cpp_stats.get("fetch_cache_put_us", 0)
        stats.fetch_vid_resolution_us = cpp_stats.get("fetch_vid_resolution_us", 0)

        # Cache stats
        cache = cpp_stats.get("cache", {})
        if cache:
            stats.cache_vec_hits = cache.get("vec_hits", 0)
            stats.cache_vec_misses = cache.get("vec_misses", 0)
            stats.cache_nbr_hits = cache.get("nbr_hits", 0)
            stats.cache_nbr_misses = cache.get("nbr_misses", 0)
            stats.cache_vec_evictions = cache.get("vec_evictions", 0)
            stats.cache_nbr_evictions = cache.get("nbr_evictions", 0)

        return stats

    def _aggregate_stats(self, all_stats: List[dict]) -> dict:
        if not all_stats:
            return {}

        result = {}
        keys = all_stats[0].keys()

        for key in keys:
            values = [s[key] for s in all_stats if key in s]
            if not values:
                continue

            if isinstance(values[0], (int, float)):
                entry = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p99": float(np.percentile(values, 99)),
                    "p999": float(np.percentile(values, 99.9)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                if isinstance(values[0], int):
                    entry["sum"] = int(np.sum(values))
                    entry["count"] = len(values)
                result[key] = entry

        if "total_us" in result:
            total_mean = result["total_us"]["mean"]
            if total_mean > 0:
                for key in ["neighbor_fetch_us", "distance_calc_us", "queue_update_us"]:
                    if key in result:
                        result[key]["pct"] = 100.0 * result[key]["mean"] / total_mean

        return result

    def _print_summary(self, stats: dict, title: str):
        """Print compact pipeline summary."""
        if "total_us" not in stats:
            return

        total = stats["total_us"]
        routing = stats.get("routing", {})
        self.log(f"\n  {title}:")
        self.log(f"    Queries: {routing.get('routed_queries', '?')}/{routing.get('total_queries', '?')}")
        self.log(f"    Total:         {total['mean']:8.1f} us  "
                 f"(p50={total['p50']:7.1f}, p99={total['p99']:7.1f})")

        for stage_name, key in [("neighbor_fetch", "neighbor_fetch_us"),
                                ("distance_calc", "distance_calc_us"),
                                ("queue_update", "queue_update_us")]:
            if key in stats:
                v = stats[key]
                self.log(f"    {stage_name:18s} {v['mean']:8.1f} us  ({v.get('pct', 0):5.1f}%)")

        rdma = stats.get("rdma_total_us", {}).get("mean", 0)
        hops = stats.get("n_hops", {}).get("mean", 0)
        cmps = stats.get("n_cmps", {}).get("mean", 0)
        self.log(f"    RDMA total: {rdma:.1f} us, hops: {hops:.1f}, cmps: {cmps:.0f}")

        # Cache stats (if present)
        vec_hits = stats.get("cache_vec_hits", {})
        if isinstance(vec_hits, dict) and vec_hits.get("sum", 0) > 0:
            vh = vec_hits.get("sum", 0)
            vm = stats.get("cache_vec_misses", {}).get("sum", 0)
            nh = stats.get("cache_nbr_hits", {}).get("sum", 0)
            nm = stats.get("cache_nbr_misses", {}).get("sum", 0)
            vhr = vh / (vh + vm) * 100 if (vh + vm) > 0 else 0
            nhr = nh / (nh + nm) * 100 if (nh + nm) > 0 else 0
            self.log(f"    Cache vec: {vh}/{vh+vm} ({vhr:.1f}%), "
                     f"nbr: {nh}/{nh+nm} ({nhr:.1f}%)")

    def _save_results(self, all_results: dict, K: int, num_queries: int):
        """Save search results to JSON."""
        # Partition info
        partition_info = {}
        wrapper = self.rdma_store_wrapper
        if wrapper.partition_ids is not None:
            unique, counts = np.unique(wrapper.partition_ids, return_counts=True)
            partition_info = {
                "my_partition": wrapper.my_partition,
                "num_local_vectors": int(len(wrapper.local_global_ids)),
                "total_vectors": int(len(wrapper.partition_ids)),
                "partition_sizes": {int(p): int(c) for p, c in zip(unique, counts)},
            }

        output = {
            "node_id": self.node_id,
            "timestamp": self.forced_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": self.config,
            "cluster": {
                "num_nodes": self.cluster.num_nodes,
                "coordinator": self.cluster.coordinator_node,
                "node_ids": sorted(self.cluster.nodes.keys()),
            },
            "partition": partition_info,
            "search_params": {
                "K": K,
                "num_queries": num_queries,
            },
            "results": all_results,
            "rdma_stats": self.rdma_store_wrapper.get_latency_stats(),
        }

        result_file = self.output_dir / f"results_node{self.node_id}.json"
        with open(result_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        self.log(f"Results saved to: {result_file}")

        # Summary table
        self.log(f"\n{'='*80}")
        self.log("SUMMARY")
        self.log(f"{'='*80}")
        has_recall = any("recall" in v for v in all_results.values())
        if has_recall:
            self.log(f"  {'L':>6}  {'Mean(us)':>10}  {'P50(us)':>10}  "
                     f"{'P99(us)':>10}  {'RDMA(us)':>10}  {'Hops':>6}  "
                     f"{'Recall':>8}  {'Queries':>8}")
            self.log(f"  {'-'*85}")
        else:
            self.log(f"  {'L':>6}  {'Mean(us)':>10}  {'P50(us)':>10}  "
                     f"{'P99(us)':>10}  {'RDMA(us)':>10}  {'Hops':>6}  {'Queries':>8}")
            self.log(f"  {'-'*75}")
        for key, val in sorted(all_results.items(), key=lambda x: int(x[0].replace("L=", ""))):
            if "total_us" not in val:
                continue
            t = val['total_us']
            rdma = val.get("rdma_total_us", {}).get("mean", 0)
            hops = val.get("n_hops", {}).get("mean", 0)
            num_q = val.get("routing", {}).get("routed_queries", 0)
            recall = val.get("recall")
            if has_recall:
                recall_str = f"{recall:.4f}" if recall is not None else "N/A"
                self.log(f"  {key:>6}  {t['mean']:>10.1f}  {t['p50']:>10.1f}  "
                         f"{t['p99']:>10.1f}  {rdma:>10.1f}  {hops:>6.1f}  "
                         f"{recall_str:>8}  {num_q:>8}")
            else:
                self.log(f"  {key:>6}  {t['mean']:>10.1f}  {t['p50']:>10.1f}  "
                         f"{t['p99']:>10.1f}  {rdma:>10.1f}  {hops:>6.1f}  {num_q:>8}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Per-node worker for distributed pipeline stages')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--node-id', type=int, required=True, help='Node ID (1, 2, 3, 4)')
    parser.add_argument('--nodes', type=str, required=True,
                        help='Comma-separated active node IDs (e.g., 1,2,3,4)')
    parser.add_argument('--coordinator', type=int, default=None,
                        help='Coordinator node ID (default: lowest node ID). '
                             'Use 0 for external orchestrator on orchestrator-host.')
    parser.add_argument('--orchestrator-host', type=str, default=None,
                        help='Orchestrator hostname/IP (orchestrator-host). Required when --coordinator=0.')
    parser.add_argument('--orchestrator-port', type=int, default=19500,
                        help='Orchestrator TCP sync port (default: 19500)')
    parser.add_argument('--stage', type=str, default='search',
                        choices=['search', 'distributed_engine'],
                        help='Stage to execute (default: search)')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp string for output dir (from launcher)')
    parser.add_argument('--numa-node', type=int, default=None,
                        help='Bind process to NUMA node (CPU + memory via first-touch)')
    parser.add_argument('--project-root', type=str, default=None,
                        help='Project root for resolving relative paths (default: /workspace)')
    parser.add_argument(
        '--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
        help='Override a config key via dotted path before execution',
    )

    args = parser.parse_args()

    # NUMA binding (before any data allocation)
    if args.numa_node is not None:
        numa_id = args.numa_node
        cpulist_path = f'/sys/devices/system/node/node{numa_id}/cpulist'
        try:
            with open(cpulist_path) as f:
                cpulist_str = f.read().strip()
            # Parse "0-31,64-95" format
            cpus = set()
            for part in cpulist_str.split(','):
                if '-' in part:
                    lo, hi = part.split('-')
                    cpus.update(range(int(lo), int(hi) + 1))
                else:
                    cpus.add(int(part))
            os.sched_setaffinity(0, cpus)
            print(f"[NUMA] Bound to node {numa_id}: {len(cpus)} CPUs ({cpulist_str})")
        except Exception as e:
            print(f"[WARN] NUMA binding failed: {e}")

    if not _HAS_RDMA:
        print("ERROR: RDMA bindings not available. "
              "This script must run inside Docker on a DPU node.")
        sys.exit(1)

    active_nodes = [int(x.strip()) for x in args.nodes.split(',')]

    worker = NodeWorker(
        config_path=args.config,
        node_id=args.node_id,
        active_nodes=active_nodes,
        coordinator=args.coordinator,
        stage=args.stage,
        timestamp=args.timestamp,
        orchestrator_host=args.orchestrator_host,
        orchestrator_port=args.orchestrator_port,
        project_root=args.project_root,
        overrides=args.overrides,
    )
    worker.run()


if __name__ == '__main__':
    main()
