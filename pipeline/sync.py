"""
Inter-node synchronization for the DANNS distributed pipeline.

Provides SyncProtocol ABC and TCPSync implementation.
"""

import socket
import struct
import time
from abc import ABC, abstractmethod
from typing import Dict

from pipeline.config import ClusterConfig


class SyncProtocol(ABC):
    """Abstract synchronization protocol for inter-node coordination."""

    @abstractmethod
    def setup(self):
        """Initialize connections."""
        ...

    @abstractmethod
    def barrier(self, name: str = ""):
        """Blocking barrier — all nodes must reach this point before proceeding."""
        ...

    @abstractmethod
    def broadcast_bytes(self, data: bytes):
        """Coordinator sends raw bytes to all workers."""
        ...

    @abstractmethod
    def recv_bytes(self) -> bytes:
        """Worker receives raw bytes from coordinator."""
        ...

    @abstractmethod
    def gather_bytes(self) -> Dict[int, bytes]:
        """Coordinator gathers raw bytes from all workers."""
        ...

    @abstractmethod
    def send_to_coordinator(self, data: bytes):
        """Worker sends raw bytes to coordinator."""
        ...

    @abstractmethod
    def barrier_with_status(self, name: str, success: bool) -> bool:
        """Barrier with success/failure exchange. True only if all nodes succeeded."""
        ...

    @abstractmethod
    def close(self):
        """Close all connections."""
        ...


class TCPSync(SyncProtocol):
    """TCP-based synchronization between nodes (star topology via coordinator)."""

    def __init__(self, cluster: ClusterConfig, my_id: int):
        self.cluster = cluster
        self.my_id = my_id
        self.is_coordinator = (my_id == cluster.coordinator_node)
        self.conns: Dict[int, socket.socket] = {}
        self.server_socket = None

    @staticmethod
    def _enable_keepalive(sock):
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

    def _recv_exact(self, conn, length: int) -> bytes:
        buf = b''
        while len(buf) < length:
            chunk = conn.recv(min(length - len(buf), 65536))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            buf += chunk
        return buf

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

    def broadcast_bytes(self, data: bytes):
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can broadcast_bytes")
        for conn in self.conns.values():
            conn.sendall(struct.pack('!I', len(data)))
            conn.sendall(data)

    def recv_bytes(self) -> bytes:
        if self.is_coordinator:
            raise RuntimeError("Only workers can recv_bytes")
        conn = self.conns[self.cluster.coordinator_node]
        length = struct.unpack('!I', self._recv_exact(conn, 4))[0]
        return self._recv_exact(conn, length)

    def gather_bytes(self) -> Dict[int, bytes]:
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can gather_bytes")
        result = {}
        for peer_id, conn in self.conns.items():
            length = struct.unpack('!I', self._recv_exact(conn, 4))[0]
            result[peer_id] = self._recv_exact(conn, length)
        return result

    def send_to_coordinator(self, data: bytes):
        if self.is_coordinator:
            raise RuntimeError("Coordinator should not send_to_coordinator")
        conn = self.conns[self.cluster.coordinator_node]
        conn.sendall(struct.pack('!I', len(data)))
        conn.sendall(data)

    def barrier_with_status(self, name: str, success: bool) -> bool:
        if self.is_coordinator:
            all_ok = success
            for peer_id, conn in self.conns.items():
                status_byte = self._recv_exact(conn, 1)
                peer_ok = struct.unpack('!?', status_byte)[0]
                all_ok = all_ok and peer_ok
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
