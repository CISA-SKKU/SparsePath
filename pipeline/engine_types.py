"""Shared types for the distributed operation engine."""

from dataclasses import dataclass, field


@dataclass
class OperationStats:
    """Per-operation statistics."""
    op_name: str = ""
    seq: int = 0
    wall_time: float = 0.0
    status: str = "pending"      # pending / success / failed
    items_processed: int = 0
    items_failed: int = 0
    error: str = ""
    custom: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "op": self.op_name,
            "seq": self.seq,
            "wall_time_s": round(self.wall_time, 3),
            "status": self.status,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
        }
        if self.error:
            d["error"] = self.error
        if self.custom:
            d.update(self.custom)
        return d


@dataclass
class RDMACounters:
    """Python-side RDMA call counters.

    Reset per operation by OperationEngine._dispatch().
    """
    # Actual RDMA fetch_entry calls
    fetch_entry_calls: int = 0
    fetch_entry_items: int = 0
    # Logical access counts (local vs remote)
    vector_access_local: int = 0
    vector_access_remote: int = 0
    neighbor_access_local: int = 0
    neighbor_access_remote: int = 0
    # Write operations (insert/consolidation)
    write_neighbors_local: int = 0
    write_neighbors_remote: int = 0
    search_colocation_calls: int = 0

    def reset(self):
        self.fetch_entry_calls = 0
        self.fetch_entry_items = 0
        self.vector_access_local = 0
        self.vector_access_remote = 0
        self.neighbor_access_local = 0
        self.neighbor_access_remote = 0
        self.write_neighbors_local = 0
        self.write_neighbors_remote = 0
        self.search_colocation_calls = 0

    def to_dict(self) -> dict:
        return {
            "fetch_entry": {"calls": self.fetch_entry_calls,
                            "items": self.fetch_entry_items},
            "vector_accesses": {"local": self.vector_access_local,
                                "remote": self.vector_access_remote},
            "neighbor_accesses": {"local": self.neighbor_access_local,
                                  "remote": self.neighbor_access_remote},
            "write_neighbors": {"local": self.write_neighbors_local,
                                "remote": self.write_neighbors_remote},
            "search_colocation_calls": self.search_colocation_calls,
        }
