"""Node-to-partition mapping helpers.

The default contract is intentionally simple: sorted worker node ids map to
partition ids 0..N-1.  Large runs can override it with
colocation.node_partition_map or the distributed colocation build CLI.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def parse_node_partition_map(value: Any) -> dict[int, int]:
    """Parse node -> partition mapping from config/CLI shapes.

    Accepted forms:
      - "1:0,2:1,3:2,4:3"
      - {1: 0, 2: 1} or {"1": 0, "2": 1}
      - [{"node_id": 1, "partition": 0}, ...]
    """
    if value is None or value == "":
        return {}

    if isinstance(value, str):
        result: dict[int, int] = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    f"Invalid node_partition_map item {item!r}; expected NODE:PARTITION"
                )
            node_s, part_s = item.split(":", 1)
            result[int(node_s.strip())] = int(part_s.strip())
        return result

    if isinstance(value, Mapping):
        return {int(node): int(part) for node, part in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = {}
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "List node_partition_map entries must be mappings with "
                    "'node_id' and 'partition'"
                )
            node = item.get("node_id", item.get("node"))
            part = item.get("partition", item.get("partition_id"))
            if node is None or part is None:
                raise ValueError(
                    "List node_partition_map entries require node_id and partition"
                )
            result[int(node)] = int(part)
        return result

    raise ValueError(f"Unsupported node_partition_map value: {value!r}")


def resolve_node_partition_map(
    node_ids: Sequence[int],
    num_partitions: int,
    configured: Any = None,
) -> dict[int, int]:
    """Return a validated node_id -> partition_id map.

    The map is one-to-one and must cover every active worker and every
    partition.  That keeps runtime, build orchestration, and metadata validation
    aligned for both 4-node/4-partition and future 8-node/8-partition runs.
    """
    sorted_nodes = sorted(int(n) for n in node_ids)
    if not sorted_nodes:
        raise ValueError("node_partition_map requires at least one worker node")

    if configured is None or configured == "":
        mapping = {node_id: idx for idx, node_id in enumerate(sorted_nodes)}
    else:
        mapping = parse_node_partition_map(configured)

    expected_nodes = set(sorted_nodes)
    actual_nodes = set(mapping)
    missing_nodes = expected_nodes - actual_nodes
    extra_nodes = actual_nodes - expected_nodes
    if missing_nodes or extra_nodes:
        parts = []
        if missing_nodes:
            parts.append(f"missing nodes {sorted(missing_nodes)}")
        if extra_nodes:
            parts.append(f"unknown nodes {sorted(extra_nodes)}")
        raise ValueError("Invalid node_partition_map: " + ", ".join(parts))

    partitions = list(mapping.values())
    invalid = sorted(p for p in partitions if p < 0 or p >= num_partitions)
    if invalid:
        raise ValueError(
            f"Invalid node_partition_map: partitions out of range 0.."
            f"{num_partitions - 1}: {invalid}"
        )

    duplicates = sorted({p for p in partitions if partitions.count(p) > 1})
    if duplicates:
        raise ValueError(
            f"Invalid node_partition_map: duplicate partitions {duplicates}"
        )

    expected_partitions = set(range(num_partitions))
    missing_partitions = expected_partitions - set(partitions)
    if missing_partitions:
        raise ValueError(
            "Invalid node_partition_map: missing partitions "
            f"{sorted(missing_partitions)}"
        )

    return mapping


def mapping_from_config(config: dict, node_ids: Sequence[int]) -> dict[int, int]:
    coloc_cfg = config.get("colocation", {}) or {}
    num_partitions = int(config.get("partition", {}).get("num_partitions", 0))
    if num_partitions <= 0:
        raise ValueError("partition.num_partitions must be positive")
    configured = coloc_cfg.get("node_partition_map")
    return resolve_node_partition_map(node_ids, num_partitions, configured)
