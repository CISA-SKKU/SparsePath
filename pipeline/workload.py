#!/usr/bin/env python3
"""Shared distributed workload parsing.

The orchestrator and workers must derive the same operation sequence from a
config. Keeping this parser shared prevents protocol drift where one side
waits for an operation the other side never dispatches.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


SUPPORTED_OPS = {
    "insert",
    "delete",
    "consolidate",
    "concurrent",
    "search",
    "checkpoint",
}

LEGACY_WORKLOAD_KEYS = {
    "max_iterations",
    "points_to_insert",
    "points_to_delete",
    "consolidate_every",
}


@dataclass(frozen=True)
class WorkloadSpec:
    op: str
    params: Dict[str, Any]
    source_index: int
    legacy: bool = False

    def as_tuple(self):
        return self.op, dict(self.params)

    def as_dict(self) -> Dict[str, Any]:
        return {"op": self.op, **self.params}


def parse_workload_specs(config: dict) -> List[WorkloadSpec]:
    """Parse distributed_engine workload specs from a full pipeline config.

    Missing or empty distributed_engine means an empty workload. The legacy
    iteration-based fallback is enabled only when one of the legacy keys is
    explicitly present.
    """
    du_cfg = config.get("distributed_engine")
    if not du_cfg:
        return []

    if "workload" in du_cfg:
        workload = du_cfg.get("workload") or []
        if not isinstance(workload, list):
            raise ValueError("distributed_engine.workload must be a list")
        return [_parse_explicit_spec(spec, idx) for idx, spec in enumerate(workload)]

    if any(k in du_cfg for k in LEGACY_WORKLOAD_KEYS):
        return _build_legacy_specs(du_cfg)

    return []


def _parse_explicit_spec(spec: dict, idx: int) -> WorkloadSpec:
    if not isinstance(spec, dict):
        raise ValueError(f"distributed_engine.workload[{idx}] must be a mapping")
    op_name = spec.get("op", "")
    if op_name not in SUPPORTED_OPS:
        raise ValueError(
            f"Unknown operation: {op_name!r}. Available: {sorted(SUPPORTED_OPS)}"
        )
    params = {k: v for k, v in spec.items() if k != "op"}
    return WorkloadSpec(op=op_name, params=params, source_index=idx)


def _build_legacy_specs(du_cfg: dict) -> List[WorkloadSpec]:
    max_iter = du_cfg.get("max_iterations", 10)
    ins_count = du_cfg.get("points_to_insert", 5000)
    del_count = du_cfg.get("points_to_delete", 5000)
    consolidate_every = du_cfg.get("consolidate_every", 5)

    specs: List[WorkloadSpec] = []
    for i in range(max_iter):
        if ins_count > 0:
            specs.append(WorkloadSpec(
                op="insert",
                params={"count": ins_count, "seed": 42 + i},
                source_index=len(specs),
                legacy=True,
            ))
        if del_count > 0:
            specs.append(WorkloadSpec(
                op="delete",
                params={"count": del_count, "seed": 1000 + i},
                source_index=len(specs),
                legacy=True,
            ))
        if consolidate_every > 0 and (i + 1) % consolidate_every == 0:
            specs.append(WorkloadSpec(
                op="consolidate",
                params={},
                source_index=len(specs),
                legacy=True,
            ))

    specs.append(WorkloadSpec(
        op="checkpoint",
        params={},
        source_index=len(specs),
        legacy=True,
    ))
    return specs

