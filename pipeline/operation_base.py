"""Base operation type for the distributed engine."""

from __future__ import annotations

from typing import Any, Optional

from pipeline.engine_types import OperationStats


class Operation:
    """Base class for all distributed graph operations.

    Subclasses implement prepare() and/or execute(). The OperationEngine
    calls prepare() on the coordinator only, broadcasts the result, then
    calls execute() on all nodes.
    """
    name: str = "base"
    needs_broadcast: bool = True

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        self.stats = OperationStats()

    def prepare(self, ctx: UpdateContext) -> Any:
        """Coordinator only: prepare data to broadcast.

        Returns a picklable object that will be sent to all nodes.
        Return None if no broadcast is needed.
        """
        return None

    def execute(self, ctx: UpdateContext, payload: Any):
        """All nodes: execute the operation with broadcast payload."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"
