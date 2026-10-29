"""Checkpoint serialization helpers (pure Python — host-testable).

File formats deliberately match build_colocation output so a checkpoint
directory can later serve as a colocation directory (reload is out of
scope for now; see the 2026-06-12 design doc).

t2l format:          [max_tag: uint32][table: uint32[max_tag+1]]
deleted-tags format: sorted uint32 tags, no header (empty file = none)
"""

from __future__ import annotations

import numpy as np

from pipeline.slot_manager import INVALID_LOC


def save_tag_to_local(table: np.ndarray, path) -> int:
    """Write a tag_to_local table. Returns the mapped-entry count.

    An empty table writes nothing and returns 0 (parity with the old
    engine_context._save_tag_to_local behavior).
    """
    table = np.asarray(table, dtype=np.uint32)
    if table.size == 0:
        return 0
    with open(path, 'wb') as f:
        np.array([table.size - 1], dtype=np.uint32).tofile(f)
        table.tofile(f)
    return int(np.count_nonzero(table != INVALID_LOC))


def load_tag_to_local(path) -> np.ndarray:
    """Read a t2l file back into a uint32 table (header stripped)."""
    raw = np.fromfile(str(path), dtype=np.uint32)
    if raw.size == 0:
        raise ValueError(f"empty tag_to_local file: {path}")
    max_tag, table = int(raw[0]), raw[1:]
    if table.size != max_tag + 1:
        raise ValueError(
            f"tag_to_local size mismatch in {path}: header implies "
            f"{max_tag + 1} entries, file has {table.size}")
    return table


def save_deleted_tags(tags, path) -> int:
    """Write lazy-deleted (pre-consolidation) tags. Returns the count.

    An empty set still writes a 0-byte file so the analysis can tell
    "no deletes" apart from "file missing".
    """
    arr = np.array(sorted(int(t) for t in tags), dtype=np.uint32)
    arr.tofile(str(path))
    return int(arr.size)


def load_deleted_tags(path) -> np.ndarray:
    """Read a deleted-tags file (uint32 array, possibly empty)."""
    return np.fromfile(str(path), dtype=np.uint32)
