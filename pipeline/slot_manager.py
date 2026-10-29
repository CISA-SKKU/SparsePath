"""Colocation slot manager for the DANNS distributed pipeline."""

import numpy as np
from typing import List, Optional

INVALID_LOC = np.uint32(0xFFFFFFFF)


class ColocationSlotManager:
    """Manages free/active slots in a pre-allocated co-location binary.

    Each partition's .coloc file has `capacity` entries. Active entries have
    a valid VectorID; empty slots have VectorID = INVALID_VID.

    Uses numpy arrays for tag_to_local and local_to_tag mappings for O(1)
    lookups with minimal memory overhead and fast initialization.
    """

    def __init__(self, capacity: int, active_count: int,
                 tag_to_local_array: np.ndarray):
        self.capacity = capacity
        self.tag_to_local = tag_to_local_array
        self.local_to_tag = np.full(capacity, INVALID_LOC, dtype=np.uint32)
        active_tags = np.where(tag_to_local_array != INVALID_LOC)[0].astype(np.uint32)
        active_locs = tag_to_local_array[active_tags]
        mask = active_locs < capacity
        self.local_to_tag[active_locs[mask]] = active_tags[mask]
        occupied_mask = np.zeros(capacity, dtype=np.bool_)
        occupied_mask[active_locs[mask]] = True
        self.free_slots: List[int] = np.where(~occupied_mask)[0].tolist()
        self.active_count = int(np.sum(occupied_mask))

    @property
    def num_free(self) -> int:
        return len(self.free_slots)

    def allocate(self, tag: int) -> int:
        """Allocate a free slot for a new entry. Returns local_loc."""
        if not self.free_slots:
            raise RuntimeError(
                f"ColocationSlotManager: no free slots "
                f"(capacity={self.capacity}, active={self.active_count})")
        local_loc = self.free_slots.pop()
        if tag >= len(self.tag_to_local):
            new_size = tag + 1
            extended = np.full(new_size - len(self.tag_to_local), INVALID_LOC, dtype=np.uint32)
            self.tag_to_local = np.concatenate([self.tag_to_local, extended])
        self.tag_to_local[tag] = np.uint32(local_loc)
        self.local_to_tag[local_loc] = np.uint32(tag)
        self.active_count += 1
        return local_loc

    def release(self, tag: int) -> Optional[int]:
        """Release a slot back to the free list. Returns local_loc or None."""
        if tag >= len(self.tag_to_local) or self.tag_to_local[tag] == INVALID_LOC:
            return None
        local_loc = int(self.tag_to_local[tag])
        self.tag_to_local[tag] = INVALID_LOC
        self.local_to_tag[local_loc] = INVALID_LOC
        self.free_slots.append(local_loc)
        self.active_count -= 1
        return local_loc

    def get_local_loc(self, tag: int) -> Optional[int]:
        """Get local_loc for a tag, or None if not mapped."""
        if tag >= len(self.tag_to_local):
            return None
        loc = self.tag_to_local[tag]
        return None if loc == INVALID_LOC else int(loc)
