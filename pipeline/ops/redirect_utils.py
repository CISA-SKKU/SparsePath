"""Live-redirect settlement helpers (pure Python — host-testable).

These own the Python-side slot/mapping bookkeeping after a redirect window
closes (C++ updated its runtime tables live during the batch). TDD'd on the
host against the real ColocationSlotManager (pure numpy).

Design: docs/features/live_redirect_design.md §6.
"""

from __future__ import annotations

import numpy as np

from pipeline.slot_manager import INVALID_LOC


def settle_redirected_out(redirected, slot_manager, tag_to_local_arr,
                          new_local_to_tag, my_partition):
    """Settle this node's redirected-OUT tags after the window closed:
    release each provisional slot and keep the tag out of the broadcast
    (the TARGET publishes the final mapping via adopt_redirected_in).

    redirected:        list of (tag, m_assign) the C++ batch moved
    slot_manager:      ColocationSlotManager for my_partition (mutated)
    tag_to_local_arr:  ctx.tag_to_local[my_partition] numpy array (mutated)
    new_local_to_tag:  ctx._new_local_to_tag list of (part, loc, tag)
    my_partition:      this node's partition (= m_search)

    Returns (released, filtered_new_l2t):
      released         list of (tag, local_loc) actually released (C++
                       already cleared its runtime tables at redirect time)
      filtered_new_l2t new_local_to_tag without the redirected tags (the
                       post-op broadcast must never publish the provisional
                       m_search mapping)
    """
    redirected_tags = {int(tag) for tag, _ in redirected}

    released = []
    for tag in sorted(redirected_tags):
        local_loc = slot_manager.release(tag)
        if local_loc is None:
            # Not mapped (defensive — e.g. double settlement); nothing to
            # release, but the tag must still be kept out of the broadcast.
            continue
        if tag < len(tag_to_local_arr):
            tag_to_local_arr[tag] = INVALID_LOC
        released.append((tag, local_loc))

    filtered_new_l2t = [(p, loc, t) for (p, loc, t) in new_local_to_tag
                        if t not in redirected_tags]
    return released, filtered_new_l2t


def adopt_redirected_in(granted, slot_manager, tag_to_local_arr,
                        new_local_to_tag, my_partition):
    """Adopt slots the alloc server granted DURING the batch (live redirect).

    The C++ alloc server handed out these slots from the lent reserve —
    they never went through slot_manager.allocate(), so Python's books
    don't know them. For each (tag, floc): record both mappings, bump
    active_count, and append the final mapping to new_local_to_tag so the
    post-op broadcast publishes it (the slot OWNER publishes — origins
    only retract their provisional entries via settle_redirected_out).

    granted:           list of (tag, floc) from the RedirectLedger
    slot_manager:      this node's ColocationSlotManager (mutated)
    tag_to_local_arr:  ctx.tag_to_local[my_partition] numpy array (mutated;
                       caller must have pre-sized it past max granted tag)
    new_local_to_tag:  ctx._new_local_to_tag list (mutated: appended)
    Returns the number of adopted slots.
    """
    adopted = 0
    for tag, floc in granted:
        tag, floc = int(tag), int(floc)
        # slot manager books: mirror what allocate() would have recorded
        if tag >= len(slot_manager.tag_to_local):
            ext = np.full(tag + 1 - len(slot_manager.tag_to_local),
                          INVALID_LOC, dtype=np.uint32)
            slot_manager.tag_to_local = np.concatenate(
                [slot_manager.tag_to_local, ext])
        slot_manager.tag_to_local[tag] = np.uint32(floc)
        slot_manager.local_to_tag[floc] = np.uint32(tag)
        slot_manager.active_count += 1
        # ctx-level array + broadcast list (owner publishes the final home)
        if tag < len(tag_to_local_arr):
            tag_to_local_arr[tag] = np.uint32(floc)
        new_local_to_tag.append((int(my_partition), floc, tag))
        adopted += 1
    return adopted
