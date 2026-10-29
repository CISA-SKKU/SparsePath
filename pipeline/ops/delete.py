"""Delete operation for the distributed engine."""

from __future__ import annotations

import time
from typing import Any

from pipeline.danns_types import encode_vector_id
from pipeline.operation_base import Operation


class DeleteOp(Operation):
    """Delete points from the distributed graph.

    params:
        count: int — number of points to delete
        seed: Optional[int] — RNG seed for reproducibility
    """
    name = "delete"
    needs_broadcast = True

    def prepare(self, ctx: UpdateContext) -> dict:
        count = self.params.get('count', 0)
        seed = self.params.get('seed', None)
        return ctx._prepare_delete_tags(count, seed=seed)

    def execute(self, ctx: UpdateContext, payload: dict):
        delete_tags = payload["tags"]
        delete_vids = payload["vids"]

        my_deletes = delete_tags.get(ctx.my_partition, [])
        ctx.log(f"DeleteOp: {len(my_deletes)} tags for partition {ctx.my_partition}")

        t0 = time.time()
        for tag in my_deletes:
            if ctx.slot_manager is not None and ctx.slot_manager.get_local_loc(tag) is not None:
                local_loc = ctx.slot_manager.get_local_loc(tag)
                vid = encode_vector_id(ctx.my_partition, local_loc)
                ctx.rdma_store.mark_vid_deleted(vid)
                ctx.delete_tags.add(tag)
                self.stats.items_processed += 1

        # Mark ALL partitions' deleted VIDs in local C++ store,
        # so cross-partition beam search excludes them from results.
        # VIDs are pre-computed by coordinator (who has full tag_to_local).
        remote_marked = 0
        for p, vids in delete_vids.items():
            if p == ctx.my_partition:
                continue
            for vid in vids:
                ctx.rdma_store.mark_vid_deleted(vid)
                remote_marked += 1

        # Mark ALL deleted tags in C++ (tag-based filter for result extraction)
        all_del_tags = 0
        for p, tags in delete_tags.items():
            for tag in tags:
                ctx.rdma_store.mark_tag_deleted(int(tag))
                all_del_tags += 1

        elapsed = time.time() - t0
        ctx.log(f"DeleteOp done: {self.stats.items_processed} local + "
                f"{remote_marked} remote VIDs + {all_del_tags} tags marked in {elapsed:.2f}s")

        # Accumulate global delete tags for consolidation (across multiple DeleteOps)
        for p, tags in delete_tags.items():
            if p not in ctx._global_delete_tags:
                ctx._global_delete_tags[p] = set()
            ctx._global_delete_tags[p].update(tags)
