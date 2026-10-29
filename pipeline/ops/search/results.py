"""Search result merge and serialization helpers."""

from __future__ import annotations

import pickle
from typing import Dict, Iterable

import numpy as np

from pipeline.slot_manager import INVALID_LOC


def make_result_payload(results_by_L: Dict[int, list], my_partition: int) -> bytes:
    return pickle.dumps({
        "results_by_L": {L: res for L, res in results_by_L.items()},
        "partition": my_partition,
    })


def merge_gathered_results(ctx, results_by_L: Dict[int, list],
                           l_values: Iterable[int]) -> Dict[int, list]:
    gathered = ctx.sync.gather_bytes()
    all_results_by_L: Dict[int, list] = {L: list(results_by_L[L]) for L in l_values}

    for peer_data in gathered.values():
        peer_payload = pickle.loads(peer_data)
        for L in l_values:
            peer_results = peer_payload["results_by_L"].get(L, [])
            all_results_by_L[L].extend(peer_results)

    return all_results_by_L


def active_global_ids(ctx) -> np.ndarray:
    """Build active global IDs from tag_to_local, excluding pending deletes."""
    gid_arrays = []
    for p, t2l in ctx.tag_to_local.items():
        active_tags = np.where(t2l != INVALID_LOC)[0]
        del_tags_for_p = set()
        if ctx.delete_tags:
            del_tags_for_p.update(ctx.delete_tags)
        if p in ctx._global_delete_tags:
            del_tags_for_p.update(ctx._global_delete_tags[p])
        if del_tags_for_p:
            del_arr = np.array(list(del_tags_for_p), dtype=np.int64)
            active_tags = active_tags[~np.isin(active_tags, del_arr)]
        gid_arrays.append(active_tags.astype(np.int64) - 1)
    return np.sort(np.concatenate(gid_arrays)) if gid_arrays else np.array([], dtype=np.int64)


def save_search_results(ctx, all_results_by_L: Dict[int, list], K: int, seq: int):
    """Save raw search results to npz for offline recall computation."""
    try:
        run_dir = ctx.run_dir
        if run_dir is None:
            return
        save_dict = {}
        for L, results in all_results_by_L.items():
            if not results:
                continue
            sorted_results = sorted(results, key=lambda x: x[0])
            qids = np.array([r[0] for r in sorted_results], dtype=np.int32)
            max_k = max(len(r[1]) for r in sorted_results) if sorted_results else K
            gids = np.full((len(sorted_results), max_k), -1, dtype=np.int64)
            dists = np.full((len(sorted_results), max_k), np.inf, dtype=np.float32)
            for i, r in enumerate(sorted_results):
                result_gids = r[1]
                gids[i, :len(result_gids)] = result_gids[:max_k]
                if len(r) > 2 and r[2]:
                    dl = r[2]
                    dists[i, :len(dl)] = dl[:max_k]
            save_dict[f"qids_L{L}"] = qids
            save_dict[f"gids_L{L}"] = gids
            save_dict[f"dists_L{L}"] = dists
        save_dict["L_values"] = np.array(sorted(all_results_by_L.keys()), dtype=np.int32)
        save_dict["K"] = np.array([K], dtype=np.int32)
        save_dict["partition"] = np.array([ctx.my_partition], dtype=np.int32)
        results_dir = run_dir / "search_results"
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / f"seq{seq}_p{ctx.my_partition}.npz"
        np.savez_compressed(str(save_path), **save_dict)
        ctx.log(f"SearchOp: saved raw results to {save_path.name} "
                f"({len(all_results_by_L)} L values)")
    except Exception as e:
        ctx.log(f"[WARN] Failed to save search results: {e}")
