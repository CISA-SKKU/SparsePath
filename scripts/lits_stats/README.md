# LITS stats — single-node graph-structure measurement (no RDMA, no rebuild)

Collects LITS-related stats from **already-built** graphs. The headline is the
**inter-partition ("remote") edge fraction**, which the paper reports as
deep100M **lits 7.56%** vs **vamana+ncb 24.8%** vs **vamana+rr 75.0%**.

## Why this is the right shape

The single-node LITS **graph builder** (`diskann-lits`) already produced the
`lits_k{K}_iter{N}` checkpoints (+ per-iteration `.labels` partition labels). LITS
is an update-path feature whose *effect* is the resulting graph's cross-partition
edge count — a pure **graph-structure** property. So we don't re-run anything: we
load the built graph + its labels and count. Inter-partition edges are exact
single-node (no RDMA needed — "remote" just means the two endpoints live in
different partitions/machines).

This is distinct from the *colocation runtime* LITS (`update/insertion.cpp`,
RDMA/ColocationStore) — that is the distributed insert path, a separate track.

## What it measures (`graph_stats` in `pq_search_bindings`)

Per graph + partition-label pair:
- **inter_pct** — inter-partition out-edges / total out-edges (the remote-edge %)
- **avg_degree**, **zero_degree_nodes** (a known lits characteristic)
- **balance** — max/min active nodes per partition
- **partition_node_counts** — active nodes per partition
- **n_skipped** — nodes whose partition index was out of range (should be 0)

Graphs are loaded with **`load_topology`** (adjacency + tags only, no base
vectors), so post-update checkpoints (iter150 tags are non-sequential, gids >
100M base) load fine and fast.

## Files

| file | role |
|---|---|
| `../../bindings/pq_search_bindings.cpp` | `load_topology()` + `graph_stats()` |
| `run_lits_stats.py` | driver: per (graph, partition) inter-edge% + degree/balance table |
| `datasets_deep100M.yaml` | ready deep100M config (lits vs vamana+ncb vs vamana+rr) |

## Build & run

```bash
docker exec -w /workspace/.claude/worktrees/single-node-features danns ./build_all.sh
python scripts/lits_stats/run_lits_stats.py \
    --config scripts/lits_stats/datasets_deep100M.yaml --dataset deep100M --out results_lits_deep100M
```

Outputs: `<ds>_lits_stats.json`, combined `LITS_stats_table.md`.

## Partition sources
- **lits**: `checkpoints/lits_k{K}_iter{N}.labels` — `kind: labels`, TAG-indexed
  (header `[n u32][K u32]`; tags non-sequential after updates).
- **vamana**: `partition/{M}_partitions_{nearest_centroid_balanced,round_robin}/partition_ids.npy`
  — `kind: partition_ids`, GID-indexed (gid = tag-1); the deep100M arrays are
  length 300M (full pool), covering the iter150 gids.

## Gating
- Single-node, RDMA-free, ~26 GB RAM (graph adjacency + labels; no vectors).
  Building the `.so` is fine; firing runs waits for the user's **"fire"**.
- Recall (search) is a separate concern needing per-iteration GT + the matching
  pool base; this harness focuses on structure. Use the M10/M11 drivers for recall.
