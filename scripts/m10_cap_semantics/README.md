# M10 — Convergence-Aware Prefetching semantics (single node, no RDMA)

Answers reviewer **R7·D3**: after convergence the search visits `B` unvisited
candidates as a batch, which changes the order nodes are visited vs. standard
best-first. Show **why this does not hurt recall** and **how the convergence
threshold is selected across datasets**.

Design rationale + code map: `00_Docs/SparsePath/02_Action_items/gates/M10_cap_semantics.md`.

## What this is

The **same** `pq_search_bindings` harness M11 uses (loads the real Vamana graph,
mirrors `search/src/beam_search.cpp` ColocationBeamSearch), driven with the
harness's **`batch_size` > 0 path** — a line-for-line mirror of beam_search.cpp's
converge-phase batch expansion (pop B closest-unexpanded as a batch → insert all
their neighbors in one round → `>5%`-improvement fallback). The **PQ filter is
OFF** (`filter_on=False`) so this isolates the *batching* effect; PQ filtering is
M11's axis.

**Why single-node is faithful:** batching only changes *which* candidates are
expanded together (a recall property), never *how* a fetch is transported (a
performance property). Recall is layout-independent; RDMA/colocation only decide
what fetch is batched. Every vector is local here, so recall(B) is exact and
recall(B=0) is the exact best-first oracle to compare against.

## Files

| file | role |
|---|---|
| `../../bindings/pq_search_bindings.cpp` | the C++ module; `search_batch(..., batch_size=B)` is the CAP path |
| `run_m10.py` | driver: B × conv-threshold × L sweep, recall(B) vs recall(B=0), round-reduction (QPS proxy) |
| `datasets.example.yaml` | per-dataset path schema (copy → `datasets.yaml`, fill). No PQ needed. |
| `datasets_deep100M.yaml` | ready deep100M config (container paths) |

## Build

Same as everything else (the `batch_size` extension is in the existing module):

```bash
# worker-host danns container (see memory: reference_build_command)
docker exec -w /workspace/.claude/worktrees/single-node-features danns ./build_all.sh
# -> lib/pq_search_bindings.cpython-*.so
```

## Run

```bash
python run_m10.py --config datasets_deep100M.yaml --dataset deep100M --out results_m10_deep100M
```

Outputs per dataset: `<ds>_rows.json`, `<ts>_<ds>_m10_cap.png` (recall Δ vs B;
fetch-round reduction vs B, one line per threshold), plus combined `M10_table.md`.

### What to read
- **`recall_delta` (= recall(B=0) − recall(B))** should be ≈0 across B and
  threshold → *batching after convergence does not hurt recall* (R7·D3, part 1).
- **`round_reduction`** (avg fetch rounds at B=0 / at B) > 1 → the RDMA-round
  saving batching buys; this is the QPS proxy (real QPS comes from the cluster).
- **conv-threshold sweep** → the per-dataset selection guide (R7·D3, part 2): the
  threshold where recall Δ is still ≈0 while rounds drop the most.

## Gating / rules
- Single-node and RDMA-free, but still a big-RAM box (holds base vectors, e.g.
  deep100M ~38 GB). Launch measurement runs per the user's **"fire"** convention.
  Building the `.so` is fine; firing the runs waits for the go-ahead.
- Requires each dataset's `iteration0` graph + `iteration0.tags` +
  `iteration0.frozen` sidecar. **No PQ artifacts needed** (unlike M11).
