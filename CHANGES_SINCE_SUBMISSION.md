# Changes since the submission artifact

The submission package (2026-05) was cut from commit `2fd29e5`. This package
is cut from `main` (see `VERSION`), 189 commits later. Summary of what moved
in ways a reviewer can observe; refactors that do not change behaviour are omitted.

## Changes that alter measured numbers (default ON)

| Change | Where | Effect |
|---|---|---|
| **FrozenSeedCache** — the frozen start point's remote neighbours are fetched once per search op instead of once per query | `search/`, `distributed_search.frozen_seed_cache: true` | +20–30 % QPS for SparsePath at equal recall; recall-neutral (same bytes, same order) |
| **Consolidation cache** — search warms a per-deleted-node cache that consolidation reads first | `store/consolidation_cache.h`, `update/consolidation.cpp`, `consolidation_cache:` block | consolidation wall time ↓ (paper §6.4); off unless enabled in the config |
| **Build flags** `DANNS_ARCH=icelake-server`, `-Ofast` | `CMakeLists.txt`, `build_all.sh` | replaces `-mavx2 -mfma -O3`; ~10–30 % engine speed-up |
| `lits_tau_include_deleted: true` | `update/lits_policy.cpp` | LITS consolidation filter counts deleted vectors when computing τ; changes the consolidated graph slightly |

## New features (default OFF, config-gated)

Load-balanced insert with live redirect (`load_balancer:`), work-stealing
query distribution (`exchange_claim`), hop-interleaved multi-query search
(`interleave_factor`), search-during-consolidation workload op (`concurrent`),
16-worker socket-split topology (not shipped), NUMA-bound payload allocation,
single-node quantized-search harness (`lib/single_engine.so`,
`lib/pq_search_bindings.so`).

## Correctness fixes

- Owner-node bound check (`kMaxOwnerNodes`): the submission code silently
  dropped neighbours owned by node id ≥ 16 (only reachable in 16p setups).
- VID bounds check in `resolve_vid` under concurrent consolidation.
- Frozen-point seeding fixes for the FROZEN_ONLY entry mode (`entry_mode: 1`),
  which is now the documented convention; centroid seeding
  (`set_partition_entry_points`) was removed.

## Removed

Overlap-mode / PASS3 search path, `remote_skip_*` probes, the search-time
`ColocationCache` LRU, `vector_store`, `PHASE_AWARE` fetch mode, the
`RandomStrategy` insert policy, the `BwProbeOp` workload op.

## Config schema

The YAML schema is **not backward compatible** with the submission's
`scripts/configs/vldb/*.yaml`: `update:`, `distributed_engine:`-as-toggle,
`batch_prefetch:` and `pq_filter:` were replaced by `distributed_engine.workload`,
`convergence_aware_prefetch:`, `consolidation_cache:`, `distributed_search:`,
`gt:` and `benchmark_probes:`. `scripts/configs/TEMPLATE.yaml` documents every key.

## Orchestration

`scripts/pipeline/` moved to `pipeline/` and the monolithic `engine.py` was
split into `pipeline/ops/{insert,delete,consolidate,checkpoint,concurrent}.py`
and `pipeline/ops/search/`. The launcher gained `--set KEY=VALUE` overrides
and `--postfix`. Results moved from `scripts/experiments/` to
`experiments/results/<dataset>/<experiment>/<timestamp>/`.
