# M11 — PQ False-Filtering measurement harness (single node, no RDMA)

Answers reviewer **R7·D4 / R3·D7**: show that PQ "false filtering" (the
`d_worst*(1+α)` filter dropping candidates whose true distance would have been
useful) has **negligible recall impact**, including on **LAION 512-d**.

Design rationale + code map: `00_Docs/SparsePath/02_Action_items/gates/M11_pq_false_filtering_design.md`.

## What this is

A **separate** pybind module `pq_search_bindings` (kept apart from
`diskann_bindings` for functional separation) that:

- loads the real global Vamana graph (`load_graph_only`) + the real PQ
  codebook/codes (`danns::PQFilter`, reused verbatim),
- runs a **single-node beam search** mirroring `ColocationBeamSearch`'s
  converge-gated PQ filter (same `pq_dist > d_worst*(1+α)` rule, same
  `ConvergenceDetector`, squared-L2 units),
- and — because every vector is local — computes the PQ-approx **and** the exact
  distance for every dropped candidate, so false filtering is directly
  classifiable (a drop is *false* iff `exact ≤ d_worst`).

Why single-node is valid here: the filter **decision** and its recall
consequence are layout-independent; RDMA/colocation only change *what fetch is
skipped* (performance), not *which candidates are dropped* (recall). See the
design doc §2 (LOCAL_MMAP dead-end) for why the real distributed path can't do
this without a cluster.

**filter-all mode**: this harness filters *every* candidate (the all-remote /
M→∞ conservative upper bound). The real M=4 system only filters remote-owned
candidates, so real false-filtering is ≤ what we measure here. If it's negligible
here, it's negligible at M=4.

## Files

| file | role |
|---|---|
| `../../bindings/pq_search_bindings.cpp` | the C++ module (built to `lib/pq_search_bindings*.so`) |
| `run_m11.py` | driver: α sweep, ON/OFF recall, false-filter rate, PQ-error CDF, M11 table |
| `datasets.example.yaml` | per-dataset artifact-path schema (copy → `datasets.yaml`, fill) |

## Build

The module builds with the rest of the C++ (added as a target in
`bindings/CMakeLists.txt`). Build the same way as everything else:

```bash
# on orchestrator-host, inside the danns container (see memory: reference_build_command)
docker exec -w /workspace danns ./build_all.sh
# produces lib/pq_search_bindings.cpython-*.so  (/workspace is NFS-shared worker hosts)
```

(Requires the worktree's code to be on the build box. If building from this
worktree, deploy/checkout it where `build_all.sh` runs.)

## Run

```bash
cp datasets.example.yaml datasets.yaml     # then fill <PLACEHOLDER> paths
python3 run_m11.py \
    --config datasets.yaml --dataset deep100M --out results_m11
```

Outputs per dataset: `<ds>_rows.json`, `<ds>_drop_samples.npz`,
`<ts>_<ds>_m11_false_filtering.png` (false-filter% & recall Δ vs α; PQ-error CDF
vs the 1+α boundary), plus a combined `M11_table.md`.

### Sanity checks the driver prints (read these first)
- **`recall(OFF, oracle)`** should be high (e.g. > 0.99). If low, the
  `id_offset` / GT id-space mapping is wrong (GT ids are *global*; `mapped_id =
  tag-1+id_offset`).
- **`[sanity] pq-vs-exact corr`** should be > 0.5. If ~0, the `gid = tag-1`
  → PQ-code mapping is wrong (PQ distances uncorrelated with truth).

## RAM budget ⚠️ (matters for LAION)

`load_graph_only` holds the **base vectors** in RAM (aligned to dim):

| dataset | base RAM (100M) | fits 256 GB node? |
|---|---|---|
| deep 96-d f32 | ~38 GB | yes |
| sift 128-d u8 | ~13 GB | yes |
| spacev 100-d i8 | ~10 GB | yes |
| **laion 512-d f32** | **~205 GB** | **no (+ graph ~26 GB + PQ ~1.3 GB → OOM)** |

For **LAION**, use a **smaller-scale subset (still 512-d)** — e.g. laion 10M–50M
(512-d f32 = ~20–103 GB). False filtering is driven by the **dimensionality**
(PQ error at 512-d), not the point count, so a 512-d subset faithfully answers
R3·D7 at a fraction of the RAM. Build the subset graph + PQ the same way and
point `datasets.yaml` at it. (Or run full LAION on a >256 GB box.)

## Gating / rules
- This is single-node and RDMA-free, but the runs still consume a big-RAM box.
  Launch per the user's **"fire"** convention for cluster/heavy jobs
  (memory: feedback-redanns-fire-to-run-experiments). Building the `.so` is fine;
  firing the measurement runs waits for the go-ahead.
- Requires each dataset's `iteration0` graph + `iteration0.tags` +
  `iteration0.frozen` sidecar (data-less checkpoint) + `pq_codebook.bin` +
  `pq_all_codes.bin`. Verify these exist before running.
