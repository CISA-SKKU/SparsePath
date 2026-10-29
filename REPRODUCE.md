# REPRODUCE — Phase B (data + graphs) and Phase C (runs + figure)

Assumes Phase A ([INSTALL.md](INSTALL.md)) is done: containers up on all hosts,
`lib/*.so` built, `experiments/configs/nodes.yaml` filled, `.env` exported.
All host-side commands below run **on the orchestrator host**, from the
artifact root unless stated otherwise. `DANNS_EXPERIMENT_ROOT` defaults to
`./experiments` (container path `/experiments`).

## Phase B

### B-1. Dataset slice (once)

The graphs were built over the first 300M vectors of DEEP1B (100M initial
window + 200M insert pool), stored as one DiskANN `.fbin`:

```bash
RAW_DIR=$RAW_DIR DATASETS_ROOT=$DATASETS_ROOT bash data/download_and_extract.sh deep1B
ls -l $DATASETS_ROOT/DEEP1B/deep300M_{base,query}.fbin      # 115 GB + 3.8 MB
```

Only the raw shards covering the first 300M vectors are downloaded
(`data/README.md`). The container sees the slice at
`/workspace/datasets_nfs/DEEP1B/` (mount set by `DATASETS_ROOT`).
The slice must be byte-identical to ours because checkpoints are data-less
(vectors are re-read as `base[tag-1]`); the bundle's `BUNDLE.sha256` includes
the expected checksum of `deep300M_base.fbin`.

### B-2. Graphs — choose one

**B-fast (recommended): prebuilt bundle (~49 GB).**
Follow `experiments/artifacts/deep100M/README.md`; afterwards:

```
experiments/artifacts/deep100M/graph/checkpoints/{iteration150,lits_k4_iter150}{,.tags,.frozen}  (+ lits_k4_iter150.labels)
experiments/artifacts/deep100M/gt/gt_{iteration150,lits_k4_iter150}_K10_Q10000.{npy,json,_queries.npy}
```

The two graphs share one active set, so the `gt_lits_k4_iter150_*` files are
symlinks to the `gt_iteration150_*` files (`tar` preserves them).

**B-full: rebuild the update-stream graphs (days per graph, orchestrator, ≥ 256 GB DRAM).**

```bash
docker compose exec danns ./build_lits.sh                 # LITS-aware bindings → third_party/lits_build_tree/lib
docker compose exec danns bash -c '
  cd /workspace && B=/workspace/datasets_nfs/DEEP1B/deep300M_base.fbin; A=/experiments/artifacts/deep100M
  # SparsePath graph: cluster labels/centroids for the initial window, then the LITS update stream
  python3 tools/graph_build/prep_lits_cluster.py --base $B --dtype float32 --dim 96 --initial-size 100000000 \
      --k 4 --out-prefix $A/lits_assets/deep100M_k4
  python3 tools/graph_build/build_updatecycle_lits.py --base $B --dtype float32 --dim 96 \
      --out-ckpt-dir $A/graph/checkpoints --initial-size 100000000 --max-capacity 120000000 \
      --points-insert 1000000 --points-delete 1000000 --consolidate-every 5 --max-iterations 150 --save-every 50 \
      --R 64 --L 100 --alpha 1.2 --num-threads 0 \
      --lits-labels $A/lits_assets/deep100M_k4.labels --lits-centroids $A/lits_assets/deep100M_k4.centroids \
      --lits-clusters 4 --lits-threshold 0.7 --lits-filter-mode long_inter_only --lits-tau-scale 1.0 \
      --ckpt-name-prefix lits_k4_iter --bindings-lib /workspace/third_party/lits_build_tree/lib
  # Balanced graph: same stream without LITS (mainline bindings)
  python3 tools/graph_build/build_updatecycle.py --base $B --dtype float32 --dim 96 \
      --out-ckpt-dir $A/graph/checkpoints --initial-size 100000000 --max-capacity 120000000 \
      --points-insert 1000000 --points-delete 1000000 --consolidate-every 5 --max-iterations 150 --save-every 50 \
      --R 64 --L 100 --alpha 1.2 --num-threads 0 --bindings-lib /workspace/lib
'
```

Exact flags are recorded in the bundle's `build_config_lits_k4.yaml`
(as-run dump). `prep_lits_cluster.py --help` / `build_updatecycle*.py --help`
list all options.

### B-3. Partition, colocation, PQ codes, ground truth (per arm, ~1–2 h total; ~260 GB of colocation payload for both arms)

```bash
cd scripts
./prepare_data.sh configs/ae/deep100M_4p_balanced.yaml   --from partition --skip-deploy
./prepare_data.sh configs/ae/deep100M_4p_sparsepath.yaml --from partition --skip-deploy
cd ..
docker compose exec danns python3 pipeline/run_pipeline.py \
    --config /experiments/configs/ae/deep100M_4p_sparsepath.yaml --stages pq_build gt
docker compose exec danns python3 pipeline/run_pipeline.py \
    --config /experiments/configs/ae/deep100M_4p_balanced.yaml --stages gt
```

| Stage | Output under `experiments/artifacts/deep100M/` | Note |
|---|---|---|
| `partition` | `partition/4_partitions_{round_robin,lits_balanced}/` | k-means on the active window + insert-pool assignment |
| `colocation` | `colocation/4_partitions_<strategy>/<graph_variant>/partition{0..3}.coloc, tag_to_local_p*.bin, local_to_tag_p*.bin, global_to_vectorid.bin, colocation_meta.json` | **~32 GB per partition, ~130 GB per arm** (vectors + neighbour lists with slack) |
| `pq_build` | `pq/M24_active/{pq_codebook.bin,pq_all_codes.bin,pq_meta.json}` | SparsePath arm only (PQ filtering, M=24) |
| `gt` | `gt/gt_<graph_variant>_K10_Q10000.*` | skipped if the bundle already provides it |

`--from partition` skips the `build` stage explicitly; with the checkpoints in
place `skip_if_exists: true` would skip it anyway. Then verify every worker
can see its payload **before** launching (a missing `partition<p>.coloc` only
fails after the RDMA handshake):

```bash
docker compose exec danns python3 /experiments/analysis/check_colocation.py /experiments/configs/ae/deep100M_4p_balanced.yaml
docker compose exec danns python3 /experiments/analysis/check_colocation.py /experiments/configs/ae/deep100M_4p_sparsepath.yaml
```

Workers read the payload from the shared `/experiments` mount. To load from
node-local SSD instead, copy the `colocation/.../<variant>/` directory to each
worker's `WORKER_LOCAL_SSD` and uncomment `colocation.local_dir` in the config.

## Phase C

### C-1. Run the two arms

```bash
cd scripts
RUN_RECALL_POSTHOC=1 ./run_experiment.sh configs/ae/deep100M_4p_balanced.yaml   --nodes 1,2,3,4
RUN_RECALL_POSTHOC=1 ./run_experiment.sh configs/ae/deep100M_4p_sparsepath.yaml --nodes 1,2,3,4
```

Each launch: starts `pipeline/run_pipeline.py` in the orchestrator container,
SSHes into the 4 workers and starts `pipeline/node_worker.py` in their `danns`
containers, completes the TCP barrier + RDMA QP setup, loads the partition
payloads, then runs the workload: one `search` op, 10 000 queries, K=10,
16 threads per worker, L = 10…300 (30 points), 5 timing loops per L.
`RUN_RECALL_POSTHOC=1` computes recall against the GT afterwards and emits
`results.csv`. Engine wall time ≈ 3.5–4.5 min per arm on the authors' cluster (payload load + 30 L × 5 loops); the recall post-hoc step adds a few minutes.

Run directory: `experiments/results/deep100M/ae_deep100M_4p_<arm>/<UTC timestamp>/` with
`engine_report.txt` (per-op summary + RECALL SUMMARY), `node_stats/node*_seq*.json`,
`recall_posthoc.json`, `results.csv` (one row per L × worker; columns include
`seq, node_id, T, L, recall_10, qps, remote_vector_accesses, …`), `logs/`.

Useful knobs: `--set distributed_engine.workload.0.num_threads=32` (thread
count), `--set distributed_search.frozen_seed_cache=false` (A/B of the
frozen-seed cache), `--postfix LABEL` (suffix the run dir).

### C-2. Plot

```bash
./plot/plot_qps_recall.py \
    --label Balanced   ../experiments/results/deep100M/ae_deep100M_4p_balanced/<ts>/results.csv \
    --label SparsePath ../experiments/results/deep100M/ae_deep100M_4p_sparsepath/<ts>/results.csv \
    -o ../experiments/results/fig13_deep100M_4p.pdf
```

QPS is **summed over the 4 workers** per L (the paper's Σ QPS convention);
the script also prints iso@0.95 (highest Σ QPS with recall ≥ 0.95) per arm,
which is the number the paper's speed-up ratios are built from.

### What success looks like

Measured on the authors' cluster with this package's configs (2026-09-02,
4 workers × 16 threads, ConnectX-6 RoCEv2), so a reviewer can calibrate:

| Arm | rows in `results.csv` | recall@10 at L=100 | iso@0.95 Σ QPS (L) | run wall time |
|---|---|---|---|---|
| SparsePath 4p | 120 (30 L × 4 workers) | 0.966 | ≈ 57 k (L=80) | ≈ 3.5 min (+ ~2 min recall post-hoc) |
| Balanced 4p | 120 | ≈ 0.97 | well below SparsePath at equal recall | ≈ 4 min (+ post-hoc) |

- Recall is monotone in L, ≈ 0.75 at L=10, ≥ 0.95 from L ≈ 80, ≈ 0.99 at L=300,
  and it is a property of the graph + GT, not of the engine: re-running an arm
  reproduces its recall column exactly, while Σ QPS varies by a few percent
  between runs (NIC/NUMA state). A recall of 0.0 means a colocation/GT
  mismatch — rerun the B-3 checks.
- `node_stats/node*_seq0.json` of both arms show `frozen_seed_cache_hits` in
  the hundreds of thousands per worker (FROZEN_ONLY seeding + cache active);
  the SparsePath arm additionally shows `pq_filtered` in the tens of millions
  per L and far fewer `remote_vector_accesses` than Balanced.
- The SparsePath curve lies above the Balanced curve over the whole recall
  range; the paper's deep100M/4p speed-up is the ratio of the two iso@0.95 Σ QPS
  values (§6.2.1 — absolute numbers depend on NIC speed and core count).

## What else can be reproduced

| Tier | Figures / sections | How | Needs |
|---|---|---|---|
| **CORE** (this guide) | Fig. 13 deep100M/4p (2 of 4 arms) | above | 4 RDMA workers |
| CORE-build-only | Fig. 15 inter-partition edge ratio / imbalance | `prepare_data.sh … --from partition` + `experiments/analysis/check_colocation.py`; edge stats via `scripts/lits_stats/` | orchestrator only |
| CORE-extra | Fig. 19 ablation (batching / PQ / LITS), Fig. 20 α, B sensitivity | same configs with `--set convergence_aware_prefetch.*` overrides | 4 workers |
| CORE-extra | Fig. 18 consolidation cache (4p subset) | `consolidation_cache.enabled` + workload `delete → search → consolidate` (see `scripts/configs/TEMPLATE.yaml`); a 4p config is not shipped in this version | 4 workers |
| EXTENDED | 8p/16p scaling, 200M/400M, τ sensitivity, other datasets | same pipeline; build graphs per dataset (`data/`, `tools/graph_build/`), regenerate partition/colocation for N workers | 8–16 workers, more storage |
| SINGLE-NODE | §6.2.3 false filtering (M11), §6.2.4 quantized search (Q1/W3), CAP semantics (M10) | `scripts/m11_pq_false_filtering/`, `scripts/q1_quantization_pareto/`, `scripts/convergence_quantization/`, `scripts/m10_cap_semantics/` (each has a README; manifests reference `/experiments/artifacts/...`) | one host, `lib/single_engine.so` |
| EXTERNAL | RED-ANNS / CoTra baselines | not included (third-party systems, separate ports) | — |

## Limitations

- Four workers are hard-wired; a different worker count needs new
  partition + colocation outputs (`partition.num_partitions`) and 4-way
  colocation payloads are not reusable.
- No single-node fallback for the distributed engine.
- Baseline systems (RED-ANNS, CoTra) and the 8p/16p/400M campaigns are out of
  scope of this package.
- Absolute QPS depends on NIC line rate, core count per worker and NUMA
  placement; recall values and the relative ordering of the arms are the
  reproducible quantities.
