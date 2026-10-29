# SparsePath — artifact (PVLDB / VLDB 2027)

Source code and reproduction scripts for **SparsePath**, a distributed
graph-based ANN index with RDMA-colocated partitions that stays fast under
continuous inserts and deletes. The paper has been **accepted to PVLDB (VLDB 2027)**;
this repository is its artifact. Changes since the originally submitted code
are listed in [CHANGES_SINCE_SUBMISSION.md](CHANGES_SINCE_SUBMISSION.md).

> **Artifact status (v0.1, work in progress).**
> Verified so far: the C++ engine and Python pipeline build from this tree in
> the provided container, and the two guided configurations
> (`experiments/configs/ae/*.yaml`) run end-to-end on the authors' 4-worker
> RDMA cluster and reproduce the paper's deep100M search curve shape.
> Still being prepared for the reproducibility submission: the prebuilt graph
> bundle (~49 GB, see `experiments/artifacts/deep100M/README.md`), a
> one-command preflight/driver script, the consolidation-cache configuration
> (Fig. 18), and a full dry run of Phase B from a fresh cluster. Until then,
> Phase B as documented is correct but has been exercised only on the authors'
> machines. Citation information will be added with the camera-ready version.

The guided reproduction path is the paper's headline figure:
**QPS vs. recall@10 on deep100M with 4 workers**, comparing the *Balanced*
baseline (round-robin placement, plain update graph) with *SparsePath*
(LITS-aware update graph + LITS-balanced placement + convergence-aware
prefetch), both after 150 update iterations (1% insert + 1% delete each).
Other figures are classified by reproducibility tier in
[REPRODUCE.md](REPRODUCE.md#what-else-can-be-reproduced).

## System configuration

Tested cluster (all numbers in the paper):

- **1 orchestrator host** — coordinates runs over TCP/SSH, builds partitions,
  no RDMA NIC needed.
- **4 worker hosts** — Intel Xeon (Ice Lake / Sapphire Rapids), 256–384 GB DRAM,
  one NVIDIA ConnectX-6 (RoCEv2, 100/200 Gbps) each, node-local NVMe SSD, no GPU.
- Ubuntu 22.04, Docker ≥ 24 with Compose v2, Mellanox OFED ≥ 5.8.
- The artifact root and `experiments/` are on a filesystem shared by all
  hosts (NFS). Non-shared setups work but need manual copying (see INSTALL.md §3).

The worker count is fixed to **4** by the shipped configs
(`partition.num_partitions: 4`, four entries in `nodes.yaml`). Any RDMA
cluster with ≥ 4 workers, ≥ 64 GB DRAM per worker and ≥ 200 GB of shared
storage can run the guided path.

## Repository layout

```
.
├── README.md  INSTALL.md  REPRODUCE.md        — read in this order
├── CHANGES_SINCE_SUBMISSION.md                — what moved since the submission package
├── PORTABILITY.md                             — every site-specific knob / known remnant
├── VERSION  MANIFEST.sha256  PACKAGE_CHECK.txt — provenance, checksums, leak-scan log
├── Dockerfile  entrypoint.sh  build_all.sh    — image + search-side build (lib/*.so)
├── build_lits.sh                              — graph-build-side build (only for Phase B-full)
├── docker-compose.yml                         — orchestrator container
├── docker-compose.worker.yml.template         — worker container (RDMA)
├── .env.example                               — host paths, RDMA device, build flags
├── CMakeLists.txt + core/ rdma/ trace/ store/ search/ update/ cluster/ bindings/
│                                              — C++ engine (search + update + RDMA)
├── third_party/diskann/                       — vendored DiskANN (search-side)
├── third_party/lits_build_tree/               — LITS-aware DiskANN bindings (graph build only)
├── pipeline/                                  — orchestrator, node worker, stages, workload ops
├── scripts/
│   ├── run_experiment.sh  prepare_data.sh  run_recall.sh  cleanup.sh
│   ├── common/                                — data loader, partition paths, RDMA/PMU monitors
│   ├── configs/TEMPLATE.yaml                  — annotated config reference
│   ├── plot/plot_qps_recall.py                — figure from results.csv
│   └── m10_* m11_* q1_* convergence_* lits_stats/ p1_*  — single-node review harnesses (tier SINGLE-NODE)
├── tools/graph_build/                         — update-stream graph builders (Phase B-full)
├── tests/                                     — pytest unit tests for the pipeline
├── data/                                      — dataset download + slicing
└── experiments/                               — DANNS_EXPERIMENT_ROOT (mounted at /experiments)
    ├── configs/nodes.yaml.template  configs/ae/*.yaml
    ├── analysis/                              — stats_to_csv, check_colocation, nic_bw_monitor
    ├── artifacts/deep100M/                    — prebuilt graphs + GT go here (README inside)
    └── results/                               — one directory per run
```

## Reproduction in three phases

| Phase | Document | What | Time |
|---|---|---|---|
| A | [INSTALL.md](INSTALL.md) | cluster inventory, containers, C++ build, RDMA sanity | ~1 h |
| B | [REPRODUCE.md](REPRODUCE.md#phase-b) | dataset slice, prebuilt graphs (or rebuild), partition + colocation + GT | 2–5 h (B-fast) / days (B-full) |
| C | [REPRODUCE.md](REPRODUCE.md#phase-c) | two search runs (Balanced, SparsePath) + plot | ~20 min |

## Quick start (cluster already set up, bundle downloaded)

```bash
cd scripts
./prepare_data.sh configs/ae/deep100M_4p_balanced.yaml   --from partition --skip-deploy
./prepare_data.sh configs/ae/deep100M_4p_sparsepath.yaml --from partition --skip-deploy
docker compose -f ../docker-compose.yml exec danns python3 pipeline/run_pipeline.py \
    --config /experiments/configs/ae/deep100M_4p_sparsepath.yaml --stages pq_build gt
RUN_RECALL_POSTHOC=1 ./run_experiment.sh configs/ae/deep100M_4p_balanced.yaml   --nodes 1,2,3,4
RUN_RECALL_POSTHOC=1 ./run_experiment.sh configs/ae/deep100M_4p_sparsepath.yaml --nodes 1,2,3,4
./plot/plot_qps_recall.py --label Balanced   ../experiments/results/deep100M/ae_deep100M_4p_balanced/<ts>/results.csv \
                          --label SparsePath ../experiments/results/deep100M/ae_deep100M_4p_sparsepath/<ts>/results.csv \
                          -o fig13_deep100M_4p.pdf
```

## Provenance

`VERSION` records the exact git commits this tree was cut from;
`MANIFEST.sha256` covers every file. The package is generated by a script
(`tools/make_post_revision_package.sh` in the authors' archive), so later
versions are regenerated from the same recipe. Licensed under the MIT License
(see [LICENSE](LICENSE)); vendored DiskANN code keeps its own MIT notice.
