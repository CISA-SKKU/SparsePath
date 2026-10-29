# PORTABILITY — site-specific knobs and known remnants

This tree was cut from a research repository that runs on one lab cluster.
Everything cluster-specific is meant to be reachable through the items below.
If you hit a hard-coded path that is not listed here, please report it.

## 1. Environment variables (host side)

| Variable | Read by | Default | Purpose |
|---|---|---|---|
| `DATASETS_ROOT` | compose | — (required) | host dir mounted read-only at `/workspace/datasets_nfs` |
| `EXPERIMENTS_ROOT` | compose | `./experiments` | mounted at `/experiments`; must be shared or synced across hosts |
| `WORKER_LOCAL_SSD` | worker compose | — (required on workers) | mounted at `/workspace/colocation_local` |
| `RDMA_DEVICE`, `RDMA_GID_INDEX` | worker compose → engine (`cluster/src/cluster_setup.cpp`) | `mlx5_0`, `3` | verbs device / GID |
| `RDMA_RENAME_PAIRS` | `entrypoint.sh` | unset | optional `hostdev:mlx5_N` renames inside the container |
| `HOST_UID`, `HOST_GID`, `USERNAME` | `entrypoint.sh` | 1000/1000/sparsepath | in-container user matching the host owner |
| `DANNS_ARCH`, `DANNS_FAST_MATH` | `build_all.sh` (via compose env) | `icelake-server`, `ON` | ISA target / `-Ofast` |
| `DANNS_EXPERIMENT_ROOT` | `scripts/*.sh`, `pipeline/cleanup.sh` | `<artifact>/experiments` | host path of the experiments root |
| `DANNS_EXPERIMENT_CONTAINER_ROOT` | same | `/experiments` | container path of the same |
| `DANNS_COMPOSE_FILE` | `scripts/run_experiment.sh` | `<artifact>/docker-compose.yml` | orchestrator compose file |
| `DANNS_PLAIN_DOCKER=1` | same | unset | use `docker exec danns` instead of `docker compose exec` |
| `RUN_RECALL_POSTHOC=1` | same | unset | compute recall + `results.csv` after a run |
| `PIPELINE_PMU=1` | same | unset | per-run PMU counters (needs `perf_event_paranoid=0`) |
| `PYTHON` | `pipeline/run_pipeline.sh` | `python3` | host interpreter for the local launcher |
| `BUILD_CONTAINER` | `pipeline/run_pipeline.sh` | `danns` | container for LOCAL stages |
| `DANNS_NODES` | `scripts/cleanup.sh` | empty | space-separated ssh names for the orphan sweep |
| `REMOTE_ARTIFACT_ROOT` | `danns-exec.sh` | — | artifact root on remote hosts |
| `RAW_DIR` | `data/download_and_extract.sh` | — | raw 1B downloads |
| `DANNS_SLACK_WEBHOOK`, `SLACK_WEBHOOK` | `scripts/run_experiment.sh`, `pipeline/run_eval.sh` | unset (no notifications) | optional Slack incoming-webhook for run notifications |

Cluster identity (worker IPs, ssh names, containers, ports) lives only in
`experiments/configs/nodes.yaml`.

## 2. Edits applied to the package copy (relative to the git tree)

| File | Edit | Why |
|---|---|---|
| `scripts/run_experiment.sh`, `scripts/prepare_data.sh`, `scripts/run_recall.sh`, `scripts/mark_failed_runs.sh`, `pipeline/cleanup.sh` | default experiments root `../02_SparsePath_experiments` → `$PROJECT_ROOT/experiments` | bundled `experiments/` |
| `pipeline/config.py` | default orchestrator host `orchestrator-host` → `localhost` | lab hostname |
| `pipeline/run_pipeline.sh` | `BUILD_CONTAINER` default `vecdb-build` → `danns` | stale name |
| `scripts/cleanup.sh` | hard-coded node lists → `$DANNS_NODES` | lab hostnames |
| `experiments/analysis/check_colocation.py` | lab roots → `DANNS_EXPERIMENT_ROOT` / `WORKER_LOCAL_SSD` | lab paths |
| `danns-exec.sh` | replaced by the submission's env-driven version | stale hard-coded path |
| `pipeline/run_eval.sh`, `scripts/run_experiment.sh` | hard-coded Slack webhook URL → `$SLACK_WEBHOOK` / `$DANNS_SLACK_WEBHOOK`, no-op when unset | credential removal |
| `docker-compose*.yml` | 22 per-host files replaced by one orchestrator file + one worker template | lab mounts/IPs |

## 3. Excluded from the package

`scripts/common/vldb_notify.sh` (Slack notifier), 16-worker socket-split
launcher/compose files, `scripts/bf2_perf.py` (NIC saturation harness),
`scripts/plan_artifact_migration.py`, `scripts/rerun_sharded_broadcast.sh`,
`scripts/**/fire_*.sh`, `scripts/**/dispatch_*.sh` (lab queue wrappers), per-host compose files.

## 4. Leak gate

The packaging script rewrites or removes every reference to the authors'
environment (home paths, dataset roots, host names, RDMA IPs, notification
webhooks) and then **fails the build if a single one remains**; the gate's
patterns and its empty result are in `PACKAGE_CHECK.txt`. What that means in
the tree:

- Lab-only automation is gone: Slack notifications, the run-inventory hook,
  the paper-eval runner, queue dispatchers (`fire_*.sh`, `dispatch_*.sh`),
  the 16-worker socket-split launcher, per-host compose files.
- Legacy fallback paths were removed from `pipeline/generate_gt.py`,
  `pipeline/node_worker.py` and `scripts/common/data_loader.py`; the GT tool
  is `build/third_party/diskann/apps/utils/compute_groundtruth_with_tags`,
  built by `build_all.sh`.
- Defaults that pointed at lab directories now point at the container mounts
  (`/experiments/...`, `/workspace/lib`, `/workspace/third_party/lits_build_tree/lib`)
  or `/tmp`; host names in comments were replaced by generic role names.
- `scripts/prepare_data.sh --deploy-only` still assumes the legacy container
  mount `/workspace/datasets_local` for per-node copies; the shipped worker
  compose mounts `/workspace/colocation_local`. The AE configs read the payload
  from the shared `/experiments` path instead (see REPRODUCE.md B-3).
- The single-node harness directories (`scripts/m10_*`, `m11_*`, `q1_*`,
  `convergence_*`, `lits_stats/`, `p1_*`) ship their as-run manifests; their
  `graph_file` / `base_file` entries are container-mount paths and assume
  graphs/PQ/GT beyond the core bundle. Their plot scripts default to
  `/experiments/results/...`; pass `--out` where offered.

## 5. Container internals worth knowing

- `entrypoint.sh` creates a user with `HOST_UID/HOST_GID`, grants passwordless
  `sudo` inside the container, and activates the conda env `danns`.
- Containers run `privileged` with `network_mode: host` for verbs and for the
  TCP barrier ports (`sync_port`, `rdma_port` in `nodes.yaml`).
