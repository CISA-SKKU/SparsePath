# INSTALL — Phase A, one-time cluster setup

Everything here is done once per cluster. Phase B/C are in [REPRODUCE.md](REPRODUCE.md).

## 1. Prerequisites

Every host (orchestrator + 4 workers):
- Ubuntu 22.04 (only distro exercised), Docker ≥ 24 with the Compose v2 plugin.
- Key-based, non-interactive SSH from the orchestrator to every worker
  (`ssh <worker> true` must return without a prompt).
- The artifact root **at the same path on every host**. A shared filesystem
  (NFS) is the tested layout and also makes `experiments/` (graphs, colocation,
  results) visible everywhere without copying.

Workers in addition:
- One RDMA NIC on a common RoCEv2/InfiniBand fabric; Mellanox OFED ≥ 5.8
  (`ibv_devinfo` lists an active port; note the device name, e.g. `mlx5_0`).
- ≥ 64 GB DRAM (deep100M/4p uses ~30 GB per worker), ≥ 16 physical cores.
- `ulimit -l unlimited` capability (the compose files set `memlock: -1`).

Orchestrator:
- ≥ 128 GB DRAM for the partition/colocation/GT stages over the 100M active set.
- ~250 GB free under `DATASETS_ROOT` (300M-vector slice) and **~450 GB** under
  `experiments/` (bundle 49 GB + colocation payload ~130 GB per arm + PQ 7 GB
  + partitions 5 GB). Colocation payload can be deleted after the runs.

No GPU is required or used.

## 2. Unpack and configure

```bash
ARTIFACT_ROOT=/srv/sparsepath/artifact        # same on every host
tar -xf sparsepath-artifact-postrev.tar -C "$ARTIFACT_ROOT" && cd "$ARTIFACT_ROOT"

cp .env.example .env && $EDITOR .env          # DATASETS_ROOT, EXPERIMENTS_ROOT, WORKER_LOCAL_SSD,
                                              # RDMA_DEVICE, RDMA_GID_INDEX, HOST_UID/GID, DANNS_ARCH
cp experiments/configs/nodes.yaml.template experiments/configs/nodes.yaml && $EDITOR experiments/configs/nodes.yaml
```

`nodes.yaml` fields:

| Field | Meaning |
|---|---|
| `orchestrator.host` | hostname/IP of the orchestrator, TCP-reachable from every worker |
| `nodes[i].host` | worker *i*'s IP on the RDMA rail (the address the QP handshake advertises) |
| `nodes[i].ssh_host` | name the orchestrator uses to `ssh` into worker *i* |
| `nodes[i].container` | must stay `danns` (the launcher execs by name) |
| `nodes[i].rdma_device` | `ibv_devinfo` device on that worker |
| `rdma.gid_index` | GID index of the RoCEv2/IPv4 entry (`show_gids`); usually 3 |
| `port_offset` | 0 on a dedicated cluster |

`DANNS_ARCH` in `.env` must be an ISA every host that loads `lib/*.so`
supports (`icelake-server` for Ice Lake + Sapphire Rapids; never `native`
across heterogeneous hosts — see `CMakeLists.txt`).

## 3. Containers

Build the image once and bring up the orchestrator container:

```bash
docker compose build          # ~15 min; Miniconda env `danns`, DiskANN deps, RDMA userspace
docker compose up -d
docker compose exec danns true && echo "orchestrator container OK"
```

On every worker (image via a registry, or `docker save danns:latest | ssh <worker> docker load`):

```bash
cd "$ARTIFACT_ROOT"
cp docker-compose.worker.yml.template docker-compose.worker.yml
docker compose -f docker-compose.worker.yml up -d
docker compose -f docker-compose.worker.yml exec danns ibv_devinfo | head   # device visible inside
```

The worker compose mounts `/dev/infiniband`, the shared `experiments/`
directory at `/experiments`, and `WORKER_LOCAL_SSD` at `/workspace/colocation_local`.

## 4. Build the engine

```bash
docker compose exec danns ./build_all.sh          # → lib/{distributed_bindings,diskann_bindings,pq_search_bindings,single_engine}.so
```

`build_all.sh` reads `DANNS_ARCH` / `DANNS_FAST_MATH` from the container
environment (set through `.env` → compose). If the artifact root is **not**
shared, copy `lib/` and `build/` to every worker (or run `build_all.sh` in
each worker container — the inputs are identical).

`./build_lits.sh` is only needed for Phase B-full (rebuilding the SparsePath
graph from scratch); skip it otherwise.

Unit tests (host, any Python 3.10+ with `pyyaml numpy pytest`):

```bash
python3 -m pytest -q tests/
```

## 5. RDMA sanity check

```bash
# worker 1
docker compose -f docker-compose.worker.yml exec danns ib_send_bw -d $RDMA_DEVICE -F
# worker 2
docker compose -f docker-compose.worker.yml exec danns ib_send_bw -d $RDMA_DEVICE -F <worker1_rdma_ip>
```

Expect near line rate. If this fails the engine's QP handshake will fail too;
fix the fabric (GID index, MTU, subnet manager) before Phase B.

## 6. Optional: PMU counters

Only for the optional per-run PMU breakdown (`PIPELINE_PMU=1`):
`echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid` on every worker.
The guided figure does not need it.
