"""Cluster configuration for the DANNS distributed pipeline.

Single authority for config/nodes.yaml parsing. All other modules
(orchestrator, node_worker, launcher) use this module's API.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ClusterNode:
    id: int
    host: str
    rdma_port: int
    sync_port: int
    ssh_host: str = ""
    container: str = ""
    gpu: bool = False
    local_ssd: str = ""
    rdma_device: str = ""
    gid_index: Optional[int] = None


@dataclass
class NodesConfig:
    """Complete parsed result from nodes.yaml."""
    nodes: List[ClusterNode]       # DPU worker nodes (port_offset applied)
    orchestrator: ClusterNode      # Orchestrator node (port NOT offset-adjusted)
    rdma_cfg: dict = field(default_factory=dict)
    port_offset: int = 0


def parse_nodes_file(nodes_path: str) -> NodesConfig:
    """Parse nodes.yaml once and return everything.

    port_offset is applied to DPU node ports (rdma_port, sync_port).
    Orchestrator sync_port is used as-is (already final value in nodes.yaml).
    """
    with open(nodes_path) as f:
        data = yaml.safe_load(f)

    port_offset = data.get('port_offset', 0)

    # DPU worker nodes
    dpu_nodes = []
    for n in data.get('nodes', []):
        nd = ClusterNode(**n)
        nd.rdma_port += port_offset
        nd.sync_port += port_offset
        dpu_nodes.append(nd)

    # Orchestrator node
    orch_data = data.get('orchestrator', {})
    orch_node = ClusterNode(
        id=orch_data.get('id', 0),
        host=orch_data.get('host', 'localhost'),
        rdma_port=0,
        sync_port=orch_data.get('sync_port', 19700),
        ssh_host=orch_data.get('ssh_host', orch_data.get('host', 'localhost')),
    )

    rdma_cfg = data.get('rdma', {})

    return NodesConfig(
        nodes=dpu_nodes,
        orchestrator=orch_node,
        rdma_cfg=rdma_cfg,
        port_offset=port_offset,
    )


class ClusterConfig:
    def __init__(self, num_nodes: int, coordinator_node: int, nodes: List[ClusterNode]):
        self.num_nodes = num_nodes
        self.coordinator_node = coordinator_node
        self.nodes: Dict[int, ClusterNode] = {n.id: n for n in nodes}

    @classmethod
    def from_config(cls, cluster_cfg: dict, config_dir: str,
                    coordinator_node: int = None,
                    active_nodes: List[int] = None) -> Tuple['ClusterConfig', dict]:
        """Parse nodes.yaml and build ClusterConfig (DPU nodes only, no orchestrator)."""
        nodes_file = cluster_cfg.get('nodes_file', '')
        if nodes_file:
            nodes_path = str(Path(config_dir) / nodes_file)
            parsed = parse_nodes_file(nodes_path)
            all_nodes = parsed.nodes
            rdma_cfg = parsed.rdma_cfg
        else:
            all_nodes = [ClusterNode(**{k: v for k, v in n.items() if k != 'role'})
                         for n in cluster_cfg.get('nodes', [])]
            rdma_cfg = {}

        if active_nodes is not None:
            active_set = set(active_nodes)
            nodes = [n for n in all_nodes if n.id in active_set]
            missing = active_set - {n.id for n in nodes}
            if missing:
                raise ValueError(f"Node IDs {missing} not found in nodes.yaml")
        else:
            nodes = all_nodes

        if coordinator_node is None:
            coordinator_node = cluster_cfg.get('coordinator_node')
        if coordinator_node is None:
            coordinator_node = min(n.id for n in nodes)

        return cls(len(nodes), coordinator_node, nodes), rdma_cfg

    @classmethod
    def build_full(cls, cluster_cfg: dict, config_dir: str,
                   coordinator_node: int = 0,
                   active_nodes: List[int] = None) -> Tuple['ClusterConfig', dict]:
        """Build ClusterConfig with orchestrator node 0 injected.

        Use this instead of from_config() when running with an external orchestrator.
        """
        nodes_file = cluster_cfg.get('nodes_file', '')
        if not nodes_file:
            raise ValueError("cluster.nodes_file is required for build_full()")

        nodes_path = str(Path(config_dir) / nodes_file)
        parsed = parse_nodes_file(nodes_path)

        # Filter to active DPU nodes
        if active_nodes is not None:
            active_set = set(active_nodes)
            nodes = [n for n in parsed.nodes if n.id in active_set]
            missing = active_set - {n.id for n in nodes}
            if missing:
                raise ValueError(f"Node IDs {missing} not found in nodes.yaml")
        else:
            nodes = parsed.nodes

        # Inject orchestrator as node 0
        all_nodes = [parsed.orchestrator] + nodes

        config = cls(len(all_nodes), coordinator_node, all_nodes)
        return config, parsed.rdma_cfg

    def get_node(self, node_id: int) -> Optional[ClusterNode]:
        return self.nodes.get(node_id)

    def get_peers(self, my_id: int) -> List[ClusterNode]:
        return sorted([n for n in self.nodes.values() if n.id != my_id], key=lambda n: n.id)




BUILD_STAGE_NAMES = {"build", "partition", "colocation", "pq_build", "gt"}
ENGINE_STAGE_NAME = "distributed_engine"


def apply_config_overrides(
    config: dict,
    overrides: List[str],
    *,
    log: bool = False,
) -> None:
    """Apply dotted `KEY=VALUE` overrides to a loaded config in place.

    Values are parsed via YAML so common CLI shapes like `true`, `1.2`,
    `[1, 2]`, and plain strings work consistently across launchers.
    """
    for kv in overrides or []:
        if "=" not in kv:
            raise ValueError(f"--set expected KEY=VALUE, got: {kv!r}")
        key, raw = kv.split("=", 1)
        path = [p for p in key.split(".") if p]
        if not path:
            raise ValueError(f"--set has empty key: {kv!r}")

        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw

        node = config
        for k in path[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[path[-1]] = value

        if log:
            print(f"  [override] {key} = {value!r}")


def normalize_canonical_config(config: dict) -> None:
    """Normalize canonical pipeline/feature config while preserving aliases.

    Canonical pipeline split:
      - build_pipeline: creates the initial graph/layout/artifacts
      - engine_pipeline: runs workload over an existing graph/layout

    Legacy sections (`pipeline.stages`, `distributed_engine.workload`) are
    still populated in-place so existing launchers and workers keep working.

    Canonical shape for Convergence-Aware Prefetching follows the paper:

      convergence_aware_prefetch:
        enabled: true
        batched_fetching: {...}
        pq_filtering: {...}

    The runtime still has older Python/C++ boundaries named `batch_prefetch`
    and `pq_filter`, so this function backfills those legacy sections.

    Benchmark probes are kept outside the search optimization taxonomy:

      benchmark_probes:
        compute_skip: false

    Older configs may still put `compute_skip` under `search`; that remains
    supported as a compatibility input.
    """
    # ---- Build Pipeline / Engine Pipeline ----------------------------
    build_pipe = config.get('build_pipeline')
    engine_pipe = config.get('engine_pipeline')
    legacy_pipeline = config.setdefault('pipeline', {})
    legacy_engine = config.setdefault('distributed_engine', {})

    if build_pipe is not None or engine_pipe is not None:
        if build_pipe is not None:
            build_pipe = build_pipe or {}
            build_stages = list(build_pipe.get('stages', []))
        else:
            build_stages = []

        if engine_pipe is not None:
            engine_pipe = engine_pipe or {}
            engine_enabled = bool(engine_pipe.get(
                'enabled', bool(engine_pipe.get('workload'))))
            engine_stages = [ENGINE_STAGE_NAME] if engine_enabled else []
        else:
            engine_stages = []

        if 'stages' not in legacy_pipeline:
            legacy_pipeline['stages'] = build_stages + engine_stages

    if build_pipe is not None:
        build_pipe = build_pipe or {}
        if 'skip_if_exists' in build_pipe and 'skip_if_exists' not in legacy_pipeline:
            legacy_pipeline['skip_if_exists'] = build_pipe['skip_if_exists']

    if engine_pipe is not None:
        engine_pipe = engine_pipe or {}
        if 'workload' in engine_pipe and 'workload' not in legacy_engine:
            legacy_engine['workload'] = engine_pipe.get('workload') or []
        for key in ('enabled', 'partition_policy', 'save_checkpoint',
                    'fail_fast'):
            if key in engine_pipe and key not in legacy_engine:
                legacy_engine[key] = engine_pipe[key]

        input_cfg = engine_pipe.get('input', {}) or {}
        if input_cfg.get('experiment_name') and not config.get('experiment_name'):
            config['experiment_name'] = input_cfg['experiment_name']
        if input_cfg.get('graph_variant'):
            config.setdefault('build', {}).setdefault(
                'graph_variant', input_cfg['graph_variant'])

    # Backfill canonical sections from legacy configs for reports/templates.
    if 'build_pipeline' not in config:
        stages = list(legacy_pipeline.get('stages', []))
        build_stages = [s for s in stages if s in BUILD_STAGE_NAMES]
        if build_stages:
            config['build_pipeline'] = {
                'stages': build_stages,
                'skip_if_exists': legacy_pipeline.get('skip_if_exists', True),
            }

    if 'engine_pipeline' not in config and legacy_engine.get('workload') is not None:
        config['engine_pipeline'] = {
            'enabled': ENGINE_STAGE_NAME in legacy_pipeline.get(
                'stages', [ENGINE_STAGE_NAME]),
            'input': {
                'experiment_name': config.get('experiment_name', 'unnamed'),
                'graph_variant': config.get('build', {}).get(
                    'graph_variant', 'iteration0'),
            },
            'workload': legacy_engine.get('workload', []),
        }

    # ---- Convergence-Aware Prefetching -------------------------------
    cap = config.get('convergence_aware_prefetch')

    # Transitional alias from the short-lived name introduced during refactor.
    short_cap = config.get('convergence_prefetch')
    legacy_batch = config.get('batch_prefetch')
    legacy_pq = config.get('pq_filter')

    if cap is None:
        if short_cap is not None:
            sc = short_cap or {}
            batch_size = int(sc.get('batch_size', 0) or 0)
            cap = {
                'enabled': bool(sc.get('enabled', batch_size > 0)),
                'batched_fetching': {
                    'enabled': bool(sc.get('enabled', batch_size > 0)),
                    'batch_size': batch_size,
                    'convergence_window': sc.get(
                        'convergence_window', sc.get('conv_window', 3)),
                    'convergence_threshold_pct': sc.get(
                        'convergence_threshold_pct',
                        sc.get('conv_threshold_pct', 1.0)),
                },
                'pq_filtering': dict(legacy_pq or {'enabled': False}),
            }
        elif legacy_batch is not None or legacy_pq is not None:
            lb = legacy_batch or {}
            lpq = legacy_pq or {}
            batch_size = int(lb.get('batch_size', 0) or 0)
            cap = {
                'enabled': batch_size > 0 or bool(lpq.get('enabled', False)),
                'batched_fetching': {
                    'enabled': batch_size > 0,
                    'batch_size': batch_size,
                    'convergence_window': lb.get(
                        'convergence_window', lb.get('conv_window', 3)),
                    'convergence_threshold_pct': lb.get(
                        'convergence_threshold_pct',
                        lb.get('conv_threshold_pct', 1.0)),
                },
                'pq_filtering': dict(lpq or {'enabled': False}),
            }
        else:
            cap = {'enabled': False}
        config['convergence_aware_prefetch'] = cap

    cap = cap or {}
    cap_enabled = bool(cap.get('enabled', False))
    bf = cap.get('batched_fetching', {}) or {}
    pq = cap.get('pq_filtering', {}) or {}

    bf_batch_size = int(bf.get('batch_size', 0) or 0)
    bf_enabled = cap_enabled and bool(bf.get('enabled', bf_batch_size > 0))
    if not bf_enabled:
        bf_batch_size = 0
    config['batch_prefetch'] = {
        'batch_size': bf_batch_size,
        'conv_window': bf.get(
            'conv_window', bf.get('convergence_window', 3)),
        'conv_threshold_pct': bf.get(
            'conv_threshold_pct', bf.get('convergence_threshold_pct', 1.0)),
    }

    pq_backfill = dict(pq)
    pq_backfill['enabled'] = cap_enabled and bool(pq.get('enabled', False))
    config['pq_filter'] = pq_backfill

    # ---- Benchmark Probes -------------------------------------------
    search = config.setdefault('search', {})
    probes = config.get('benchmark_probes')
    if probes is None:
        probes = {
            'compute_skip': bool(search.get('compute_skip', False)),
        }
        config['benchmark_probes'] = probes
    else:
        probes = probes or {}
        compute_skip = bool(probes.get(
            'compute_skip', search.get('compute_skip', False)))
        probes['compute_skip'] = compute_skip
        config['benchmark_probes'] = probes
        search.setdefault('compute_skip', compute_skip)

    # Legacy overlap search modes were removed. Existing configs often carry
    # overlap_mode: none; accept and drop that no-op, but fail loudly if an
    # old experiment asks for a real overlap mode.
    overlap_mode = search.pop('overlap_mode', None)
    if str(overlap_mode or 'none').lower() not in ('none', ''):
        raise ValueError(
            "search.overlap_mode is deprecated and unsupported; "
            "use convergence_aware_prefetch.batched_fetching instead")

    # ---- Consolidation Cache -----------------------------------------
    # This is separate from search-side convergence-aware prefetching. The
    # canonical cache implementation is per-deleted-node CSR.
    consol = config.setdefault('consolidation', {})
    # De-overlapped ("sync fetch") baseline. The OFF path's remote vector reads
    # normally ride a 256-deep RDMA WR pipeline inside post_read_batch, which
    # hides nearly all per-read RTT; true issues them one at a time, each
    # drained to completion. Bit-identical results, much slower wall clock.
    # Applies to BOTH arms: with the cache interceptor on it reaches only the
    # MISS fetches (hits never touch a fetch site), which is what the controlled
    # "naive vs naive+cache" pair needs.
    # SUPERSEDED by fetch_window below (sync_fetch=true == fetch_window=1);
    # kept working unchanged for backward compatibility.
    consol['sync_fetch'] = bool(consol.get('sync_fetch', False))
    # Fetch in-flight window depth K. Generalizes sync_fetch from a two-state
    # switch into the whole pipelining-depth axis between its two extremes
    # (deep100M/8p measured: naive 4,662 s <-> batched 1,175 s, a 3.97x cliff):
    #   0   = unlimited / the current batched behaviour (default)
    #   1   = exactly sync_fetch=true — the same C++ code path, so the measured
    #         naive baseline IS the K=1 point of the curve, not a neighbour
    #   >=2 = sliding window: keep at most K reads in flight, post one more per
    #         completion reaped. Clamped C++-side to the SQ cap (256).
    # Bit-identical results for every K; only wall clock moves. Same scope as
    # sync_fetch: the remote VECTOR fetch on both arms (cache ON reaches only
    # the interceptor's MISS fetches).
    # Precedence: fetch_window > 0 wins over sync_fetch (C++ logs the override);
    # fetch_window == 0 with sync_fetch true is mapped to K=1 C++-side.
    consol['fetch_window'] = max(0, int(consol.get('fetch_window', 0) or 0))
    # Buffer-reuse arm. The OFF path normally malloc/frees one heap block per
    # fetched remote vector (~4.5e9 pairs on deep100M/8p @ 10M deletes, each
    # value-initialized to zero and then immediately overwritten by the memcpy);
    # true stages them in a per-thread arena instead. Bit-identical results,
    # strictly less wall clock. Orthogonal to sync_fetch.
    # OFF-path only — ignored C++-side when the cache interceptor is active.
    consol['reuse_fetch_buffers'] = bool(consol.get('reuse_fetch_buffers', False))
    cache = config.get('consolidation_cache')
    if cache is None:
        config['consolidation_cache'] = {
            'enabled': bool(consol.get('enable_vec_cache', False)),
            'mode': 'per_deleted_node',
            'neighbor_records_per_thread': 0,
            'global_vector_bytes': 0,
        }
    else:
        cache = cache or {}
        consol['enable_vec_cache'] = bool(cache.get('enabled', False))
        mode = str(cache.get('mode', 'per_deleted_node'))
        if mode != 'per_deleted_node':
            raise ValueError(
                "consolidation_cache.mode must be 'per_deleted_node'; "
                "older vector-cache implementations were removed")
        for key in ('neighbor_records_per_thread', 'global_vector_bytes'):
            cache[key] = int(cache.get(key, 0) or 0)
            if cache[key] < 0:
                raise ValueError(f"consolidation_cache.{key} must be >= 0")


def get_pq_filter_config(config: dict) -> dict:
    """Return canonical PQ-filtering config, with legacy fallback.

    PQ filtering is a sub-mechanism of convergence-aware prefetching. Older
    runtime boundaries still expose a top-level ``pq_filter`` section, so this
    helper keeps call sites from depending on that legacy name directly.
    """
    cap = config.get('convergence_aware_prefetch', {}) or {}
    pq = dict(cap.get('pq_filtering', {}) or {})
    cap_enabled = bool(cap.get('enabled', False))
    if pq:
        pq['enabled'] = cap_enabled and bool(pq.get('enabled', False))
        return pq
    return dict(config.get('pq_filter', {}) or {})


# ============================================================================
# Shared utilities (used by orchestrator.py and node_worker.py)
# ============================================================================

def resolve_config_paths(config: dict, project_root: Path) -> None:
    """Resolve relative paths and normalize feature aliases in-place.

    Dataset files resolve against dataset.data_root when set, else project_root.
    output/results/artifacts roots and colocation.output_dir resolve against
    project_root.
    """
    normalize_canonical_config(config)
    root = Path(project_root)

    ds = config.get('dataset', {})
    data_root = ds.get('data_root', '')
    ds_root = Path(data_root) if data_root else root
    for key in ('base_file', 'query_file', 'centroids_file', 'groundtruth_file'):
        val = ds.get(key, '')
        if val and not os.path.isabs(val):
            ds[key] = str((ds_root / val).resolve())

    out = config.get('output', {})
    for key in ('base_dir', 'run_base_dir', 'results_base_dir',
                'artifact_base_dir', 'artifacts_base_dir'):
        val = out.get(key, '')
        if val and not os.path.isabs(val):
            out[key] = str((root / val).resolve())

    for section_name in ('results', 'artifacts'):
        section = config.get(section_name, {}) or {}
        val = section.get('base_dir', '')
        if val and not os.path.isabs(val):
            section['base_dir'] = str((root / val).resolve())
        if section:
            config[section_name] = section

    coloc = config.get('colocation', {})
    val = coloc.get('output_dir', '')
    if val and not os.path.isabs(val):
        coloc['output_dir'] = str((root / val).resolve())


def get_launch_info(config_path: str, overrides: List[str] = None) -> dict:
    """Extract all info the launcher needs. Called once via docker exec."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    apply_config_overrides(config, overrides or [])
    resolve_config_paths(config, Path('/workspace'))

    try:
        from pipeline.experiment_paths import (
            dataset_id, experiment_label, has_split_results, results_root,
        )
    except ModuleNotFoundError:
        from experiment_paths import (
            dataset_id, experiment_label, has_split_results, results_root,
        )

    config_dir = str(Path(config_path).parent)
    cluster_cfg = config.get('cluster', {})
    nodes_file = cluster_cfg.get('nodes_file', 'nodes.yaml')
    nodes_path = str(Path(config_dir) / nodes_file)

    parsed = parse_nodes_file(nodes_path)

    return {
        'orch_host': parsed.orchestrator.host,
        'orch_port': parsed.orchestrator.sync_port,
        'node_ids': [n.id for n in parsed.nodes],
        'ssh_hosts': {n.id: n.ssh_host for n in parsed.nodes},
        'local_ssds': {n.id: n.local_ssd for n in parsed.nodes},
        'containers': {n.id: n.container for n in parsed.nodes},
        'experiment_name': config.get('experiment_name', 'unnamed'),
        'dataset_id': dataset_id(config),
        'experiment_label': experiment_label(config),
        'split_results': has_split_results(config),
        'results_root': str(results_root(config)),
        'num_partitions': config.get('partition', {}).get('num_partitions', 4),
    }


# ============================================================================
# CLI: outputs bash-sourceable key=value lines for the launcher
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Emit bash-sourceable launch variables for run_experiment.sh"
    )
    parser.add_argument("config", help="Pipeline YAML config")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config key via dotted path before extracting launch info",
    )
    args = parser.parse_args()

    info = get_launch_info(args.config, args.overrides)

    print(f"ORCH_HOST={info['orch_host']}")
    print(f"ORCH_PORT={info['orch_port']}")
    print(f"EXPERIMENT_NAME={info['experiment_name']}")
    print(f"DATASET_ID={info['dataset_id']}")
    print(f"EXPERIMENT_LABEL={info['experiment_label']}")
    print(f"SPLIT_RESULTS={1 if info['split_results'] else 0}")
    print(f"RESULTS_ROOT_CONTAINER={info['results_root']}")

    node_ids = ','.join(str(i) for i in info['node_ids'])
    print(f"NODES_CSV={node_ids}")

    for nid, ssh in info['ssh_hosts'].items():
        print(f"NODE_SSH_{nid}={ssh}")

    for nid, ssd in info['local_ssds'].items():
        print(f"NODE_LOCAL_SSD_{nid}={ssd}")

    for nid, ctr in info['containers'].items():
        print(f"NODE_CONTAINER_{nid}={ctr}")

    print(f"NUM_PARTITIONS={info['num_partitions']}")
