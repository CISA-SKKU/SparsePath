#!/usr/bin/env python3
"""Distributed local-SSD colocation builder.

Builds each partition's .coloc payload on the worker that owns it while
replicating metadata/mapping files on every worker.  The default mode is a
dry-run; pass --execute to run SSH/docker commands.
"""
from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
_scripts_dir = _project_root / "scripts"
for _p in [str(_this_dir), str(_project_root), str(_scripts_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.partition_paths import (  # noqa: E402
    colocation_dir as build_colocation_dir,
    get_insert_strategy,
    partition_dir as build_partition_dir,
)
from pipeline.config import normalize_canonical_config, parse_nodes_file  # noqa: E402
from pipeline.experiment_paths import (  # noqa: E402
    artifact_experiment_dir,
    has_split_artifacts,
)
from pipeline.partition_mapping import resolve_node_partition_map  # noqa: E402


DEFAULT_LOCAL_ROOT = "/workspace/colocation_local"
DEFAULT_HOST_LINK_ROOT = "/data/ssd0/sparsepath/colocation"
DEFAULT_HOST_STORE_ROOT = "/data/ssd0/sparsepath/colocation"


@dataclass(frozen=True)
class Worker:
    node_id: int
    partition: int
    ssh_host: str
    container: str


def _container_path(path: str, base: str = "/workspace") -> str:
    if not path:
        return base
    if posixpath.isabs(path):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(base, path))


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _ssh(host: str, command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", host, command]


def _docker_exec(container: str, script: str) -> list[str]:
    return ["docker", "exec", container, "bash", "-lc", script]


def _ssh_docker(host: str, container: str, script: str) -> list[str]:
    return _ssh(host, shlex.join(_docker_exec(container, script)))


def _print_command(label: str, argv: Sequence[str]) -> None:
    print(f"[DRY RUN] {label}:")
    print(f"  {shlex.join(list(argv))}")


def _run(argv: Sequence[str], label: str) -> None:
    print(f"[RUN] {label}")
    result = subprocess.run(list(argv), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def _run_parallel(commands: list[tuple[str, list[str]]]) -> None:
    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for label, argv in commands:
        print(f"[RUN] {label}")
        procs.append((label, subprocess.Popen(argv, text=True)))

    failures = []
    for label, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append(f"{label} exited {rc}")

    if failures:
        raise RuntimeError("; ".join(failures))


def _prepare_storage_script(experiment: str, link_root: str, store_root: str) -> str:
    link = posixpath.join(link_root, experiment)
    target = posixpath.join(store_root, experiment)
    if posixpath.normpath(link) == posixpath.normpath(target):
        return "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {_q(target)}",
            ]
        )
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {_q(link_root)} {_q(target)}",
            f"if [ -L {_q(link)} ]; then",
            f"  current=$(readlink {_q(link)})",
            f'  if [ "$current" != {_q(target)} ]; then',
            f'    echo "ERROR: {link} points to a different target: $current"; exit 1',
            "  fi",
            f"elif [ -e {_q(link)} ]; then",
            f"  echo {_q('ERROR: local colocation link path exists and is not a symlink: ' + link)}",
            "  exit 1",
            "else",
            f"  ln -s {_q(target)} {_q(link)}",
            "fi",
        ]
    )

def _build_script(
    *,
    experiment_base: str,
    local_coloc_dir: str,
    partition_dir: str,
    variant: str,
    partition: int,
    base_file: str,
    data_type: str,
    r_value: int,
    slack_factor: float,
    alignment: int,
    max_capacity_factor: float,
    tag_indexed: bool,
) -> str:
    graph_a = posixpath.join(experiment_base, "graph", variant)
    graph_b = posixpath.join(experiment_base, "graph", "checkpoints", variant)
    cmd = [
        "python3",
        "pipeline/build_colocation.py",
        "--graph-file",
        "$GRAPH_FILE",
        "--tags-file",
        "$TAGS_FILE",
        "--vectors-file",
        base_file,
        "--partition-dir",
        partition_dir,
        "--output-dir",
        local_coloc_dir,
        "--R",
        str(r_value),
        "--slack-factor",
        str(slack_factor),
        "--alignment",
        str(alignment),
        "--max-capacity-factor",
        str(max_capacity_factor),
        "--data-type",
        data_type,
        "--partition",
        str(partition),
    ]
    if tag_indexed:
        cmd.append("--tag-indexed")

    quoted_cmd = " ".join(_q(x) if x not in ("$GRAPH_FILE", "$TAGS_FILE") else x for x in cmd)
    return "\n".join(
        [
            "set -euo pipefail",
            "cd /workspace",
            f"mkdir -p {_q(local_coloc_dir)}",
            f"GRAPH_FILE={_q(graph_a)}",
            f"if [ ! -f \"$GRAPH_FILE\" ]; then GRAPH_FILE={_q(graph_b)}; fi",
            'TAGS_FILE="${GRAPH_FILE}.tags"',
            'test -f "$GRAPH_FILE"',
            'test -f "$TAGS_FILE"',
            f"test -f {_q(base_file)}",
            f"test -d {_q(partition_dir)}",
            quoted_cmd,
        ]
    )


def _validate_script(coloc_dir: str, partition: int, num_partitions: int) -> str:
    checks = [
        "set -euo pipefail",
        f"COLOC_DIR={_q(coloc_dir)}",
        f"PARTITION={partition}",
        f"NUM_PARTITIONS={num_partitions}",
        'test -f "$COLOC_DIR/colocation_meta.json"',
        'test -f "$COLOC_DIR/global_to_vectorid.bin"',
        'test -f "$COLOC_DIR/partition${PARTITION}.coloc"',
        # 16p socket-split (2026-08-11): the exactly-one .coloc guard is gone.
        # With two logical nodes per box the SAME local dir legitimately holds
        # two payloads (partition N-1 and N+7 on box N). Owning-partition
        # presence is still asserted above; stray payloads are harmless to the
        # loader (it opens partition${PARTITION}.coloc by name).
        'p=0; while [ "$p" -lt "$NUM_PARTITIONS" ]; do',
        '  test -f "$COLOC_DIR/tag_to_local_p${p}.bin"',
        '  test -f "$COLOC_DIR/local_to_tag_p${p}.bin"',
        "  p=$((p + 1))",
        "done",
    ]
    return "\n".join(checks)


def _copy_metadata_script(local_coloc_dir: str, nfs_coloc_dir: str) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"LOCAL={_q(local_coloc_dir)}",
            f"NFS={_q(nfs_coloc_dir)}",
            'mkdir -p "$NFS"',
            'cp "$LOCAL/colocation_meta.json" "$NFS/"',
            'cp "$LOCAL/global_to_vectorid.bin" "$NFS/"',
            'cp "$LOCAL"/tag_to_local_p*.bin "$NFS/"',
            'cp "$LOCAL"/local_to_tag_p*.bin "$NFS/"',
            'if [ -f "$LOCAL/centroids.fbin" ]; then cp "$LOCAL/centroids.fbin" "$NFS/"; fi',
            'echo "metadata/mapping copied to $NFS (partition*.coloc excluded)"',
        ]
    )


def _load_workers(config: dict, config_path: Path, mapping_arg: str | None) -> list[Worker]:
    cluster_cfg = config.get("cluster", {}) or {}
    nodes_file = cluster_cfg.get("nodes_file", "nodes.yaml")
    nodes_path = (config_path.parent / nodes_file).resolve()
    parsed = parse_nodes_file(str(nodes_path))
    node_ids = [n.id for n in parsed.nodes]
    num_partitions = int(config.get("partition", {}).get("num_partitions", 0))
    configured = mapping_arg if mapping_arg is not None else (
        config.get("colocation", {}) or {}
    ).get("node_partition_map")
    mapping = resolve_node_partition_map(node_ids, num_partitions, configured)

    workers = []
    for node in sorted(parsed.nodes, key=lambda n: n.id):
        workers.append(
            Worker(
                node_id=node.id,
                partition=mapping[node.id],
                ssh_host=node.ssh_host or node.host,
                container=node.container or "danns",
            )
        )
    return workers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build partition-local colocation payloads on worker SSDs"
    )
    parser.add_argument("config", help="Experiment YAML config")
    parser.add_argument("--variant", help="Graph/colocation variant, e.g. iteration150")
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--mapping", help="Node-to-partition map, e.g. 1:0,2:1")
    parser.add_argument("--execute", action="store_true",
                        help="Run SSH/docker commands. Omit for dry-run.")
    parser.add_argument("--skip-storage-prep", action="store_true")
    parser.add_argument("--host-link-root", default=DEFAULT_HOST_LINK_ROOT)
    parser.add_argument("--host-store-root", default=DEFAULT_HOST_STORE_ROOT)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    normalize_canonical_config(config)

    experiment = config.get("experiment_name", "unnamed")
    if has_split_artifacts(config):
        # Split-artifacts layout: graph/partition inputs and the NFS metadata
        # copy live under artifacts.base_dir/<dataset_id> (e.g.
        # /experiments/artifacts/deep100M), not output.base_dir/experiment_name.
        # `experiment` still names the node-local payload dir
        # (colocation_local/vldb_<ds>/...).
        experiment_base = str(artifact_experiment_dir(config))
    else:
        output_base = _container_path(
            str((config.get("output", {}) or {}).get("base_dir", ".")),
            "/workspace",
        )
        experiment_base = posixpath.join(output_base, experiment)
    variant = args.variant or (config.get("build", {}) or {}).get("graph_variant") or "iteration0"
    num_partitions = int((config.get("partition", {}) or {}).get("num_partitions", 0))
    strategy = get_insert_strategy(config)

    dataset_cfg = config.get("dataset", {}) or {}
    data_root = _container_path(str(dataset_cfg.get("data_root", "/workspace")), "/workspace")
    base_file = str(dataset_cfg.get("base_file", ""))
    if not posixpath.isabs(base_file):
        base_file = posixpath.normpath(posixpath.join(data_root, base_file))

    local_base = posixpath.join(args.local_root, experiment)
    local_coloc_dir = str(build_colocation_dir(local_base, num_partitions, strategy, variant=variant))
    nfs_coloc_dir = str(build_colocation_dir(experiment_base, num_partitions, strategy, variant=variant))
    partition_dir = str(build_partition_dir(experiment_base, num_partitions, strategy))

    coloc_cfg = config.get("colocation", {}) or {}
    build_cfg = config.get("build", {}) or {}
    workers = _load_workers(config, config_path, args.mapping)

    print("=" * 72)
    print("Distributed local colocation build")
    print("=" * 72)
    print(f"Config:      {config_path}")
    print(f"Experiment:  {experiment}")
    print(f"Variant:     {variant}")
    print(f"Strategy:    {strategy}")
    print(f"Local dir:   {local_coloc_dir}")
    print(f"NFS meta dir:{nfs_coloc_dir}")
    print(f"Mode:        {'EXECUTE' if args.execute else 'DRY RUN'}")
    for w in workers:
        print(f"  node {w.node_id} ({w.ssh_host}) -> partition {w.partition}")

    commands: list[tuple[str, list[str]]] = []

    if not args.skip_storage_prep:
        for w in workers:
            script = _prepare_storage_script(
                experiment, args.host_link_root, args.host_store_root
            )
            commands.append(
                (
                    f"prepare storage node {w.node_id}",
                    _ssh(w.ssh_host, script),
                )
            )

    build_commands = []
    for w in workers:
        script = _build_script(
            experiment_base=experiment_base,
            local_coloc_dir=local_coloc_dir,
            partition_dir=partition_dir,
            variant=variant,
            partition=w.partition,
            base_file=base_file,
            data_type=str(dataset_cfg.get("data_type", "float32")),
            r_value=int(build_cfg.get("R", 64)),
            slack_factor=float(coloc_cfg.get("slack_factor", 1.3)),
            alignment=int(coloc_cfg.get("alignment", 64)),
            max_capacity_factor=float(coloc_cfg.get("max_capacity_factor", 1.0)),
            tag_indexed=bool(coloc_cfg.get("tag_indexed", False)),
        )
        build_commands.append(
            (
                f"build node {w.node_id} partition {w.partition}",
                _ssh_docker(w.ssh_host, w.container, script),
            )
        )

    validate_commands = []
    for w in workers:
        script = _validate_script(local_coloc_dir, w.partition, num_partitions)
        validate_commands.append(
            (
                f"validate node {w.node_id} partition {w.partition}",
                _ssh_docker(w.ssh_host, w.container, script),
            )
        )

    first = workers[0]
    copy_command = (
        "copy metadata/mapping to NFS",
        _ssh_docker(
            first.ssh_host,
            first.container,
            _copy_metadata_script(local_coloc_dir, nfs_coloc_dir),
        ),
    )

    if not args.execute:
        for label, argv in commands:
            _print_command(label, argv)
        for label, argv in build_commands:
            _print_command(label, argv)
        for label, argv in validate_commands:
            _print_command(label, argv)
        _print_command(copy_command[0], copy_command[1])
        print("\nDry run complete. Re-run with --execute to start builds.")
        return

    for label, argv in commands:
        _run(argv, label)
    _run_parallel(build_commands)
    _run_parallel(validate_commands)
    _run(copy_command[1], copy_command[0])
    print("\nDistributed local colocation build complete.")


if __name__ == "__main__":
    main()
