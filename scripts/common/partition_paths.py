"""
Centralized partition/colocation directory naming.

Strategy is embedded in the path so multiple insert-strategies can coexist
on disk without clobbering each other.

    partition/{N}_partitions_{strategy}/
    colocation/{N}_partitions_{strategy}/<variant>/
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

DEFAULT_INSERT_STRATEGY = "nearest_centroid"

PathLike = Union[str, Path]


def get_insert_strategy(config: dict) -> str:
    """Read insert_strategy from config['partition']. Default: nearest_centroid."""
    return config.get("partition", {}).get("insert_strategy", DEFAULT_INSERT_STRATEGY)


def partition_subdir_name(num_partitions: int, strategy: str) -> str:
    return f"{num_partitions}_partitions_{strategy}"


def partition_dir(
    experiment_base: PathLike,
    num_partitions: int,
    strategy: str,
) -> Path:
    """experiment_base = output.base_dir / experiment_name (already joined)."""
    return (
        Path(experiment_base)
        / "partition"
        / partition_subdir_name(num_partitions, strategy)
    )


def colocation_dir(
    experiment_base: PathLike,
    num_partitions: int,
    strategy: str,
    variant: str | None = None,
) -> Path:
    """experiment_base = output.base_dir / experiment_name (already joined)."""
    d = (
        Path(experiment_base)
        / "colocation"
        / partition_subdir_name(num_partitions, strategy)
    )
    if variant:
        d = d / variant
    return d


def colocation_dir_candidates(
    base: PathLike,
    experiment_name: str,
    num_partitions: int,
    strategy: str,
    variant: str | None = None,
) -> list[Path]:
    """Candidate colocation payload directories for NFS and local-SSD layouts.

    Historical pipeline artifacts use:
      <root>/<experiment>/colocation/<N>_partitions_<strategy>/<variant>

    Swift local-SSD coloc builds may use a flatter payload layout:
      <root>/<experiment>/<N>_partitions_<strategy>/<variant>

    ``base`` may be either the root that contains experiment directories or an
    already-selected experiment directory, so include both interpretations.
    """
    root = Path(base)
    subdir = partition_subdir_name(num_partitions, strategy)
    raw = [
        colocation_dir(root / experiment_name, num_partitions, strategy, variant),
        root / experiment_name / subdir / variant if variant else root / experiment_name / subdir,
        colocation_dir(root, num_partitions, strategy, variant),
        root / subdir / variant if variant else root / subdir,
    ]
    out: list[Path] = []
    seen = set()
    for path in raw:
        norm = str(path)
        if norm not in seen:
            seen.add(norm)
            out.append(path)
    return out


def pq_dir(
    experiment_base: PathLike,
    num_chunks: int,
    variant: str | None = None,
) -> Path:
    """PQ artifacts (codebook, all_codes, meta) live outside colocation so they
    survive colocation rebuilds and can be built in parallel with colocation.

    Layout: {experiment_base}/pq/M{num_chunks}/[{variant}/]
    The variant subdir (e.g. 'iteration150') is optional. When absent the PQ
    encoding is treated as dataset-stable (single file per (dataset, num_chunks)).
    When present it keys to a specific checkpoint (rare; reserved for experiments
    where the encoded set must track insertions/deletions exactly).
    """
    d = Path(experiment_base) / "pq" / f"M{num_chunks}"
    if variant:
        d = d / variant
    return d
