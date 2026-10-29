"""Canonical experiment artifact/result path helpers.

New configs can separate reusable artifacts from per-run results:

  artifacts:
    base_dir: /experiments/artifacts
    set: lits_updated
  results:
    base_dir: /experiments/results
  dataset:
    id: deep100M
  experiment:
    name: local_coloc_update_consol

Legacy configs that only define output.base_dir keep the historical layout:

  <output.base_dir>/<experiment_name>/<stage>/<timestamp>

When split layout is enabled and output.base_dir still points at runs/, the
sibling roots are used by default:

  <experiment_root>/artifacts/<dataset>/
  <experiment_root>/artifacts/<dataset>/colocation/<set>/
  <experiment_root>/results/<dataset>/<experiment>_<timestamp>
"""

from __future__ import annotations

import re
from pathlib import Path


def _slug(value: object, default: str = "unnamed") -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or default


def legacy_experiment_name(config: dict) -> str:
    return _slug(config.get("experiment_name", "unnamed"))


def dataset_id(config: dict) -> str:
    dataset_cfg = config.get("dataset", {}) or {}
    explicit = (
        dataset_cfg.get("id")
        or config.get("dataset_id")
        or (config.get("output", {}) or {}).get("dataset_id")
    )
    if explicit:
        return _slug(explicit)

    name = legacy_experiment_name(config)
    if name.startswith("vldb_"):
        name = name[len("vldb_"):]
    return _slug(name)


def experiment_label(config: dict) -> str:
    experiment_cfg = config.get("experiment", {}) or {}
    return _slug(
        experiment_cfg.get("name")
        or config.get("run_name")
        or config.get("experiment_label")
        or legacy_experiment_name(config)
    )


def artifact_set(config: dict) -> str:
    artifacts_cfg = config.get("artifacts", {}) or {}
    return _slug(
        artifacts_cfg.get("set")
        or config.get("artifact_set")
        or "default"
    )


def output_base_dir(config: dict) -> Path:
    return Path((config.get("output", {}) or {}).get("base_dir", "."))


def experiment_root(config: dict) -> Path:
    """Root that contains runs/, artifacts/, and results/."""
    base = output_base_dir(config)
    if base.name == "runs":
        return base.parent
    return base


def has_split_results(config: dict) -> bool:
    output_cfg = config.get("output", {}) or {}
    return bool(
        config.get("results")
        or output_cfg.get("run_base_dir")
        or output_cfg.get("results_base_dir")
        or (config.get("experiment", {}) or {}).get("name")
    )


def has_split_artifacts(config: dict) -> bool:
    output_cfg = config.get("output", {}) or {}
    return bool(
        config.get("artifacts")
        or output_cfg.get("artifact_base_dir")
        or output_cfg.get("artifacts_base_dir")
    )


def results_root(config: dict) -> Path:
    results_cfg = config.get("results", {}) or {}
    output_cfg = config.get("output", {}) or {}
    explicit = (
        results_cfg.get("base_dir")
        or output_cfg.get("run_base_dir")
        or output_cfg.get("results_base_dir")
    )
    if explicit:
        return Path(explicit)
    if has_split_results(config):
        return experiment_root(config) / "results"
    return output_base_dir(config)


def artifacts_root(config: dict) -> Path:
    artifacts_cfg = config.get("artifacts", {}) or {}
    output_cfg = config.get("output", {}) or {}
    explicit = (
        artifacts_cfg.get("base_dir")
        or output_cfg.get("artifact_base_dir")
        or output_cfg.get("artifacts_base_dir")
    )
    if explicit:
        return Path(explicit)
    if has_split_artifacts(config):
        return experiment_root(config) / "artifacts"
    return output_base_dir(config)


def legacy_experiment_dir(config: dict) -> Path:
    return output_base_dir(config) / legacy_experiment_name(config)


def artifact_experiment_dir(config: dict) -> Path:
    if has_split_artifacts(config):
        return artifacts_root(config) / dataset_id(config)
    return legacy_experiment_dir(config)


def artifact_colocation_dir(config: dict) -> Path:
    if has_split_artifacts(config):
        return artifact_experiment_dir(config) / "colocation" / artifact_set(config)
    return artifact_experiment_dir(config)


def artifact_dir_candidates(config: dict) -> list[Path]:
    """Return the single canonical artifact root for this config.

    Split-layout configs are intentionally strict: once artifacts.base_dir/set
    is configured, runtime must read from that path only. Legacy fallback would
    make partial migrations ambiguous.
    """
    return [artifact_experiment_dir(config)]


def run_dir(config: dict, stage: str, timestamp: str) -> Path:
    if has_split_results(config):
        return results_root(config) / dataset_id(config) / experiment_label(config) / _slug(timestamp)
    return legacy_experiment_dir(config) / stage / _slug(timestamp)
