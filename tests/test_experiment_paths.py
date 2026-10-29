import unittest
from pathlib import Path

from pipeline.config import resolve_config_paths
from pipeline.experiment_paths import (
    artifact_colocation_dir,
    artifact_dir_candidates,
    artifact_experiment_dir,
    artifacts_root,
    dataset_id,
    experiment_label,
    has_split_artifacts,
    has_split_results,
    results_root,
    run_dir,
)


class ExperimentPathsTest(unittest.TestCase):
    def test_split_layout_result_path_omits_stage(self):
        cfg = {
            "experiment_name": "vldb_deep100M",
            "dataset": {"id": "deep100M"},
            "experiment": {"name": "local_coloc_update_consol"},
            "results": {"base_dir": "/experiments/results"},
            "artifacts": {
                "base_dir": "/experiments/artifacts",
                "set": "lits_updated",
            },
        }
        self.assertEqual(dataset_id(cfg), "deep100M")
        self.assertEqual(experiment_label(cfg), "local_coloc_update_consol")
        self.assertEqual(
            run_dir(cfg, "distributed_engine", "20260606_130034"),
            Path("/experiments/results/deep100M/local_coloc_update_consol/20260606_130034"),
        )

    def test_split_artifacts_are_strict_single_candidate(self):
        cfg = {
            "experiment_name": "vldb_deep100M",
            "dataset": {"id": "deep100M"},
            "artifacts": {
                "base_dir": "/experiments/artifacts",
                "set": "lits_updated",
            },
            "output": {"base_dir": "/experiments/runs"},
        }
        expected = Path("/experiments/artifacts/deep100M")
        self.assertEqual(artifact_experiment_dir(cfg), expected)
        self.assertEqual(artifact_dir_candidates(cfg), [expected])
        self.assertEqual(
            artifact_colocation_dir(cfg),
            Path("/experiments/artifacts/deep100M/colocation/lits_updated"),
        )

    def test_split_defaults_are_siblings_of_runs(self):
        cfg = {
            "experiment_name": "vldb_deep100M",
            "dataset": {"id": "deep100M"},
            "experiment": {"name": "local_coloc_update_consol"},
            "artifacts": {"set": "lits_updated"},
            "output": {"base_dir": "/experiments/runs"},
        }
        self.assertEqual(
            run_dir(cfg, "distributed_engine", "20260606_130034"),
            Path("/experiments/results/deep100M/local_coloc_update_consol/20260606_130034"),
        )
        self.assertEqual(
            artifact_experiment_dir(cfg),
            Path("/experiments/artifacts/deep100M"),
        )
        self.assertEqual(
            artifact_colocation_dir(cfg),
            Path("/experiments/artifacts/deep100M/colocation/lits_updated"),
        )

    def test_representative_split_config_resolution(self):
        cfg = {
            "experiment_name": "vldb_deep100M",
            "output": {"base_dir": "/experiments/runs"},
            "dataset": {"id": "deep100M"},
            "experiment": {"name": "local_coloc_update_consol"},
            "artifacts": {"set": "lits_updated"},
        }
        resolve_config_paths(cfg, Path("/workspace"))

        self.assertTrue(has_split_artifacts(cfg))
        self.assertTrue(has_split_results(cfg))
        self.assertEqual(results_root(cfg), Path("/experiments/results"))
        self.assertEqual(artifacts_root(cfg), Path("/experiments/artifacts"))
        self.assertEqual(
            artifact_experiment_dir(cfg),
            Path("/experiments/artifacts/deep100M"),
        )
        self.assertEqual(
            artifact_colocation_dir(cfg),
            Path("/experiments/artifacts/deep100M/colocation/lits_updated"),
        )
        self.assertEqual(
            run_dir(cfg, "search_rdma", "20260607_010203"),
            Path("/experiments/results/deep100M/local_coloc_update_consol/20260607_010203"),
        )

    def test_legacy_layout_when_split_not_configured(self):
        cfg = {
            "experiment_name": "vldb_deep100M",
            "output": {"base_dir": "/experiments/runs"},
        }
        self.assertEqual(dataset_id(cfg), "deep100M")
        self.assertEqual(
            run_dir(cfg, "distributed_engine", "20260606_130034"),
            Path("/experiments/runs/vldb_deep100M/distributed_engine/20260606_130034"),
        )
        self.assertEqual(
            artifact_dir_candidates(cfg),
            [Path("/experiments/runs/vldb_deep100M")],
        )
        self.assertEqual(
            artifact_colocation_dir(cfg),
            Path("/experiments/runs/vldb_deep100M"),
        )


if __name__ == "__main__":
    unittest.main()
