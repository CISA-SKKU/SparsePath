import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.partition_paths import colocation_dir_candidates  # noqa: E402


class PartitionPathsTest(unittest.TestCase):
    def test_colocation_candidates_include_nfs_and_flat_local_root(self):
        candidates = colocation_dir_candidates(
            "/workspace/colocation_local",
            "vldb_deep100M",
            4,
            "lits_balanced",
            "iteration150",
        )
        self.assertIn(
            Path("/workspace/colocation_local/vldb_deep100M/colocation/4_partitions_lits_balanced/iteration150"),
            candidates,
        )
        self.assertIn(
            Path("/workspace/colocation_local/vldb_deep100M/4_partitions_lits_balanced/iteration150"),
            candidates,
        )

    def test_colocation_candidates_include_experiment_dir_base(self):
        candidates = colocation_dir_candidates(
            "/workspace/colocation_local/vldb_deep100M",
            "vldb_deep100M",
            4,
            "lits_balanced",
            "iteration150",
        )
        self.assertIn(
            Path("/workspace/colocation_local/vldb_deep100M/4_partitions_lits_balanced/iteration150"),
            candidates,
        )


if __name__ == "__main__":
    unittest.main()
