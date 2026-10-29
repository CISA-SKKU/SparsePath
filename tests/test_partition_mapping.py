import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.partition_mapping import (  # noqa: E402
    parse_node_partition_map,
    resolve_node_partition_map,
)


class PartitionMappingTest(unittest.TestCase):
    def test_default_sorted_nodes_map_to_dense_partitions(self):
        self.assertEqual(
            resolve_node_partition_map([4, 1, 3, 2], 4),
            {1: 0, 2: 1, 3: 2, 4: 3},
        )

    def test_explicit_non_identity_mapping(self):
        self.assertEqual(
            resolve_node_partition_map([1, 2, 3, 4], 4, "1:3,2:2,3:1,4:0"),
            {1: 3, 2: 2, 3: 1, 4: 0},
        )

    def test_parse_dict_and_list_shapes(self):
        self.assertEqual(parse_node_partition_map({"1": 0, 2: 1}), {1: 0, 2: 1})
        self.assertEqual(
            parse_node_partition_map(
                [{"node_id": 1, "partition": 0}, {"node": 2, "partition_id": 1}]
            ),
            {1: 0, 2: 1},
        )

    def test_duplicate_partition_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "duplicate partitions"):
            resolve_node_partition_map([1, 2, 3, 4], 4, "1:0,2:1,3:1,4:3")

    def test_missing_partition_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "missing partitions"):
            resolve_node_partition_map([1, 2, 3], 4, "1:0,2:1,3:2")


if __name__ == "__main__":
    unittest.main()
