import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "numpy is required for query routing tests")
class TestNearestCentroidRouting(unittest.TestCase):
    def setUp(self):
        from pipeline.danns_types import NearestCentroidPolicy
        self.policy_cls = NearestCentroidPolicy

    def test_batch_matches_single_point_assignment(self):
        rng = np.random.default_rng(123)
        points = rng.normal(size=(257, 7)).astype(np.float32)
        centroids = rng.normal(size=(4, 7)).astype(np.float32)

        policy = self.policy_cls()
        expected = np.array(
            [policy.assign(point, centroids) for point in points],
            dtype=np.int32,
        )

        actual = policy.assign_batch(points, centroids)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.int32)

    def test_batch_routes_across_internal_chunk_boundary(self):
        points = np.zeros((1_000_001, 2), dtype=np.float32)
        points[:500_000, 0] = -1.0
        points[500_000:, 0] = 1.0
        centroids = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        actual = self.policy_cls().assign_batch(points, centroids)

        self.assertTrue(np.all(actual[:500_000] == 0))
        self.assertTrue(np.all(actual[500_000:] == 1))

    def test_empty_batch(self):
        points = np.empty((0, 3), dtype=np.float32)
        centroids = np.ones((2, 3), dtype=np.float32)

        actual = self.policy_cls().assign_batch(points, centroids)

        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
