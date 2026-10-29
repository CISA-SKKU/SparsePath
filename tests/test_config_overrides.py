import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import apply_config_overrides  # noqa: E402


class ConfigOverridesTest(unittest.TestCase):
    def test_applies_dotted_override_and_yaml_value(self):
        config = {"colocation": {"local_dir": "/old"}}

        apply_config_overrides(
            config,
            [
                "colocation.local_dir=/workspace/colocation_local",
                "search.num_queries=1000",
                "features.enabled=true",
            ],
        )

        self.assertEqual(
            config["colocation"]["local_dir"],
            "/workspace/colocation_local",
        )
        self.assertEqual(config["search"]["num_queries"], 1000)
        self.assertIs(config["features"]["enabled"], True)

    def test_invalid_override_requires_equals(self):
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            apply_config_overrides({}, ["colocation.local_dir"])


if __name__ == "__main__":
    unittest.main()
