import json
import tempfile
import unittest
from pathlib import Path

from mmwm.config import PlatformConfig


class ConfigTest(unittest.TestCase):
    def test_roundtrip(self):
        cfg = PlatformConfig.from_dict({
            "workdir": "w", "model": {"d_model": 32}, "train": {"rounds": 3},
            "cluster": {"backend": "lsf", "num_shards": 10000, "queue": "q"},
        })
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            cfg.save(p)
            again = PlatformConfig.load(p)
        self.assertEqual(cfg, again)
        self.assertEqual(again.cluster.num_shards, 10000)

    def test_unknown_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown key"):
            PlatformConfig.from_dict({"cluster": {"num_shard": 5}})  # typo
        with self.assertRaisesRegex(ValueError, "top-level"):
            PlatformConfig.from_dict({"clusterr": {}})

    def test_validation(self):
        for bad in (
            {"cluster": {"backend": "slurm"}},
            {"cluster": {"num_shards": 0}},
            {"cluster": {"min_shard_fraction": 0}},
            {"train": {"lr": -1}},
            {"model": {"d_model": 0}},
            {"eval": {"horizon": 0}},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PlatformConfig.from_dict(bad)

    def test_shipped_configs_are_valid(self):
        root = Path(__file__).resolve().parents[2] / "configs"
        files = list(root.glob("*.json"))
        self.assertTrue(files)
        for f in files:
            with self.subTest(config=f.name):
                PlatformConfig.from_dict(json.loads(f.read_text()))


if __name__ == "__main__":
    unittest.main()
