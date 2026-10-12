import re
import shlex
import unittest

from mmwm.config import PlatformConfig
from mmwm.workflow import (Layout, bsub_command, effective_fan_in, group_shards,
                           num_groups, plan)


def make(**cluster):
    return PlatformConfig.from_dict({
        "workdir": "/tmp/mmwm-plan-test",
        "train": {"rounds": 2},
        "cluster": {"backend": "lsf", **cluster},
    })


class PlanTest(unittest.TestCase):
    def test_10k_shards_split_into_arrays_within_lsf_limit(self):
        cfg = make(num_shards=10000, max_array_size=1000, max_concurrent=500, queue="normal")
        jobs = plan(cfg, "rid", Layout(cfg.workdir))
        trains = [j for j in jobs if "-r0-train" in j.name]
        self.assertEqual(len(trains), 10)
        self.assertTrue(all(j.array_size == 1000 for j in trains))
        # Offsets cover 0..9999 exactly once.
        offsets = sorted(int(j.argv[j.argv.index("--shard-offset") + 1]) for j in trains)
        self.assertEqual(offsets, list(range(0, 10000, 1000)))
        covered = sum(j.array_size for j in trains)
        self.assertEqual(covered, 10000)

    def test_uneven_last_array(self):
        cfg = make(num_shards=2500, max_array_size=1000)
        sizes = [j.array_size for j in plan(cfg, "r", Layout(cfg.workdir)) if "-r0-train" in j.name]
        self.assertEqual(sizes, [1000, 1000, 500])

    def test_reduce_tree_partitions_all_shards(self):
        cfg = make(num_shards=10000, reduce_fan_in=100)
        self.assertEqual(num_groups(cfg), 100)
        seen = [i for g in range(num_groups(cfg)) for i in group_shards(cfg, g)]
        self.assertEqual(seen, list(range(10000)))

    def test_fan_in_raised_when_groups_exceed_array_limit(self):
        cfg = make(num_shards=100000, max_array_size=1000, reduce_fan_in=10)
        self.assertEqual(effective_fan_in(cfg), 100)
        self.assertLessEqual(num_groups(cfg), 1000)

    def test_dependency_chain(self):
        cfg = make(num_shards=2000, max_array_size=1000)
        jobs = {j.name: j for j in plan(cfg, "rid", Layout(cfg.workdir))}
        reduce0 = jobs["mmwm-rid-r0-reduce"]
        self.assertEqual(set(reduce0.depends_on), {"mmwm-rid-r0-train0", "mmwm-rid-r0-train1"})
        # Round 1 trains only after round 0's merge succeeded.
        self.assertEqual(jobs["mmwm-rid-r1-train0"].after_ok, ("mmwm-rid-r0-merge",))
        self.assertEqual(jobs["mmwm-rid-eval"].after_ok, ("mmwm-rid-r1-merge",))
        self.assertEqual(jobs["mmwm-rid-r0-merge"].depends_on, ("mmwm-rid-r0-reduce",))

    def test_bsub_command(self):
        cfg = make(num_shards=2000, max_array_size=1000, max_concurrent=250, queue="gpu",
                   project="wm", walltime="01:30", cores=4, mem_mb=8192,
                   extra_bsub=["-app", "py"])
        layout = Layout(cfg.workdir)
        jobs = {j.name: j for j in plan(cfg, "rid", layout)}
        cmd = bsub_command(jobs["mmwm-rid-r1-train1"], cfg, layout)
        self.assertEqual(cmd[0], "bsub")
        self.assertEqual(cmd[cmd.index("-J") + 1], "mmwm-rid-r1-train1[1-1000]%250")
        self.assertEqual(cmd[cmd.index("-n") + 1], "4")
        self.assertEqual(cmd[cmd.index("-R") + 1], "rusage[mem=8192]")
        self.assertEqual(cmd[cmd.index("-q") + 1], "gpu")
        self.assertEqual(cmd[cmd.index("-P") + 1], "wm")
        self.assertEqual(cmd[cmd.index("-W") + 1], "01:30")
        self.assertEqual(cmd[cmd.index("-w") + 1], 'done("mmwm-rid-r0-merge")')
        self.assertIn("%J.%I", cmd[cmd.index("-o") + 1])
        self.assertEqual(cmd[cmd.index("-app") + 1], "py")
        inner = shlex.split(cmd[-1])
        self.assertIn("mmwm.cli", inner)
        self.assertEqual(inner[inner.index("--shard-offset") + 1], "1000")

        red = bsub_command(jobs["mmwm-rid-r0-reduce"], cfg, layout)
        deps = red[red.index("-w") + 1]
        self.assertEqual(re.findall(r'ended\("([^"]+)"\)', deps),
                         ["mmwm-rid-r0-train0", "mmwm-rid-r0-train1"])

        single = bsub_command(jobs["mmwm-rid-r0-merge"], cfg, layout)
        self.assertEqual(single[single.index("-J") + 1], "mmwm-rid-r0-merge")  # no array spec
        self.assertNotIn("%I", single[single.index("-o") + 1])


if __name__ == "__main__":
    unittest.main()
