"""End-to-end tests against the real native library (skipped if not built)."""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mmwm import _native, workflow
from mmwm.config import PlatformConfig

try:
    _native.find_library()
    HAVE_LIB = True
except _native.NativeError:
    HAVE_LIB = False

MODEL = {"d_model": 32, "vocab": 64, "text_len": 6, "image_patches": 4, "patch_dim": 12,
         "audio_frames": 5, "audio_dim": 10, "action_dim": 4, "seed": 3}


def cfg_for(workdir, **cluster):
    return PlatformConfig.from_dict({
        "workdir": str(workdir), "model": MODEL,
        "train": {"rounds": 2, "steps": 30, "batch_size": 16, "lr": 0.05},
        "cluster": {"backend": "local", "num_shards": 6, "reduce_fan_in": 4,
                    "local_parallelism": 3, **cluster},
        "eval": {"samples": 64, "horizon": 3},
    })


@unittest.skipUnless(HAVE_LIB, "libmmwm not built")
class WorkflowE2E(unittest.TestCase):
    def test_full_local_run_improves_over_untrained(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(d)
            layout = workflow.Layout(d)
            cfg.save(layout.config)
            jobs = workflow.plan(cfg, "t", layout)
            failures = workflow.run_local(jobs, cfg, layout)
            self.assertTrue(all(v == 0 for v in failures.values()), failures)

            report = json.loads(layout.eval_json().read_text())
            self.assertLess(report["trained"]["model_mse"], report["untrained"]["model_mse"])
            self.assertLess(report["trained"]["model_mse"], report["trained"]["copy_mse"])
            merged = json.loads(layout.merged_json(1).read_text())
            self.assertEqual(merged["shards"], 6)
            self.assertEqual(merged["samples"], 6 * 30 * 16)

    def test_worker_is_idempotent_and_handles_padding(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(d)
            layout = workflow.Layout(d)
            m1 = workflow.run_worker(cfg, layout, 0, 2)
            mtime = layout.shard_ckpt(0, 2).stat().st_mtime_ns
            m2 = workflow.run_worker(cfg, layout, 0, 2)  # skip-existing
            self.assertEqual(m1, m2)
            self.assertEqual(layout.shard_ckpt(0, 2).stat().st_mtime_ns, mtime)
            self.assertIsNone(workflow.run_worker(cfg, layout, 0, 6))  # beyond num_shards

    def test_tolerates_lost_shards_above_threshold_but_not_below(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(d, num_shards=8, reduce_fan_in=8, min_shard_fraction=0.75)
            layout = workflow.Layout(d)
            for i in range(8):
                workflow.run_worker(cfg, layout, 0, i)
            for i in (0, 5):  # lose two of eight => 6/8 = 0.75, still acceptable
                layout.shard_ckpt(0, i).unlink()
            s = workflow.run_reduce_level0(cfg, layout, 0, 0)
            self.assertEqual((s["shards"], s["expected"]), (6, 8))
            self.assertEqual(workflow.run_reduce_final(cfg, layout, 0)["shards"], 6)

            layout.shard_ckpt(0, 7).unlink()  # 5/8 < 0.75
            # Level 0 merges survivors; the global budget is enforced at the end.
            self.assertEqual(workflow.run_reduce_level0(cfg, layout, 0, 0)["shards"], 5)
            with self.assertRaisesRegex(RuntimeError, "min_shard_fraction"):
                workflow.run_reduce_final(cfg, layout, 0)

    def test_failures_clustered_in_one_group_do_not_drop_the_group(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(d, num_shards=8, reduce_fan_in=4, min_shard_fraction=0.7)
            layout = workflow.Layout(d)
            for i in range(8):
                workflow.run_worker(cfg, layout, 0, i)
            for i in (0, 1):  # half of group 0 lost: 6/8 = 0.75 overall, within budget
                layout.shard_ckpt(0, i).unlink()
            for g in range(2):
                workflow.run_reduce_level0(cfg, layout, 0, g)
            self.assertEqual(workflow.run_reduce_final(cfg, layout, 0)["shards"], 6)

            for i in (2, 3):  # group 0 now has no shards at all
                layout.shard_ckpt(0, i).unlink()
            with self.assertRaisesRegex(RuntimeError, "none of its"):
                workflow.run_reduce_level0(cfg, layout, 0, 0)

    def test_hierarchical_reduce_matches_flat_reduce(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(d, num_shards=6, reduce_fan_in=2)
            layout = workflow.Layout(d)
            for i in range(6):
                workflow.run_worker(cfg, layout, 0, i)
            for g in range(workflow.num_groups(cfg)):
                workflow.run_reduce_level0(cfg, layout, 0, g)
            workflow.run_reduce_final(cfg, layout, 0)
            flat = Path(d) / "flat.ckpt"
            _native.reduce([layout.shard_ckpt(0, i) for i in range(6)], flat)

            def floats(p):
                import array
                a = array.array("f")
                raw = Path(p).read_bytes()[40:]  # header is 40 bytes
                a.frombytes(raw)
                return a

            tree, ref = floats(layout.merged_ckpt(0)), floats(flat)
            self.assertEqual(len(tree), len(ref))
            self.assertLess(max(abs(x - y) for x, y in zip(tree, ref)), 1e-5)

    def test_cli_worker_uses_lsb_jobindex(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(d)
            layout = workflow.Layout(d)
            cfg.save(layout.config)
            env = dict(os.environ, LSB_JOBINDEX="3",
                       PYTHONPATH=str(Path(__file__).resolve().parents[1]))
            r = subprocess.run(
                [sys.executable, "-m", "mmwm.cli", "worker", "--config", str(layout.config),
                 "--round", "0", "--shard-offset", "2"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(layout.shard_ckpt(0, 4).exists())  # offset 2 + index 3 - 1

    def test_lsf_submission_invokes_bsub_for_every_job(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "bsub.log"
            fake = Path(d) / "bsub"
            fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\necho "Job <1> is submitted"\n')
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            cfg = cfg_for(Path(d) / "run", backend="lsf", bsub=str(fake))
            layout = workflow.Layout(cfg.workdir)
            jobs = workflow.plan(cfg, "t", layout)
            workflow.submit_lsf(jobs, cfg, layout)
            lines = log.read_text().strip().splitlines()
            self.assertEqual(len(lines), len(jobs))
            self.assertTrue(layout.logs.is_dir())

    def test_failed_bsub_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "bsub"
            fake.write_text("#!/bin/sh\necho 'queue closed' >&2\nexit 1\n")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            cfg = cfg_for(Path(d) / "run", backend="lsf", bsub=str(fake))
            layout = workflow.Layout(cfg.workdir)
            with self.assertRaisesRegex(RuntimeError, "queue closed"):
                workflow.submit_lsf(workflow.plan(cfg, "t", layout), cfg, layout)

    def test_native_errors_surface_as_exceptions(self):
        with self.assertRaises(_native.NativeError):
            _native.reduce(["/nonexistent.ckpt"], "/tmp/never.ckpt")
        with self.assertRaises(_native.NativeError):
            _native.evaluate(PlatformConfig().model, "/nonexistent.ckpt", samples=4, horizon=1)


if __name__ == "__main__":
    unittest.main()
