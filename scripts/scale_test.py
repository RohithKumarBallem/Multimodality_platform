#!/usr/bin/env python3
"""Run the real workflow at 10,000-shard scale on one machine, with fault injection.

This validates the *orchestration* at scale: job planning, shard file layout,
per-process worker launch, the two-level reduction tree, tolerance to lost
shards, and that the hierarchical merge equals a flat merge over the survivors.
It does not (cannot) validate cluster scheduling or multi-node compute.

  python scripts/scale_test.py --shards 10000 --fail-rate 0.015 --workdir /tmp/mmwm-scale
"""
import argparse
import array
import json
import random
import resource
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mmwm import _native, workflow  # noqa: E402
from mmwm.config import PlatformConfig  # noqa: E402

HEADER_BYTES = 40  # see cpp/include/mmwm/checkpoint.h


def floats(path: Path) -> array.array:
    a = array.array("f")
    a.frombytes(path.read_bytes()[HEADER_BYTES:])
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=10000)
    ap.add_argument("--fail-rate", type=float, default=0.015,
                    help="fraction of shard checkpoints deleted before reduction")
    ap.add_argument("--min-fraction", type=float, default=0.97)
    ap.add_argument("--parallelism", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    work = Path(a.workdir)
    if work.exists():
        shutil.rmtree(work)
    cfg = PlatformConfig.from_dict({
        "workdir": str(work),
        "model": {"d_model": a.d_model, "vocab": 128, "text_len": 4, "image_patches": 4,
                  "patch_dim": 8, "audio_frames": 4, "audio_dim": 8, "action_dim": 8, "seed": 5},
        "train": {"rounds": 1, "steps": 2, "batch_size": 4, "lr": 0.05},
        "cluster": {"backend": "local", "num_shards": a.shards, "max_array_size": 1000,
                    "reduce_fan_in": 100, "min_shard_fraction": a.min_fraction,
                    "local_parallelism": a.parallelism},
        "eval": {"samples": 32, "horizon": 2},
    })
    layout = workflow.Layout(work)
    cfg.save(layout.config)
    jobs = workflow.plan(cfg, "scale", layout)
    train = [j for j in jobs if "-train" in j.name]
    rest = [j for j in jobs if "-train" not in j.name]
    print(f"planned {len(jobs)} jobs: {len(train)} train arrays x "
          f"{train[0].array_size}, {workflow.num_groups(cfg)} reducers")

    t0 = time.time()
    fails = workflow.run_local(train, cfg, layout)
    t_train = time.time() - t0
    produced = sum(1 for i in range(a.shards) if layout.shard_ckpt(0, i).exists())
    print(f"training: {produced}/{a.shards} shard checkpoints in {t_train:.0f}s "
          f"({a.shards / t_train:.0f} shards/s); worker failures: {sum(fails.values())}")

    # Fault injection: lose a random subset of finished shards (node death).
    rng = random.Random(a.seed)
    lost = rng.sample(range(a.shards), int(a.shards * a.fail_rate))
    for i in lost:
        layout.shard_ckpt(0, i).unlink(missing_ok=True)
        layout.shard_json(0, i).unlink(missing_ok=True)
    survivors = [i for i in range(a.shards) if layout.shard_ckpt(0, i).exists()]
    per_group = {}
    for i in lost:
        per_group[i // workflow.effective_fan_in(cfg)] = per_group.get(
            i // workflow.effective_fan_in(cfg), 0) + 1
    print(f"fault injection: removed {len(lost)} shards ({a.fail_rate:.1%}); worst group lost "
          f"{max(per_group.values())}/{workflow.effective_fan_in(cfg)}")

    t0 = time.time()
    fails = workflow.run_local(rest, cfg, layout)
    t_reduce = time.time() - t0
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    check("reduce/merge/eval jobs all succeeded", all(v == 0 for v in fails.values()),
          str({k: v for k, v in fails.items() if v}))
    merged = json.loads(layout.merged_json(0).read_text()) if layout.merged_json(0).exists() else {}
    check("merged includes every surviving shard", merged.get("shards") == len(survivors),
          f"(merged {merged.get('shards')}, survivors {len(survivors)})")

    if layout.merged_ckpt(0).exists():
        flat = work / "flat.ckpt"
        t0 = time.time()
        # Flat merge in chunks small enough for the C ABI call; exactness is
        # restored by merging chunk results with their sample weights.
        parts = []
        for c in range(0, len(survivors), 500):
            p = work / f"flat_{c}.ckpt"
            _native.reduce([layout.shard_ckpt(0, i) for i in survivors[c:c + 500]], p)
            parts.append(p)
        _native.reduce(parts, flat)
        tree, ref = floats(layout.merged_ckpt(0)), floats(flat)
        err = max(abs(x - y) for x, y in zip(tree, ref))
        check("hierarchical merge == flat merge over survivors", err < 1e-5,
              f"(max |diff| {err:.2e}, {time.time() - t0:.0f}s)")

    rss_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (
        1024 * 1024 if sys.platform == "darwin" else 1024)
    files = sum(1 for _ in work.rglob("*.ckpt"))
    size_mb = sum(p.stat().st_size for p in work.rglob("*.ckpt")) / 1e6
    print(f"reduce+merge+eval: {t_reduce:.0f}s; largest child RSS {rss_mb:.0f} MB; "
          f"{files} checkpoints, {size_mb:.0f} MB on disk")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
