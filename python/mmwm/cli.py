"""Command line entry point:  python -m mmwm.cli <command> ..."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

from . import _native, workflow
from .config import PlatformConfig


def _load(path: str) -> PlatformConfig:
    return PlatformConfig.load(path)


def _index(args: argparse.Namespace) -> int:
    """1-based array index from --index or $LSB_JOBINDEX."""
    if args.index is not None:
        return args.index
    env = os.environ.get("LSB_JOBINDEX")
    if env is None:
        raise SystemExit("error: pass --index or run under an LSF job array (LSB_JOBINDEX)")
    return int(env)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    if args.backend:
        cfg.cluster.backend = args.backend
    if args.shards:
        cfg.cluster.num_shards = args.shards
    cfg.validate()

    layout = workflow.Layout(cfg.workdir)
    run_id = time.strftime("%Y%m%d-%H%M%S")  # unique: avoids stale ended() matches in LSF
    cfg.workdir = str(layout.root)
    jobs = workflow.plan(cfg, run_id, layout)

    if args.dry_run:
        if cfg.cluster.backend == "lsf":
            for cmd in workflow.submit_lsf(jobs, cfg, layout, dry_run=True):
                print(shlex.join(cmd))
        else:
            for j in jobs:
                arr = f"[1-{j.array_size}]" if j.array_size else ""
                print(f"{j.name}{arr}: {shlex.join(j.argv)}")
        return 0

    layout.root.mkdir(parents=True, exist_ok=True)
    cfg.save(layout.config)
    layout.run_id_file.write_text(run_id + "\n")

    if cfg.cluster.backend == "lsf":
        workflow.submit_lsf(jobs, cfg, layout)
        print(f"submitted {len(jobs)} job(s); results will appear under {layout.root}")
        return 0

    failures = workflow.run_local(jobs, cfg, layout)
    if any(v != 0 for v in failures.values()) and not layout.eval_json().exists():
        print("run incomplete; see logs in", layout.logs, file=sys.stderr)
        return 1
    print(json.dumps(json.loads(layout.eval_json().read_text()), indent=2))
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    if cfg.train.use_blas:
        # BLAS libraries start their own thread pools; cap them to this worker's
        # allotment so 10,000 workers don't each spawn one thread per node core.
        # Must be set before the native library is first loaded.
        limit = str(max(1, cfg.train.threads))
        for var in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
            os.environ.setdefault(var, limit)
    shard = args.shard_offset + _index(args) - 1
    m = workflow.run_worker(cfg, workflow.Layout(cfg.workdir), args.round, shard)
    print(json.dumps(m) if m else f"shard {shard} out of range; nothing to do")
    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    layout = workflow.Layout(cfg.workdir)
    if args.level == 0:
        s = workflow.run_reduce_level0(cfg, layout, args.round, _index(args) - 1)
    else:
        s = workflow.run_reduce_final(cfg, layout, args.round)
    print(json.dumps(s))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    print(json.dumps(workflow.run_eval(cfg, workflow.Layout(cfg.workdir), args.round), indent=2))
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    print(f"libmmwm: {_native.find_library()} (v{_native.version()})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mmwm", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="plan and execute (or submit) a distributed training run")
    r.add_argument("--config", required=True)
    r.add_argument("--backend", choices=["local", "lsf"])
    r.add_argument("--shards", type=int, help="override cluster.num_shards")
    r.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    r.set_defaults(fn=cmd_run)

    w = sub.add_parser("worker", help="train one shard (LSF array element)")
    w.add_argument("--config", required=True)
    w.add_argument("--round", type=int, required=True)
    w.add_argument("--shard-offset", type=int, default=0)
    w.add_argument("--index", type=int, help="1-based array index (default $LSB_JOBINDEX)")
    w.set_defaults(fn=cmd_worker)

    d = sub.add_parser("reduce", help="merge shard (level 0) or partial (level 1) checkpoints")
    d.add_argument("--config", required=True)
    d.add_argument("--round", type=int, required=True)
    d.add_argument("--level", type=int, choices=[0, 1], required=True)
    d.add_argument("--index", type=int)
    d.set_defaults(fn=cmd_reduce)

    e = sub.add_parser("eval", help="evaluate the merged checkpoint of a round")
    e.add_argument("--config", required=True)
    e.add_argument("--round", type=int, required=True)
    e.set_defaults(fn=cmd_eval)

    i = sub.add_parser("info", help="show the native library in use")
    i.set_defaults(fn=cmd_info)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (_native.NativeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
