"""Distributed training workflow: job planning, worker/reducer logic, backends.

A run is `train.rounds` synchronisation rounds. Each round is a DAG:

    train arrays (shards, split to fit max_array_size)
        -> level-0 reduce array (fan-in shards each)
            -> final merge (weighted average of partials)  -> merged.ckpt
                 -> next round's train arrays start from merged.ckpt

and the run ends with an evaluation job. The same planned commands are used by
the LSF backend (bsub job arrays with `ended()` dependencies) and the local
backend (subprocess pool), so the local path exercises the production code.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import _native
from .config import PlatformConfig

SHARDS_PER_DIR = 1000  # keep directories small for parallel filesystems


# --------------------------------------------------------------------------- layout
class Layout:
    def __init__(self, workdir: str | os.PathLike):
        self.root = Path(workdir).resolve()

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def run_id_file(self) -> Path:
        return self.root / "run_id"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def round_dir(self, r: int) -> Path:
        return self.root / f"round_{r:03d}"

    def shard_ckpt(self, r: int, i: int) -> Path:
        return self.round_dir(r) / f"g{i // SHARDS_PER_DIR:04d}" / f"shard_{i:06d}.ckpt"

    def shard_json(self, r: int, i: int) -> Path:
        return self.shard_ckpt(r, i).with_suffix(".json")

    def partial_ckpt(self, r: int, g: int) -> Path:
        return self.round_dir(r) / "partials" / f"partial_{g:05d}.ckpt"

    def partial_json(self, r: int, g: int) -> Path:
        return self.partial_ckpt(r, g).with_suffix(".json")

    def merged_ckpt(self, r: int) -> Path:
        return self.round_dir(r) / "merged.ckpt"

    def merged_json(self, r: int) -> Path:
        return self.merged_ckpt(r).with_suffix(".json")

    def eval_json(self) -> Path:
        return self.root / "eval.json"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------- planning
@dataclass(frozen=True)
class Job:
    name: str
    argv: tuple[str, ...]
    array_size: int | None = None  # None: single job; else LSB_JOBINDEX = 1..array_size
    depends_on: tuple[str, ...] = ()  # run once these have *ended* (success or not)
    after_ok: tuple[str, ...] = ()    # run only if these finished successfully
    description: str = ""


def effective_fan_in(cfg: PlatformConfig) -> int:
    """Fan-in, raised if needed so the level-0 reducers fit in one array."""
    c = cfg.cluster
    return max(c.reduce_fan_in, math.ceil(c.num_shards / c.max_array_size))


def num_groups(cfg: PlatformConfig) -> int:
    return math.ceil(cfg.cluster.num_shards / effective_fan_in(cfg))


def group_shards(cfg: PlatformConfig, g: int) -> range:
    fan = effective_fan_in(cfg)
    return range(g * fan, min((g + 1) * fan, cfg.cluster.num_shards))


def _runner(cfg: PlatformConfig) -> list[str]:
    """`env PYTHONPATH=... MMWM_LIB=... python -m mmwm.cli` prefix for node commands."""
    pkg_parent = str(Path(__file__).resolve().parents[1])
    prefix = ["env", f"PYTHONPATH={pkg_parent}"]
    try:
        prefix.append(f"MMWM_LIB={_native.find_library()}")
    except _native.NativeError:
        pass  # plan can be inspected before the library is built
    py = cfg.cluster.python or sys.executable
    return prefix + [py, "-m", "mmwm.cli"]


def plan(cfg: PlatformConfig, run_id: str, layout: Layout) -> list[Job]:
    cfg.validate()
    c = cfg.cluster
    run = _runner(cfg)
    cfgp = str(layout.config)
    jobs: list[Job] = []
    prev_merge: str | None = None

    for r in range(cfg.train.rounds):
        train_names: list[str] = []
        arrays = math.ceil(c.num_shards / c.max_array_size)
        for a in range(arrays):
            offset = a * c.max_array_size
            size = min(c.max_array_size, c.num_shards - offset)
            name = f"mmwm-{run_id}-r{r}-train{a}"
            jobs.append(Job(
                name=name,
                argv=tuple(run + ["worker", "--config", cfgp, "--round", str(r),
                                  "--shard-offset", str(offset)]),
                array_size=size,
                after_ok=(prev_merge,) if prev_merge else (),
                description=f"round {r} train shards {offset}..{offset + size - 1}",
            ))
            train_names.append(name)

        red = f"mmwm-{run_id}-r{r}-reduce"
        jobs.append(Job(
            name=red,
            argv=tuple(run + ["reduce", "--config", cfgp, "--round", str(r), "--level", "0"]),
            array_size=num_groups(cfg),
            depends_on=tuple(train_names),
            description=f"round {r} level-0 reduce ({effective_fan_in(cfg)} shards each)",
        ))
        merge = f"mmwm-{run_id}-r{r}-merge"
        jobs.append(Job(
            name=merge,
            argv=tuple(run + ["reduce", "--config", cfgp, "--round", str(r), "--level", "1"]),
            depends_on=(red,),
            description=f"round {r} final merge",
        ))
        prev_merge = merge

    jobs.append(Job(
        name=f"mmwm-{run_id}-eval",
        argv=tuple(run + ["eval", "--config", cfgp, "--round", str(cfg.train.rounds - 1)]),
        after_ok=(prev_merge,) if prev_merge else (),
        description="evaluate merged checkpoint",
    ))
    return jobs


# ------------------------------------------------------------------ node-side logic
def run_worker(cfg: PlatformConfig, layout: Layout, rnd: int, shard: int,
               skip_existing: bool = True) -> dict | None:
    """Train one shard. Returns its metrics, or None if the shard is out of range
    (array padding). Idempotent so failed shards can simply be resubmitted."""
    if shard >= cfg.cluster.num_shards:
        return None
    out = layout.shard_ckpt(rnd, shard)
    meta = layout.shard_json(rnd, shard)
    if skip_existing and out.exists() and meta.exists():
        return json.loads(meta.read_text())

    init = None
    if rnd > 0:
        init = layout.merged_ckpt(rnd - 1)
        if not init.exists():
            raise FileNotFoundError(f"round {rnd - 1} merged checkpoint missing: {init}")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = cfg.train
    res = _native.train_shard(
        cfg.model, steps=t.steps, batch_size=t.batch_size, lr=t.lr, round=rnd,
        shard_index=shard, out_ckpt=out, init_ckpt=init, threads=t.threads)
    metrics = {"shard": shard, "round": rnd, "samples": res.samples,
               "first_loss": res.first_loss, "last_loss": res.last_loss,
               "seconds": res.seconds}
    _write_json(meta, metrics)  # written after the ckpt: its presence marks completion
    return metrics


def _min_required(expected: int, fraction: float) -> int:
    return max(1, math.ceil(expected * fraction))


def run_reduce_level0(cfg: PlatformConfig, layout: Layout, rnd: int, group: int) -> dict:
    shards = group_shards(cfg, group)
    if len(shards) == 0:
        raise ValueError(f"reduce group {group} is out of range (have {num_groups(cfg)})")
    present = [i for i in shards
               if layout.shard_ckpt(rnd, i).exists() and layout.shard_json(rnd, i).exists()]
    need = _min_required(len(shards), cfg.cluster.min_shard_fraction)
    if len(present) < need:
        raise RuntimeError(
            f"group {group}: only {len(present)}/{len(shards)} shards present "
            f"(< min_shard_fraction={cfg.cluster.min_shard_fraction})")
    out = layout.partial_ckpt(rnd, group)
    out.parent.mkdir(parents=True, exist_ok=True)
    _native.reduce([layout.shard_ckpt(rnd, i) for i in present], out)

    ms = [json.loads(layout.shard_json(rnd, i).read_text()) for i in present]
    summary = {
        "group": group, "shards": len(present), "expected": len(shards),
        "samples": sum(m["samples"] for m in ms),
        "first_loss": sum(m["first_loss"] for m in ms) / len(ms),
        "last_loss": sum(m["last_loss"] for m in ms) / len(ms),
        "seconds_max": max(m["seconds"] for m in ms),
    }
    _write_json(layout.partial_json(rnd, group), summary)
    return summary


def run_reduce_final(cfg: PlatformConfig, layout: Layout, rnd: int) -> dict:
    groups = range(num_groups(cfg))
    present = [g for g in groups
               if layout.partial_ckpt(rnd, g).exists() and layout.partial_json(rnd, g).exists()]
    summaries = [json.loads(layout.partial_json(rnd, g).read_text()) for g in present]
    included = sum(s["shards"] for s in summaries)
    need = _min_required(cfg.cluster.num_shards, cfg.cluster.min_shard_fraction)
    if included < need:
        raise RuntimeError(
            f"round {rnd}: only {included}/{cfg.cluster.num_shards} shards survived "
            f"(< min_shard_fraction={cfg.cluster.min_shard_fraction})")
    out = layout.merged_ckpt(rnd)
    _native.reduce([layout.partial_ckpt(rnd, g) for g in present], out)

    summary = {
        "round": rnd, "shards": included, "expected": cfg.cluster.num_shards,
        "samples": sum(s["samples"] for s in summaries),
        "first_loss": sum(s["first_loss"] * s["shards"] for s in summaries) / included,
        "last_loss": sum(s["last_loss"] * s["shards"] for s in summaries) / included,
    }
    _write_json(layout.merged_json(rnd), summary)
    return summary


def run_eval(cfg: PlatformConfig, layout: Layout, rnd: int) -> dict:
    ev = cfg.eval
    kw = dict(samples=ev.samples, horizon=ev.horizon, threads=max(1, cfg.train.threads))
    base = _native.evaluate(cfg.model, None, **kw)
    final = _native.evaluate(cfg.model, layout.merged_ckpt(rnd), **kw)
    report = {
        "round": rnd,
        "untrained": base.__dict__,
        "trained": final.__dict__,
        "one_step_improvement_vs_copy_pct": 100.0 * (1.0 - final.model_mse / final.copy_mse),
        "rollout_improvement_vs_copy_pct":
            100.0 * (1.0 - final.rollout_mse / final.rollout_copy_mse),
    }
    _write_json(layout.eval_json(), report)
    return report


# --------------------------------------------------------------------- backends
def bsub_command(job: Job, cfg: PlatformConfig, layout: Layout) -> list[str]:
    """argv for submitting `job` with bsub (pure function; unit-tested)."""
    c = cfg.cluster
    cmd = [c.bsub, "-J"]
    if job.array_size is not None:
        spec = f"{job.name}[1-{job.array_size}]"
        if c.max_concurrent:
            spec += f"%{c.max_concurrent}"
        cmd.append(spec)
        stem = f"{layout.logs}/{job.name}.%J.%I"
    else:
        cmd.append(job.name)
        stem = f"{layout.logs}/{job.name}.%J"
    cmd += ["-n", str(c.cores), "-R", f"rusage[mem={c.mem_mb}]", "-o", f"{stem}.out",
            "-e", f"{stem}.err"]
    if c.queue:
        cmd += ["-q", c.queue]
    if c.project:
        cmd += ["-P", c.project]
    if c.walltime:
        cmd += ["-W", c.walltime]
    # ended(): reducers still run when some workers failed and enforce
    # min_shard_fraction themselves. done(): later rounds / eval only start from
    # a merge that succeeded (otherwise they stay PEND instead of failing en masse).
    deps = [f'ended("{d}")' for d in job.depends_on] + [f'done("{d}")' for d in job.after_ok]
    if deps:
        cmd += ["-w", " && ".join(deps)]
    cmd += list(c.extra_bsub)
    cmd.append(shlex.join(job.argv))
    return cmd


def submit_lsf(jobs: Sequence[Job], cfg: PlatformConfig, layout: Layout,
               dry_run: bool = False) -> list[list[str]]:
    cmds = [bsub_command(j, cfg, layout) for j in jobs]
    if dry_run:
        return cmds
    layout.logs.mkdir(parents=True, exist_ok=True)  # bsub -o fails if the dir is missing
    for job, cmd in zip(jobs, cmds):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"bsub failed for {job.name} (exit {proc.returncode}): {proc.stderr.strip()}")
        print(proc.stdout.strip() or f"submitted {job.name}")
    return cmds


def run_local(jobs: Sequence[Job], cfg: PlatformConfig, layout: Layout) -> dict[str, int]:
    """Executes the planned DAG in order with a subprocess pool.

    Mirrors LSF semantics: failed array elements do not stop dependants; the
    reducers decide whether enough shards survived. Jobs with unmet `after_ok`
    dependencies are skipped. Returns failed-element counts per job (-1: skipped).
    """
    log_dir = layout.logs
    log_dir.mkdir(parents=True, exist_ok=True)
    failures: dict[str, int] = {}

    def run_one(job: Job, index: int | None) -> bool:
        env = dict(os.environ)
        if index is not None:
            env["LSB_JOBINDEX"] = str(index)
        tag = job.name if index is None else f"{job.name}.{index}"
        with open(log_dir / f"{tag}.log", "w") as log:
            return subprocess.run(list(job.argv), env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode == 0

    succeeded: set[str] = set()
    for job in jobs:
        t0 = time.time()
        blocked = [d for d in job.after_ok if d not in succeeded]
        if blocked:
            failures[job.name] = -1
            print(f"skipping {job.name}: dependency {blocked[0]} did not succeed")
            continue
        if job.array_size is None:
            ok = [run_one(job, None)]
        else:
            with ThreadPoolExecutor(max_workers=cfg.cluster.local_parallelism) as ex:
                ok = list(ex.map(lambda i: run_one(job, i), range(1, job.array_size + 1)))
        bad = ok.count(False)
        failures[job.name] = bad
        if bad == 0:
            succeeded.add(job.name)
        print(f"[{time.time() - t0:7.1f}s] {job.description or job.name}: "
              f"{len(ok) - bad}/{len(ok)} ok")
    return failures
