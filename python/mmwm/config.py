"""Typed, JSON-serialisable platform configuration.

Unknown keys are rejected so a typo in a config file fails loudly instead of
silently falling back to a default on a 10,000-job submission.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _from_dict(cls, data: dict[str, Any], where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected an object, got {type(data).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; valid keys: {sorted(known)}")
    return cls(**data)


@dataclass
class ModelConfig:
    d_model: int = 128
    vocab: int = 512
    text_len: int = 16
    image_patches: int = 16
    patch_dim: int = 48
    audio_frames: int = 20
    audio_dim: int = 40
    action_dim: int = 8
    seed: int = 1

    def validate(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, int) or v < 0 or (f.name != "seed" and v == 0):
                raise ValueError(f"model.{f.name} must be a positive integer, got {v!r}")


@dataclass
class TrainConfig:
    rounds: int = 1            # synchronisation rounds (parameter averaging)
    steps: int = 100           # SGD steps per shard per round
    batch_size: int = 32
    lr: float = 0.05
    threads: int = 1           # threads per worker process

    def validate(self) -> None:
        if self.rounds < 1 or self.steps < 1 or self.batch_size < 1 or self.threads < 0:
            raise ValueError("train.rounds/steps/batch_size must be >= 1 and threads >= 0")
        if not self.lr > 0:
            raise ValueError("train.lr must be > 0")


@dataclass
class ClusterConfig:
    backend: str = "local"           # "local" | "lsf"
    num_shards: int = 8              # parallel training workers (10_000 for the full run)
    max_array_size: int = 1000       # LSF MAX_JOB_ARRAY_SIZE; larger runs are split into several arrays
    max_concurrent: int = 0          # LSF array throttle "%N"; 0 = unlimited
    reduce_fan_in: int = 100         # shards merged per level-0 reducer
    min_shard_fraction: float = 0.98  # fraction of shards that must survive to accept a merge
    queue: str | None = None
    project: str | None = None
    cores: int = 1
    mem_mb: int = 2048
    walltime: str | None = None      # "HH:MM"
    bsub: str = "bsub"
    python: str | None = None        # interpreter on the compute nodes (default: this one)
    local_parallelism: int = 4       # local backend only
    extra_bsub: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.backend not in ("local", "lsf"):
            raise ValueError(f"cluster.backend must be 'local' or 'lsf', got {self.backend!r}")
        for name in ("num_shards", "max_array_size", "reduce_fan_in", "cores", "mem_mb",
                     "local_parallelism"):
            if getattr(self, name) < 1:
                raise ValueError(f"cluster.{name} must be >= 1")
        if self.max_concurrent < 0:
            raise ValueError("cluster.max_concurrent must be >= 0")
        if not 0.0 < self.min_shard_fraction <= 1.0:
            raise ValueError("cluster.min_shard_fraction must be in (0, 1]")


@dataclass
class EvalConfig:
    samples: int = 256
    horizon: int = 4


@dataclass
class PlatformConfig:
    workdir: str = "runs/default"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> "PlatformConfig":
        self.model.validate()
        self.train.validate()
        self.cluster.validate()
        if self.eval.samples < 1 or self.eval.horizon < 1:
            raise ValueError("eval.samples and eval.horizon must be >= 1")
        return self

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlatformConfig":
        data = dict(data)
        sub = {
            "model": ModelConfig,
            "train": TrainConfig,
            "cluster": ClusterConfig,
            "eval": EvalConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            if key in data:
                kwargs[key] = _from_dict(klass, data.pop(key), key)
        if "workdir" in data:
            kwargs["workdir"] = data.pop("workdir")
        if data:
            raise ValueError(f"config: unknown top-level key(s) {sorted(data)}")
        return cls(**kwargs).validate()

    @classmethod
    def load(cls, path: str | Path) -> "PlatformConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
