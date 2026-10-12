"""ctypes bindings for libmmwm (the C ABI in cpp/include/mmwm/c_api.h)."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import ModelConfig

_ERR_LEN = 1024


class NativeError(RuntimeError):
    pass


class _CModelConfig(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "d_model", "vocab", "text_len", "image_patches", "patch_dim",
        "audio_frames", "audio_dim", "action_dim", "seed")]


class _CTrainArgs(ctypes.Structure):
    _fields_ = [
        ("model", _CModelConfig),
        ("steps", ctypes.c_uint64),
        ("batch_size", ctypes.c_uint64),
        ("lr", ctypes.c_float),
        ("round", ctypes.c_uint64),
        ("shard_index", ctypes.c_uint64),
        ("threads", ctypes.c_uint32),
        ("reference_mode", ctypes.c_uint32),
    ]


class _CTrainResult(ctypes.Structure):
    _fields_ = [("first_loss", ctypes.c_double), ("last_loss", ctypes.c_double),
                ("seconds", ctypes.c_double), ("samples", ctypes.c_uint64)]


class _CEvalResult(ctypes.Structure):
    _fields_ = [("model_mse", ctypes.c_double), ("copy_mse", ctypes.c_double),
                ("rollout_mse", ctypes.c_double), ("rollout_copy_mse", ctypes.c_double),
                ("samples", ctypes.c_uint64), ("horizon", ctypes.c_uint64)]


@dataclass
class TrainResult:
    first_loss: float
    last_loss: float
    seconds: float
    samples: int


@dataclass
class EvalResult:
    model_mse: float
    copy_mse: float
    rollout_mse: float
    rollout_copy_mse: float
    samples: int
    horizon: int


def _lib_name() -> list[str]:
    return ["libmmwm.dylib", "libmmwm.so"] if sys.platform == "darwin" else ["libmmwm.so", "libmmwm.dylib"]


def find_library() -> Path:
    """Locate libmmwm: $MMWM_LIB, then common build directories in the repo."""
    env = os.environ.get("MMWM_LIB")
    if env:
        p = Path(env)
        if not p.is_file():
            raise NativeError(f"MMWM_LIB={env} does not exist")
        return p
    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    for d in [root / "build", root / "bazel-bin", *sorted(root.glob("build-*"))]:
        candidates += [d / n for n in _lib_name()]
    for c in candidates:
        if c.is_file():
            return c
    raise NativeError(
        "libmmwm not found. Build it with `cmake -S . -B build && cmake --build build` "
        "(or `bazel build //:libmmwm.so`) or set MMWM_LIB to its path."
    )


_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(str(find_library()))
    lib.mmwm_version.restype = ctypes.c_char_p
    lib.mmwm_train_shard.argtypes = [
        ctypes.POINTER(_CTrainArgs), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(_CTrainResult), ctypes.c_char_p, ctypes.c_size_t]
    lib.mmwm_train_shard.restype = ctypes.c_int
    lib.mmwm_reduce.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.mmwm_reduce.restype = ctypes.c_int
    lib.mmwm_evaluate.argtypes = [
        ctypes.POINTER(_CModelConfig), ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint32, ctypes.POINTER(_CEvalResult), ctypes.c_char_p, ctypes.c_size_t]
    lib.mmwm_evaluate.restype = ctypes.c_int
    _lib = lib
    return lib


def _c_model(m: ModelConfig) -> _CModelConfig:
    return _CModelConfig(**{n: getattr(m, n) for n, _ in _CModelConfig._fields_})


def _enc(p: str | os.PathLike | None) -> bytes | None:
    return None if p is None else os.fsencode(os.fspath(p))


def version() -> str:
    return _load().mmwm_version().decode()


def train_shard(model: ModelConfig, *, steps: int, batch_size: int, lr: float,
                round: int, shard_index: int, out_ckpt: str | os.PathLike,
                init_ckpt: str | os.PathLike | None = None, threads: int = 1,
                reference: bool = False) -> TrainResult:
    lib = _load()
    args = _CTrainArgs(_c_model(model), steps, batch_size, lr, round, shard_index,
                       threads, 1 if reference else 0)
    res = _CTrainResult()
    err = ctypes.create_string_buffer(_ERR_LEN)
    rc = lib.mmwm_train_shard(ctypes.byref(args), _enc(init_ckpt), _enc(out_ckpt),
                              ctypes.byref(res), err, _ERR_LEN)
    if rc != 0:
        raise NativeError(f"train_shard failed: {err.value.decode(errors='replace')}")
    return TrainResult(res.first_loss, res.last_loss, res.seconds, res.samples)


def reduce(inputs: Sequence[str | os.PathLike], out_ckpt: str | os.PathLike) -> None:
    if not inputs:
        raise ValueError("reduce: no input checkpoints")
    lib = _load()
    arr = (ctypes.c_char_p * len(inputs))(*[os.fsencode(os.fspath(p)) for p in inputs])
    err = ctypes.create_string_buffer(_ERR_LEN)
    rc = lib.mmwm_reduce(arr, len(inputs), _enc(out_ckpt), err, _ERR_LEN)
    if rc != 0:
        raise NativeError(f"reduce failed: {err.value.decode(errors='replace')}")


def evaluate(model: ModelConfig, ckpt: str | os.PathLike | None, *, samples: int,
             horizon: int, threads: int = 1) -> EvalResult:
    lib = _load()
    cm = _c_model(model)
    res = _CEvalResult()
    err = ctypes.create_string_buffer(_ERR_LEN)
    rc = lib.mmwm_evaluate(ctypes.byref(cm), _enc(ckpt), samples, horizon, threads,
                           ctypes.byref(res), err, _ERR_LEN)
    if rc != 0:
        raise NativeError(f"evaluate failed: {err.value.decode(errors='replace')}")
    return EvalResult(res.model_mse, res.copy_mse, res.rollout_mse, res.rollout_copy_mse,
                      res.samples, res.horizon)
