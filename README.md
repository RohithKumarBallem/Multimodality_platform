# Multimodal World Models Research Platform

C++20 core + Python/LSF orchestration for training and evaluating **multimodal
latent world models** (text + image + audio → latent state → action-conditioned
dynamics) across many parallel workers, with a Bazel/CMake build, memory
checking, profiling, and a regression-release gate.

```
 text tokens ─┐
 image patches ┼─ encoders ─ gated fusion ─► z_t ─┐
 audio frames ─┘                                  ├─ dynamics ─► z_{t+1}   (trained)
                                       action a_t ┘
```

* Encoders are fixed random-feature projections; the trainable part is the
  residual dynamics `z' = z + tanh([z; a] W + b)`, trained with SGD on one-step
  latent prediction. Any subset of modalities may be present.
* Data is a deterministic synthetic world (`cpp/include/mmwm/synthetic_world.h`)
  so the whole system runs with no external datasets.

## Layout

| Path | What |
|---|---|
| `cpp/include/mmwm`, `cpp/src` | Core: arena allocator, thread pool, scoped profiler, kernels (`ExecContext`), world model, trainer, checkpoints, C ABI |
| `cpp/tests` | Dependency-free unit tests (runtime, kernels, model, training, checkpoints, profiler) |
| `cpp/bench` | Reference-vs-optimized latency benchmark (JSON output, `--profile`) |
| `python/mmwm` | ctypes bindings, typed config, workflow planner, LSF + local backends, CLI |
| `configs/` | `smoke.json` (local), `lsf_10k.json` (10,000 shards on LSF) |
| `scripts/` | `memcheck.sh`, `profile.sh`, `compare_bench.py`, `regression_release.sh` |
| `CMakeLists.txt`, `BUILD.bazel`, `MODULE.bazel`, `.bazelrc` | Both build systems |

## Technologies

| Technology | How it is used here | Status on the dev machine |
|---|---|---|
| **C++20** | Core library built with `-std=c++20`; uses `<span>` (`Matrix::as_span`), `<ranges>` (`std::ranges::sort` with a projection), `<numbers>` (`pi_v`), `<bit>` (`std::has_single_bit` static_assert), plus threads/atomics/RAII | Compiles clean under `-Wall -Wextra -Wpedantic -Wshadow -Werror` (CMake and Bazel) |
| **Python 3.10+** | Orchestration (`python/mmwm`): typed config, workflow planner, LSF/local backends, CLI, `ctypes` binding to the C ABI. Standard library only | 18 unit/e2e tests pass (3.12) |
| **LSF** | `bsub` job arrays (`name[1-N]%throttle`), `ended()`/`done()` dependencies, `LSB_JOBINDEX` shard mapping, `rusage[mem]`, queue/project/walltime | Command generation unit-tested and exercised against a fake `bsub`; **not run on a real LSF cluster** |
| **CMake** (≥ 3.20, Ninja) | Primary build: static core, shared C-ABI library, ctest, `MMWM_SANITIZE`, `MMWM_WERROR` | Verified |
| **Bazel** (bzlmod, 8.x) | `cc_library` / `cc_binary` / `cc_test`, `--config=opt|ci|asan|ubsan|tsan`; `//:libmmwm.so` for Python | `bazel test --config=opt --config=ci //...` passes; also stage 3 of the release gate |
| **Git** | Source control; the release script stamps the short SHA into the artifact name and, with `--tag`, cuts an annotated `v<version>` tag from a clean tree after all gates pass | Verified |
| **Memory checking** | ASan + LSan, UBSan, TSan, valgrind memcheck via `scripts/memcheck.sh`, also `--config=asan` in Bazel | **UBSan verified locally.** ASan/TSan runtimes crash on this machine's toolchain and valgrind is not installed, so those are skipped locally and run (`STRICT=1`) in CI |
| **Performance profiling** | Built-in scoped-timer profiler (`MMWM_PROFILE`, `mmwm_bench_latency --profile`); `scripts/profile.sh` adds `perf record` or `valgrind --tool=callgrind` where installed | Built-in profiler verified; `perf`/callgrind not available locally |
| **GitHub Actions** | `.github/workflows/ci.yml`: Linux (CMake, sanitizers, valgrind, latency gate), Bazel (opt + ASan), macOS | Written, **not yet run** |

Deliberately *not* used: GPUs/CUDA, MPI, PyTorch/other ML frameworks, pybind11,
GoogleTest. The model is a small random-feature + residual-dynamics model, and
the tests use a dependency-free harness to keep the Bazel/CMake builds hermetic.

## Build and test

```bash
# CMake
cmake -S . -B build -G Ninja -DMMWM_WERROR=ON
cmake --build build
ctest --test-dir build --output-on-failure
PYTHONPATH=python python3 -m unittest discover -s python/tests   # uses build/libmmwm.*

# Bazel (8.x)
bazel test --config=opt //...
bazel build --config=opt //:libmmwm.so    # then MMWM_LIB=bazel-bin/libmmwm.so
```

## Run a distributed training workflow

```bash
export PYTHONPATH=python
python -m mmwm.cli run --config configs/smoke.json            # local process pool
python -m mmwm.cli run --config configs/lsf_10k.json --dry-run  # print the bsub plan
python -m mmwm.cli run --config configs/lsf_10k.json            # submit to LSF
```

Each synchronisation round is a DAG (identical commands on both backends):

```
train arrays (shards, split to fit MAX_JOB_ARRAY_SIZE)
  └─► level-0 reduce array (reduce_fan_in shards per reducer)
        └─► final merge → merged.ckpt ─► next round's workers start from it
                                      └─► eval
```

* **10,000 shards** are submitted as 10 job arrays of 1,000 (`max_array_size`),
  throttled by `max_concurrent`, merged by a 100-way reduction tree.
* Reducers run on `ended()` dependencies and tolerate lost shards down to
  `min_shard_fraction`; later rounds require `done()` on the previous merge.
* Workers are idempotent (completed shards are skipped), so failed array
  elements can be resubmitted by re-running the same command.
* Parameter averaging is sample-weighted and hierarchical-equals-flat (tested).

## Memory checking, profiling, release gate

```bash
scripts/memcheck.sh all      # ASan(+LSan), UBSan, TSan, valgrind; STRICT=1 makes skips fatal
scripts/profile.sh           # built-in scope timers + perf/callgrind when available
scripts/regression_release.sh            # 7 gated stages below
scripts/regression_release.sh --tag      # ...then cut annotated tag v<version> (clean tree only)
```

Stages: CMake `-Werror` build → ctest → Bazel build+test → memcheck → Python
tests → latency gate → package (`dist/mmwm-<version>-<sha>-<os>-<arch>.tar.gz`).
`STRICT=1` makes an unavailable tool (Bazel, a sanitizer, valgrind) a failure
instead of a skip.

The bench gate fails the release if the optimized path is not ≥ 40 % lower
latency than the reference path, or its optimized/reference ratio regresses > 15 %
against `bench/baseline.json` (same thread count only).

## Measured results (this repo, one machine)

Intel i7-9750H (12 hardware threads), macOS, Apple clang 17, Release.
Encode + 8-step rollout, batch 64, d_model 256, text 32 / 64 patches×192 / 50 frames×80.

| Path | p50 latency |
|---|---|
| Reference (naive ijk matmul, per-call heap allocation, 1 thread) | ~280–335 ms |
| Optimized, 1 thread (register-tiled matmul + arena) | ~84 ms (**−73 %**, 3.8×) |
| Optimized, 12 threads | ~11.4 ms (**−96 %**, ~25×) |

Outputs of the two paths agree (max abs diff in the bench is checked).

**How to read these numbers.** The "reference" is the deliberately unoptimised
implementation in this repository, not an external system or a production model;
the percentages show what tiling, arena allocation and threading buy on this
workload and give the regression gate something to compare against. They are not
a claim about any other model. Reproduce with `build/mmwm_bench_latency` (add
`--threads 1` to isolate the single-thread gain).

Training sanity (smoke config, 8 shards × 2 rounds): the merged model's one-step
latent error is ~42 % below a copy-last-latent baseline (~50 % over a 4-step rollout).

## What has and hasn't been verified

Verified locally: CMake `-Werror` build, C++ tests, UBSan run, Bazel 8.1.1
build/test, Python tests (including a fake-`bsub` submission test), the full
local multi-round workflow, and all 7 release-gate stages.

Not verified here:

* **A real LSF cluster.** Submission is exercised against a fake `bsub` and the
  command shape is unit-tested, but nothing ran on 10,000 nodes.
* **ASan / LSan / TSan / valgrind**, because the sanitizer runtimes on the
  development machine crash on a trivial program. `memcheck.sh` detects that and
  skips; CI runs them with `STRICT=1` on Linux.
* **`perf` / callgrind** profiling paths in `scripts/profile.sh`.
* The **GitHub Actions** workflow itself.
