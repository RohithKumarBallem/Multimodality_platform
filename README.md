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
| `scripts/` | `memcheck.sh`, `profile.sh`, `compare_bench.py`, `regression_release.sh`, `scale_test.py` |
| `CMakeLists.txt`, `BUILD.bazel`, `MODULE.bazel`, `.bazelrc` | Both build systems |

## Technologies

| Technology | How it is used here | Status on the dev machine |
|---|---|---|
| **C++20** | Core library built with `-std=c++20`; uses `<span>` (`Matrix::as_span`), `<ranges>` (`std::ranges::sort` with a projection), `<numbers>` (`pi_v`), `<bit>` (`std::has_single_bit` static_assert), plus threads/atomics/RAII | Compiles clean under `-Wall -Wextra -Wpedantic -Wshadow -Werror` (CMake and Bazel) |
| **Python 3.10+** | Orchestration (`python/mmwm`): typed config, workflow planner, LSF/local backends, CLI, `ctypes` binding to the C ABI. Standard library only | 18 unit/e2e tests pass (3.12) |
| **LSF** | `bsub` job arrays (`name[1-N]%throttle`), `ended()`/`done()` dependencies, `LSB_JOBINDEX` shard mapping, `rusage[mem]`, queue/project/walltime | Command generation unit-tested and exercised against a fake `bsub`; **not run on a real LSF cluster** |
| **Vendor BLAS** | Optional `sgemm` stage: Accelerate on macOS, any BLAS with `cblas.h` elsewhere (`-DMMWM_USE_BLAS`); worker caps BLAS threads to `train.threads` | Verified on macOS (Accelerate); Linux path compiled out in CI |
| **CMake** (≥ 3.20, Ninja) | Primary build: static core, shared C-ABI library, ctest, `MMWM_SANITIZE`, `MMWM_WERROR` | Verified |
| **Bazel** (bzlmod, 8.x) | `cc_library` / `cc_binary` / `cc_test`, `--config=opt|ci|asan|ubsan|tsan`; `//:libmmwm.so` for Python | `bazel test --config=opt --config=ci //...` passes (macOS locally; Linux opt + ASan in CI); also stage 3 of the release gate |
| **Git** | Source control; the release script stamps the short SHA into the artifact name and, with `--tag`, cuts an annotated `v<version>` tag from a clean tree after all gates pass | Verified |
| **Memory checking** | ASan + LSan, UBSan, TSan, valgrind memcheck via `scripts/memcheck.sh`, also `--config=asan` in Bazel | **All four pass in CI on Linux** (`STRICT=1`, so a skip would fail the job). Locally on the macOS dev machine only UBSan runs: its ASan/TSan runtimes crash on a trivial program and valgrind isn't installed, so `memcheck.sh` skips those there |
| **Performance profiling** | Built-in scoped-timer profiler (`MMWM_PROFILE`, `mmwm_bench_latency --profile`); `scripts/profile.sh` adds `perf record` or `valgrind --tool=callgrind` where installed | Built-in profiler verified; `perf`/callgrind not available locally |
| **GitHub Actions** | `.github/workflows/ci.yml`: Linux (CMake, sanitizers, valgrind, latency gate), Bazel (opt + ASan), macOS | Passing: Linux (CMake `-Werror`, ctest, Python tests, sanitizers, valgrind, latency gate), Bazel (opt + ASan), macOS |

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

The bench gate fails the release if the portable optimised path is not ≥ 40 %
below the reference (a sanity floor), not ≥ 10 % below v1 (the profile-guided
gain), or its optimised/reference ratio regresses > 15 % against
`bench/baseline.json` (same thread count only).

## Measured results (one machine, reproducible)

Intel i7-9750H (6 cores / 12 threads, AVX2+FMA capable), macOS, Apple clang 17,
Release build. Workload: encode + 8-step rollout, batch 64, d_model 256, text
32 / 64 image patches × 192 / 50 audio frames × 80. `mmwm_bench_latency` runs all
stages interleaved and checks each against the reference numerically.

### Profile-guided optimisation ladder

The starting point ("v1") is the tiled + arena + thread-pool implementation. A
single-thread profile of v1 (`--profile`) put ~55 ms per call in `encode`, nearly
all of it the image/audio projections; a microbenchmark put libm `tanhf` at
~7.8 ms per million elements, i.e. ~14 ms (≈ 25 %) of that encode; and the default
build targets baseline SSE2 only. Each stage below was kept only because it
measured faster.

| Stage | 12 threads vs v1 | 1 thread vs v1 | Notes |
|---|---|---|---|
| v1: tiled matmul + arena + thread pool | — (≈ 10–15 ms) | — (≈ 43–56 ms) | |
| + fast rational `tanh` (`fast_tanhf`) | **−23…−27 %** | **−22…−28 %** | portable; accuracy tested < 2e-6 |
| + vendor sgemm (`use_blas`, Accelerate/cblas) | **−45…−59 %** | **−84…−87 %** | needs a BLAS build; manages its own threads |
| *tried, removed:* fused bias+tanh+pool epilogue | no gain (within noise) | no gain | hypothesis (memory traffic) was wrong; code deleted |

Build-flag note: with `-DMMWM_NATIVE=ON` (`-march=native`, not portable) my own
kernels get AVX2+FMA, and v1 → +fast-tanh is −43 % at 12 threads (≈ 8.0 → 4.6 ms)
and −46 % single-threaded; vendor BLAS is still ~3× faster on one thread but does
not scale across threads at these small matrix sizes, so at 12 threads the two are
within noise (4.6 vs 5.0 ms).

So a ≥ 40 % latency reduction relative to the pre-profiling implementation is
reproducible here, but it takes either vendor BLAS or a `-march=native` build; the
portable default build gets ~23–27 %. Ranges above are min…max over 4–5
interleaved runs; absolute times drift ±20 % with machine load, so treat single
percentages as ±5 points.

The reference path (naive ijk matmul, per-call heap allocation, 1 thread) is
~280–470 ms; it exists as a correctness oracle and an explicit "unoptimised"
point, **not** as something the optimisations should be credited against. It is
deliberately slow, so reductions against it (≈ 96–98 %) are not meaningful as a
performance claim.

### Orchestration at 10,000 shards (`scripts/scale_test.py`)

The real workflow (job planning, worker processes, shard layout, two-level
reduction tree) run at full scale on one machine, with fault injection:

| | |
|---|---|
| Shards trained | 10,000 / 10,000 (10 arrays × 1,000), 14 shards/s, 690 s |
| Injected loss | 150 random shard checkpoints (1.5 %); worst reducer group lost 5/100 |
| Reduction | 100 level-0 reducers → final merge; merged 9,850 = all survivors |
| Correctness | hierarchical merge equals flat merge over the survivors (max \|diff\| 7e-9) |
| Footprint | largest process RSS 19 MB; 187 MB of checkpoints (tiny model) |

This test found a real design bug: level-0 reducers originally enforced
`min_shard_fraction` per group, so a group with a few lost shards was dropped
wholesale. The budget is now global and enforced once at the final merge.

What this does **not** show: cluster scheduling, node failure modes, shared-FS
behaviour, or multi-node compute: each "worker" here is a local process on one
machine, and the model is deliberately tiny.

Training sanity (smoke config, 8 shards × 2 rounds): the merged model's one-step
latent error is ~42 % below a copy-last-latent baseline (~50 % over a 4-step rollout).

## What has and hasn't been verified

Verified locally (macOS): CMake `-Werror` build, C++ tests (incl. all
fast-tanh/BLAS flag combinations against the reference), UBSan run, Bazel 8.1.1
build/test, Python tests (incl. a fake-`bsub` submission test and clustered-failure
cases), the full local multi-round workflow, the **10,000-shard fault-injection
run**, and all 7 release-gate stages.

Verified in GitHub Actions (Linux + macOS): CMake `-Werror` build and tests,
ASan + LSan, UBSan, TSan and valgrind with `STRICT=1`, Bazel opt and ASan,
Python tests, and the latency gate. (The BLAS stage is compiled out on the Linux
runner, which has no `cblas.h`.)

Not verified:

* **A real LSF cluster.** Submission is exercised against a fake `bsub` and the
  command shape is unit-tested; the scale test uses local processes.
* **Real data.** The world is a deterministic synthetic generator and the
  encoders are fixed random projections; this validates the system, not model
  quality on real multimodal data.
* **`perf` / callgrind** profiling paths in `scripts/profile.sh`.

## Author

Rohith Kumar Ballem · <brohithkum@gmail.com> · [github.com/RohithKumarBallem](https://github.com/RohithKumarBallem)
