// ExecContext selects how a forward/backward pass is executed.
//
//   Reference : single-threaded, naive (ijk, strided) matmul, a fresh heap
//               allocation per intermediate. The correctness oracle and the
//               latency baseline.
//   Optimized : row-tiled cache-friendly matmul, thread pool, arena-backed
//               intermediates (no per-step malloc/free), plus the stages in
//               OptFlags. Each stage is kept only because a measurement showed a
//               gain (a bias+tanh+pool fusion was tried and removed: no win).
//
// Both modes compute the same function; tests assert agreement.
#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "mmwm/arena.h"
#include "mmwm/fastmath.h"
#include "mmwm/tensor.h"
#include "mmwm/thread_pool.h"

namespace mmwm {

enum class ExecMode { kReference, kOptimized };

// Optional optimisation stages of the optimised path. Defaults are the
// portable fast configuration; the benchmark toggles them one by one to report
// what each stage buys.
struct OptFlags {
  bool fast_tanh = true;  // vectorisable rational tanh instead of libm
  bool blas = false;      // vendor sgemm (Accelerate / cblas) when compiled in
};

class ExecContext {
 public:
  static ExecContext Reference();
  // `arena` must outlive the context; `pool` may be null (single-threaded).
  // Throws std::runtime_error if flags.blas is set but BLAS was not compiled in.
  static ExecContext Optimized(Arena& arena, ThreadPool* pool, OptFlags flags = {});
  static constexpr bool has_blas() {
#if defined(MMWM_HAVE_BLAS)
    return true;
#else
    return false;
#endif
  }

  ExecMode mode() const { return mode_; }
  ThreadPool* pool() const { return pool_; }
  const OptFlags& flags() const { return flags_; }
  float tanh(float v) const { return flags_.fast_tanh ? fast_tanhf(v) : std::tanh(v); }

  // Zero-initialised scratch matrix. Valid until reset().
  Matrix alloc(size_t rows, size_t cols);
  // Releases heap scratch (reference) or rewinds the arena (optimized).
  void reset();

  // C = A * B  (C must be pre-sized [A.rows, B.cols]; contents overwritten).
  void matmul(const Matrix& A, const Matrix& B, Matrix& C) const;
  // C = A^T * B (C pre-sized [A.cols, B.cols]; contents overwritten).
  void matmul_atb(const Matrix& A, const Matrix& B, Matrix& C) const;
  // out[s, :] = tanh(mean over the `group` rows of s of tanh(x*W + bias)),
  // for s in [0, out.rows). x is [out.rows * group, W.rows]. Used by the
  // image and audio encoders.
  void project_pool_tanh(const Matrix& x, const Matrix& W, const float* bias, size_t group,
                         Matrix& out);
  // Runs fn over [0, n) in chunks (inline in reference mode).
  void parallel_for(size_t n, size_t grain, const ThreadPool::RangeFn& fn) const;

 private:
  ExecContext() = default;

  ExecMode mode_ = ExecMode::kReference;
  OptFlags flags_{false, false};
  Arena* arena_ = nullptr;
  ThreadPool* pool_ = nullptr;
  std::vector<std::unique_ptr<Buffer>> heap_;
};

}  // namespace mmwm
