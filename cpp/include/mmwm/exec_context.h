// ExecContext selects how a forward/backward pass is executed.
//
//   Reference : single-threaded, naive (ijk, strided) matmul, a fresh heap
//               allocation per intermediate. The correctness oracle and the
//               latency baseline.
//   Optimized : row-tiled cache-friendly matmul, thread pool, arena-backed
//               intermediates (no per-step malloc/free).
//
// Both modes compute the same function; tests assert agreement.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "mmwm/arena.h"
#include "mmwm/tensor.h"
#include "mmwm/thread_pool.h"

namespace mmwm {

enum class ExecMode { kReference, kOptimized };

class ExecContext {
 public:
  static ExecContext Reference();
  // `arena` must outlive the context; `pool` may be null (single-threaded).
  static ExecContext Optimized(Arena& arena, ThreadPool* pool);

  ExecMode mode() const { return mode_; }
  ThreadPool* pool() const { return pool_; }

  // Zero-initialised scratch matrix. Valid until reset().
  Matrix alloc(size_t rows, size_t cols);
  // Releases heap scratch (reference) or rewinds the arena (optimized).
  void reset();

  // C = A * B  (C must be pre-sized [A.rows, B.cols]; contents overwritten).
  void matmul(const Matrix& A, const Matrix& B, Matrix& C) const;
  // C = A^T * B (C pre-sized [A.cols, B.cols]; contents overwritten).
  void matmul_atb(const Matrix& A, const Matrix& B, Matrix& C) const;
  // Runs fn over [0, n) in chunks (inline in reference mode).
  void parallel_for(size_t n, size_t grain, const ThreadPool::RangeFn& fn) const;

 private:
  ExecContext() = default;

  ExecMode mode_ = ExecMode::kReference;
  Arena* arena_ = nullptr;
  ThreadPool* pool_ = nullptr;
  std::vector<std::unique_ptr<Buffer>> heap_;
};

}  // namespace mmwm
