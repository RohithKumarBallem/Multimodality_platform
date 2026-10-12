#include "mmwm/exec_context.h"

#include <algorithm>
#include <cstring>

#include "mmwm/profiler.h"

namespace mmwm {

namespace {

// Naive baseline: i-j-k with a strided walk down B's columns.
void matmul_reference(const Matrix& A, const Matrix& B, Matrix& C) {
  const size_t M = A.rows, K = A.cols, N = B.cols;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) acc += A.data[i * K + k] * B.data[k * N + j];
      C.data[i * N + j] = acc;
    }
  }
}

// 4 rows of A at a time so each B row is loaded once and reused four times;
// inner loop is a contiguous axpy the compiler vectorises without needing
// fast-math (every C element has an independent accumulation chain).
void matmul_rows(const Matrix& A, const Matrix& B, Matrix& C, size_t r0, size_t r1) {
  const size_t K = A.cols, N = B.cols;
  size_t i = r0;
  for (; i + 4 <= r1; i += 4) {
    float* c0 = C.data + (i + 0) * N;
    float* c1 = C.data + (i + 1) * N;
    float* c2 = C.data + (i + 2) * N;
    float* c3 = C.data + (i + 3) * N;
    std::memset(c0, 0, 4 * N * sizeof(float));  // rows are contiguous
    const float* a0 = A.data + (i + 0) * K;
    const float* a1 = A.data + (i + 1) * K;
    const float* a2 = A.data + (i + 2) * K;
    const float* a3 = A.data + (i + 3) * K;
    for (size_t k = 0; k < K; ++k) {
      const float* __restrict b = B.data + k * N;
      const float v0 = a0[k], v1 = a1[k], v2 = a2[k], v3 = a3[k];
      for (size_t j = 0; j < N; ++j) {
        const float bj = b[j];
        c0[j] += v0 * bj;
        c1[j] += v1 * bj;
        c2[j] += v2 * bj;
        c3[j] += v3 * bj;
      }
    }
  }
  for (; i < r1; ++i) {
    float* c = C.data + i * N;
    std::memset(c, 0, N * sizeof(float));
    const float* a = A.data + i * K;
    for (size_t k = 0; k < K; ++k) {
      const float* __restrict b = B.data + k * N;
      const float v = a[k];
      for (size_t j = 0; j < N; ++j) c[j] += v * b[j];
    }
  }
}

}  // namespace

ExecContext ExecContext::Reference() { return ExecContext(); }

ExecContext ExecContext::Optimized(Arena& arena, ThreadPool* pool) {
  ExecContext c;
  c.mode_ = ExecMode::kOptimized;
  c.arena_ = &arena;
  c.pool_ = pool;
  return c;
}

Matrix ExecContext::alloc(size_t rows, size_t cols) {
  if (mode_ == ExecMode::kOptimized) return arena_->alloc_zeroed(rows, cols);
  heap_.push_back(std::make_unique<Buffer>(rows, cols));
  return heap_.back()->view();
}

void ExecContext::reset() {
  heap_.clear();
  if (arena_) arena_->reset();
}

void ExecContext::matmul(const Matrix& A, const Matrix& B, Matrix& C) const {
  MMWM_PROFILE("matmul");
  if (mode_ == ExecMode::kReference) {
    matmul_reference(A, B, C);
    return;
  }
  const size_t M = A.rows;
  // ~64k MACs per chunk keeps dispatch overhead negligible; chunks are
  // multiples of 4 rows to preserve the register tile.
  const size_t work_per_row = std::max<size_t>(1, A.cols * B.cols);
  size_t grain = std::max<size_t>(4, (65536 / work_per_row + 3) / 4 * 4);
  if (pool_ && pool_->size() > 1) {
    pool_->parallel_for(0, M, grain,
                        [&](size_t lo, size_t hi) { matmul_rows(A, B, C, lo, hi); });
  } else {
    matmul_rows(A, B, C, 0, M);
  }
}

void ExecContext::matmul_atb(const Matrix& A, const Matrix& B, Matrix& C) const {
  MMWM_PROFILE("matmul_atb");
  // C[i,:] = sum_n A[n,i] * B[n,:]; parallel over output rows i.
  const size_t Nn = A.rows, I = A.cols, J = B.cols;
  auto rows = [&](size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) {
      float* c = C.data + i * J;
      std::memset(c, 0, J * sizeof(float));
      for (size_t n = 0; n < Nn; ++n) {
        const float v = A.data[n * I + i];
        const float* b = B.data + n * J;
        for (size_t j = 0; j < J; ++j) c[j] += v * b[j];
      }
    }
  };
  if (mode_ == ExecMode::kOptimized && pool_) {
    pool_->parallel_for(0, I, 8, rows);
  } else {
    rows(0, I);
  }
}

void ExecContext::parallel_for(size_t n, size_t grain, const ThreadPool::RangeFn& fn) const {
  if (mode_ == ExecMode::kOptimized && pool_) {
    pool_->parallel_for(0, n, grain, fn);
  } else if (n > 0) {
    fn(0, n);
  }
}

}  // namespace mmwm
