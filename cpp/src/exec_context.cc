#include "mmwm/exec_context.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "mmwm/profiler.h"

#if defined(MMWM_HAVE_BLAS)
#if defined(MMWM_BLAS_ACCELERATE)
// Use Accelerate's long-standing cblas entry points (macOS 10.15+). Apple marks
// them deprecated in favour of the "new LAPACK" interface, which requires
// macOS 13.3+; the legacy symbols remain and have the same signature.
#include <Accelerate/Accelerate.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#else
#include <cblas.h>
#endif
#endif

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
void matmul_rows(const float* A, size_t K, const float* B, size_t N, float* C, size_t rows) {
  size_t i = 0;
  for (; i + 4 <= rows; i += 4) {
    float* c0 = C + (i + 0) * N;
    float* c1 = C + (i + 1) * N;
    float* c2 = C + (i + 2) * N;
    float* c3 = C + (i + 3) * N;
    std::memset(c0, 0, 4 * N * sizeof(float));  // rows are contiguous
    const float* a0 = A + (i + 0) * K;
    const float* a1 = A + (i + 1) * K;
    const float* a2 = A + (i + 2) * K;
    const float* a3 = A + (i + 3) * K;
    for (size_t k = 0; k < K; ++k) {
      const float* __restrict b = B + k * N;
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
  for (; i < rows; ++i) {
    float* c = C + i * N;
    std::memset(c, 0, N * sizeof(float));
    const float* a = A + i * K;
    for (size_t k = 0; k < K; ++k) {
      const float* __restrict b = B + k * N;
      const float v = a[k];
      for (size_t j = 0; j < N; ++j) c[j] += v * b[j];
    }
  }
}

template <bool Fast>
inline float th(float v) {
  if constexpr (Fast) {
    return fast_tanhf(v);
  } else {
    return std::tanh(v);
  }
}

// In place: h[r, :] = tanh(h[r, :] + bias) for rows [lo, hi).
template <bool Fast>
void bias_tanh_rows(float* h, size_t d, const float* bias, size_t lo, size_t hi) {
  for (size_t r = lo; r < hi; ++r) {
    float* row = h + r * d;
    for (size_t j = 0; j < d; ++j) row[j] = th<Fast>(row[j] + bias[j]);
  }
}

// out[s, :] = tanh(mean_g h[s*group + g, :]) for samples [lo, hi); h is
// already activated.
template <bool Fast>
void pool_samples(const float* h, size_t group, size_t d, float* out, size_t lo, size_t hi) {
  const float inv = 1.0f / static_cast<float>(group);
  for (size_t s = lo; s < hi; ++s) {
    float* o = out + s * d;
    std::memset(o, 0, d * sizeof(float));
    for (size_t g = 0; g < group; ++g) {
      const float* r = h + (s * group + g) * d;
      for (size_t j = 0; j < d; ++j) o[j] += r[j];
    }
    for (size_t j = 0; j < d; ++j) o[j] = th<Fast>(o[j] * inv);
  }
}

}  // namespace

ExecContext ExecContext::Reference() { return ExecContext(); }

ExecContext ExecContext::Optimized(Arena& arena, ThreadPool* pool, OptFlags flags) {
  if (flags.blas && !has_blas())
    throw std::runtime_error("OptFlags::blas requested but built without BLAS");
  ExecContext c;
  c.mode_ = ExecMode::kOptimized;
  c.arena_ = &arena;
  c.pool_ = pool;
  c.flags_ = flags;
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
#if defined(MMWM_HAVE_BLAS)
  if (flags_.blas) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(A.rows),
                static_cast<int>(B.cols), static_cast<int>(A.cols), 1.0f, A.data,
                static_cast<int>(A.cols), B.data, static_cast<int>(B.cols), 0.0f, C.data,
                static_cast<int>(C.cols));
    return;
  }
#endif
  const size_t M = A.rows;
  // ~64k MACs per chunk keeps dispatch overhead negligible; chunks are
  // multiples of 4 rows to preserve the register tile.
  const size_t work_per_row = std::max<size_t>(1, A.cols * B.cols);
  const size_t grain = std::max<size_t>(4, (65536 / work_per_row + 3) / 4 * 4);
  auto rows = [&](size_t lo, size_t hi) {
    matmul_rows(A.data + lo * A.cols, A.cols, B.data, B.cols, C.data + lo * B.cols, hi - lo);
  };
  if (pool_ && pool_->size() > 1) {
    pool_->parallel_for(0, M, grain, rows);
  } else {
    rows(0, M);
  }
}

void ExecContext::matmul_atb(const Matrix& A, const Matrix& B, Matrix& C) const {
  MMWM_PROFILE("matmul_atb");
#if defined(MMWM_HAVE_BLAS)
  if (mode_ == ExecMode::kOptimized && flags_.blas) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(A.cols),
                static_cast<int>(B.cols), static_cast<int>(A.rows), 1.0f, A.data,
                static_cast<int>(A.cols), B.data, static_cast<int>(B.cols), 0.0f, C.data,
                static_cast<int>(C.cols));
    return;
  }
#endif
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

void ExecContext::project_pool_tanh(const Matrix& x, const Matrix& W, const float* bias,
                                    size_t group, Matrix& out) {
  const size_t n = out.rows, d = out.cols;

  auto body = [&](auto fast_tag) {
    constexpr bool Fast = decltype(fast_tag)::value;

    // Whole-tensor matmul, then a bias+tanh pass, then a pooling pass.
    Matrix h = alloc(n * group, d);
    matmul(x, W, h);
    MMWM_PROFILE("project.epilogue");
    parallel_for(n * group, 64, [&](size_t lo, size_t hi) {
      bias_tanh_rows<Fast>(h.data, d, bias, lo, hi);
    });
    parallel_for(n, 8, [&](size_t lo, size_t hi) {
      pool_samples<Fast>(h.data, group, d, out.data, lo, hi);
    });
  };

  if (flags_.fast_tanh) {
    body(std::true_type{});
  } else {
    body(std::false_type{});
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
