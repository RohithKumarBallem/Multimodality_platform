// Minimal row-major float matrix types.
//
// `Matrix` is a non-owning view; storage comes from an `Arena` (hot path) or a
// `Buffer` (reference path / long-lived parameters).
#pragma once

#include <cstddef>
#include <vector>

namespace mmwm {

struct Matrix {
  float* data = nullptr;
  size_t rows = 0;
  size_t cols = 0;

  size_t size() const { return rows * cols; }
  float* row(size_t r) { return data + r * cols; }
  const float* row(size_t r) const { return data + r * cols; }
  float& at(size_t r, size_t c) { return data[r * cols + c]; }
  float at(size_t r, size_t c) const { return data[r * cols + c]; }
};

// Owning heap matrix.
class Buffer {
 public:
  Buffer() = default;
  Buffer(size_t rows, size_t cols) : rows_(rows), cols_(cols), store_(rows * cols, 0.0f) {}

  Matrix view() { return Matrix{store_.data(), rows_, cols_}; }
  Matrix view() const {
    return Matrix{const_cast<float*>(store_.data()), rows_, cols_};
  }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  std::vector<float>& vec() { return store_; }
  const std::vector<float>& vec() const { return store_; }

 private:
  size_t rows_ = 0;
  size_t cols_ = 0;
  std::vector<float> store_;
};

}  // namespace mmwm
