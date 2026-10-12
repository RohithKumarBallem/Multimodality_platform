// Bump-pointer arena for per-step activations.
//
// Removes allocator traffic from the hot path: a training/inference step
// allocates all intermediates from the arena and calls reset() when done.
// Not thread-safe; allocate on the dispatching thread, then fan out work.
#pragma once

#include <bit>
#include <cstddef>
#include <memory>
#include <vector>

#include "mmwm/tensor.h"

namespace mmwm {

class Arena {
 public:
  static constexpr size_t kAlignment = 64;
  static_assert(std::has_single_bit(kAlignment), "alignment must be a power of two");

  explicit Arena(size_t chunk_bytes = 1u << 20);
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // Uninitialised storage.
  void* allocate(size_t bytes);
  Matrix alloc(size_t rows, size_t cols);
  Matrix alloc_zeroed(size_t rows, size_t cols);

  // Invalidates every view handed out so far; keeps the chunks for reuse.
  void reset();

  size_t bytes_in_use() const { return in_use_; }
  size_t high_water_bytes() const { return high_water_; }
  size_t chunk_count() const { return chunks_.size(); }

 private:
  struct Chunk {
    std::byte* base;
    size_t capacity;
    size_t offset;
  };
  void add_chunk(size_t min_bytes);

  size_t chunk_bytes_;
  std::vector<Chunk> chunks_;
  size_t current_ = 0;
  size_t in_use_ = 0;
  size_t high_water_ = 0;
};

}  // namespace mmwm
