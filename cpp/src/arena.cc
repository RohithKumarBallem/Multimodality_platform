#include "mmwm/arena.h"

#include <algorithm>
#include <cstring>
#include <new>

namespace mmwm {

namespace {
size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }
}  // namespace

Arena::Arena(size_t chunk_bytes) : chunk_bytes_(align_up(chunk_bytes, kAlignment)) {}

Arena::~Arena() {
  for (auto& c : chunks_) {
    ::operator delete(c.base, std::align_val_t{kAlignment});
  }
}

void Arena::add_chunk(size_t min_bytes) {
  const size_t cap = std::max(chunk_bytes_, align_up(min_bytes, kAlignment));
  auto* base = static_cast<std::byte*>(::operator new(cap, std::align_val_t{kAlignment}));
  chunks_.push_back(Chunk{base, cap, 0});
  current_ = chunks_.size() - 1;
}

void* Arena::allocate(size_t bytes) {
  const size_t need = align_up(bytes == 0 ? 1 : bytes, kAlignment);
  // Try the current chunk, then any later (already allocated, post-reset) chunk.
  while (current_ < chunks_.size()) {
    Chunk& c = chunks_[current_];
    if (c.capacity - c.offset >= need) {
      void* p = c.base + c.offset;
      c.offset += need;
      in_use_ += need;
      high_water_ = std::max(high_water_, in_use_);
      return p;
    }
    ++current_;
  }
  add_chunk(need);
  return allocate(bytes);
}

Matrix Arena::alloc(size_t rows, size_t cols) {
  return Matrix{static_cast<float*>(allocate(rows * cols * sizeof(float))), rows, cols};
}

Matrix Arena::alloc_zeroed(size_t rows, size_t cols) {
  Matrix m = alloc(rows, cols);
  std::memset(m.data, 0, m.size() * sizeof(float));
  return m;
}

void Arena::reset() {
  for (auto& c : chunks_) c.offset = 0;
  current_ = 0;
  in_use_ = 0;
}

}  // namespace mmwm
