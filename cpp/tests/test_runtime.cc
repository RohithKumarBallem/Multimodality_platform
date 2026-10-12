#include <atomic>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "mmwm/arena.h"
#include "mmwm/exec_context.h"
#include "mmwm/rng.h"
#include "mmwm/thread_pool.h"
#include "test_util.h"

using namespace mmwm;

static void test_arena() {
  Arena a(256);
  Matrix m1 = a.alloc(4, 4);
  Matrix m2 = a.alloc_zeroed(2, 8);
  CHECK(m1.data != m2.data);
  CHECK(reinterpret_cast<uintptr_t>(m1.data) % Arena::kAlignment == 0);
  CHECK(reinterpret_cast<uintptr_t>(m2.data) % Arena::kAlignment == 0);
  for (size_t i = 0; i < m2.size(); ++i) CHECK(m2.data[i] == 0.0f);

  // Larger than the chunk size forces a dedicated chunk.
  Matrix big = a.alloc(64, 64);
  CHECK(big.data != nullptr);
  CHECK(a.chunk_count() >= 2);
  const size_t hw = a.high_water_bytes();
  CHECK(hw >= 64 * 64 * sizeof(float));

  // After reset the same chunks are reused: no new chunks, same first pointer.
  const size_t chunks = a.chunk_count();
  float* first = m1.data;
  a.reset();
  CHECK(a.bytes_in_use() == 0);
  Matrix again = a.alloc(4, 4);
  CHECK(again.data == first);
  CHECK(a.chunk_count() == chunks);
}

static void test_thread_pool() {
  ThreadPool pool(4);
  CHECK(pool.size() == 4);
  const size_t n = 100000;
  std::vector<int> hits(n, 0);
  for (int round = 0; round < 50; ++round) {  // exercises generation hand-off
    pool.parallel_for(0, n, 997, [&](size_t lo, size_t hi) {
      for (size_t i = lo; i < hi; ++i) hits[i] += 1;
    });
  }
  bool all = true;
  for (int h : hits) all = all && (h == 50);
  CHECK(all);

  CHECK_THROWS(pool.parallel_for(0, 1000, 10, [](size_t lo, size_t) {
    if (lo >= 500) throw std::runtime_error("boom");
  }));
  // Pool remains usable after an exception.
  std::atomic<size_t> sum{0};
  pool.parallel_for(0, 100, 7, [&](size_t lo, size_t hi) { sum += hi - lo; });
  CHECK(sum.load() == 100);

  ThreadPool single(1);
  size_t calls = 0;
  single.parallel_for(0, 10, 3, [&](size_t lo, size_t hi) { calls += hi - lo; });
  CHECK(calls == 10);
}

static void test_matmul_modes_agree() {
  Rng rng(7);
  for (size_t M : {1u, 3u, 4u, 7u, 33u}) {
    const size_t K = 19, N = 23;
    Buffer A(M, K), B(K, N);
    for (auto& v : A.vec()) v = rng.normal();
    for (auto& v : B.vec()) v = rng.normal();

    ExecContext ref = ExecContext::Reference();
    Matrix c_ref = ref.alloc(M, N);
    ref.matmul(A.view(), B.view(), c_ref);

    Arena arena;
    ThreadPool pool(3);
    ExecContext opt = ExecContext::Optimized(arena, &pool);
    Matrix c_opt = opt.alloc(M, N);
    opt.matmul(A.view(), B.view(), c_opt);

    for (size_t i = 0; i < c_ref.size(); ++i) CHECK_NEAR(c_ref.data[i], c_opt.data[i], 1e-4);

    // A^T B against a direct computation.
    Buffer D(M, N);
    for (auto& v : D.vec()) v = rng.normal();
    Matrix t = opt.alloc(K, N);
    opt.matmul_atb(A.view(), D.view(), t);
    for (size_t i = 0; i < K; ++i)
      for (size_t j = 0; j < N; ++j) {
        double acc = 0;
        for (size_t m = 0; m < M; ++m) acc += double(A.view().at(m, i)) * D.view().at(m, j);
        CHECK_NEAR(t.at(i, j), acc, 1e-3);
      }
  }
}

static void test_rng_determinism() {
  Rng a(42), b(42), c(43);
  for (int i = 0; i < 100; ++i) CHECK(a.next_u64() == b.next_u64());
  CHECK(Rng(42).next_u64() != c.next_u64());
  Rng r(1);
  double s = 0, s2 = 0;
  const int n = 20000;
  for (int i = 0; i < n; ++i) {
    const double x = r.normal();
    s += x;
    s2 += x * x;
  }
  CHECK_NEAR(s / n, 0.0, 0.05);
  CHECK_NEAR(s2 / n, 1.0, 0.05);
}

int main() {
  test_arena();
  test_thread_pool();
  test_matmul_modes_agree();
  test_rng_determinism();
  TEST_MAIN_END();
}
