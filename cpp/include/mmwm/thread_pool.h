// Fixed-size pool with a blocking parallel_for. The calling thread
// participates, so a pool of size N uses N threads total.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mmwm {

class ThreadPool {
 public:
  using RangeFn = std::function<void(size_t lo, size_t hi)>;

  // threads == 0 -> hardware_concurrency(); threads == 1 -> fully inline.
  explicit ThreadPool(unsigned threads = 0);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return static_cast<unsigned>(workers_.size()) + 1; }

  // Splits [begin, end) into chunks of at most `grain` and runs fn on each.
  // Blocks until all chunks finish; rethrows the first exception from fn.
  // Calls are serialised, so it is safe (if pointless) to share a pool.
  void parallel_for(size_t begin, size_t end, size_t grain, const RangeFn& fn);

 private:
  void worker_loop();
  void run_chunks();

  std::vector<std::thread> workers_;
  std::mutex call_mu_;  // serialises parallel_for callers

  std::mutex mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  bool stop_ = false;
  uint64_t generation_ = 0;
  size_t workers_pending_ = 0;

  // Current job (valid while a parallel_for is in flight).
  const RangeFn* fn_ = nullptr;
  size_t end_ = 0;
  size_t grain_ = 1;
  std::atomic<size_t> next_{0};
  std::exception_ptr error_;
};

}  // namespace mmwm
