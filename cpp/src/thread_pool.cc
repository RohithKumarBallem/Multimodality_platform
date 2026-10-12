#include "mmwm/thread_pool.h"

#include <algorithm>

namespace mmwm {

ThreadPool::ThreadPool(unsigned threads) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  workers_.reserve(threads - 1);
  for (unsigned i = 1; i < threads; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_work_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::run_chunks() {
  const RangeFn& fn = *fn_;
  while (true) {
    const size_t lo = next_.fetch_add(grain_, std::memory_order_relaxed);
    if (lo >= end_) return;
    const size_t hi = std::min(lo + grain_, end_);
    try {
      fn(lo, hi);
    } catch (...) {
      std::lock_guard<std::mutex> lk(mu_);
      if (!error_) error_ = std::current_exception();
      next_.store(end_, std::memory_order_relaxed);  // drain remaining work
      return;
    }
  }
}

void ThreadPool::worker_loop() {
  uint64_t seen = 0;
  while (true) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_work_.wait(lk, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    run_chunks();
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (--workers_pending_ == 0) cv_done_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(size_t begin, size_t end, size_t grain, const RangeFn& fn) {
  if (begin >= end) return;
  grain = std::max<size_t>(grain, 1);
  if (workers_.empty() || end - begin <= grain) {
    fn(begin, end);
    return;
  }

  std::lock_guard<std::mutex> call_lock(call_mu_);
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    end_ = end;
    grain_ = grain;
    next_.store(begin, std::memory_order_relaxed);
    error_ = nullptr;
    workers_pending_ = workers_.size();
    ++generation_;
  }
  cv_work_.notify_all();

  run_chunks();

  std::exception_ptr err;
  {
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return workers_pending_ == 0; });
    err = error_;
    fn_ = nullptr;
  }
  if (err) std::rethrow_exception(err);
}

}  // namespace mmwm
