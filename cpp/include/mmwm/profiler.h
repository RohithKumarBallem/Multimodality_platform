// Lightweight scoped-timer profiler. Zero cost when disabled (one relaxed
// atomic load per scope). Use MMWM_PROFILE("name") at the top of a scope.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mmwm {

class Profiler {
 public:
  struct Entry {
    std::string name;
    uint64_t calls = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
  };

  static Profiler& instance();

  void set_enabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
  bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

  void record(const char* name, double ms);
  void reset();
  std::vector<Entry> snapshot() const;  // sorted by total_ms, descending
  std::string report() const;           // human-readable table
  std::string to_json() const;

 private:
  std::atomic<bool> enabled_{false};
  mutable std::mutex mu_;
  std::unordered_map<std::string, Entry> entries_;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(const char* name)
      : name_(name), active_(Profiler::instance().enabled()) {
    if (active_) start_ = std::chrono::steady_clock::now();
  }
  ~ScopedTimer() {
    if (!active_) return;
    const auto dt = std::chrono::steady_clock::now() - start_;
    Profiler::instance().record(name_, std::chrono::duration<double, std::milli>(dt).count());
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  const char* name_;
  bool active_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace mmwm

#define MMWM_PROFILE_CAT2(a, b) a##b
#define MMWM_PROFILE_CAT(a, b) MMWM_PROFILE_CAT2(a, b)
#define MMWM_PROFILE(name) \
  ::mmwm::ScopedTimer MMWM_PROFILE_CAT(mmwm_scope_, __LINE__)(name)
