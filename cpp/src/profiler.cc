#include "mmwm/profiler.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <sstream>

namespace mmwm {

Profiler& Profiler::instance() {
  static Profiler p;
  return p;
}

void Profiler::record(const char* name, double ms) {
  std::lock_guard<std::mutex> lk(mu_);
  Entry& e = entries_[name];
  if (e.name.empty()) e.name = name;
  ++e.calls;
  e.total_ms += ms;
  e.max_ms = std::max(e.max_ms, ms);
}

void Profiler::reset() {
  std::lock_guard<std::mutex> lk(mu_);
  entries_.clear();
}

std::vector<Profiler::Entry> Profiler::snapshot() const {
  std::vector<Entry> out;
  {
    std::lock_guard<std::mutex> lk(mu_);
    out.reserve(entries_.size());
    for (const auto& kv : entries_) out.push_back(kv.second);
  }
  std::ranges::sort(out, std::greater<>{}, &Entry::total_ms);
  return out;
}

std::string Profiler::report() const {
  std::ostringstream os;
  char line[160];
  std::snprintf(line, sizeof line, "%-28s %10s %12s %10s %10s\n", "scope", "calls",
                "total_ms", "mean_ms", "max_ms");
  os << line;
  for (const auto& e : snapshot()) {
    std::snprintf(line, sizeof line, "%-28s %10llu %12.3f %10.4f %10.4f\n", e.name.c_str(),
                  static_cast<unsigned long long>(e.calls), e.total_ms,
                  e.total_ms / static_cast<double>(e.calls), e.max_ms);
    os << line;
  }
  return os.str();
}

std::string Profiler::to_json() const {
  std::ostringstream os;
  os << "[";
  bool first = true;
  for (const auto& e : snapshot()) {
    if (!first) os << ",";
    first = false;
    os << "{\"name\":\"" << e.name << "\",\"calls\":" << e.calls
       << ",\"total_ms\":" << e.total_ms << ",\"max_ms\":" << e.max_ms << "}";
  }
  os << "]";
  return os.str();
}

}  // namespace mmwm
