// Small deterministic RNG (splitmix64). Used instead of <random> distributions
// so results are bit-identical across standard libraries and platforms.
#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>

namespace mmwm {

class Rng {
 public:
  explicit Rng(uint64_t seed = 0) : state_(seed) {}

  uint64_t next_u64() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  // Uniform in [0, 1).
  float uniform() { return static_cast<float>(next_u64() >> 40) * (1.0f / 16777216.0f); }

  // Standard normal (Box-Muller; the second variate is discarded so the stream
  // position does not depend on call parity).
  float normal() {
    float u1 = uniform();
    if (u1 < 1e-7f) u1 = 1e-7f;
    const float u2 = uniform();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * std::numbers::pi_v<float> * u2);
  }

  // Derive an independent stream seed from two values (e.g. seed, shard).
  static uint64_t mix(uint64_t a, uint64_t b) {
    Rng r(a ^ (b + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2)));
    return r.next_u64();
  }

 private:
  uint64_t state_;
};

}  // namespace mmwm
