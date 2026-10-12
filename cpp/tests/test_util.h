// Tiny dependency-free test harness (keeps the build hermetic for Bazel/CMake).
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace mmwm_test {
inline int& failures() {
  static int n = 0;
  return n;
}
}  // namespace mmwm_test

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++mmwm_test::failures();                                                   \
    }                                                                            \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
  do {                                                                           \
    const double va_ = (a), vb_ = (b);                                           \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                      \
      std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s=%g vs %s=%g (tol %g)\n", \
                   __FILE__, __LINE__, #a, va_, #b, vb_, static_cast<double>(tol)); \
      ++mmwm_test::failures();                                                   \
    }                                                                            \
  } while (0)

#define CHECK_THROWS(expr)                                                       \
  do {                                                                           \
    bool threw_ = false;                                                         \
    try {                                                                        \
      (void)(expr);                                                              \
    } catch (...) {                                                              \
      threw_ = true;                                                             \
    }                                                                            \
    if (!threw_) {                                                               \
      std::fprintf(stderr, "%s:%d: expected exception: %s\n", __FILE__, __LINE__, #expr); \
      ++mmwm_test::failures();                                                   \
    }                                                                            \
  } while (0)

#define TEST_MAIN_END()                                                          \
  do {                                                                           \
    if (mmwm_test::failures()) {                                                 \
      std::fprintf(stderr, "%d check(s) failed\n", mmwm_test::failures());       \
      return 1;                                                                  \
    }                                                                            \
    std::printf("ok\n");                                                         \
    return 0;                                                                    \
  } while (0)
