// Vectorisable math approximations used on the optimised path.
#pragma once

#include <algorithm>

namespace mmwm {

// Rational (13,6) minimax approximation of tanh for float, the formulation
// used by Eigen. Branch-free, so loops over it auto-vectorise, unlike libm's
// tanhf. Max abs error vs std::tanh is checked in tests (< 1e-6).
inline float fast_tanhf(float x) {
  x = std::clamp(x, -7.90531110763549805f, 7.90531110763549805f);
  const float x2 = x * x;
  float p = -2.76076847742355e-16f;
  p = p * x2 + 2.00018790482477e-13f;
  p = p * x2 + -8.60467152213735e-11f;
  p = p * x2 + 5.12229709037114e-08f;
  p = p * x2 + 1.48572235717979e-05f;
  p = p * x2 + 6.37261928875436e-04f;
  p = p * x2 + 4.89352455891786e-03f;
  p = p * x;
  float q = 1.19825839466702e-06f;
  q = q * x2 + 1.18534705686654e-04f;
  q = q * x2 + 2.26843463243900e-03f;
  q = q * x2 + 4.89352518554385e-03f;
  return p / q;
}

}  // namespace mmwm
