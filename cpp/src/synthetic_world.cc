#include "mmwm/synthetic_world.h"

#include <algorithm>
#include <cmath>

namespace mmwm {

SyntheticWorld::SyntheticWorld(const ModelConfig& cfg) : cfg_(cfg), k_(cfg.action_dim) {
  Rng rng(Rng::mix(cfg.seed, 0x574F524C44ULL));  // "WORLD"
  rot_.resize((k_ + 1) / 2);
  for (float& a : rot_) a = 0.1f + 0.4f * rng.uniform();
  b_.resize(k_ * cfg.action_dim);
  for (float& v : b_) v = rng.normal() / std::sqrt(static_cast<float>(cfg.action_dim));
  auto fill = [&](std::vector<float>& v, size_t rows) {
    v.resize(rows * k_);
    for (float& x : v) x = rng.normal() / std::sqrt(static_cast<float>(k_));
  };
  fill(p_img_, cfg.image_patches * cfg.patch_dim);
  fill(p_aud_, cfg.audio_frames * cfg.audio_dim);
  fill(p_tok_, cfg.text_len);
}

void SyntheticWorld::render(const float* s, size_t i, Batch& out) const {
  const size_t k = k_;
  if (out.modalities & kText) {
    for (size_t t = 0; t < cfg_.text_len; ++t) {
      float dot = 0.0f;
      for (size_t j = 0; j < k; ++j) dot += p_tok_[t * k + j] * s[j];
      const float u = 0.5f * (std::tanh(2.0f * dot) + 1.0f);  // (0, 1)
      int32_t tok = static_cast<int32_t>(u * static_cast<float>(cfg_.vocab));
      out.tokens[i * cfg_.text_len + t] =
          std::min<int32_t>(tok, static_cast<int32_t>(cfg_.vocab) - 1);
    }
  }
  if (out.modalities & kImage) {
    const size_t rows = cfg_.image_patches * cfg_.patch_dim;
    float* dst = out.image.data() + i * rows;
    for (size_t r = 0; r < rows; ++r) {
      float dot = 0.0f;
      for (size_t j = 0; j < k; ++j) dot += p_img_[r * k + j] * s[j];
      dst[r] = std::tanh(dot);
    }
  }
  if (out.modalities & kAudio) {
    const size_t rows = cfg_.audio_frames * cfg_.audio_dim;
    float* dst = out.audio.data() + i * rows;
    for (size_t r = 0; r < rows; ++r) {
      float dot = 0.0f;
      for (size_t j = 0; j < k; ++j) dot += p_aud_[r * k + j] * s[j];
      dst[r] = std::tanh(dot);
    }
  }
}

std::vector<Batch> SyntheticWorld::sample_sequence(size_t n, size_t horizon, Rng& rng,
                                                   uint8_t modalities) const {
  const size_t k = k_, A = cfg_.action_dim;
  std::vector<Batch> seq(horizon + 1);
  for (auto& b : seq) {
    b.n = n;
    b.modalities = modalities;
    if (modalities & kText) b.tokens.assign(n * cfg_.text_len, 0);
    if (modalities & kImage) b.image.assign(n * cfg_.image_patches * cfg_.patch_dim, 0.0f);
    if (modalities & kAudio) b.audio.assign(n * cfg_.audio_frames * cfg_.audio_dim, 0.0f);
    b.action.assign(n * A, 0.0f);
  }

  std::vector<float> s(k), nxt(k);
  for (size_t i = 0; i < n; ++i) {
    for (float& v : s) v = rng.normal();
    for (size_t t = 0; t <= horizon; ++t) {
      render(s.data(), i, seq[t]);
      if (t == horizon) break;
      float* a = seq[t].action.data() + i * A;
      for (size_t j = 0; j < A; ++j) a[j] = rng.normal();
      // rotate pairs, contract, add control.
      for (size_t j = 0; j + 1 < k; j += 2) {
        const float c = std::cos(rot_[j / 2]), sn = std::sin(rot_[j / 2]);
        nxt[j] = 0.9f * (c * s[j] - sn * s[j + 1]);
        nxt[j + 1] = 0.9f * (sn * s[j] + c * s[j + 1]);
      }
      if (k % 2) nxt[k - 1] = 0.9f * s[k - 1];
      for (size_t r = 0; r < k; ++r) {
        float acc = 0.0f;
        for (size_t j = 0; j < A; ++j) acc += b_[r * A + j] * a[j];
        nxt[r] += 0.3f * acc;
      }
      s = nxt;
    }
  }
  return seq;
}

}  // namespace mmwm
