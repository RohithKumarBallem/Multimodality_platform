#include "mmwm/world_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mmwm/profiler.h"
#include "mmwm/rng.h"

namespace mmwm {

namespace {

void fill_normal(Buffer& b, Rng& rng, float scale) {
  for (float& v : b.vec()) v = rng.normal() * scale;
}

}  // namespace

WorldModel::WorldModel(const ModelConfig& cfg)
    : cfg_(cfg),
      text_embed_(cfg.vocab, cfg.d_model),
      w_img_(cfg.patch_dim, cfg.d_model),
      b_img_(1, cfg.d_model),
      w_aud_(cfg.audio_dim, cfg.d_model),
      b_aud_(1, cfg.d_model),
      gate_(1, cfg.d_model),
      w_dyn_(cfg.d_model + cfg.action_dim, cfg.d_model),
      b_dyn_(1, cfg.d_model) {
  if (cfg.d_model == 0 || cfg.vocab == 0 || cfg.text_len == 0 || cfg.image_patches == 0 ||
      cfg.patch_dim == 0 || cfg.audio_frames == 0 || cfg.audio_dim == 0 ||
      cfg.action_dim == 0) {
    throw std::invalid_argument("ModelConfig: all dimensions must be > 0");
  }
  Rng rng(Rng::mix(cfg.seed, 0x4D4F44454CULL));  // "MODEL"
  fill_normal(text_embed_, rng, 1.0f);
  fill_normal(w_img_, rng, 1.0f / std::sqrt(static_cast<float>(cfg.patch_dim)));
  fill_normal(b_img_, rng, 0.1f);
  fill_normal(w_aud_, rng, 1.0f / std::sqrt(static_cast<float>(cfg.audio_dim)));
  fill_normal(b_aud_, rng, 0.1f);
  fill_normal(gate_, rng, 1.0f);
  fill_normal(w_dyn_, rng, 0.1f / std::sqrt(static_cast<float>(cfg.d_model + cfg.action_dim)));
  // b_dyn_ stays zero.
}

Matrix WorldModel::encode(const Batch& b, ExecContext& ctx) const {
  MMWM_PROFILE("encode");
  const size_t n = b.n, d = cfg_.d_model;
  if (b.modalities == 0) throw std::invalid_argument("encode: no modalities present");

  const bool has_text = b.modalities & kText;
  const bool has_img = b.modalities & kImage;
  const bool has_aud = b.modalities & kAudio;
  if (has_text && b.tokens.size() != n * cfg_.text_len)
    throw std::invalid_argument("encode: tokens size mismatch");
  if (has_img && b.image.size() != n * cfg_.image_patches * cfg_.patch_dim)
    throw std::invalid_argument("encode: image size mismatch");
  if (has_aud && b.audio.size() != n * cfg_.audio_frames * cfg_.audio_dim)
    throw std::invalid_argument("encode: audio size mismatch");

  Matrix e_text, e_img, e_aud;

  if (has_text) {
    MMWM_PROFILE("encode.text");
    e_text = ctx.alloc(n, d);
    const size_t L = cfg_.text_len;
    const float inv = 1.0f / static_cast<float>(L);
    ctx.parallel_for(n, 16, [&](size_t lo, size_t hi) {
      for (size_t i = lo; i < hi; ++i) {
        float* o = e_text.row(i);
        for (size_t t = 0; t < L; ++t) {
          int32_t tok = b.tokens[i * L + t];
          tok = std::clamp<int32_t>(tok, 0, static_cast<int32_t>(cfg_.vocab) - 1);
          const float* r = text_embed_.view().row(static_cast<size_t>(tok));
          for (size_t j = 0; j < d; ++j) o[j] += r[j];
        }
        for (size_t j = 0; j < d; ++j) o[j] = ctx.tanh(o[j] * inv);
      }
    });
  }

  if (has_img) {
    MMWM_PROFILE("encode.image");
    const size_t P = cfg_.image_patches;
    Matrix x{const_cast<float*>(b.image.data()), n * P, cfg_.patch_dim};
    Matrix w = w_img_.view();
    e_img = ctx.alloc(n, d);
    ctx.project_pool_tanh(x, w, b_img_.vec().data(), P, e_img);
  }

  if (has_aud) {
    MMWM_PROFILE("encode.audio");
    const size_t F = cfg_.audio_frames;
    Matrix x{const_cast<float*>(b.audio.data()), n * F, cfg_.audio_dim};
    Matrix w = w_aud_.view();
    e_aud = ctx.alloc(n, d);
    ctx.project_pool_tanh(x, w, b_aud_.vec().data(), F, e_aud);
  }

  // Gated fusion: softmax over present modalities of <e_m, gate>/sqrt(d),
  // convex combination, then layer-norm.
  MMWM_PROFILE("encode.fuse");
  Matrix z = ctx.alloc(n, d);
  const float* gate = gate_.vec().data();
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  ctx.parallel_for(n, 16, [&](size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) {
      const float* e[3];
      float s[3];
      size_t m = 0;
      auto add = [&](const Matrix& mat) {
        const float* r = mat.row(i);
        float dot = 0.0f;
        for (size_t j = 0; j < d; ++j) dot += r[j] * gate[j];
        e[m] = r;
        s[m] = dot * inv_sqrt_d;
        ++m;
      };
      if (has_text) add(e_text);
      if (has_img) add(e_img);
      if (has_aud) add(e_aud);

      float mx = s[0];
      for (size_t k = 1; k < m; ++k) mx = std::max(mx, s[k]);
      float denom = 0.0f;
      for (size_t k = 0; k < m; ++k) {
        s[k] = std::exp(s[k] - mx);
        denom += s[k];
      }
      float* o = z.row(i);
      for (size_t k = 0; k < m; ++k) {
        const float a = s[k] / denom;
        for (size_t j = 0; j < d; ++j) o[j] += a * e[k][j];
      }
      float mean = 0.0f;
      for (size_t j = 0; j < d; ++j) mean += o[j];
      mean /= static_cast<float>(d);
      float var = 0.0f;
      for (size_t j = 0; j < d; ++j) var += (o[j] - mean) * (o[j] - mean);
      var /= static_cast<float>(d);
      const float rs = 1.0f / std::sqrt(var + 1e-5f);
      for (size_t j = 0; j < d; ++j) o[j] = (o[j] - mean) * rs;
    }
  });
  return z;
}

Matrix WorldModel::step(const Matrix& z, const float* action, ExecContext& ctx) const {
  MMWM_PROFILE("dynamics.step");
  const size_t n = z.rows, d = cfg_.d_model, A = cfg_.action_dim;
  Matrix x = ctx.alloc(n, d + A);
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(x.row(i), z.row(i), d * sizeof(float));
    std::memcpy(x.row(i) + d, action + i * A, A * sizeof(float));
  }
  Matrix u = ctx.alloc(n, d);
  Matrix w = w_dyn_.view();
  ctx.matmul(x, w, u);
  const float* bias = b_dyn_.vec().data();
  Matrix out = ctx.alloc(n, d);
  ctx.parallel_for(n, 64, [&](size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) {
      const float* ur = u.row(i);
      const float* zr = z.row(i);
      float* o = out.row(i);
      for (size_t j = 0; j < d; ++j) o[j] = zr[j] + ctx.tanh(ur[j] + bias[j]);
    }
  });
  return out;
}

Matrix WorldModel::rollout(const Matrix& z0, const float* actions, size_t horizon,
                           ExecContext& ctx) const {
  MMWM_PROFILE("rollout");
  const size_t n = z0.rows, d = cfg_.d_model, A = cfg_.action_dim;
  Matrix traj = ctx.alloc(horizon * n, d);
  Matrix cur = z0;
  for (size_t t = 0; t < horizon; ++t) {
    Matrix nxt = step(cur, actions + t * n * A, ctx);
    std::memcpy(traj.row(t * n), nxt.data, n * d * sizeof(float));
    cur = nxt;
  }
  return traj;
}

}  // namespace mmwm
