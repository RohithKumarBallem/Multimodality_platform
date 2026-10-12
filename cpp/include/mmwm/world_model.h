// Multimodal latent world model.
//
//   text tokens ─┐
//   image patches ─┼─ encoders ─ gated fusion ─ z_t ──┐
//   audio frames ─┘                                   ├─ dynamics ─ z_{t+1}
//                                         action a_t ─┘
//
// Encoders are fixed random-feature projections derived from `seed`; the
// trainable part is the residual latent dynamics  z' = z + tanh([z;a] W + b).
// Any subset of modalities may be present (graceful modality dropout).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mmwm/exec_context.h"
#include "mmwm/tensor.h"

namespace mmwm {

enum Modality : uint8_t { kText = 1, kImage = 2, kAudio = 4, kAllModalities = 7 };

struct ModelConfig {
  size_t d_model = 128;
  size_t vocab = 512;
  size_t text_len = 16;
  size_t image_patches = 16;
  size_t patch_dim = 48;
  size_t audio_frames = 20;
  size_t audio_dim = 40;
  size_t action_dim = 8;
  uint64_t seed = 1;
};

// A batch of observations for one timestep. Layouts are sample-major.
struct Batch {
  size_t n = 0;
  uint8_t modalities = kAllModalities;
  std::vector<int32_t> tokens;  // [n, text_len]
  std::vector<float> image;     // [n, image_patches, patch_dim]
  std::vector<float> audio;     // [n, audio_frames, audio_dim]
  std::vector<float> action;    // [n, action_dim]  (action taken from this obs)
};

class WorldModel {
 public:
  explicit WorldModel(const ModelConfig& cfg);

  const ModelConfig& config() const { return cfg_; }

  // Trainable parameters.
  Buffer& dyn_weight() { return w_dyn_; }  // [d_model + action_dim, d_model]
  Buffer& dyn_bias() { return b_dyn_; }    // [1, d_model]
  const Buffer& dyn_weight() const { return w_dyn_; }
  const Buffer& dyn_bias() const { return b_dyn_; }

  // Fused latent for a batch -> [n, d_model] allocated from ctx.
  Matrix encode(const Batch& b, ExecContext& ctx) const;

  // One dynamics step. `action` is [n, action_dim]; returns [n, d_model].
  Matrix step(const Matrix& z, const float* action, ExecContext& ctx) const;

  // Open-loop rollout. `actions` is [horizon][n][action_dim]; the result is
  // [horizon * n, d_model] with step t in rows [t*n, (t+1)*n).
  Matrix rollout(const Matrix& z0, const float* actions, size_t horizon, ExecContext& ctx) const;

 private:
  ModelConfig cfg_;
  Buffer text_embed_;  // [vocab, d]
  Buffer w_img_;       // [patch_dim, d]
  Buffer b_img_;       // [1, d]
  Buffer w_aud_;       // [audio_dim, d]
  Buffer b_aud_;       // [1, d]
  Buffer gate_;        // [1, d]
  Buffer w_dyn_;
  Buffer b_dyn_;
};

}  // namespace mmwm
