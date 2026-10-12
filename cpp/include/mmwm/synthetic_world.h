// Deterministic synthetic multimodal environment so the platform runs
// end-to-end without external datasets.
//
// A hidden state s_t (dim = action_dim) evolves as
//     s_{t+1} = 0.9 * Rot(s_t) + 0.3 * B a_t
// and is rendered into text tokens, image patches and audio frames through
// fixed nonlinear projections. The "laws" (Rot, B, projections) depend only on
// the model seed, so every shard of a distributed run sees the same world but
// different trajectories.
#pragma once

#include <cstdint>
#include <vector>

#include "mmwm/rng.h"
#include "mmwm/world_model.h"

namespace mmwm {

class SyntheticWorld {
 public:
  explicit SyntheticWorld(const ModelConfig& cfg);

  // Returns horizon + 1 observation batches (t = 0..horizon). Batch t carries
  // the action applied at step t (the last batch's action is zero).
  std::vector<Batch> sample_sequence(size_t n, size_t horizon, Rng& rng,
                                     uint8_t modalities = kAllModalities) const;

 private:
  void render(const float* s, size_t i, Batch& out) const;

  ModelConfig cfg_;
  size_t k_;                    // hidden state dim
  std::vector<float> rot_;      // [k/2] angles
  std::vector<float> b_;        // [k, A]
  std::vector<float> p_img_;    // [P * pd, k]
  std::vector<float> p_aud_;    // [F * ad, k]
  std::vector<float> p_tok_;    // [L, k]
};

}  // namespace mmwm
