// On-disk format for the trainable parameters (dynamics weight + bias).
//
//   bytes 0..7    magic "MMWMCKPT"
//   u32 version, u32 d_model, u32 action_dim, u32 reserved
//   u64 seed      (model/world seed; shards with different seeds can't merge)
//   u64 samples   (training samples seen; the weight used when averaging)
//   f32 weight[(d_model + action_dim) * d_model], f32 bias[d_model]
//
// Little-endian, written to a temp file then renamed so readers on a shared
// filesystem never observe a partial checkpoint.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mmwm/world_model.h"

namespace mmwm {

struct Checkpoint {
  uint32_t d_model = 0;
  uint32_t action_dim = 0;
  uint64_t seed = 0;
  uint64_t samples = 0;
  std::vector<float> weight;
  std::vector<float> bias;
};

// Throws std::runtime_error on I/O or format errors.
void save_checkpoint(const std::string& path, const Checkpoint& ckpt);
Checkpoint load_checkpoint(const std::string& path);

Checkpoint checkpoint_from_model(const WorldModel& m, uint64_t samples);
// Throws if the checkpoint's shape/seed does not match the model.
void apply_checkpoint(const Checkpoint& ckpt, WorldModel& m);

// Sample-weighted average of compatible checkpoints (accumulated in double).
Checkpoint merge_checkpoints(const std::vector<Checkpoint>& parts);

}  // namespace mmwm
