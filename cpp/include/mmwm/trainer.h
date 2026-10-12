// Shard-level training and evaluation.
#pragma once

#include <cstdint>
#include <string>

#include "mmwm/checkpoint.h"
#include "mmwm/world_model.h"

namespace mmwm {

struct TrainOptions {
  size_t steps = 100;
  size_t batch_size = 32;
  float lr = 0.5f;
  uint64_t round = 0;        // distinguishes data streams across sync rounds
  uint64_t shard_index = 0;  // which slice of the global data stream
  unsigned threads = 1;      // 0 -> hardware_concurrency
  ExecMode mode = ExecMode::kOptimized;
};

struct TrainResult {
  double first_loss = 0.0;  // mean over the first 10% of steps
  double last_loss = 0.0;   // mean over the last 10% of steps
  uint64_t samples = 0;
  double seconds = 0.0;
};

struct EvalResult {
  double model_mse = 0.0;  // one-step latent prediction error
  double copy_mse = 0.0;   // baseline: predict z_{t+1} = z_t
  double rollout_mse = 0.0;       // mean error over the rollout horizon
  double rollout_copy_mse = 0.0;  // copy baseline over the same horizon
  size_t samples = 0;
  size_t horizon = 0;
};

// Trains `model` in place with SGD on the one-step latent prediction loss.
TrainResult train_shard(WorldModel& model, const TrainOptions& opt);

// Evaluates on a fixed held-out stream (independent of shard/round).
EvalResult evaluate(const WorldModel& model, size_t samples, size_t horizon,
                    unsigned threads = 1);

}  // namespace mmwm
