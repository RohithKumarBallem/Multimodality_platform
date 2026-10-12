#include "mmwm/c_api.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "mmwm/checkpoint.h"
#include "mmwm/trainer.h"
#include "mmwm/world_model.h"

namespace {

void set_err(char* err, size_t len, const char* msg) {
  if (!err || len == 0) return;
  std::snprintf(err, len, "%s", msg);
}

mmwm::ModelConfig to_cpp(const mmwm_model_config& c) {
  mmwm::ModelConfig m;
  m.d_model = c.d_model;
  m.vocab = c.vocab;
  m.text_len = c.text_len;
  m.image_patches = c.image_patches;
  m.patch_dim = c.patch_dim;
  m.audio_frames = c.audio_frames;
  m.audio_dim = c.audio_dim;
  m.action_dim = c.action_dim;
  m.seed = c.seed;
  return m;
}

template <typename F>
int guarded(char* err, size_t err_len, F&& f) {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    set_err(err, err_len, e.what());
    return 1;
  } catch (...) {
    set_err(err, err_len, "unknown error");
    return 2;
  }
}

}  // namespace

extern "C" {

const char* mmwm_version(void) { return "0.1.0"; }

int mmwm_train_shard(const mmwm_train_args* a, const char* init_ckpt, const char* out_ckpt,
                     mmwm_train_result* res, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!a || !out_ckpt || !res) throw std::invalid_argument("null argument");
    mmwm::WorldModel model(to_cpp(a->model));
    if (init_ckpt && init_ckpt[0]) mmwm::apply_checkpoint(mmwm::load_checkpoint(init_ckpt), model);

    mmwm::TrainOptions opt;
    opt.steps = a->steps;
    opt.batch_size = a->batch_size;
    opt.lr = a->lr;
    opt.round = a->round;
    opt.shard_index = a->shard_index;
    opt.threads = a->threads;
    opt.mode = a->reference_mode ? mmwm::ExecMode::kReference : mmwm::ExecMode::kOptimized;
    const mmwm::TrainResult r = mmwm::train_shard(model, opt);

    mmwm::save_checkpoint(out_ckpt, mmwm::checkpoint_from_model(model, r.samples));
    res->first_loss = r.first_loss;
    res->last_loss = r.last_loss;
    res->seconds = r.seconds;
    res->samples = r.samples;
  });
}

int mmwm_reduce(const char* const* inputs, size_t n, const char* out_ckpt, char* err,
                size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!inputs || !out_ckpt || n == 0) throw std::invalid_argument("no inputs");
    std::vector<mmwm::Checkpoint> parts;
    parts.reserve(n);
    for (size_t i = 0; i < n; ++i) parts.push_back(mmwm::load_checkpoint(inputs[i]));
    mmwm::save_checkpoint(out_ckpt, mmwm::merge_checkpoints(parts));
  });
}

int mmwm_evaluate(const mmwm_model_config* cfg, const char* ckpt, uint64_t samples,
                  uint64_t horizon, uint32_t threads, mmwm_eval_result* res, char* err,
                  size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!cfg || !res) throw std::invalid_argument("null argument");
    mmwm::WorldModel model(to_cpp(*cfg));
    if (ckpt && ckpt[0]) mmwm::apply_checkpoint(mmwm::load_checkpoint(ckpt), model);
    const mmwm::EvalResult r = mmwm::evaluate(model, samples, horizon, threads);
    res->model_mse = r.model_mse;
    res->copy_mse = r.copy_mse;
    res->rollout_mse = r.rollout_mse;
    res->rollout_copy_mse = r.rollout_copy_mse;
    res->samples = r.samples;
    res->horizon = r.horizon;
  });
}

}  // extern "C"
