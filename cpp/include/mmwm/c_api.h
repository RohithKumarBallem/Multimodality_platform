/* C ABI over the training / reduction / evaluation entry points.
 * Consumed from Python via ctypes (python/mmwm/_native.py). All functions
 * return 0 on success and a non-zero code on failure, writing a message into
 * `err` (may be NULL). Never throws across the boundary. */
#ifndef MMWM_C_API_H_
#define MMWM_C_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define MMWM_EXPORT __declspec(dllexport)
#else
#define MMWM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmwm_model_config {
  uint64_t d_model, vocab, text_len, image_patches, patch_dim, audio_frames,
      audio_dim, action_dim, seed;
} mmwm_model_config;

typedef struct mmwm_train_args {
  mmwm_model_config model;
  uint64_t steps, batch_size;
  float lr;
  uint64_t round, shard_index;
  uint32_t threads;
  uint32_t reference_mode; /* 1 = naive reference path (for A/B timing) */
} mmwm_train_args;

typedef struct mmwm_train_result {
  double first_loss, last_loss, seconds;
  uint64_t samples;
} mmwm_train_result;

typedef struct mmwm_eval_result {
  double model_mse, copy_mse, rollout_mse, rollout_copy_mse;
  uint64_t samples, horizon;
} mmwm_eval_result;

MMWM_EXPORT const char* mmwm_version(void);

/* Trains one shard. `init_ckpt` may be NULL/"" (fresh init). Writes `out_ckpt`. */
MMWM_EXPORT int mmwm_train_shard(const mmwm_train_args* args, const char* init_ckpt,
                                 const char* out_ckpt, mmwm_train_result* result,
                                 char* err, size_t err_len);

/* Sample-weighted average of `n_inputs` checkpoints into `out_ckpt`. */
MMWM_EXPORT int mmwm_reduce(const char* const* inputs, size_t n_inputs, const char* out_ckpt,
                            char* err, size_t err_len);

/* Evaluates a checkpoint (NULL/"" evaluates the untrained init). */
MMWM_EXPORT int mmwm_evaluate(const mmwm_model_config* model, const char* ckpt,
                              uint64_t samples, uint64_t horizon, uint32_t threads,
                              mmwm_eval_result* result, char* err, size_t err_len);

#ifdef __cplusplus
}
#endif
#endif /* MMWM_C_API_H_ */
