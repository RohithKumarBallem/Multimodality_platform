#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "mmwm/checkpoint.h"
#include "mmwm/profiler.h"
#include "mmwm/synthetic_world.h"
#include "mmwm/trainer.h"
#include "mmwm/world_model.h"
#include "test_util.h"

using namespace mmwm;

static ModelConfig small_cfg() {
  ModelConfig c;
  c.d_model = 32;
  c.vocab = 64;
  c.text_len = 6;
  c.image_patches = 4;
  c.patch_dim = 12;
  c.audio_frames = 5;
  c.audio_dim = 10;
  c.action_dim = 4;
  c.seed = 11;
  return c;
}

static std::string tmp_path(const char* name) {
  const char* dir = std::getenv("TMPDIR");
  return std::string(dir ? dir : "/tmp") + "/mmwm_" + std::to_string(::getpid()) + "_" + name;
}

static void test_reference_and_optimized_agree() {
  const ModelConfig cfg = small_cfg();
  WorldModel model(cfg);
  SyntheticWorld world(cfg);
  Rng rng(3);
  auto seq = world.sample_sequence(9, 3, rng);

  ExecContext ref = ExecContext::Reference();
  Matrix z_ref = model.encode(seq[0], ref);

  Arena arena;
  ThreadPool pool(3);
  ExecContext opt = ExecContext::Optimized(arena, &pool);
  Matrix z_opt = model.encode(seq[0], opt);
  for (size_t i = 0; i < z_ref.size(); ++i) CHECK_NEAR(z_ref.data[i], z_opt.data[i], 1e-4);

  std::vector<float> actions;
  for (size_t t = 0; t < 3; ++t)
    actions.insert(actions.end(), seq[t].action.begin(), seq[t].action.end());
  Matrix r_ref = model.rollout(z_ref, actions.data(), 3, ref);
  Matrix r_opt = model.rollout(z_opt, actions.data(), 3, opt);
  CHECK(r_ref.rows == 27 && r_ref.cols == cfg.d_model);
  for (size_t i = 0; i < r_ref.size(); ++i) CHECK_NEAR(r_ref.data[i], r_opt.data[i], 1e-3);

  // Layer-norm output: zero mean, unit variance per row.
  for (size_t i = 0; i < z_opt.rows; ++i) {
    double m = 0, v = 0;
    for (size_t j = 0; j < z_opt.cols; ++j) m += z_opt.at(i, j);
    m /= z_opt.cols;
    for (size_t j = 0; j < z_opt.cols; ++j) v += (z_opt.at(i, j) - m) * (z_opt.at(i, j) - m);
    CHECK_NEAR(m, 0.0, 1e-4);
    CHECK_NEAR(v / z_opt.cols, 1.0, 1e-2);
  }
}

static void test_all_opt_flag_combinations_match_reference() {
  const ModelConfig cfg = small_cfg();
  WorldModel model(cfg);
  SyntheticWorld world(cfg);
  Rng rng(9);
  auto seq = world.sample_sequence(10, 2, rng);
  std::vector<float> actions;
  for (size_t t = 0; t < 2; ++t)
    actions.insert(actions.end(), seq[t].action.begin(), seq[t].action.end());

  ExecContext ref = ExecContext::Reference();
  Matrix z_ref = model.encode(seq[0], ref);
  Matrix r_ref = model.rollout(z_ref, actions.data(), 2, ref);

  for (bool fast : {false, true}) {
    for (bool blas : {false, true}) {
      if (blas && !ExecContext::has_blas()) {
        Arena a;
        CHECK_THROWS(ExecContext::Optimized(a, nullptr, OptFlags{fast, true}));
        continue;
      }
      for (unsigned threads : {1u, 3u}) {
        Arena arena;
        ThreadPool pool(threads);
        ExecContext opt = ExecContext::Optimized(arena, &pool, OptFlags{fast, blas});
        Matrix z = model.encode(seq[0], opt);
        Matrix r = model.rollout(z, actions.data(), 2, opt);
        for (size_t i = 0; i < z_ref.size(); ++i) CHECK_NEAR(z_ref.data[i], z.data[i], 2e-4);
        for (size_t i = 0; i < r_ref.size(); ++i) CHECK_NEAR(r_ref.data[i], r.data[i], 2e-3);
      }
    }
  }
}

static void test_blas_training_matches_builtin() {
  if (!ExecContext::has_blas()) return;
  const ModelConfig cfg = small_cfg();
  TrainOptions opt;
  opt.steps = 15;
  opt.batch_size = 16;
  opt.lr = 0.05f;
  WorldModel a(cfg), b(cfg);
  train_shard(a, opt);
  opt.use_blas = true;
  train_shard(b, opt);
  for (size_t i = 0; i < a.dyn_weight().vec().size(); ++i)
    CHECK_NEAR(a.dyn_weight().vec()[i], b.dyn_weight().vec()[i], 5e-3);
}

static void test_modality_subsets() {
  const ModelConfig cfg = small_cfg();
  WorldModel model(cfg);
  SyntheticWorld world(cfg);
  ExecContext ctx = ExecContext::Reference();

  for (uint8_t mask : {uint8_t(kText), uint8_t(kImage), uint8_t(kAudio),
                       uint8_t(kText | kAudio), uint8_t(kAllModalities)}) {
    Rng rng(5);
    auto seq = world.sample_sequence(4, 1, rng, mask);
    Matrix z = model.encode(seq[0], ctx);
    CHECK(z.rows == 4 && z.cols == cfg.d_model);
    for (size_t i = 0; i < z.size(); ++i) CHECK(std::isfinite(z.data[i]));
  }

  Batch empty;
  empty.n = 1;
  empty.modalities = 0;
  CHECK_THROWS(model.encode(empty, ctx));

  Batch bad;
  bad.n = 2;
  bad.modalities = kText;
  bad.tokens.assign(3, 0);  // wrong size
  CHECK_THROWS(model.encode(bad, ctx));
}

static void test_training_reduces_loss_and_beats_copy() {
  const ModelConfig cfg = small_cfg();
  WorldModel model(cfg);
  const EvalResult before = evaluate(model, 128, 3);

  TrainOptions opt;
  opt.steps = 150;
  opt.batch_size = 32;
  opt.lr = 0.05f;
  opt.threads = 2;
  const TrainResult tr = train_shard(model, opt);
  CHECK(tr.last_loss < tr.first_loss * 0.8);

  const EvalResult after = evaluate(model, 128, 3);
  CHECK(after.model_mse < before.model_mse);
  CHECK(after.model_mse < after.copy_mse);
  CHECK(after.rollout_mse < after.rollout_copy_mse);
  CHECK(after.horizon == 3);
}

static void test_training_is_deterministic_across_modes_and_threads() {
  const ModelConfig cfg = small_cfg();
  TrainOptions opt;
  opt.steps = 10;
  opt.batch_size = 8;
  opt.lr = 0.05f;

  WorldModel a(cfg), b(cfg), c(cfg);
  opt.mode = ExecMode::kOptimized;
  opt.threads = 1;
  train_shard(a, opt);
  opt.threads = 4;
  train_shard(b, opt);
  opt.mode = ExecMode::kReference;
  train_shard(c, opt);

  for (size_t i = 0; i < a.dyn_weight().vec().size(); ++i) {
    CHECK_NEAR(a.dyn_weight().vec()[i], b.dyn_weight().vec()[i], 1e-5);
    CHECK_NEAR(a.dyn_weight().vec()[i], c.dyn_weight().vec()[i], 1e-3);
  }

  // Different shards must see different data.
  WorldModel d(cfg);
  opt.mode = ExecMode::kOptimized;
  opt.threads = 1;
  opt.shard_index = 1;
  train_shard(d, opt);
  double diff = 0;
  for (size_t i = 0; i < a.dyn_weight().vec().size(); ++i)
    diff += std::fabs(a.dyn_weight().vec()[i] - d.dyn_weight().vec()[i]);
  CHECK(diff > 1e-3);
}

static void test_checkpoint_roundtrip_and_merge() {
  const ModelConfig cfg = small_cfg();
  WorldModel m(cfg);
  TrainOptions opt;
  opt.steps = 3;
  opt.batch_size = 4;
  train_shard(m, opt);

  const std::string p = tmp_path("a.ckpt");
  Checkpoint c = checkpoint_from_model(m, 100);
  save_checkpoint(p, c);
  Checkpoint r = load_checkpoint(p);
  CHECK(r.d_model == c.d_model && r.seed == c.seed && r.samples == 100);
  CHECK(r.weight == c.weight && r.bias == c.bias);

  WorldModel fresh(cfg);
  apply_checkpoint(r, fresh);
  CHECK(fresh.dyn_weight().vec() == m.dyn_weight().vec());

  // Weighted merge: 1/4 * A + 3/4 * B.
  Checkpoint a = c, b = c;
  a.samples = 100;
  b.samples = 300;
  for (auto& v : a.weight) v = 1.0f;
  for (auto& v : b.weight) v = 5.0f;
  for (auto& v : a.bias) v = 0.0f;
  for (auto& v : b.bias) v = 4.0f;
  Checkpoint merged = merge_checkpoints({a, b});
  CHECK(merged.samples == 400);
  CHECK_NEAR(merged.weight[0], 4.0, 1e-6);
  CHECK_NEAR(merged.bias[0], 3.0, 1e-6);

  // Hierarchical merge equals flat merge (needed for the reduction tree).
  Checkpoint c3 = c;
  c3.samples = 50;
  for (auto& v : c3.weight) v = -2.0f;
  Checkpoint flat = merge_checkpoints({a, b, c3});
  Checkpoint tree = merge_checkpoints({merge_checkpoints({a, b}), c3});
  for (size_t i = 0; i < flat.weight.size(); ++i)
    CHECK_NEAR(flat.weight[i], tree.weight[i], 1e-5);

  // Error handling.
  Checkpoint other = a;
  other.seed = 999;
  CHECK_THROWS(merge_checkpoints({a, other}));
  CHECK_THROWS(merge_checkpoints({}));
  CHECK_THROWS(load_checkpoint(tmp_path("does_not_exist.ckpt")));
  ModelConfig cfg2 = cfg;
  cfg2.d_model = 16;
  WorldModel wrong(cfg2);
  CHECK_THROWS(apply_checkpoint(r, wrong));
  std::remove(p.c_str());
}

static void test_profiler() {
  Profiler& p = Profiler::instance();
  p.reset();
  p.set_enabled(true);
  const ModelConfig cfg = small_cfg();
  WorldModel m(cfg);
  TrainOptions opt;
  opt.steps = 2;
  opt.batch_size = 4;
  train_shard(m, opt);
  p.set_enabled(false);
  bool saw_step = false;
  for (const auto& e : p.snapshot()) saw_step = saw_step || (e.name == "train.step" && e.calls == 2);
  CHECK(saw_step);
  CHECK(p.report().find("matmul") != std::string::npos);
  CHECK(p.to_json().front() == '[');
  p.reset();
}

int main() {
  test_reference_and_optimized_agree();
  test_all_opt_flag_combinations_match_reference();
  test_blas_training_matches_builtin();
  test_modality_subsets();
  test_training_reduces_loss_and_beats_copy();
  test_training_is_deterministic_across_modes_and_threads();
  test_checkpoint_roundtrip_and_merge();
  test_profiler();
  TEST_MAIN_END();
}
