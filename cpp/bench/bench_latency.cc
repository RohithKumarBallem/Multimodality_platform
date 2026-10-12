// Latency benchmark: reference (naive) vs optimized execution of the
// encode + rollout inference path. Emits JSON consumed by
// scripts/compare_bench.py. Also verifies both paths agree numerically.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mmwm/profiler.h"
#include "mmwm/synthetic_world.h"
#include "mmwm/world_model.h"

using namespace mmwm;
using Clock = std::chrono::steady_clock;

namespace {

struct Stats {
  double p50 = 0, p95 = 0, mean = 0;
};

Stats summarize(std::vector<double> ms) {
  std::ranges::sort(ms);
  Stats s;
  s.p50 = ms[ms.size() / 2];
  s.p95 = ms[std::min(ms.size() - 1, static_cast<size_t>(ms.size() * 0.95))];
  for (double v : ms) s.mean += v;
  s.mean /= static_cast<double>(ms.size());
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  size_t batch = 64, horizon = 8, iters = 30, d_model = 256;
  unsigned threads = 0;
  std::string out_path;
  bool profile = false;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--batch")) batch = std::strtoull(next("--batch"), nullptr, 10);
    else if (!std::strcmp(argv[i], "--horizon")) horizon = std::strtoull(next("--horizon"), nullptr, 10);
    else if (!std::strcmp(argv[i], "--iters")) iters = std::strtoull(next("--iters"), nullptr, 10);
    else if (!std::strcmp(argv[i], "--d-model")) d_model = std::strtoull(next("--d-model"), nullptr, 10);
    else if (!std::strcmp(argv[i], "--threads")) threads = static_cast<unsigned>(std::strtoul(next("--threads"), nullptr, 10));
    else if (!std::strcmp(argv[i], "--out")) out_path = next("--out");
    else if (!std::strcmp(argv[i], "--profile")) profile = true;
    else {
      std::fprintf(stderr, "unknown flag %s\n", argv[i]);
      return 2;
    }
  }

  ModelConfig cfg;
  cfg.d_model = d_model;
  cfg.vocab = 1024;
  cfg.text_len = 32;
  cfg.image_patches = 64;
  cfg.patch_dim = 192;
  cfg.audio_frames = 50;
  cfg.audio_dim = 80;
  cfg.action_dim = 16;
  cfg.seed = 7;

  WorldModel model(cfg);
  SyntheticWorld world(cfg);
  Rng rng(1);
  auto seq = world.sample_sequence(batch, horizon, rng);
  std::vector<float> actions;
  for (size_t t = 0; t < horizon; ++t)
    actions.insert(actions.end(), seq[t].action.begin(), seq[t].action.end());

  Arena arena(8u << 20);
  ThreadPool pool(threads);
  ExecContext ref = ExecContext::Reference();
  ExecContext opt = ExecContext::Optimized(arena, &pool);

  auto run = [&](ExecContext& ctx, std::vector<float>* keep) {
    ctx.reset();
    Matrix z = model.encode(seq[0], ctx);
    Matrix traj = model.rollout(z, actions.data(), horizon, ctx);
    if (keep) keep->assign(traj.data, traj.data + traj.size());
  };

  std::vector<float> out_ref, out_opt;
  run(ref, &out_ref);
  run(opt, &out_opt);
  double max_abs = 0.0;
  for (size_t i = 0; i < out_ref.size(); ++i)
    max_abs = std::max(max_abs, static_cast<double>(std::fabs(out_ref[i] - out_opt[i])));
  if (!(max_abs < 1e-2)) {
    std::fprintf(stderr, "reference/optimized mismatch: max |diff| = %g\n", max_abs);
    return 1;
  }

  auto time_mode = [&](ExecContext& ctx) {
    for (int w = 0; w < 3; ++w) run(ctx, nullptr);  // warm-up
    std::vector<double> ms;
    for (size_t i = 0; i < iters; ++i) {
      const auto t0 = Clock::now();
      run(ctx, nullptr);
      ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return summarize(ms);
  };

  // Interleave so thermal/frequency drift hits both modes equally.
  const Stats s_ref = time_mode(ref);
  const Stats s_opt = time_mode(opt);
  const double reduction = 100.0 * (1.0 - s_opt.p50 / s_ref.p50);

  char buf[1024];
  std::snprintf(buf, sizeof buf,
                "{\n"
                "  \"config\": {\"batch\": %zu, \"horizon\": %zu, \"d_model\": %zu, "
                "\"iters\": %zu, \"threads\": %u},\n"
                "  \"reference\": {\"p50_ms\": %.4f, \"p95_ms\": %.4f, \"mean_ms\": %.4f},\n"
                "  \"optimized\": {\"p50_ms\": %.4f, \"p95_ms\": %.4f, \"mean_ms\": %.4f},\n"
                "  \"speedup\": %.3f,\n"
                "  \"latency_reduction_pct\": %.2f,\n"
                "  \"max_abs_diff\": %.3g\n"
                "}\n",
                batch, horizon, d_model, iters, pool.size(), s_ref.p50, s_ref.p95, s_ref.mean,
                s_opt.p50, s_opt.p95, s_opt.mean, s_ref.p50 / s_opt.p50, reduction, max_abs);
  std::fputs(buf, stdout);
  if (!out_path.empty()) {
    FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) {
      std::perror("fopen");
      return 1;
    }
    std::fputs(buf, f);
    std::fclose(f);
  }

  if (profile) {  // per-scope breakdown of the optimized path, on stderr
    Profiler::instance().reset();
    Profiler::instance().set_enabled(true);
    for (size_t i = 0; i < std::min<size_t>(iters, 10); ++i) run(opt, nullptr);
    Profiler::instance().set_enabled(false);
    std::fprintf(stderr, "\nOptimized path, per-scope wall time on the dispatching thread:\n%s",
                 Profiler::instance().report().c_str());
  }
  return 0;
}
