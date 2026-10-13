// Latency benchmark for the encode + rollout inference path.
//
// Runs the naive reference path and a ladder of optimisation stages, verifies
// every stage against the reference numerically, and reports the p50/p95 of
// each so the contribution of every stage is visible:
//
//   reference            naive ijk matmul, heap allocation per intermediate, 1 thread
//   v1-tiled-arena-mt    register-tiled matmul + arena + thread pool (std::tanh, unfused)
//   +fast-tanh           vectorisable rational tanh (portable; reported as "optimized")
//   +blas-sgemm          vendor sgemm (Accelerate/cblas) when compiled in
//
// Emits JSON consumed by scripts/compare_bench.py.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
  s.p95 = ms[std::min(ms.size() - 1, static_cast<size_t>(static_cast<double>(ms.size()) * 0.95))];
  for (double v : ms) s.mean += v;
  s.mean /= static_cast<double>(ms.size());
  return s;
}

struct Stage {
  std::string name;
  std::unique_ptr<ExecContext> ctx;
  Stats stats;
  double max_abs_diff = 0.0;
};

std::string stats_json(const Stats& s) {
  char buf[160];
  std::snprintf(buf, sizeof buf, "{\"p50_ms\": %.4f, \"p95_ms\": %.4f, \"mean_ms\": %.4f}",
                s.p50, s.p95, s.mean);
  return buf;
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

  Arena arena(16u << 20);
  ThreadPool pool(threads);

  std::vector<Stage> stages;
  auto add = [&](const char* name, std::unique_ptr<ExecContext> ctx) {
    stages.push_back(Stage{name, std::move(ctx), {}, 0.0});
  };
  add("reference", std::make_unique<ExecContext>(ExecContext::Reference()));
  add("v1-tiled-arena-mt", std::make_unique<ExecContext>(
          ExecContext::Optimized(arena, &pool, OptFlags{false, false})));
  add("+fast-tanh", std::make_unique<ExecContext>(
          ExecContext::Optimized(arena, &pool, OptFlags{true, false})));
  if (ExecContext::has_blas()) {
    add("+blas-sgemm", std::make_unique<ExecContext>(
            ExecContext::Optimized(arena, &pool, OptFlags{true, true})));
  }
  constexpr size_t kV1 = 1, kOptimized = 2;  // indices into `stages`

  auto run = [&](ExecContext& ctx, std::vector<float>* keep) {
    ctx.reset();
    Matrix z = model.encode(seq[0], ctx);
    Matrix traj = model.rollout(z, actions.data(), horizon, ctx);
    if (keep) keep->assign(traj.data, traj.data + traj.size());
  };

  // Correctness first: every stage must agree with the reference.
  std::vector<float> ref_out, out;
  run(*stages[0].ctx, &ref_out);
  for (auto& st : stages) {
    run(*st.ctx, &out);
    for (size_t i = 0; i < ref_out.size(); ++i)
      st.max_abs_diff = std::max(st.max_abs_diff, static_cast<double>(std::fabs(ref_out[i] - out[i])));
    if (!(st.max_abs_diff < 1e-2)) {
      std::fprintf(stderr, "stage %s disagrees with reference: max |diff| = %g\n",
                   st.name.c_str(), st.max_abs_diff);
      return 1;
    }
  }

  // Interleaved timing: one run of each stage per round, so frequency /
  // thermal drift hits every stage equally.
  std::vector<std::vector<double>> samples(stages.size());
  for (auto& st : stages)
    for (int w = 0; w < 3; ++w) run(*st.ctx, nullptr);
  for (size_t it = 0; it < iters; ++it) {
    for (size_t s = 0; s < stages.size(); ++s) {
      const auto t0 = Clock::now();
      run(*stages[s].ctx, nullptr);
      samples[s].push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
  }
  for (size_t s = 0; s < stages.size(); ++s) stages[s].stats = summarize(samples[s]);

  const Stats& ref = stages[0].stats;
  const Stats& v1 = stages[kV1].stats;
  const Stats& opt = stages[kOptimized].stats;

  std::fprintf(stderr, "\n%-20s %10s %10s %14s %14s\n", "stage", "p50 ms", "p95 ms",
               "vs reference", "vs v1");
  for (const auto& st : stages) {
    std::fprintf(stderr, "%-20s %10.2f %10.2f %13.1f%% %13.1f%%\n", st.name.c_str(), st.stats.p50,
                 st.stats.p95, 100.0 * (1.0 - st.stats.p50 / ref.p50),
                 100.0 * (1.0 - st.stats.p50 / v1.p50));
  }
  std::fprintf(stderr, "(threads=%u batch=%zu horizon=%zu d_model=%zu iters=%zu)\n", pool.size(),
               batch, horizon, d_model, iters);

  std::string json;
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "{\n  \"config\": {\"batch\": %zu, \"horizon\": %zu, \"d_model\": %zu, "
                "\"iters\": %zu, \"threads\": %u, \"blas\": %s},\n",
                batch, horizon, d_model, iters, pool.size(),
                ExecContext::has_blas() ? "true" : "false");
  json += buf;
  json += "  \"reference\": " + stats_json(ref) + ",\n";
  json += "  \"v1\": " + stats_json(v1) + ",\n";
  json += "  \"optimized\": " + stats_json(opt) + ",\n";
  std::snprintf(buf, sizeof buf,
                "  \"speedup\": %.3f,\n  \"latency_reduction_pct\": %.2f,\n"
                "  \"reduction_vs_v1_pct\": %.2f,\n  \"max_abs_diff\": %.3g,\n",
                ref.p50 / opt.p50, 100.0 * (1.0 - opt.p50 / ref.p50),
                100.0 * (1.0 - opt.p50 / v1.p50), stages[kOptimized].max_abs_diff);
  json += buf;
  json += "  \"stages\": [\n";
  for (size_t s = 0; s < stages.size(); ++s) {
    std::snprintf(buf, sizeof buf, "    {\"name\": \"%s\", \"p50_ms\": %.4f, \"p95_ms\": %.4f, "
                  "\"max_abs_diff\": %.3g}%s\n", stages[s].name.c_str(), stages[s].stats.p50,
                  stages[s].stats.p95, stages[s].max_abs_diff, s + 1 < stages.size() ? "," : "");
    json += buf;
  }
  json += "  ]\n}\n";
  std::fputs(json.c_str(), stdout);
  if (!out_path.empty()) {
    FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) {
      std::perror("fopen");
      return 1;
    }
    std::fputs(json.c_str(), f);
    std::fclose(f);
  }

  if (profile) {  // per-scope breakdown of the portable optimised path, on stderr
    Profiler::instance().reset();
    Profiler::instance().set_enabled(true);
    for (size_t i = 0; i < std::min<size_t>(iters, 10); ++i) run(*stages[kOptimized].ctx, nullptr);
    Profiler::instance().set_enabled(false);
    std::fprintf(stderr, "\n%s path, per-scope wall time on the dispatching thread:\n%s",
                 stages[kOptimized].name.c_str(), Profiler::instance().report().c_str());
  }
  return 0;
}
