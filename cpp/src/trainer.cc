#include "mmwm/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>

#include "mmwm/profiler.h"
#include "mmwm/rng.h"
#include "mmwm/synthetic_world.h"

namespace mmwm {

namespace {

double mse(const Matrix& a, const Matrix& b) {
  const std::span<const float> x = a.as_span(), y = b.as_span();
  double acc = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double d = static_cast<double>(x[i]) - y[i];
    acc += d * d;
  }
  return acc / static_cast<double>(x.size());
}

struct Runtime {
  Arena arena{4u << 20};
  std::unique_ptr<ThreadPool> pool;
  ExecContext ctx;

  Runtime(ExecMode mode, unsigned threads)
      : pool(mode == ExecMode::kOptimized ? std::make_unique<ThreadPool>(threads) : nullptr),
        ctx(mode == ExecMode::kOptimized ? ExecContext::Optimized(arena, pool.get())
                                         : ExecContext::Reference()) {}
};

}  // namespace

TrainResult train_shard(WorldModel& model, const TrainOptions& opt) {
  const ModelConfig& cfg = model.config();
  const size_t d = cfg.d_model, A = cfg.action_dim, B = opt.batch_size;
  SyntheticWorld world(cfg);
  Rng rng(Rng::mix(Rng::mix(cfg.seed, 0x5452414EULL /* "TRAN" */ + opt.round), opt.shard_index));
  Runtime rt(opt.mode, opt.threads);
  ExecContext& ctx = rt.ctx;

  const size_t window = std::max<size_t>(1, opt.steps / 10);
  double first_sum = 0.0, last_sum = 0.0;
  size_t first_n = 0, last_n = 0;

  const auto t0 = std::chrono::steady_clock::now();
  for (size_t step = 0; step < opt.steps; ++step) {
    MMWM_PROFILE("train.step");
    std::vector<Batch> seq = world.sample_sequence(B, 1, rng);
    ctx.reset();

    Matrix z_t = model.encode(seq[0], ctx);
    Matrix z_next = model.encode(seq[1], ctx);  // frozen target

    // Forward through dynamics, keeping tanh activations for the backward pass.
    Matrix x = ctx.alloc(B, d + A);
    for (size_t i = 0; i < B; ++i) {
      std::memcpy(x.row(i), z_t.row(i), d * sizeof(float));
      std::memcpy(x.row(i) + d, seq[0].action.data() + i * A, A * sizeof(float));
    }
    Matrix u = ctx.alloc(B, d);
    Matrix w = model.dyn_weight().view();
    ctx.matmul(x, w, u);
    const float* bias = model.dyn_bias().vec().data();

    // loss = mean((z + tanh(u) - z_next)^2);  du = dloss/du
    Matrix du = ctx.alloc(B, d);
    double loss = 0.0;
    const double inv_count = 1.0 / static_cast<double>(B * d);
    for (size_t i = 0; i < B; ++i) {
      for (size_t j = 0; j < d; ++j) {
        const float t = std::tanh(u.at(i, j) + bias[j]);
        const float diff = z_t.at(i, j) + t - z_next.at(i, j);
        loss += static_cast<double>(diff) * diff;
        du.at(i, j) = static_cast<float>(2.0 * diff * inv_count) * (1.0f - t * t);
      }
    }
    loss *= inv_count;

    Matrix dW = ctx.alloc(d + A, d);
    ctx.matmul_atb(x, du, dW);

    float* wp = model.dyn_weight().vec().data();
    float* bp = model.dyn_bias().vec().data();
    // Gradients above are averaged over B*d, so lr is on a comparable scale
    // regardless of batch size; multiply by d to keep per-feature scale.
    const float lr = opt.lr * static_cast<float>(d);
    for (size_t i = 0; i < dW.size(); ++i) wp[i] -= lr * dW.data[i];
    for (size_t i = 0; i < B; ++i)
      for (size_t j = 0; j < d; ++j) bp[j] -= lr * du.at(i, j);

    if (step < window) {
      first_sum += loss;
      ++first_n;
    }
    if (step + window >= opt.steps) {
      last_sum += loss;
      ++last_n;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();

  TrainResult r;
  r.first_loss = first_n ? first_sum / static_cast<double>(first_n) : 0.0;
  r.last_loss = last_n ? last_sum / static_cast<double>(last_n) : 0.0;
  r.samples = static_cast<uint64_t>(opt.steps) * B;
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  return r;
}

EvalResult evaluate(const WorldModel& model, size_t samples, size_t horizon, unsigned threads) {
  const ModelConfig& cfg = model.config();
  const size_t d = cfg.d_model, A = cfg.action_dim;
  SyntheticWorld world(cfg);
  Rng rng(Rng::mix(cfg.seed, 0x4556414CULL));  // "EVAL": fixed stream
  Runtime rt(ExecMode::kOptimized, threads);
  ExecContext& ctx = rt.ctx;

  const size_t H = std::max<size_t>(1, horizon);
  const size_t chunk = std::min<size_t>(samples, 64);
  EvalResult out;
  out.horizon = H;

  double one = 0.0, one_copy = 0.0, roll = 0.0, roll_copy = 0.0;
  size_t done = 0, chunks = 0;
  while (done < samples) {
    const size_t n = std::min(chunk, samples - done);
    std::vector<Batch> seq = world.sample_sequence(n, H, rng);
    ctx.reset();

    std::vector<Matrix> z(H + 1);
    for (size_t t = 0; t <= H; ++t) z[t] = model.encode(seq[t], ctx);

    Matrix z1 = model.step(z[0], seq[0].action.data(), ctx);
    one += mse(z1, z[1]);
    one_copy += mse(z[0], z[1]);

    std::vector<float> actions(H * n * A);
    for (size_t t = 0; t < H; ++t)
      std::memcpy(actions.data() + t * n * A, seq[t].action.data(), n * A * sizeof(float));
    Matrix traj = model.rollout(z[0], actions.data(), H, ctx);
    double r = 0.0, rc = 0.0;
    for (size_t t = 0; t < H; ++t) {
      Matrix pred{traj.row(t * n), n, d};
      r += mse(pred, z[t + 1]);
      rc += mse(z[0], z[t + 1]);
    }
    roll += r / static_cast<double>(H);
    roll_copy += rc / static_cast<double>(H);

    done += n;
    ++chunks;
  }
  const double c = static_cast<double>(std::max<size_t>(1, chunks));
  out.model_mse = one / c;
  out.copy_mse = one_copy / c;
  out.rollout_mse = roll / c;
  out.rollout_copy_mse = roll_copy / c;
  out.samples = samples;
  return out;
}

}  // namespace mmwm
