#include "mmwm/checkpoint.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <unistd.h>

namespace mmwm {

namespace {

constexpr char kMagic[8] = {'M', 'M', 'W', 'M', 'C', 'K', 'P', 'T'};
constexpr uint32_t kVersion = 1;

template <typename T>
void put(std::ofstream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
void get(std::ifstream& f, T& v, const std::string& path) {
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!f) throw std::runtime_error("truncated checkpoint: " + path);
}

}  // namespace

void save_checkpoint(const std::string& path, const Checkpoint& c) {
  const size_t expect_w = static_cast<size_t>(c.d_model + c.action_dim) * c.d_model;
  if (c.weight.size() != expect_w || c.bias.size() != c.d_model)
    throw std::runtime_error("save_checkpoint: tensor sizes do not match header");

  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open for write: " + tmp);
    f.write(kMagic, sizeof kMagic);
    put(f, kVersion);
    put(f, c.d_model);
    put(f, c.action_dim);
    put(f, uint32_t{0});
    put(f, c.seed);
    put(f, c.samples);
    f.write(reinterpret_cast<const char*>(c.weight.data()),
            static_cast<std::streamsize>(c.weight.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(c.bias.data()),
            static_cast<std::streamsize>(c.bias.size() * sizeof(float)));
    f.flush();
    if (!f) {
      std::remove(tmp.c_str());
      throw std::runtime_error("write failed: " + tmp);
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    throw std::runtime_error("rename failed: " + tmp + " -> " + path);
  }
}

Checkpoint load_checkpoint(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open checkpoint: " + path);
  char magic[8];
  f.read(magic, sizeof magic);
  if (!f || std::memcmp(magic, kMagic, sizeof kMagic) != 0)
    throw std::runtime_error("bad checkpoint magic: " + path);
  uint32_t version = 0, reserved = 0;
  Checkpoint c;
  get(f, version, path);
  if (version != kVersion)
    throw std::runtime_error("unsupported checkpoint version in " + path);
  get(f, c.d_model, path);
  get(f, c.action_dim, path);
  get(f, reserved, path);
  get(f, c.seed, path);
  get(f, c.samples, path);
  // Guard against absurd headers before allocating.
  if (c.d_model == 0 || c.d_model > (1u << 16) || c.action_dim == 0 || c.action_dim > (1u << 16))
    throw std::runtime_error("implausible checkpoint dimensions: " + path);
  c.weight.resize(static_cast<size_t>(c.d_model + c.action_dim) * c.d_model);
  c.bias.resize(c.d_model);
  f.read(reinterpret_cast<char*>(c.weight.data()),
         static_cast<std::streamsize>(c.weight.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(c.bias.data()),
         static_cast<std::streamsize>(c.bias.size() * sizeof(float)));
  if (!f) throw std::runtime_error("truncated checkpoint: " + path);
  return c;
}

Checkpoint checkpoint_from_model(const WorldModel& m, uint64_t samples) {
  Checkpoint c;
  c.d_model = static_cast<uint32_t>(m.config().d_model);
  c.action_dim = static_cast<uint32_t>(m.config().action_dim);
  c.seed = m.config().seed;
  c.samples = samples;
  c.weight = m.dyn_weight().vec();
  c.bias = m.dyn_bias().vec();
  return c;
}

void apply_checkpoint(const Checkpoint& c, WorldModel& m) {
  const auto& cfg = m.config();
  if (c.d_model != cfg.d_model || c.action_dim != cfg.action_dim || c.seed != cfg.seed)
    throw std::runtime_error("checkpoint does not match model config (d_model/action_dim/seed)");
  m.dyn_weight().vec() = c.weight;
  m.dyn_bias().vec() = c.bias;
}

Checkpoint merge_checkpoints(const std::vector<Checkpoint>& parts) {
  if (parts.empty()) throw std::runtime_error("merge_checkpoints: no inputs");
  const Checkpoint& first = parts.front();
  uint64_t total = 0;
  for (const auto& p : parts) {
    if (p.d_model != first.d_model || p.action_dim != first.action_dim || p.seed != first.seed)
      throw std::runtime_error("merge_checkpoints: incompatible checkpoints");
    total += p.samples;
  }
  std::vector<double> w(first.weight.size(), 0.0), b(first.bias.size(), 0.0);
  for (const auto& p : parts) {
    // Zero-sample parts contribute nothing unless every part is empty.
    const double coef = total ? static_cast<double>(p.samples) / static_cast<double>(total)
                              : 1.0 / static_cast<double>(parts.size());
    for (size_t i = 0; i < w.size(); ++i) w[i] += coef * p.weight[i];
    for (size_t i = 0; i < b.size(); ++i) b[i] += coef * p.bias[i];
  }
  Checkpoint out;
  out.d_model = first.d_model;
  out.action_dim = first.action_dim;
  out.seed = first.seed;
  out.samples = total;
  out.weight.assign(w.begin(), w.end());
  out.bias.assign(b.begin(), b.end());
  return out;
}

}  // namespace mmwm
