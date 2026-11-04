#pragma once

// Phase-scaled fundamental matrices: state is (position, velocity / angular frequency).
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>

namespace rf_detail
{
using Matrix = std::array<double, 4>;
using State = std::array<double, 2>;
constexpr double period = 2 * std::numbers::pi;
inline Matrix multiply(const Matrix &a, const Matrix &b)
{
  return {(a[0] * b[0]) + (a[1] * b[2]), (a[0] * b[1]) + (a[1] * b[3]),
          (a[2] * b[0]) + (a[3] * b[2]), (a[2] * b[1]) + (a[3] * b[3])};
}
inline State matvec(const Matrix &a, const State &b)
{
  return {(a[0] * b[0]) + (a[1] * b[1]), (a[2] * b[0]) + (a[3] * b[1])};
}
inline Matrix inverse(const Matrix &a)
{
  const double d = (a[0] * a[3]) - (a[1] * a[2]);
  if (!std::isfinite(d) || std::abs(d) < 1e-10)
    throw std::runtime_error("Ill-conditioned Mathieu basis");
  return {a[3] / d, -a[1] / d, -a[2] / d, a[0] / d};
}
inline Matrix power(Matrix a, long long n)
{
  if (n < 0)
  {
    a = inverse(a);
    n = -n;
  }
  Matrix out{1, 0, 0, 1};
  while (n != 0)
  {
    if ((n & 1) != 0)
      out = multiply(out, a);
    n >>= 1;
    if (n != 0)
      a = multiply(a, a);
  }
  return out;
}
class Basis
{
public:
  static constexpr int default_cells = 4096;
  static constexpr int format_version = 1;

private:
  int cells;
  double h, ac, dc;
  struct Node
  {
    std::array<Matrix, 2> value, derivative;
  };
  std::vector<Node> nodes;
  Matrix derivative(const Matrix &f, double phase, int axis) const
  {
    const double k = (axis == 0 ? 1 : -1) * ((ac * std::cos(phase)) - dc);
    return {f[2], f[3], k * f[0], k * f[1]};
  }

public:
  Basis(double ac_scaled, double dc_scaled, int resolution = default_cells) : cells(resolution), h(0), ac(ac_scaled), dc(dc_scaled)
  {
    if (cells < 16 || cells > 65536 || !std::isfinite(ac) || !std::isfinite(dc))
      throw std::invalid_argument("Invalid RF basis parameters (grid must have 16 to 65536 cells)");
    h = period / cells;
    nodes.resize(cells + 1);
    for (int axis = 0; axis < 2; ++axis)
    {
      Matrix f{1, 0, 0, 1};
      for (int i = 0; i <= cells; ++i)
      {
        const double t = i * h;
        auto k1 = derivative(f, t, axis);
        nodes[i].value[axis] = f;
        nodes[i].derivative[axis] = k1;
        if (i == cells)
          break;
        f = step(f, t, h, axis);
      }
    }
    validate();
  }
  // Persist values and derivatives, in node/axis/component order. Loading checks
  // the same local accuracy and conditioning invariants as construction.
  Basis(double ac_scaled, double dc_scaled, const Eigen::ArrayXd &data)
      : cells(0), h(0), ac(ac_scaled), dc(dc_scaled)
  {
    if (data.size() < Eigen::Index{17} * 16 || data.size() > Eigen::Index{65537} * 16 || data.size() % 16 != 0 ||
        !data.isFinite().all() || !std::isfinite(ac) || !std::isfinite(dc))
      throw std::invalid_argument("Invalid persisted RF basis");
    cells = static_cast<int>(data.size() / 16) - 1;
    h = period / cells;
    nodes.resize(cells + 1);
    Eigen::Index index = 0;
    for (auto &node : nodes)
      for (auto *part : {&node.value, &node.derivative})
        for (auto &axis : *part)
          for (auto &value : axis)
            value = data[index++];
    validate();
  }
  Eigen::ArrayXd data() const
  {
    Eigen::ArrayXd result((cells + 1) * 16);
    Eigen::Index index = 0;
    for (const auto &node : nodes)
      for (const auto *part : {&node.value, &node.derivative})
        for (const auto &axis : *part)
          for (auto value : axis)
            result[index++] = value;
    return result;
  }
  bool matches(double ac_scaled, double dc_scaled) const
  {
    return ac == ac_scaled && dc == dc_scaled;
  }

private:
  Matrix step(const Matrix &f, double phase, double dt, int axis) const
  {
    auto shifted = [&](const Matrix &k, double scale)
    {
      Matrix out{};
      for (int j = 0; j < 4; ++j)
        out[j] = f[j] + (scale * k[j]);
      return out;
    };
    auto k1 = derivative(f, phase, axis);
    auto k2 = derivative(shifted(k1, dt / 2), phase + (dt / 2), axis);
    auto k3 = derivative(shifted(k2, dt / 2), phase + (dt / 2), axis);
    auto k4 = derivative(shifted(k3, dt), phase + dt, axis);
    Matrix out{};
    for (int j = 0; j < 4; ++j)
      out[j] = f[j] + (dt / 6 * (k1[j] + (2 * k2[j]) + (2 * k3[j]) + k4[j]));
    return out;
  }
  void validate() const
  {
    auto close = [](const Matrix &actual, const Matrix &expected, double tolerance)
    {
      for (int j = 0; j < 4; ++j)
        if (!std::isfinite(actual[j]) || !std::isfinite(expected[j]) ||
            std::abs(actual[j] - expected[j]) > tolerance * (1 + std::abs(expected[j])))
          throw std::invalid_argument("RF basis fails accuracy validation");
    };
    auto norm = [](const Matrix &m)
    {
      return std::max(std::abs(m[0]) + std::abs(m[1]), std::abs(m[2]) + std::abs(m[3]));
    };
    for (int axis = 0; axis < 2; ++axis)
    {
      close(nodes.front().value[axis], {1, 0, 0, 1}, 0);
      for (int i = 0; i <= cells; ++i)
      {
        const auto &f = nodes[i].value[axis];
        if (norm(f) * norm(inverse(f)) > 1e8 || std::abs((f[0] * f[3]) - (f[1] * f[2]) - 1) > 1e-7)
          throw std::invalid_argument("RF basis exceeds conditioning or Wronskian tolerance");
        close(nodes[i].derivative[axis], derivative(f, i * h, axis), 1e-13);
        if (i == cells)
          break;
        auto half = step(f, i * h, h / 2, axis);
        close(at((i + 0.5) * h, axis), half, 1e-10);
        close(nodes[i + 1].value[axis], step(half, (i + 0.5) * h, h / 2, axis), 1e-12);
      }
    }
  }
  Matrix at(double phase, int axis) const
  {
    const double x = phase / h;
    const int i = std::clamp(static_cast<int>(x), 0, cells - 1);
    const double u = x - i;
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double a = (2 * u3) - (3 * u2) + 1;
    const double b = (u3 - (2 * u2) + u) * h;
    const double c = (-2 * u3) + (3 * u2);
    const double d = (u3 - u2) * h;
    Matrix out{};
    for (int j = 0; j < 4; ++j)
      out[j] = (a * nodes[i].value[axis][j]) + (b * nodes[i].derivative[axis][j]) + (c * nodes[i + 1].value[axis][j]) + (d * nodes[i + 1].derivative[axis][j]);
    return out;
  }

public:
  struct Anchor
  {
    const Basis &basis;
    double phase;
    std::array<State, 2> coefficients;
    std::array<State, 2> evaluate(double delta) const
    {
      const double total = phase + delta;
      if (!std::isfinite(total) || std::abs(total / period) > 1e12)
        throw std::invalid_argument("RF propagation phase is out of range");
      const auto turns = static_cast<long long>(std::floor(total / period));
      const double end_phase = total - (static_cast<double>(turns) * period);
      std::array<State, 2> out{};
      for (int axis = 0; axis < 2; ++axis)
      {
        State c = coefficients[axis];
        if (turns != 0)
          c = matvec(power(basis.nodes.back().value[axis], turns), c);
        out[axis] = matvec(basis.at(end_phase, axis), c);
        if (!std::isfinite(out[axis][0]) || !std::isfinite(out[axis][1]))
          throw std::runtime_error("Nonfinite Mathieu propagation");
      }
      return out;
    }
  };
  Anchor anchor(double absolute_phase, const std::array<State, 2> &state) const
  {
    if (!std::isfinite(absolute_phase) || std::abs(absolute_phase / period) > 1e12)
      throw std::invalid_argument("RF anchor phase is out of range");
    double phase = std::fmod(absolute_phase, period);
    if (phase < 0)
      phase += period;
    return {*this, phase, {matvec(inverse(at(phase, 0)), state[0]), matvec(inverse(at(phase, 1)), state[1])}};
  }
};
} // namespace rf_detail

using RFBasis = rf_detail::Basis;
using RFBasisPtr = std::shared_ptr<const RFBasis>;
