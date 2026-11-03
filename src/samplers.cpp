#include "vib_index.h"
#include <algorithm>
#include <utility>
#include <Eigen/Dense>
#include "consts.h"


// Total collision frequency
double coll_freq(double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  if (v > 0)
    return 2.0 * pi * n * R * R * ((0.5 * ((mobility_gas / v) + v) * erf(sqrt(0.5 * mobility_gas_inv) * v)) + (sqrt(0.5 * mobility_gas / pi) * exp(-0.5 * mobility_gas_inv * v * v)));
  else
    return 2.0 * sqrt(2.0 * pi * mobility_gas) * n * R * R;
}

// Collision frequency on angle theta
double coll_freq_theta(double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  double costheta = cos(theta);
  double sintheta = sin(theta);
  return pi * n * R * R * sintheta * ((sqrt(mobility_gas * 2.0 / pi) * exp(-0.5 * mobility_gas_inv * v * v * costheta * costheta)) + (v * costheta * (erf(sqrt(0.5 * mobility_gas_inv) * v * costheta) + 1)));
}


// Collision frequency on angle theta and gas velocity
double coll_freq_theta_u(double u, double theta, double n, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  double costheta = cos(theta);
  double sintheta = sin(theta);
  return 2.0 * pi * n * R * R * sqrt(0.5 * mobility_gas_inv / pi) * (u + (v * costheta)) * exp(-0.5 * mobility_gas_inv * u * u) * sintheta;
}

// Distribution of angle theta
double distr_theta(double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  return coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v) / coll_freq(n, mobility_gas, mobility_gas_inv, R, v);
}

// Distribution of gas velocity
double distr_u(double u, double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  return coll_freq_theta_u(u, theta, n, mobility_gas_inv, R, v) / coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v);
}

VibEnergyIndex::VibEnergyIndex(VibEnergyMode mode, Eigen::ArrayXd values)
    : mode(mode), data(std::move(values))
{
  if (mode == VibEnergyMode::dos_only && data.size() != 0 && data.allFinite() && data[0] >= 0 &&
      std::is_sorted(data.data(), data.data() + data.size()))
  {
    return;
  }
  if (mode == VibEnergyMode::group_dos && data.size() >= 4 && std::has_single_bit(static_cast<size_t>(data.size())) &&
      !data.isNaN().any() && (data < std::numeric_limits<double>::infinity()).all())
  {
    for (Eigen::Index node = 1; node < data.size() / 2; ++node)
    {
      bool has_child = std::isfinite(data[2 * node]) || std::isfinite(data[(2 * node) + 1]);
      if (has_child != std::isfinite(data[node]))
      {
        throw ApiTofArgumentError("DOS tree contains inconsistent empty nodes");
      }
    }
    return;
  }
  throw ApiTofArgumentError("Invalid prepared vibrational DOS index");
}

void VibEnergyIndex::check_length(int bins) const
{
  size_t expected = mode == VibEnergyMode::dos_only ? static_cast<size_t>(bins) : 2 * std::bit_ceil(static_cast<size_t>(bins));
  if (std::cmp_not_equal(data.size(), expected))
  {
    throw ApiTofArgumentError("Vibrational DOS index length does not match histogram");
  }
}
