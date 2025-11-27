#include <Eigen/Dense>
#include "densityandrate_lib.h"

Eigen::ArrayXd dos_smoke()
{
  Eigen::ArrayXd temperatures(15);
  temperatures << 78.6292,
    156.596,
    179.502,
    663.952,
    773.544,
    944.931,
    1022.08,
    1164.52,
    1359.04,
    1492.53,
    2192.19,
    2501.24,
    4669.41,
    5204.39,
    5631.82;
  const double energy_max = 2.0e5;
  const double bin_width = 1.0;
  const int m_max = int(energy_max / bin_width);
  // Possibly a tiny bit of false sharing here
  Eigen::ArrayXd result(m_max);
  compute_density_of_states(temperatures, result, energy_max, bin_width);
  return result;
}

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> k_total_smoke()
{
#include "compute_k_total_data.h"
  Eigen::ArrayXd k_rate(m_max_rate);
  Eigen::ArrayXd k0(m_max_rate);
  compute_k_total(
    k0,
    k_rate,
    inertia_moment_1,
    inertia_moment_2,
    rotations_1,
    rotations_2,
    rho_comb,
    rho_0,
    bin_width,
    m_max_rate,
    fragmentation_energy);
  return std::make_tuple(k_rate, k0);
}
