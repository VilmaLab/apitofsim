#include <Eigen/Dense>
#include "densityandrate_lib.h"

int main()
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
  double energy_max = 2.0e5;
  double bin_width = 1.0;
  int m_max = int(energy_max / bin_width);
  // Possibly a tiny bit of false sharing here
  Eigen::ArrayXd result(m_max);
  compute_density_of_states(temperatures, result, energy_max, bin_width);
}
