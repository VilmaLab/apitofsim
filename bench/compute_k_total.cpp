#include "densityandrate_lib.h"
#include "compute_k_total_data.h"


int main()
{
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
}
