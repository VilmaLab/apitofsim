#include "densityandrate.h"
#include "benchmark_control.h"

#include <string_view>

int main(int argc, char **argv)
{
  const bool shared = argc > 1 && std::string_view(argv[1]) == "shared";
  return run_benchmark([&]
  {
    constexpr int size = 7500;
    constexpr int shift = 1000;
    constexpr int pathways = 16;
    constexpr double bin_width = 4.0;
    const MolecularCluster product(10, 0.0, Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::ArrayXd::Constant(1, 2.0), 0);
    const size_t densities = shared ? 1 : pathways;
    const std::vector<Eigen::ArrayXd> parents(densities, Eigen::ArrayXd::LinSpaced(size + shift, 1.0, 10.0).pow(8));
    const std::vector<Eigen::ArrayXd> combined(densities, Eigen::ArrayXd::LinSpaced(size, 0.0, 8.0).pow(6));
    std::vector<KTotalInput> inputs;
    for (int i = 0; i < pathways; ++i)
    {
      const size_t density = shared ? 0 : i;
      inputs.push_back({product, product, shift * bin_width, parents[density], combined[density]});
    }
    const auto rates = compute_k_total_batch(inputs, size * bin_width, bin_width);
    return rates.isFinite().all() ? 0 : 1;
  });
}
