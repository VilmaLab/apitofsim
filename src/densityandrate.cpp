#include "densityandrate.h"

#include <iostream>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <map>
#include <array>
#include "consts.h"
#include "exceptions.h"
#include "operation_context.h"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/parallel_reduce.h>

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define ACTUALLY_GCC
#endif

using namespace std;

namespace
{
double atom_rate_prefactor(double inertia_moment)
{
  return consts::boltzmann * consts::boltzmann * inertia_moment / (consts::pi * consts::hbar * consts::hbar * consts::hbar);
}
} // namespace

// FUNCTIONS

static inline void throw_invalidate_max_energies(double fragmentation_energy, double energy_max, double energy_max_rate, int n_fragmentation, int m_max, int m_max_rate)
{
  throw ApiTofArgumentError([&](auto &msg)
  {
    msg << "Density of states must be evaluated up to an energy of max rate energy + fragmentation energy\n"
        << "DOS max energy: " << energy_max << "K (" << m_max << " bins)\n"
        << "Max rate energy: " << energy_max_rate << "K (" << m_max_rate << " bins)\n"
        << "Fragmentation energy: " << fragmentation_energy << "K (" << n_fragmentation << " bins)\n";
  });
}

void validate_max_energies(double fragmentation_energy, double energy_max, double energy_max_rate, double bin_width)
{
  int m_max = static_cast<int>(energy_max / bin_width);
  int m_max_rate = static_cast<int>(energy_max_rate / bin_width);
  int n_fragmentation = static_cast<int>(fragmentation_energy / bin_width);
  if (n_fragmentation + m_max_rate > m_max)
  {
    throw_invalidate_max_energies(fragmentation_energy, energy_max, energy_max_rate, n_fragmentation, m_max, m_max_rate);
  }
}

void validate_max_energies(int n_fragmentation, int m_max, int m_max_rate, double bin_width)
{
  if (n_fragmentation + m_max_rate > m_max)
  {
    double energy_max = m_max * bin_width;
    double energy_max_rate = m_max_rate * bin_width;
    double fragmentation_energy = n_fragmentation * bin_width;
    throw_invalidate_max_energies(fragmentation_energy, energy_max, energy_max_rate, n_fragmentation, m_max, m_max_rate);
  }
}

void validate_max_energies(double fragmentation_energy, int m_max, int m_max_rate, double bin_width)
{
  int n_fragmentation = static_cast<int>(fragmentation_energy / bin_width);
  if (n_fragmentation + m_max_rate > m_max)
  {
    double energy_max = m_max * bin_width;
    double energy_max_rate = m_max_rate * bin_width;
    throw_invalidate_max_energies(fragmentation_energy, energy_max, energy_max_rate, n_fragmentation, m_max, m_max_rate);
  }
}

double get_prefactor_k_total(double inertia_moment_1, double inertia_moment_2, const Eigen::Vector3d &rotations_1, const Eigen::Vector3d &rotations_2)
{
  using consts::pi;

  double rotations_product_1 = rotations_1[0] * rotations_1[1] * rotations_1[2];
  double rotations_product_2 = rotations_2[0] * rotations_2[1] * rotations_2[2];

  return 2.0 * consts::boltzmann * consts::boltzmann * (inertia_moment_1 + inertia_moment_2) / (pi * consts::hbar * consts::hbar * consts::hbar * pow(pow(rotations_product_1, 1.0 / 3) + pow(rotations_product_2, 1.0 / 3), 1.5));
}


double get_final_rate_k_total(const Eigen::ArrayXd &k0, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m, int n_fragmentation)
{
  // Integrate over all rotation energies
  double normalization = 0.0;
#pragma omp simd reduction(+ : normalization)
  for (int i = 0; i <= n_fragmentation + m; i++)
  {
    double rotational_energy = bin_width * (i + 0.5);
    normalization += rho_0[n_fragmentation + m - i] * sqrt(rotational_energy);
  }

  double integral = 0.0;
  // Cycle over integral differential
#pragma omp simd reduction(+ : integral)
  for (int i = 0; i <= m; i++)
  {
    double rotational_energy = bin_width * (i + 0.5);
    integral += rho_0[n_fragmentation + m - i] * sqrt(rotational_energy) * k0[m - i];
  }

  return integral / normalization;
}

Eigen::ArrayXd compute_mesh_linear(double bin_width, int m_max_rate, OperationContext &operation)
{
  Eigen::ArrayXd mesh(m_max_rate);
  double prefix = 0.0;
  double integral = 0.0;
  for (int m = 0; m < m_max_rate && operation.checkpoint(); ++m)
  {
    const double rotational_energy_sqrt = sqrt(bin_width * (m + 0.5));
    // M[m] - M[m-1] = h * sum(s[0:m]) + h/2 * s[m].
    integral += bin_width * (prefix + (0.5 * rotational_energy_sqrt));
    prefix += rotational_energy_sqrt;
    mesh[m] = integral;
  }
  return mesh;
}

Eigen::ArrayXd compute_mesh(double bin_width, int m_max_rate)
{
  Eigen::ArrayXd mesh = Eigen::ArrayXd::Zero(m_max_rate);
  // It's tempting to use simd collapse(2) here, but this make things significantly slower
  for (int i = 0; i < m_max_rate; i++) // rotational energy
  {
    double rotational_energy_sqrt = sqrt(bin_width * (i + 0.5));
#pragma omp simd
    for (int j = 0; j < m_max_rate - i; j++)
    {
      double translational_energy = bin_width * (j + 0.5);
      mesh[i + j] += translational_energy * rotational_energy_sqrt;
    }
  }
  return mesh;
}

Eigen::ArrayXd compute_mesh_rearranged_presqrt(double bin_width, int m_max_rate)
{
  // A version Without precomputing these, is consistently slower than anything else
  Eigen::ArrayXd rot_energy_sqrts = Eigen::ArrayXd(m_max_rate);
#pragma omp simd
  for (int i = 0; i < m_max_rate; i++)
  {
    rot_energy_sqrts[i] = sqrt(bin_width * (i + 0.5));
  }
  Eigen::ArrayXd mesh = Eigen::ArrayXd(m_max_rate);
  for (int i_p_j = 0; i_p_j < m_max_rate; i_p_j++)
  {
    double mesh_i_p_j = 0;
#pragma omp simd reduction(+ : mesh_i_p_j)
    for (int j = 0; j <= i_p_j; j++)
    {
      int i = i_p_j - j;
      double rotational_energy_sqrt = rot_energy_sqrts[i];
      double translational_energy = bin_width * (j + 0.5);
      mesh_i_p_j += translational_energy * rotational_energy_sqrt;
    }
    mesh[i_p_j] = mesh_i_p_j;
  }
  return mesh;
}

Eigen::ArrayXd compute_mesh_tbb(double bin_width, int m_max_rate, OperationContext &operation)
{
  Eigen::ArrayXd identity = Eigen::ArrayXd::Zero(m_max_rate);
  return oneapi::tbb::parallel_reduce(
    oneapi::tbb::blocked_range<int>(0, m_max_rate),
    identity,
    [=, &operation](const oneapi::tbb::blocked_range<int> &range, Eigen::ArrayXd mesh)
  {
    for (int i = range.begin(); i != range.end() && operation.checkpoint(); ++i)
    {
      double rotational_energy_sqrt = sqrt(bin_width * (i + 0.5));
#pragma omp simd
      for (int j = 0; j < m_max_rate - i; j++)
      {
        double translational_energy = bin_width * (j + 0.5);
        mesh[i + j] += translational_energy * rotational_energy_sqrt;
      }
    }
    return mesh;
  },
    [](Eigen::ArrayXd left, const Eigen::ArrayXd &right)
  {
    return (left + right).eval();
  },
    operation.tbb_context());
}

Eigen::ArrayXd compute_mesh_rearranged_presqrt_tbb(double bin_width, int m_max_rate, OperationContext &operation)
{
  Eigen::ArrayXd rot_energy_sqrts = Eigen::ArrayXd(m_max_rate);
  oneapi::tbb::parallel_for(
    oneapi::tbb::blocked_range<int>(0, m_max_rate),
    [=, &rot_energy_sqrts, &operation](const oneapi::tbb::blocked_range<int> &range)
  {
    for (int i = range.begin(); i != range.end() && operation.checkpoint(); ++i)
    {
      rot_energy_sqrts[i] = sqrt(bin_width * (i + 0.5));
    }
  },
    operation.tbb_context());
  Eigen::ArrayXd mesh = Eigen::ArrayXd(m_max_rate);
  oneapi::tbb::parallel_for(
    oneapi::tbb::blocked_range<int>(0, m_max_rate),
    [=, &mesh, &rot_energy_sqrts, &operation](const oneapi::tbb::blocked_range<int> &range)
  {
    for (int i_p_j = range.begin(); i_p_j != range.end() && operation.checkpoint(); ++i_p_j)
    {
      double mesh_i_p_j = 0;
#pragma omp simd reduction(+ : mesh_i_p_j)
      for (int j = 0; j <= i_p_j; j++)
      {
        int i = i_p_j - j;
        double rotational_energy_sqrt = rot_energy_sqrts[i];
        double translational_energy = bin_width * (j + 0.5);
        mesh_i_p_j += translational_energy * rotational_energy_sqrt;
      }
      mesh[i_p_j] = mesh_i_p_j;
    }
  },
    operation.tbb_context());
  return mesh;
}

void compute_k_total(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, double inertia_moment_2, const Eigen::Vector3d &rotations_1, const Eigen::Vector3d &rotations_2, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy)
{
  double prefactor = get_prefactor_k_total(inertia_moment_1, inertia_moment_2, rotations_1, rotations_2);
  int n_fragmentation = static_cast<int>(fragmentation_energy / bin_width);
  for (int m = 0; m < m_max_rate; m++)
  {
    double density_cluster = rho_0[n_fragmentation + m];
    //  Compute double integral
    double integral = 0.0;
    for (int i = 0; i <= m; i++) // rotational energy
    {
      double rotational_energy_sqrt = sqrt(bin_width * (i + 0.5));
#pragma omp simd reduction(+ : integral)
      for (int j = 0; j <= m - i; j++)
      {
        double translational_energy = bin_width * (j + 0.5);
        integral += translational_energy * rotational_energy_sqrt * rho_comb[m - i - j];
      }
    }

    k0[m] = prefactor / density_cluster * integral * bin_width * bin_width;
    k_rate[m] = get_final_rate_k_total(k0, rho_0, bin_width, m, n_fragmentation);
  }
}

void compute_k_total_atom(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy)
{
  double prefactor = atom_rate_prefactor(inertia_moment_1);
  int n_fragmentation = static_cast<int>(fragmentation_energy / bin_width);
  double prefix = 0.0;
  double integral = 0.0;
  for (int m = 0; m < m_max_rate; m++)
  {
    double density_cluster = rho_0[n_fragmentation + m];
    integral += bin_width * (prefix + (0.5 * rho_comb[m]));
    prefix += rho_comb[m];

    k0[m] = prefactor / density_cluster * integral * bin_width;
    k_rate[m] = get_final_rate_k_total(k0, rho_0, bin_width, m, n_fragmentation);
  }
}


// Compute density of states from vector of frequencies neglecting the zero level energy
void compute_density_of_states_old(const Eigen::ArrayXd &frequencies, Eigen::Ref<Eigen::ArrayXd> rho, double energy_max, double bin_width)
{
  int i = 0;
  int m;
  double delta_energy;
  double energy;
  double E_m;
  int num_oscillators = frequencies.rows();

  int m_max = static_cast<int>(energy_max / bin_width);

  for (m = 0; m < m_max; m++)
  {
    rho[m] = 0.0;
  }

  if (num_oscillators == 0)
  {
    return;
  }

  Eigen::ArrayXd rho_new = Eigen::ArrayXd::Zero(m_max);

  int k_max = static_cast<int>(energy_max / frequencies[0]) + 1;

  for (int k = 0; k < k_max; k++)
  {
    energy = frequencies[0] * k;
    m = static_cast<int>(energy / bin_width);
    rho[m]++;
  }
  for (i = 1; i < num_oscillators; i++)
  {
    double frequency = frequencies[i];
    for (m = 0; m < m_max; m++)
    {
      rho_new[m] = 0.0;
      E_m = bin_width * (m + 0.5);
      k_max = static_cast<int>(E_m / frequency);
      for (int k = 0; k < k_max + 1; k++)
      {
        delta_energy = E_m - (frequency * k);
        rho_new[m] += rho[static_cast<int>(delta_energy / bin_width)];
      }
    }
    for (m = 0; m < m_max; m++)
    {
      rho[m] = rho_new[m];
    }
  }
  for (m = 0; m < m_max; m++)
  {
    rho[m] = rho[m] / bin_width;
  }
}

void compute_density_of_states(const Eigen::ArrayXd &frequencies, Eigen::Ref<Eigen::ArrayXd> rho, double energy_max, double bin_width)
{
  // This algorithm is Bayer-Swinehartt Algorithm 448
  // `Number of Multiply-Restricted Partitions`
  // https://dl.acm.org/doi/pdf/10.1145/362248.362275
  int i;
  int m;
  int num_oscillators = frequencies.rows();

  int m_max = static_cast<int>(energy_max / bin_width);

  for (m = 0; m < m_max; m++)
  {
    rho[m] = 0.0;
  }
  for (i = 0; i < num_oscillators; i++)
  {
    double frequency = frequencies[i];
    double frequency_bin_float = frequency / bin_width;
    if (frequency_bin_float >= m_max)
    {
      throw ApiTofArgumentError([&](auto &msg)
      {
        msg << "Vibrational temperature " << frequency << " K is outside the DOS histogram range [0, "
            << m_max * bin_width << " K). Increase energy_max.";
      });
    }
    int frequency_bin = static_cast<int>(frequency_bin_float);
    rho[frequency_bin]++;
    int frequency_shift = static_cast<int>(frequency_bin_float + 0.5);
#ifdef NDEBUG
    // Concurrent iterations must not read bins written by the same SIMD chunk.
    constexpr int simd_width = 4;
#pragma omp simd if (frequency_shift >= simd_width) safelen(simd_width)
#endif
    for (m = frequency_bin + 1; m < m_max; m++)
    {
      int prev_bin = m - frequency_shift;
      rho[m] += rho[prev_bin];
    }
  }
  for (m = 0; m < m_max; m++)
  {
    rho[m] = rho[m] / bin_width;
  }
}

// Combined frequencies of two products
Eigen::ArrayXd combine_frequencies(const Eigen::ArrayXd &frequencies_1, const Eigen::ArrayXd &frequencies_2)
{
  int len1 = frequencies_1.rows();
  int len2 = frequencies_2.rows();
  Eigen::ArrayXd frequencies_comb = Eigen::ArrayXd(len1 + len2);

  frequencies_comb.head(len1) = frequencies_1;
  frequencies_comb.tail(len2) = frequencies_2;
  return frequencies_comb;
}

void compute_combined_density_of_states(Eigen::Ref<Eigen::ArrayXd> rho_comb, const Eigen::ArrayXd &frequencies_1, const Eigen::ArrayXd &frequencies_2, double energy_max, double bin_width)
{
  auto frequencies_comb = combine_frequencies(frequencies_1, frequencies_2);
  compute_density_of_states(frequencies_comb, rho_comb, energy_max, bin_width);
}

namespace
{
void fill_product_dos(Eigen::Ref<Eigen::ArrayXd> rho, const MolecularProducts &products, double energy_max, double bin_width)
{
  compute_density_of_states(products.second.frequencies, rho, energy_max, bin_width);
}

void fill_product_dos(Eigen::Ref<Eigen::ArrayXd> rho, const AtomLikeProducts &products, double energy_max, double bin_width)
{
  (void)products;
  (void)energy_max;
  (void)bin_width;
  // An atom-like product has no density of states
  rho.setZero();
}

void fill_combined_dos(Eigen::Ref<Eigen::ArrayXd> rho, const MolecularProducts &products, double energy_max, double bin_width)
{
  compute_combined_density_of_states(rho, products.first.frequencies, products.second.frequencies, energy_max, bin_width);
}

void fill_combined_dos(Eigen::Ref<Eigen::ArrayXd> rho, const AtomLikeProducts &products, double energy_max, double bin_width)
{
  compute_density_of_states(products.first.frequencies, rho, energy_max, bin_width);
}
} // namespace

DensityResult compute_density_of_states_all(const MolecularCluster &cluster_0, const Products &products, double energy_max, double bin_width)
{
  OperationContext operation;
  int m_max = static_cast<int>(energy_max / bin_width);
  DensityResult rhos(m_max, 4);
  cout << endl
       << "Computing density of states of cluster, products and combined products..." << endl;
  operation.run([&]
  {
    oneapi::tbb::parallel_invoke(
      [&]
    {
      auto rhos0 = rhos.col(C0_ROW);
      compute_density_of_states(cluster_0.frequencies, rhos0, energy_max, bin_width);
    },
      [&]
    {
      auto rhos1 = rhos.col(C1_ROW);
      std::visit([&](const auto &products)
      {
        compute_density_of_states(products.first.frequencies, rhos1, energy_max, bin_width);
      }, products);
    },
      [&]
    {
      auto rhos2 = rhos.col(C2_ROW);
      std::visit([&](const auto &products)
      {
        fill_product_dos(rhos2, products, energy_max, bin_width);
      }, products);
    },
      [&]
    {
      auto rhos_comb = rhos.col(COMB_ROW);
      std::visit([&](const auto &products)
      {
        fill_combined_dos(rhos_comb, products, energy_max, bin_width);
      }, products);
    },
      operation.tbb_context());
  });
  cout << endl
       << "Done" << endl;
  return rhos;
}

Eigen::ArrayXXd compute_density_of_states_batch(std::vector<Eigen::ArrayXd> batch_frequencies, double energy_max, double bin_width, bool use_old_impl)
{
  OperationContext operation;
  int m_max = static_cast<int>(energy_max / bin_width);
  // Possibly a tiny bit of false sharing here
  Eigen::ArrayXXd result(m_max, batch_frequencies.size());
  operation.run([&]
  {
    oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(0, batch_frequencies.size()),
      [&](const oneapi::tbb::blocked_range<size_t> &range)
    {
      for (size_t i = range.begin(); i != range.end() && operation.checkpoint(); ++i)
      {
        if (use_old_impl)
        {
          compute_density_of_states_old(batch_frequencies[i], result.col(i), energy_max, bin_width);
        }
        else
        {
          compute_density_of_states(batch_frequencies[i], result.col(i), energy_max, bin_width);
        }
      }
    },
      operation.tbb_context());
  });
  return result;
}

namespace
{
// All sums retain the existing midpoint grid. Keeping them direct avoids FFT
// roundoff from high-energy DOS bins contaminating rates near threshold.
void convolve_rate_range(Eigen::Ref<const Eigen::ArrayXd> density, Eigen::Ref<const Eigen::ArrayXd> kernel, int first, Eigen::Ref<Eigen::ArrayXd> result, OperationContext &operation)
{
  for (Eigen::Index m = 0; m < result.size() && operation.checkpoint(); ++m)
  {
    const Eigen::Index energy = first + m;
    double integral = 0.0;
#pragma omp simd reduction(+ : integral)
    for (Eigen::Index i = 0; i <= energy; ++i)
    {
      integral += density[energy - i] * kernel[i];
    }
    result[m] = integral;
  }
}

Eigen::ArrayXd rate_rotation_weights(double bin_width, int size, OperationContext &operation)
{
  Eigen::ArrayXd weights(size);
  for (int i = 0; i < size && operation.checkpoint(); ++i)
  {
    weights[i] = sqrt(bin_width * (i + 0.5));
  }
  return weights;
}

void validate_rate_densities(double fragmentation_energy, Eigen::Index parent_size, Eigen::Index product_size, double bin_width, int size)
{
  if (!std::isfinite(fragmentation_energy) || fragmentation_energy < 0)
  {
    throw ApiTofArgumentError("Fragmentation energy must be finite and nonnegative");
  }
  if (fragmentation_energy / bin_width > static_cast<double>(parent_size))
  {
    throw ApiTofArgumentError("Parent DOS must cover the fragmentation energy");
  }
  validate_max_energies(fragmentation_energy, static_cast<int>(parent_size), size, bin_width);
  if (product_size < size)
  {
    throw ApiTofArgumentError("Product DOS must cover the rate histogram");
  }
}

void finish_cancelled_rate(Eigen::Ref<const Eigen::ArrayXd> numerator, Eigen::Ref<const Eigen::ArrayXd> denominator, double scale, Eigen::Ref<Eigen::ArrayXd> result, OperationContext &operation)
{
  for (Eigen::Index m = 0; m < result.size() && operation.checkpoint(); ++m)
  {
    if (!(denominator[m] > 0) || !std::isfinite(denominator[m]))
    {
      throw ApiTofArgumentError("Rate normalization must be finite and positive at every requested energy");
    }
    result[m] = scale * (numerator[m] / denominator[m]);
  }
}

struct RateNormalization
{
  const double *density;
  std::vector<int> shifts;
  Eigen::ArrayXd values;
};

struct RateNumerator
{
  const double *density;
  bool molecular;
  Eigen::ArrayXd values;
};

struct RatePlan
{
  size_t parent;
  size_t products;
  int shift;
  double scale;
};

// Total electronic energy of the two products

double products_electronic_energy(const Products &products)
{
  return std::visit([](const auto &products)
  {
    return products.first.electronic_energy + products.second.electronic_energy;
  }, products);
}
} // namespace

Eigen::ArrayXd
compute_k_total_full(const MolecularCluster &cluster_0, const Products &products, const ClusterDerived &derived_1, const ClusterDerived &derived_2, const DensityResult &rhos, double fragmentation_energy, double energy_max_rate, double bin_width)
{
  using namespace consts;
  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == 0)
  {
    fragmentation_energy = (products_electronic_energy(products) - cluster_0.electronic_energy) * hartK;
  }

  int m_max_rate = static_cast<int>(energy_max_rate / bin_width);
  validate_rate_densities(fragmentation_energy, rhos.rows(), rhos.rows(), bin_width, m_max_rate);
  const int shift = static_cast<int>(fragmentation_energy / bin_width);
  OperationContext operation;
  return operation.run([&]
  {
    const bool molecular = std::holds_alternative<MolecularProducts>(products);
    const auto weights = rate_rotation_weights(bin_width, shift + m_max_rate, operation);
    Eigen::ArrayXd kernel = compute_mesh_linear(bin_width, m_max_rate, operation);
    double scale = atom_rate_prefactor(derived_1.inertia_moment) * bin_width;
    if (molecular)
    {
      const auto &pair = std::get<MolecularProducts>(products);
      scale = get_prefactor_k_total(derived_1.inertia_moment, derived_2.inertia_moment, pair.first.rotations, pair.second.rotations) * bin_width * bin_width;
      const auto mesh = std::move(kernel);
      kernel = Eigen::ArrayXd(m_max_rate);
      convolve_rate_range(mesh, weights, 0, kernel, operation);
    }
    Eigen::ArrayXd numerator(m_max_rate);
    Eigen::ArrayXd denominator(m_max_rate);
    Eigen::ArrayXd result(m_max_rate);
    convolve_rate_range(rhos.col(COMB_ROW), kernel, 0, numerator, operation);
    convolve_rate_range(rhos.col(C0_ROW), weights, shift, denominator, operation);
    finish_cancelled_rate(numerator, denominator, scale, result, operation);
    return result;
  });
}

Eigen::ArrayXd precompute_mesh_impl(double energy_max_rate, double bin_width, MeshMode mesh_mode, OperationContext &operation)
{
  int m_max_rate = static_cast<int>(energy_max_rate / bin_width);
  if (mesh_mode == MeshMode::compute_mesh_linear)
  {
    return compute_mesh_linear(bin_width, m_max_rate, operation);
  }
  else if (mesh_mode == MeshMode::compute_mesh_single_threaded)
  {
    return compute_mesh(bin_width, m_max_rate);
  }
  else if (mesh_mode == MeshMode::compute_mesh_diagonal_single_threaded)
  {
    return compute_mesh_rearranged_presqrt(bin_width, m_max_rate);
  }
  else if (mesh_mode == MeshMode::compute_mesh_multithreaded)
  {
    return compute_mesh_tbb(bin_width, m_max_rate, operation);
  }
  else if (mesh_mode == MeshMode::compute_mesh_diagonal_multithreaded)
  {
    return compute_mesh_rearranged_presqrt_tbb(bin_width, m_max_rate, operation);
  }
  else
  {
    throw invalid_argument("mesh_mode must be be one of the compute_mesh* modes");
  }
}

Eigen::ArrayXd precompute_mesh(double energy_max_rate, double bin_width, MeshMode mesh_mode)
{
  OperationContext operation;
  return operation.run([&]
  {
    return precompute_mesh_impl(energy_max_rate, bin_width, mesh_mode, operation);
  });
}

Eigen::ArrayXXd compute_k_total_batch(const std::vector<KTotalInput> &batch_input, double energy_max_rate, double bin_width, const std::optional<Eigen::ArrayXd> &mesh, std::optional<std::function<void(size_t)>> progress_callback)
{
  OperationContext operation;
  const int size = static_cast<int>(energy_max_rate / bin_width);
  Eigen::ArrayXXd result(size, batch_input.size());
  if (batch_input.empty())
  {
    return result;
  }
  size_t completed = 0;
  std::mutex progress_mutex;
  operation.run([&]
  {
    // Identity is the start of a contiguous DOS view. Different-length views
    // with the same start have identical prefixes; validate each before reuse.
    std::map<const double *, size_t> parent_indices;
    std::array<std::map<const double *, size_t>, 2> product_indices;
    std::vector<RateNormalization> parents;
    std::vector<RateNumerator> products;
    std::vector<std::optional<RatePlan>> plans(batch_input.size());
    int weights_size = size;
    bool need_molecular = false;
    bool need_atom = false;
    for (size_t i = 0; i < batch_input.size() && operation.checkpoint(); ++i)
    {
      const auto &input = batch_input[i];
      validate_rate_densities(input.fragmentation_energy, input.rho_parent.size(), input.rho_comb.size(), bin_width, size);
      const bool molecular = std::holds_alternative<MolecularCluster>(input.cluster_2);
      if (molecular && !mesh)
      {
        continue; // Keep no_mesh as the original molecular reference algorithm.
      }
      need_molecular |= molecular;
      need_atom |= !molecular;
      const int shift = static_cast<int>(input.fragmentation_energy / bin_width);
      weights_size = std::max(weights_size, shift + size);
      const auto [parent, new_parent] = parent_indices.try_emplace(input.rho_parent.data(), parents.size());
      if (new_parent)
      {
        parents.push_back({input.rho_parent.data(), {}, {}});
      }
      parents[parent->second].shifts.push_back(shift);
      const auto [product, new_product] = product_indices[molecular].try_emplace(input.rho_comb.data(), products.size());
      if (new_product)
      {
        products.push_back({input.rho_comb.data(), molecular, Eigen::ArrayXd(size)});
      }
      const auto derived_1 = compute_derived(input.cluster_1);
      double scale = atom_rate_prefactor(derived_1.inertia_moment) * bin_width;
      if (molecular)
      {
        const auto &second = std::get<MolecularCluster>(input.cluster_2);
        const auto derived_2 = compute_derived(second);
        scale = get_prefactor_k_total(derived_1.inertia_moment, derived_2.inertia_moment, input.cluster_1.rotations, second.rotations) * bin_width * bin_width;
      }
      plans[i] = RatePlan{parent->second, product->second, shift, scale};
    }
    const auto weights = rate_rotation_weights(bin_width, (need_molecular || need_atom) ? weights_size : 0, operation);
    Eigen::ArrayXd molecular_kernel;
    Eigen::ArrayXd atom_kernel;
    // s * (rho_parent * k0) = C*h^2 * (s * mesh * rho_comb).
    // Thus the numerator is independent of the parent and fragmentation shift.
    if (need_molecular)
    {
      if (mesh->size() < size)
      {
        throw ApiTofArgumentError("Mesh must cover the rate histogram");
      }
      molecular_kernel.resize(size);
      convolve_rate_range(*mesh, weights, 0, molecular_kernel, operation);
    }
    if (need_atom)
    {
      // An explicitly supplied molecular mesh does not change atom-like rates.
      atom_kernel = compute_mesh_linear(bin_width, size, operation);
    }
    oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(0, parents.size() + products.size()),
      [&](const oneapi::tbb::blocked_range<size_t> &range)
    {
      for (size_t i = range.begin(); i != range.end() && operation.checkpoint(); ++i)
      {
        if (i < parents.size())
        {
          auto &parent = parents[i];
          std::ranges::sort(parent.shifts);
          const int first = parent.shifts.front();
          const int end = parent.shifts.back() + size;
          parent.values.resize(end - first);
          const Eigen::Map<const Eigen::ArrayXd> density(parent.density, end);
          int computed_end = first;
          for (const int shift : parent.shifts)
          {
            const int begin = std::max(shift, computed_end);
            const int count = shift + size - begin;
            // Evaluate the union of requested windows, including each bin once.
            convolve_rate_range(density, weights, begin, parent.values.segment(begin - first, count), operation);
            computed_end = shift + size;
          }
        }
        else
        {
          auto &product = products[i - parents.size()];
          const Eigen::Map<const Eigen::ArrayXd> density(product.density, size);
          convolve_rate_range(density, product.molecular ? molecular_kernel : atom_kernel, 0, product.values, operation);
        }
      }
    }, operation.tbb_context());
    oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(0, batch_input.size()),
      [&](const oneapi::tbb::blocked_range<size_t> &range)
    {
      Eigen::ArrayXd k0;
      for (size_t i = range.begin(); i != range.end() && operation.checkpoint(); ++i)
      {
        if (plans[i])
        {
          const auto &plan = *plans[i];
          const auto &parent = parents[plan.parent];
          finish_cancelled_rate(products[plan.products].values, parent.values.segment(plan.shift - parent.shifts.front(), size), plan.scale, result.col(i), operation);
        }
        else
        {
          const auto &input = batch_input[i];
          k0.resize(size);
          const auto &second = std::get<MolecularCluster>(input.cluster_2);
          const auto derived_1 = compute_derived(input.cluster_1);
          const auto derived_2 = compute_derived(second);
          compute_k_total(k0, result.col(i), derived_1.inertia_moment, derived_2.inertia_moment, input.cluster_1.rotations, second.rotations, input.rho_comb, input.rho_parent, bin_width, size, input.fragmentation_energy);
        }
        if (progress_callback && operation.should_continue())
        {
          std::scoped_lock lock(progress_mutex);
          if (!operation.should_continue())
          {
            break;
          }
          (*progress_callback)(++completed);
        }
      }
    }, operation.tbb_context());
  });
  return result;
}

Eigen::ArrayXXd compute_k_total_batch(const std::vector<KTotalInput> &batch_input, double energy_max_rate, double bin_width, MeshMode mesh_mode, std::optional<std::function<void(size_t)>> progress_callback)
{
  std::optional<Eigen::ArrayXd> mesh;
  if (mesh_mode != MeshMode::no_mesh && std::ranges::any_of(batch_input, [](const auto &input)
  { return std::holds_alternative<MolecularCluster>(input.cluster_2); }))
  {
    mesh = precompute_mesh(energy_max_rate, bin_width, mesh_mode);
  }
  return compute_k_total_batch(batch_input, energy_max_rate, bin_width, mesh, progress_callback);
}
