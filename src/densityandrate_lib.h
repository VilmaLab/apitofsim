#include <iostream>
#include <optional>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <Eigen/Dense>
#include "consts.h"

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define ACTUALLY_GCC
#endif

using namespace std;

// LIST OF FUNCTIONS

typedef Eigen::Array<double, Eigen::Dynamic, 4> DensityResult;
const int C0_ROW = 0;
const int C1_ROW = 1;
const int C2_ROW = 2;
const int COMB_ROW = 3;

void compute_density_of_states(Eigen::ArrayXd &frequencies, Eigen::Ref<Eigen::ArrayXd> rho, double energy_max, double bin_width);
void compute_combined_density_of_states(Eigen::Ref<Eigen::ArrayXd> rho_comb, Eigen::ArrayXd &frequencies_1, Eigen::ArrayXd &frequencies_2, double energy_max, double bin_width);
Eigen::ArrayXd combine_frequencies(Eigen::ArrayXd &frequencies_1, Eigen::ArrayXd &frequencies_2);
Eigen::ArrayXd prepare_energies(double bin_width, int m_max);
void compute_k_total(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, double inertia_moment_2, Eigen::Vector3d &rotations_1, Eigen::Vector3d &rotations_2, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy);
void compute_k_total_atom(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy);

// TODO: Separate struct for atom-like products
struct ClusterData
{
  int atomic_mass;
  double electronic_energy;
  Eigen::Vector3d rotations;
  Eigen::ArrayXd frequencies;

  // Computed members
  double inertia_moment;
  double radius;
  double mass;

  ClusterData()
  {
  }

  ClusterData(int atomic_mass, double electronic_energy, Eigen::Vector3d rotations, Eigen::ArrayXd frequencies) : atomic_mass(atomic_mass), electronic_energy(electronic_energy), rotations(rotations), frequencies(frequencies)
  {
  }

  void validate()
  {
    if (this->is_atom_like_product() && !this->rotations.isZero(0))
    {
      cout << "Atom-like products must have " << endl;
      exit(EXIT_FAILURE);
    }
  }

  int num_oscillators()
  {
    return this->frequencies.rows();
  }

  bool is_atom_like_product()
  {
    return this->num_oscillators() == 0;
  }

  void compute_derived()
  {
    if (this->is_atom_like_product())
    {
      // No rotations, so can't calculate inertia moment/radius
      inertia_moment = 0;
      radius = 0;
      mass = protonMass * this->atomic_mass; // proton mass * nucleons
    }
    else
    {
      inertia_moment = compute_inertia(rotations);
      compute_mass_and_radius(inertia_moment, atomic_mass, mass, radius);
    }
  }
};

Eigen::ArrayXd
compute_k_total_full(ClusterData &cluster_0, ClusterData &cluster_1, ClusterData &cluster_2, DensityResult &rhos, double fragmentation_energy, double energy_max_rate, double bin_width);
DensityResult compute_density_of_states_all(ClusterData &cluster_0, ClusterData &cluster_1, ClusterData &cluster_2, double energy_max, double bin_width);


// FUNCTIONS


double get_prefactor_k_total(double inertia_moment_1, double inertia_moment_2, Eigen::Vector3d &rotations_1, Eigen::Vector3d &rotations_2) {
  using consts::pi;

  double rotations_product_1 = rotations_1[0] * rotations_1[1] * rotations_1[2];
  double rotations_product_2 = rotations_2[0] * rotations_2[1] * rotations_2[2];

  return 2.0 * kb * kb * (inertia_moment_1 + inertia_moment_2) / (pi * hbar * hbar * hbar * pow(pow(rotations_product_1, 1.0 / 3) + pow(rotations_product_2, 1.0 / 3), 1.5));
}


double get_final_rate_k_total(Eigen::ArrayXd &k0, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m, int n_fragmentation)
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

Eigen::ArrayXd compute_mesh(double bin_width, int m_max_rate)
{
  Eigen::ArrayXd mesh = Eigen::ArrayXd::Zero(m_max_rate);
#if !defined(ACTUALLY_GCC) || __GNUC__ > 13
#pragma omp simd collapse(2)
#endif
  for (int i = 0; i < m_max_rate; i++) // rotational energy
  {
    double rotational_energy_sqrt = sqrt(bin_width * (i + 0.5));
#if defined(ACTUALLY_GCC) && __GNUC__ <= 13
#pragma omp simd
#endif
    for (int j = 0; j < m_max_rate - i; j++)
    {
      double translational_energy = bin_width * (j + 0.5);
      mesh[i + j] += translational_energy * rotational_energy_sqrt;
    }
  }
  return mesh;
}

Eigen::ArrayXd compute_mesh_rearranged(double bin_width, int m_max_rate)
{
  Eigen::ArrayXd mesh = Eigen::ArrayXd(m_max_rate);
  for (int i_p_j = 0; i_p_j < m_max_rate; i_p_j++)
  {
    double mesh_i_p_j = 0;
#pragma omp simd reduction(+ : mesh_i_p_j)
    for (int j = 0; j < i_p_j; j++)
    {
      int i = i_p_j - j;
      double rotational_energy_sqrt = sqrt(bin_width * (i + 0.5));
      double translational_energy = bin_width * (j + 0.5);
      mesh_i_p_j += translational_energy * rotational_energy_sqrt;
    }
    mesh[i_p_j] = mesh_i_p_j;
  }
  return mesh;
}

void compute_k_total_mesh(Eigen::ArrayXd &k0, Eigen::ArrayXd &mesh, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, double inertia_moment_2, Eigen::Vector3d &rotations_1, Eigen::Vector3d &rotations_2, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy)
{
  double prefactor = get_prefactor_k_total(inertia_moment_1, inertia_moment_2, rotations_1, rotations_2);
  int n_fragmentation = int(fragmentation_energy / bin_width);
  for (int m = 0; m < m_max_rate; m++)
  {
    double density_cluster = rho_0[n_fragmentation + m];
    double integral = 0.0;
#pragma omp simd reduction(+ : integral)
    for (int i_p_j = 0; i_p_j <= m; i_p_j++)
    {
      integral += mesh[i_p_j] * rho_comb[m - i_p_j];
    }

    k0[m] = prefactor / density_cluster * integral * bin_width * bin_width;
    k_rate[m] = get_final_rate_k_total(k0, rho_0, bin_width, m, n_fragmentation);
  }
}


void compute_k_total(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, double inertia_moment_2, Eigen::Vector3d &rotations_1, Eigen::Vector3d &rotations_2, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy)
{
  double prefactor = get_prefactor_k_total(inertia_moment_1, inertia_moment_2, rotations_1, rotations_2);
  int n_fragmentation = int(fragmentation_energy / bin_width);
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
  using consts::pi;

  double prefactor = kb * kb * (inertia_moment_1) / (pi * hbar * hbar * hbar);
  int n_fragmentation = int(fragmentation_energy / bin_width);
  for (int m = 0; m < m_max_rate; m++)
  {
    double density_cluster = rho_0[n_fragmentation + m];
    double integral = 0.0;
#pragma omp simd reduction(+ : integral)
    for (int i = 0; i <= m; i++) // translational energy
    {
      double translational_energy = bin_width * (i + 0.5);
      integral += translational_energy * rho_comb[m - i];
    }

    k0[m] = prefactor / density_cluster * integral * bin_width;
    k_rate[m] = get_final_rate_k_total(k0, rho_0, bin_width, m, n_fragmentation);
  }
}


Eigen::ArrayXd prepare_energies(double bin_width, int m_max)
{
  Eigen::ArrayXd energies = Eigen::ArrayXd(m_max);
  for (int m = 0; m < m_max; m++)
  {
    energies[m] = bin_width * (m + 0.5);
  }
  return energies;
}


// Compute density of states from vector of frequencies neglecting the zero level energy
void compute_density_of_states_old(Eigen::ArrayXd &frequencies, Eigen::Ref<Eigen::ArrayXd> rho, double energy_max, double bin_width)
{
  int i = 0;
  int m;
  double delta_energy;
  double energy;
  double E_m;
  int num_oscillators = frequencies.rows();

  int m_max = int(energy_max / bin_width);

  for (m = 0; m < m_max; m++)
  {
    rho[m] = 0.0;
  }

  if (num_oscillators == 0)
  {
    return;
  }

  Eigen::ArrayXd rho_new = Eigen::ArrayXd::Zero(m_max);

  int k_max = int(energy_max / frequencies[0]) + 1;

  for (int k = 0; k < k_max; k++)
  {
    energy = frequencies[0] * k;
    m = int(energy / bin_width);
    rho[m]++;
  }
  for (i = 1; i < num_oscillators; i++)
  {
    double frequency = frequencies[i];
    for (m = 0; m < m_max; m++)
    {
      rho_new[m] = 0.0;
      E_m = bin_width * (m + 0.5);
      k_max = int(E_m / frequency);
      for (int k = 0; k < k_max + 1; k++)
      {
        delta_energy = E_m - frequency * k;
        rho_new[m] += rho[int(delta_energy / bin_width)];
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

void compute_density_of_states(Eigen::ArrayXd &frequencies, Eigen::Ref<Eigen::ArrayXd> rho, double energy_max, double bin_width)
{
  // This algorithm is Bayer-Swinehartt Algorithm 448
  // `Number of Multiply-Restricted Partitions`
  // https://dl.acm.org/doi/pdf/10.1145/362248.362275
  int i, m;
  int num_oscillators = frequencies.rows();

  int m_max = int(energy_max / bin_width);

  for (m = 0; m < m_max; m++)
  {
    rho[m] = 0.0;
  }
  for (i = 0; i < num_oscillators; i++)
  {
    double frequency = frequencies[i];
    double frequency_bin_float = frequency / bin_width;
    int frequency_bin = int(frequency_bin_float);
    rho[frequency_bin]++;
    int frequency_shift = int(frequency_bin_float + 0.5);
#ifdef NDEBUG
#pragma omp simd
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
Eigen::ArrayXd combine_frequencies(Eigen::ArrayXd &frequencies_1, Eigen::ArrayXd &frequencies_2)
{
  int len1 = frequencies_1.rows();
  int len2 = frequencies_2.rows();
  Eigen::ArrayXd frequencies_comb = Eigen::ArrayXd(len1 + len2);

  frequencies_comb.head(len1) = frequencies_1;
  frequencies_comb.tail(len2) = frequencies_2;
  return frequencies_comb;
}

void compute_combined_density_of_states(Eigen::Ref<Eigen::ArrayXd> rho_comb, Eigen::ArrayXd &frequencies_1, Eigen::ArrayXd &frequencies_2, double energy_max, double bin_width)
{
  auto frequencies_comb = combine_frequencies(frequencies_1, frequencies_2);
  compute_density_of_states(frequencies_comb, rho_comb, energy_max, bin_width);
}

DensityResult compute_density_of_states_all(ClusterData &cluster_0, ClusterData &cluster_1, ClusterData &cluster_2, double energy_max, double bin_width)
{
  int m_max = int(energy_max / bin_width);
  DensityResult rhos(m_max, 4);
  cout << endl
       << "Computing density of states of cluster, products and combined products..." << endl;
#pragma omp parallel sections
  {
#pragma omp section
    {
      cluster_0.compute_derived();
      auto rhos0 = rhos.col(C0_ROW);
      compute_density_of_states(cluster_0.frequencies, rhos0, energy_max, bin_width);
    }
#pragma omp section
    {
      cluster_1.compute_derived();
      auto rhos1 = rhos.col(C1_ROW);
      compute_density_of_states(cluster_1.frequencies, rhos1, energy_max, bin_width);
    }
#pragma omp section
    {
      cluster_2.compute_derived();
      auto rhos2 = rhos.col(C2_ROW);
      compute_density_of_states(cluster_2.frequencies, rhos2, energy_max, bin_width);
    }
#pragma omp section
    {
      auto rhos_comb = rhos.col(COMB_ROW);
      compute_combined_density_of_states(rhos_comb, cluster_1.frequencies, cluster_2.frequencies, energy_max, bin_width);
    }
  }
  cout << endl
       << "Done" << endl;
  return rhos;
}

Eigen::ArrayXXd compute_density_of_states_batch(std::vector<Eigen::ArrayXd> batch_frequencies, double energy_max, double bin_width, bool use_old_impl = false)
{
  int m_max = int(energy_max / bin_width);
  // Possibly a tiny bit of false sharing here
  Eigen::ArrayXXd result(m_max, batch_frequencies.size());
#pragma omp parallel for default(none) \
  firstprivate(energy_max, bin_width, batch_frequencies, use_old_impl) \
  shared(result)
  for (size_t i = 0; i < batch_frequencies.size(); i++)
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
  return result;
}

void compute_k_total_general(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, ClusterData &cluster_1, ClusterData &cluster_2, double fragmentation_energy, Eigen::Ref<Eigen::ArrayXd> rho_parent, Eigen::Ref<Eigen::ArrayXd> rho_comb, double bin_width, double m_max_rate, std::optional<Eigen::ArrayXd> mesh = std::nullopt)
{
  if (!cluster_1.is_atom_like_product() && !cluster_2.is_atom_like_product())
  {
    if (mesh) {
      // cout << "Generic products" << endl;
      compute_k_total_mesh(k0, *mesh, k_rate, cluster_1.inertia_moment, cluster_2.inertia_moment, cluster_1.rotations, cluster_2.rotations, rho_comb, rho_parent, bin_width, m_max_rate, fragmentation_energy);
    } else {
      compute_k_total(k0, k_rate, cluster_1.inertia_moment, cluster_2.inertia_moment, cluster_1.rotations, cluster_2.rotations, rho_comb, rho_parent, bin_width, m_max_rate, fragmentation_energy);
    }
  }
  else if (!cluster_1.is_atom_like_product())
  {
    // cout << "Atom-like product" << endl;
    compute_k_total_atom(k0, k_rate, cluster_1.inertia_moment, rho_comb, rho_parent, bin_width, m_max_rate, fragmentation_energy);
  }
  else
  {
    throw invalid_argument("Only the second product may be atom-like");
  }
}


struct FragmentationPathway
{
  ClusterData &parent;
  ClusterData &product1;
  ClusterData &product2;

  double fragmentation_energy_kelvin()
  {
    return (this->product1.electronic_energy + this->product2.electronic_energy - this->parent.electronic_energy) * consts::hartK;
  }
};


Eigen::ArrayXd
compute_k_total_full(ClusterData &cluster_0, ClusterData &cluster_1, ClusterData &cluster_2, DensityResult &rhos, double fragmentation_energy, double energy_max_rate, double bin_width)
{
  using namespace consts;
  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == 0)
  {
    fragmentation_energy = (cluster_1.electronic_energy + cluster_2.electronic_energy - cluster_0.electronic_energy) * hartK;
  }

  int m_max_rate = int(energy_max_rate / bin_width);
  Eigen::ArrayXd k0 = Eigen::ArrayXd(m_max_rate);
  Eigen::ArrayXd k_rate = Eigen::ArrayXd(m_max_rate);

  auto rho_0 = rhos.col(C0_ROW);
  auto rho_comb = rhos.col(COMB_ROW);
  compute_k_total_general(k0, k_rate, cluster_1, cluster_2, fragmentation_energy, rho_0, rho_comb, bin_width, m_max_rate);
  return k_rate;
}

struct KTotalInput
{
  ClusterData cluster_1;
  ClusterData cluster_2;
  double fragmentation_energy;
  Eigen::Ref<Eigen::ArrayXd> rho_parent;
  Eigen::Ref<Eigen::ArrayXd> rho_comb;
};

Eigen::ArrayXXd compute_k_total_batch(std::vector<KTotalInput> batch_input, double energy_max_rate, double bin_width, int mesh_mode = 0)
{
  int m_max_rate = int(energy_max_rate / bin_width);
  Eigen::ArrayXXd k_rate = Eigen::ArrayXXd(m_max_rate, batch_input.size());
  std::optional<Eigen::ArrayXd> mesh;
  if (mesh_mode == 0) {
    mesh = std::nullopt;
  } else if (mesh_mode == 1) {
    mesh = compute_mesh(bin_width, m_max_rate);
  } else if (mesh_mode == 2) {
    mesh = compute_mesh_rearranged(bin_width, m_max_rate);
  } else {
    throw invalid_argument("mesh_mode must be 0, 1 or 2");
  }
#pragma omp parallel default(none) \
  firstprivate(batch_input, bin_width, m_max_rate, mesh) \
  shared(k_rate)
  {
    Eigen::ArrayXd k0 = Eigen::ArrayXd(m_max_rate);
#pragma omp for
    for (size_t i = 0; i < batch_input.size(); i++)
    {
      auto input = batch_input[i];

      input.cluster_1.compute_derived();
      input.cluster_2.compute_derived();
      compute_k_total_general(k0, k_rate.col(i), input.cluster_1, input.cluster_2, input.fragmentation_energy, input.rho_parent, input.rho_comb, bin_width, m_max_rate, mesh);
    }
  }
  return k_rate;
}
