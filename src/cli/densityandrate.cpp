#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <Eigen/Dense>
#include "apitofsim.h"
#include "densityandrate.h"
#include "common_io.h"

using namespace std;

void write_on_file(char *filename, const Eigen::Ref<const Eigen::ArrayXd> x, const Eigen::Ref<const Eigen::ArrayXd> y, int m_max);

Eigen::ArrayXd read_frequencies(const char *filename);

struct ClusterInputs
{
  std::string file_frequencies;
  std::string file_rotations;
  std::string file_electronic_energy;

  // char file_density[150];
  // char file_density_comb[150];
  // An empty frequencies file marks an atom-like product, which by definition
  // has no rotational structure.
  ClusterSpec read(int nucleon_count) const
  {
    const Eigen::ArrayXd frequencies = read_frequencies(file_frequencies.data());
    const Eigen::Vector3d rotations = read_rotations(file_rotations.data());
    const double electronic_energy = read_electronic_energy(file_electronic_energy.data());

    if (frequencies.rows() == 0)
    {
      if (!rotations.isZero(0))
      {
        cout << "Atom-like products must have 0 rotations" << endl;
        exit(EXIT_FAILURE);
      }
      return AtomLikeCluster(nucleon_count, electronic_energy, 0);
    }
    return MolecularCluster(nucleon_count, electronic_energy, rotations, frequencies, 0);
  }
};

// MAIN
int main()
{
  double energy_max;
  double energy_max_rate;
  double bin_width;
  double fragmentation_energy;
  double P1;
  double T;
  std::string file_density_0;
  std::string file_density_1;
  std::string file_density_2;
  std::string file_density_comb;
  std::string file_rate_constant;

  debug_info_on_env();

  ClusterInputs cluster_files_0;
  ClusterInputs cluster_files_1;
  ClusterInputs cluster_files_2;
  int nucleon_count_0;
  int nucleon_count_1;
  int nucleon_count_2;

  // Use read_config to read all input fields
  read_config<int, double>(
    std::cin,
    nullptr, // title
    nullptr, // cluster_charge_sign
    &nucleon_count_0,
    &nucleon_count_1,
    &nucleon_count_2,
    &T,
    &P1,
    nullptr, // pressure_second
    nullptr, // L0
    nullptr, // Lsk
    nullptr, // L1
    nullptr, // L2
    nullptr, // L3
    nullptr, // V0
    nullptr, // V1
    nullptr, // V2
    nullptr, // V3
    nullptr, // V4
    nullptr, // N
    nullptr, // dc
    nullptr, // alpha_factor
    &fragmentation_energy,
    &energy_max,
    &energy_max_rate,
    &bin_width,
    nullptr, // R_gas
    nullptr, // m_gas
    nullptr, // ga
    nullptr, // dc_field
    nullptr, // ac_field
    nullptr, // radiofrequency
    nullptr, // r_quadrupole
    nullptr, // file_skimmer
    &cluster_files_0.file_frequencies,
    &cluster_files_1.file_frequencies,
    &cluster_files_2.file_frequencies,
    &cluster_files_0.file_rotations,
    &cluster_files_1.file_rotations,
    &cluster_files_2.file_rotations,
    &cluster_files_0.file_electronic_energy,
    &cluster_files_1.file_electronic_energy,
    &cluster_files_2.file_electronic_energy,
    &file_density_0,
    &file_density_1,
    &file_density_2,
    &file_density_comb,
    &file_rate_constant,
    nullptr, // file_probabilities
    nullptr, // N_iter
    nullptr, // M_iter
    nullptr, // resolution
    nullptr // tolerance
  );

  cout << std::setprecision(3);

  // printf("]\033[F\033[J%s:%3lld%% [",text,c);
  cout << "###" << endl;
  cout << "Reading inputs..." << endl;

  // Read cluster data
  const ClusterSpec spec_0 = cluster_files_0.read(nucleon_count_0);
  const ClusterSpec spec_1 = cluster_files_1.read(nucleon_count_1);
  const ClusterSpec spec_2 = cluster_files_2.read(nucleon_count_2);

  // Combine the frequencies of two products
  const int num_oscillators_comb = num_oscillators(spec_1) + num_oscillators(spec_2);

  if (num_oscillators(spec_1) > 0 && num_oscillators(spec_2) > 0)
  {
    if (num_oscillators(spec_0) - num_oscillators_comb != 6)
    {
      cout << "Number of frequencies wrong!!!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if (num_oscillators(spec_0) - num_oscillators_comb != 3)
    {
      cout << "Number of frequencies wrong!!!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  const MolecularCluster *cluster_0 = std::get_if<MolecularCluster>(&spec_0);
  const MolecularCluster *cluster_1 = std::get_if<MolecularCluster>(&spec_1);
  if (cluster_0 == nullptr || cluster_1 == nullptr)
  {
    cout << "Only the 2nd product may be atom-like!" << endl;
    exit(EXIT_FAILURE);
  }
  const Products products = make_products(*cluster_1, spec_2);

  const ClusterDerived derived_0 = compute_derived(*cluster_0);
  const ClusterDerived derived_1 = compute_derived(*cluster_1);
  const ClusterDerived derived_2 = compute_derived(spec_2);

  DensityResult rhos = compute_density_of_states_all(*cluster_0, products, energy_max, bin_width);

  cout << endl;
  cout << "Cluster vibrational modes: " << num_oscillators(spec_0) << endl;
  cout << "First product vibrational modes: " << num_oscillators(spec_1) << endl;
  cout << "Second product vibrational modes: " << num_oscillators(spec_2) << endl;
  cout << "Combined vibrational modes of two products: " << num_oscillators_comb << " (+ " << num_oscillators(spec_0) - num_oscillators_comb << " degrees of freedom, translational and rotational)" << endl;
  cout << "Energy resolution: " << std::scientific << bin_width << " K" << endl;

  cout << "Inertia moment of cluster : " << derived_0.inertia_moment << " Kg m^2" << endl;
  cout << "Inertia moment of first product: " << derived_1.inertia_moment << " Kg m^2" << endl;
  cout << "Inertia moment of second product: " << derived_2.inertia_moment << " Kg m^2" << endl;

  cout << "Mass of cluster: " << derived_0.mass << " Kg" << endl;
  cout << "Mass of first product: " << derived_1.mass << " Kg" << endl;
  cout << "Mass of second product: " << derived_2.mass << " Kg" << endl;

  cout << "Radius of cluster: " << derived_0.radius << " m" << endl;
  cout << "Radius of first product: " << derived_1.radius << " m" << endl;
  cout << "Radius of second product: " << derived_2.radius << " m" << endl;
  cout << endl;

  cout << endl
       << "Computing total fragmentation rate constant..." << endl;

  const Eigen::ArrayXd k_rate = compute_k_total_full(*cluster_0, products, derived_1, derived_2, rhos, fragmentation_energy, energy_max_rate, bin_width);

  int m_max_rate = static_cast<int>(energy_max_rate / bin_width);
  int m_max = static_cast<int>(energy_max / bin_width);
  auto energies = prepare_energies(bin_width, m_max);
  auto energies_rate = prepare_energies(bin_width, m_max_rate);

  cout << endl
       << "END OF COMPUTATION" << endl;
  // Write density of states on files
  cout << endl;
  cout << "OUTPUTS" << endl;
  write_on_file(file_density_0.data(), energies, rhos.col(C0_ROW), m_max);
  write_on_file(file_density_1.data(), energies, rhos.col(C1_ROW), m_max);
  write_on_file(file_density_2.data(), energies, rhos.col(C2_ROW), m_max);
  write_on_file(file_density_comb.data(), energies, rhos.col(COMB_ROW), m_max);
  write_on_file(file_rate_constant.data(), energies_rate, k_rate, m_max_rate);
  cout << "###" << endl;

  return 0;
}


void write_on_file(char *filename, const Eigen::Ref<const Eigen::ArrayXd> x, const Eigen::Ref<const Eigen::ArrayXd> y, int m_max)
{
  cout << "Writing output..." << endl;
  ofstream file;
  file.open(filename);
  file << scientific;
  for (int m = 0; m < m_max; m++)
  {
    file << x[m] << " " << y[m] << endl;
  }
  file.close();
  cout << "]\033[F\033[J" << filename << endl;
}


Eigen::ArrayXd read_frequencies(const char *filename)
{
  ifstream file;
  std::string garb;
  file.open(filename);

  // Count the number of frequencies
  int num_oscillators = 0;
  while (file >> garb)
  {
    num_oscillators++;
  }
  file.close();
  Eigen::ArrayXd frequencies = Eigen::ArrayXd(num_oscillators);

  // Save the frequencies on a vector
  file.open(filename);
  for (int i = 0; i < num_oscillators; i++)
  {
    file >> frequencies[i];
  }
  file.close();
  return frequencies;
}
