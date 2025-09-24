#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>
#include "densityandrate_lib.h"

void write_on_file(char *filename, const Eigen::Ref<const Eigen::ArrayXd> x, const Eigen::Ref<const Eigen::ArrayXd> y, int m_max);

Eigen::ArrayXd read_frequencies(char *filename);
Eigen::Array3d read_rotations(char *filename);
double read_electronic_energy(char *filename);

struct ClusterInputs
{
  char file_frequencies[150];
  char file_rotations[150];
  char file_electronic_energy[150];

  // char file_density[150];
  // char file_density_comb[150];
  void read_into(ClusterData &cluster_data)
  {
    // Read frequencies
    cluster_data.frequencies = read_frequencies(file_frequencies);

    // Read rotational constants
    cluster_data.rotations = read_rotations(file_rotations);

    // Read electronic energies
    cluster_data.electronic_energy = read_electronic_energy(file_electronic_energy);
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
  char file_density_0[150];
  char file_density_1[150];
  char file_density_2[150];
  char file_density_comb[150];
  char file_rate_constant[150];

  ClusterInputs cluster_files_0;
  ClusterInputs cluster_files_1;
  ClusterInputs cluster_files_2;
  ClusterData cluster_0 = ClusterData();
  ClusterData cluster_1 = ClusterData();
  ClusterData cluster_2 = ClusterData();

  // Use read_config to read all input fields
  read_config(
    std::cin,
    nullptr, // title
    nullptr, // cluster_charge_sign
    &cluster_0.atomic_mass,
    &cluster_1.atomic_mass,
    &cluster_2.atomic_mass,
    &T,
    &P1,
    (double *)nullptr, // pressure_second
    nullptr, // L0
    (double *)nullptr, // Lsk
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
    cluster_files_0.file_frequencies,
    cluster_files_1.file_frequencies,
    cluster_files_2.file_frequencies,
    cluster_files_0.file_rotations,
    cluster_files_1.file_rotations,
    cluster_files_2.file_rotations,
    cluster_files_0.file_electronic_energy,
    cluster_files_1.file_electronic_energy,
    cluster_files_2.file_electronic_energy,
    file_density_0,
    file_density_1,
    file_density_2,
    file_density_comb,
    file_rate_constant,
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
  cluster_files_0.read_into(cluster_0);
  cluster_files_1.read_into(cluster_1);
  cluster_files_2.read_into(cluster_2);
  cluster_0.validate();
  cluster_1.validate();
  cluster_2.validate();

  // Combine the frequencies of two products
  int num_oscillators_comb = cluster_1.num_oscillators() + cluster_2.num_oscillators();

  if (cluster_1.num_oscillators() > 0 && cluster_2.num_oscillators() > 0)
  {
    if (cluster_0.num_oscillators() - num_oscillators_comb != 6)
    {
      cout << "Number of frequencies wrong!!!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if (cluster_0.num_oscillators() - num_oscillators_comb != 3)
    {
      cout << "Number of frequencies wrong!!!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  if (cluster_0.is_atom_like_product() || cluster_1.is_atom_like_product())
  {
    cout << "Only the 2nd product may be atom-like!" << endl;
    exit(EXIT_FAILURE);
  }

  DensityResult rhos = compute_density_of_states_all(cluster_0, cluster_1, cluster_2, energy_max, bin_width);

  cout << endl;
  cout << "Cluster vibrational modes: " << cluster_0.num_oscillators() << endl;
  cout << "First product vibrational modes: " << cluster_1.num_oscillators() << endl;
  cout << "Second product vibrational modes: " << cluster_2.num_oscillators() << endl;
  cout << "Combined vibrational modes of two products: " << num_oscillators_comb << " (+ " << cluster_0.num_oscillators() - num_oscillators_comb << " degrees of freedom, translational and rotational)" << endl;
  cout << "Energy resolution: " << std::scientific << bin_width << " K" << endl;

  cout << "Inertia moment of cluster : " << cluster_0.inertia_moment << " Kg m^2" << endl;
  cout << "Inertia moment of first product: " << cluster_1.inertia_moment << " Kg m^2" << endl;
  cout << "Inertia moment of second product: " << cluster_2.inertia_moment << " Kg m^2" << endl;

  cout << "Mass of cluster: " << cluster_0.mass << " Kg" << endl;
  cout << "Mass of first product: " << cluster_1.mass << " Kg" << endl;
  cout << "Mass of second product: " << cluster_2.mass << " Kg" << endl;

  cout << "Radius of cluster: " << cluster_0.radius << " m" << endl;
  cout << "Radius of first product: " << cluster_1.radius << " m" << endl;
  cout << "Radius of second product: " << cluster_2.radius << " m" << endl;
  cout << endl;

  cout << endl
       << "Computing total fragmentation rate constant..." << endl;

  const Eigen::ArrayXd k_rate = compute_k_total_full(cluster_0, cluster_1, cluster_2, rhos, fragmentation_energy, energy_max_rate, bin_width);

  int m_max_rate = int(energy_max_rate / bin_width);
  int m_max = int(energy_max / bin_width);
  auto energies = prepare_energies(bin_width, m_max);
  auto energies_rate = prepare_energies(bin_width, m_max_rate);

  cout << endl
       << "END OF COMPUTATION" << endl;
  // Write density of states on files
  cout << endl;
  cout << "OUTPUTS" << endl;
  write_on_file(file_density_0, energies, rhos.col(C0_ROW), m_max);
  write_on_file(file_density_1, energies, rhos.col(C1_ROW), m_max);
  write_on_file(file_density_2, energies, rhos.col(C2_ROW), m_max);
  write_on_file(file_density_comb, energies, rhos.col(COMB_ROW), m_max);
  write_on_file(file_rate_constant, energies_rate, k_rate, m_max_rate);
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

// Read electronic energy
double read_electronic_energy(char *filename)
{
  ifstream file;
  double electronic_energy;

  file.open(filename);

  file >> electronic_energy;

  return electronic_energy;
}


Eigen::Array3d read_rotations(char *filename)
{
  Eigen::Array3d rotations;
  ifstream file;

  file.open(filename);

  for (int i = 0; i < 3; i++)
  {
    file >> rotations[i];
  }

  file.close();
  return rotations;
}

Eigen::ArrayXd read_frequencies(char *filename)
{
  ifstream file;
  char garb[150];
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
