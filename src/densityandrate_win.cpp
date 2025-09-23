// TO DO LIST: print percentage every 1%
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <Eigen/Dense>

#define kb 1.38064852e-23 // Boltzmann constant
#define hbar 1.054571800e-34 // Reduced Planck constant
#define hart 627.509 // 1 hartree in Kcal/mol
#define R 1.9872e-3 // Gas constant in Kcal/mol/K
#define hartK 3.157732e+5 // 1 hartree in Kelvin
#define joulekcal 1.439325e+20 // 1 Joule in kcal/mol
#define kelvinkcal 1.987216e-03 // 1 K in kcal/mol
#define protonMass 1.6726219e-27 // relative mass of on proton in kg

// Define Pi if it is not already defined
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

using namespace std;

// LIST OF FUNCTIONS

void compute_density_of_states_noE0(Eigen::ArrayXd &frequencies, Eigen::ArrayXd &rho, double energy_max, double bin_width);

Eigen::ArrayXd read_frequencies(char *filename);

void write_on_file(char *filename, Eigen::ArrayXd &x, Eigen::ArrayXd &y, int m_max);

Eigen::ArrayXd combine_frequencies(Eigen::ArrayXd &frequencies_1, Eigen::ArrayXd &frequencies_2);

Eigen::ArrayXd prepare_energies(double bin_width, int m_max);

Eigen::Array3d read_rotations(char *filename);

double compute_inertia(Eigen::Vector3d &rotations);

double read_electronic_energy(char *filename);

void compute_mass_and_radius(double inertia, double amu, double &mass, double &radius);
void compute_k_total(Eigen::ArrayXd &k0, Eigen::ArrayXd &k_rate, double inertia_moment_1, double inertia_moment_2, Eigen::Vector3d &rotations_1, Eigen::Vector3d &rotations_2, Eigen::ArrayXd &rho_comb, Eigen::ArrayXd &rho_0, double bin_width, int m_max_rate, double fragmentation_energy, double max_rate);
void compute_k_total_atom(Eigen::ArrayXd &k0, Eigen::ArrayXd &k_rate, double inertia_moment_1, Eigen::ArrayXd &rho_comb, Eigen::ArrayXd &rho_0, double bin_width, int m_max_rate, double fragmentation_energy, double max_rate);

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

  Eigen::ArrayXd rho;

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

  void compute_derived(double energy_max, double bin_width)
  {
    int m_max = int(energy_max / bin_width);
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
    rho = Eigen::ArrayXd(m_max);
    compute_density_of_states_noE0(frequencies, rho, energy_max, bin_width);
  }
};

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
  int m_max;
  int m_max_rate;
  double energy_max;
  double energy_max_rate;
  double bin_width;
  double fragmentation_energy;
  double coll_freq;
  double P1;
  double T;
  double R_tot;
  double R_gas;
  double m_gas;
  double max_rate;
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
    &R_gas,
    &m_gas,
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

  m_max = int(energy_max / bin_width);
  m_max_rate = int(energy_max_rate / bin_width);

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

  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == 0)
  {
    fragmentation_energy = (cluster_1.electronic_energy + cluster_2.electronic_energy - cluster_0.electronic_energy) * hartK;
  }

  // Combine the frequencies of two products
  auto frequencies_comb = combine_frequencies(cluster_1.frequencies, cluster_2.frequencies);
  int num_oscillators_comb = frequencies_comb.rows();

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

  R_tot = cluster_0.radius + R_gas;
  coll_freq = P1 * R_tot * R_tot * sqrt(8.0 * M_PI / (kb * T * m_gas));
  max_rate = coll_freq * 1.0e3; // rate constant evaluated up to this value

  cout << "Collision frequency: " << coll_freq << " 1/s" << endl;

  Eigen::ArrayXd k0 = Eigen::ArrayXd(m_max_rate);
  Eigen::ArrayXd k_rate = Eigen::ArrayXd(m_max_rate);

  // Compute density of states neglecting zero level energy
  cout << endl
       << "Computing density of states of cluster..." << endl;
  cluster_0.compute_derived(energy_max, bin_width);
  cout << endl
       << "Computing density of states of 1st product..." << endl;
  cluster_1.compute_derived(energy_max, bin_width);
  cout << endl
       << "Computing density of states of 2nd product..." << endl;
  cluster_2.compute_derived(energy_max, bin_width);
  cout << endl
       << "Computing density of states of combined products..." << endl;
  Eigen::ArrayXd rho_comb = Eigen::ArrayXd(m_max);
  compute_density_of_states_noE0(frequencies_comb, rho_comb, energy_max, bin_width);

  cout << endl;
  cout << "Cluster vibrational modes: " << cluster_0.num_oscillators() << endl;
  cout << "First product vibrational modes: " << cluster_1.num_oscillators() << endl;
  cout << "Second product vibrational modes: " << cluster_2.num_oscillators() << endl;
  cout << "Combined vibrational modes of two products: " << num_oscillators_comb << " (+ " << cluster_0.num_oscillators() - num_oscillators_comb << " degrees of freedom, translational and rotational)" << endl;
  cout << "Fragmentation energy: " << std::scientific << fragmentation_energy << " K (" << fragmentation_energy * kelvinkcal << " kcal/mol)" << endl;
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
  if (cluster_1.num_oscillators() > 0 && cluster_2.num_oscillators() > 0)
  {
    cout << "Generic products" << endl;
    compute_k_total(k0, k_rate, cluster_1.inertia_moment, cluster_2.inertia_moment, cluster_1.rotations, cluster_2.rotations, rho_comb, cluster_0.rho, bin_width, m_max_rate, fragmentation_energy, max_rate);
  }
  else
  {
    cout << "Atom-like product" << endl;
    compute_k_total_atom(k0, k_rate, cluster_1.inertia_moment, rho_comb, cluster_0.rho, bin_width, m_max_rate, fragmentation_energy, max_rate);
  }

  auto energies_rate = prepare_energies(bin_width, m_max_rate);
  auto energies = prepare_energies(bin_width, m_max);

  cout << endl
       << "END OF COMPUTATION" << endl;
  // Write density of states on files
  cout << endl;
  cout << "OUTPUTS" << endl;
  write_on_file(file_density_0, energies, cluster_0.rho, m_max);
  write_on_file(file_density_1, energies, cluster_1.rho, m_max);
  write_on_file(file_density_2, energies, cluster_2.rho, m_max);
  write_on_file(file_density_comb, energies, rho_comb, m_max);
  write_on_file(file_rate_constant, energies_rate, k_rate, m_max_rate);
  cout << "###" << endl;

  return 0;
}

// FUNCTIONS


void compute_k_total(Eigen::ArrayXd &k0, Eigen::ArrayXd &k_rate, double inertia_moment_1, double inertia_moment_2, Eigen::Vector3d &rotations_1, Eigen::Vector3d &rotations_2, Eigen::ArrayXd &rho_comb, Eigen::ArrayXd &rho_0, double bin_width, int m_max_rate, double fragmentation_energy, double max_rate)
{
  double prefactor;
  double rotations_product_1;
  double rotations_product_2;
  int n_fragmentation;
  int progress = 500;
  double integral;
  double density_cluster;
  double rotational_energy;
  double translational_energy;
  double normalization;
  // int a=0;
  // int m=0;

  rotations_product_1 = rotations_1[0] * rotations_1[1] * rotations_1[2];
  rotations_product_2 = rotations_2[0] * rotations_2[1] * rotations_2[2];

  prefactor = 2.0 * kb * kb * (inertia_moment_1 + inertia_moment_2) / (M_PI * hbar * hbar * hbar * pow(pow(rotations_product_1, 1.0 / 3) + pow(rotations_product_2, 1.0 / 3), 1.5));
  n_fragmentation = int(fragmentation_energy / bin_width);
  for (int m = 0; m < m_max_rate; m++)
  {
    density_cluster = rho_0[n_fragmentation + m];
    // if(100*m%m_max_rate==0) cout << "]\033[F\033[J  "<< int(100.0*m/m_max_rate) << "%"<<endl;
    // cout << "]\033[F\033[J  "<< int(100.0*m/m_max_rate) << "%"<<endl;
    //  Compute double integral
    integral = 0.0;
    for (int i = 0; i <= m; i++) // rotational energy
    {
      rotational_energy = bin_width * (i + 0.5);
      for (int j = 0; j <= m - i; j++) // translational energy
      {
        translational_energy = bin_width * (j + 0.5);
        integral += translational_energy * sqrt(rotational_energy) * rho_comb[m - i - j];
      }
    }

    k0[m] = prefactor / density_cluster * integral * bin_width * bin_width;

    // if(m%1000==0 and m>0) cout << std::defaultfloat << 100.0*m/m_max_rate << "%" << endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Integrate over all rotation energies
    normalization = 0.0;
    for (int i = 0; i <= n_fragmentation + m; i++)
    {
      rotational_energy = bin_width * (i + 0.5);
      normalization += rho_0[n_fragmentation + m - i] * sqrt(rotational_energy);
    }

    integral = 0.0;
    // Cycle over integral differential
    for (int i = 0; i <= m; i++)
    {
      rotational_energy = bin_width * (i + 0.5);
      integral += rho_0[n_fragmentation + m - i] * sqrt(rotational_energy) * k0[m - i];
    }
    k_rate[m] = integral / normalization;

    // if(k_rate[m]>max_rate) a=1;

    if ((m + 1) % progress == 0)
    {
      // cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0*k_rate[m]/max_rate << "% " << string((int)50.0*k_rate[m]/max_rate,'*') << string(51-(int)50.0*k_rate[m]/max_rate,'-') << " (E="<< bin_width*(m+1) <<" K, k_rate=" << scientific << k_rate[m] << " 1/s)" << endl;
      cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 * (m + 1) / m_max_rate << "% " << string((int)(50.0 * (m + 1) / m_max_rate), '*') << string(50 - (int)(50.0 * (m + 1) / m_max_rate), '-') << " (E=" << bin_width * (m + 1) << " K, k_rate=" << scientific << k_rate[m] << " 1/s)" << endl;
    }
    // m++;
  }
  // cout << "100%" << endl;
  // m_max_rate=m;
  if ((m_max_rate) % progress != 0)
    cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 << "% " << string(50, '*') << " (E=" << bin_width * m_max_rate << " K, k_rate=" << scientific << max_rate << " 1/s)" << endl;
  // cout <<"]\033[F\033[J";
}

void compute_k_total_atom(Eigen::ArrayXd &k0, Eigen::ArrayXd &k_rate, double inertia_moment_1, Eigen::ArrayXd &rho_comb, Eigen::ArrayXd &rho_0, double bin_width, int m_max_rate, double fragmentation_energy, double max_rate)
{
  double prefactor;
  // double rotations_product_1;
  // double rotations_product_2;
  int n_fragmentation;
  int progress = 500;
  double integral;
  double density_cluster;
  double rotational_energy;
  double translational_energy;
  double normalization;
  // int a=0;
  // int m=0;

  // rotations_product_1 = rotations_1[0] * rotations_1[1] * rotations_1[2];
  // rotations_product_2 = rotations_2[0] * rotations_2[1] * rotations_2[2];

  prefactor = kb * kb * (inertia_moment_1) / (M_PI * hbar * hbar * hbar);
  n_fragmentation = int(fragmentation_energy / bin_width);
  for (int m = 0; m < m_max_rate; m++)
  {
    density_cluster = rho_0[n_fragmentation + m];
    // if(100*m%m_max_rate==0) cout << "]\033[F\033[J  "<< int(100.0*m/m_max_rate) << "%"<<endl;
    // cout << "]\033[F\033[J  "<< int(100.0*m/m_max_rate) << "%"<<endl;
    //  Compute double integral
    integral = 0.0;
    for (int i = 0; i <= m; i++) // translational energy
    {
      translational_energy = bin_width * (i + 0.5);
      integral += translational_energy * rho_comb[m - i];
    }

    k0[m] = prefactor / density_cluster * integral * bin_width;

    // if(m%1000==0 and m>0) cout << std::defaultfloat << 100.0*m/m_max_rate << "%" << endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Integrate over all rotation energies
    normalization = 0.0;
    for (int i = 0; i <= n_fragmentation + m; i++)
    {
      rotational_energy = bin_width * (i + 0.5);
      normalization += rho_0[n_fragmentation + m - i] * sqrt(rotational_energy);
    }

    integral = 0.0;
    // Cycle over integral differential
    for (int i = 0; i <= m; i++)
    {
      rotational_energy = bin_width * (i + 0.5);
      integral += rho_0[n_fragmentation + m - i] * sqrt(rotational_energy) * k0[m - i];
    }
    k_rate[m] = integral / normalization;

    // if(k_rate[m]>max_rate) a=1;

    if ((m + 1) % progress == 0)
    {
      // cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0*k_rate[m]/max_rate << "% " << string((int)50.0*k_rate[m]/max_rate,'*') << string(51-(int)50.0*k_rate[m]/max_rate,'-') << " (E="<< bin_width*(m+1) <<" K, k_rate=" << scientific << k_rate[m] << " 1/s)" << endl;
      cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 * (m + 1) / m_max_rate << "% " << string((int)(50.0 * (m + 1) / m_max_rate), '*') << string(50 - (int)(50.0 * (m + 1) / m_max_rate), '-') << " (E=" << bin_width * (m + 1) << " K, k_rate=" << scientific << k_rate[m] << " 1/s)" << endl;
    }
    // m++;
  }
  // cout << "100%" << endl;
  // m_max_rate=m;
  if ((m_max_rate) % progress != 0)
    cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 << "% " << string(50, '*') << " (E=" << bin_width * m_max_rate << " K, k_rate=" << scientific << max_rate << " 1/s)" << endl;
  // cout <<"]\033[F\033[J";
}


// Geometrical mean of moment of inertia
double compute_inertia(Eigen::Vector3d &rotations)
{
  return 0.5 * hbar * hbar / (kb * pow(rotations[0] * rotations[1] * rotations[2], 1.0 / 3));
}

// Compute radius of cluster
void compute_mass_and_radius(double inertia, double amu, double &mass, double &radius)
{
  mass = protonMass * amu; // proton mass * nucleons
  radius = sqrt(2.5 * inertia / mass);
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
void compute_density_of_states_noE0(Eigen::ArrayXd &frequencies, Eigen::ArrayXd &rho, double energy_max, double bin_width)
{
  int i = 0;
  int m;
  int progress = 10;
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
    // if((100*i/num_oscillators)%10==0)  cout << "]\033[F\033[J  "<< defaultfloat << i << "%"<< endl;
    // if((int)100.0*i%num_oscillators==0) cout << defaultfloat << int(100.0*i/num_oscillators) << "%"<< endl;
    for (m = 0; m < m_max; m++)
    {
      rho_new[m] = 0.0;
      E_m = bin_width * (m + 0.5);
      k_max = int(E_m / frequencies[i]);
      for (int k = 0; k < k_max + 1; k++)
      {
        delta_energy = E_m - frequencies[i] * k;
        rho_new[m] += rho[int(delta_energy / bin_width)];
      }
    }
    for (m = 0; m < m_max; m++)
    {
      rho[m] = rho_new[m];
    }

    if ((i + 1) % progress == 0)
    {
      cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 * (i + 1) / num_oscillators << "% " << string(i + 1, '*') << string(num_oscillators - i - 1, '-') << " (" << i + 1 << ")" << endl;
    }
    // if(i%20==0 and i>0) cout << std::defaultfloat << 100.0*i/num_oscillators << "%" << endl;
  }
  // cout << "100%" << endl;
  if ((num_oscillators % progress != 0))
    cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 << "% " << string(num_oscillators, '*') << " (" << num_oscillators << ")" << endl;
  for (m = 0; m < m_max; m++)
  {
    rho[m] = rho[m] / bin_width;
  }
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

void write_on_file(char *filename, Eigen::ArrayXd &x, Eigen::ArrayXd &y, int m_max)
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
