// TO DO LIST: print percentage every 1%
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex>
#include <string.h>
#include "utils.h"

#define kb 1.38064852e-23 // Boltzmann constant
#define hbar 1.054571800e-34 // Reduced Planck constant
#define hart 627.509 // 1 hartree in Kcal/mol
#define R 1.9872e-3 // Gas constant in Kcal/mol/K
#define hartK 3.157732e+5 // 1 hartree in Kelvin
#define joulekcal 1.439325e+20 // 1 Joule in kcal/mol
#define kelvinkcal 1.987216e-03 // 1 K in kcal/mol

// Define Pi if it is not already defined
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

using namespace std;

// LIST OF FUNCTIONS


void compute_density_of_states_noE0(double *frequencies, double *&energies, double *&rho, int num_oscillators, double energy_max, double bin_width);

void read_frequencies(char *filename, int &num_oscillators, double *&frequencies);

void write_on_file(char *filename, double *energies, double *rho, int m_max);

void combine_frequencies(double *&frequencies_comb, double *frequencies_1, double *frequencies_2, int num_oscillators_1, int num_oscillators_2, int &num_oscillators_comb);

void read_rotations(char *filename, double *rotations);

void compute_inertia(double *rotations, double &inertia_moment);

void read_electronic_energy(char *filename, double &electronic_energy);

void k0_integral(double *k_rate, double *k0, double *rho_0, int m_max_rate, double bin_width, double fragmentation_energy);

void compute_k0(double *k0, double inertia_moment_1, double inertia_moment_2, double *rotations_1, double *rotations_2, double *rho_comb, double *rho_0, double bin_width, int m_max_rate, double *energies_rate, double fragmentation_energy);

void compute_k_rate(double *k_rate, double *k0, double inertia_moment_1, double inertia_moment_2, double *rotations_1, double *rotations_2, double *rho_comb, double *rho_0, double bin_width, int m_max_rate, double *energies_rate, double fragmentation_energy);
void compute_mass_and_radius(double inertia, double amu, double &mass, double &radius);
void compute_k_total(double *k0, double *k_rate, double inertia_moment_1, double inertia_moment_2, double *rotations_1, double *rotations_2, double *rho_comb, double *rho_0, double bin_width, int &m_max_rate, double *energies_rate, double fragmentation_energy, double max_rate);
void compute_k_total_atom(double *k0, double *k_rate, double inertia_moment_1, double *rho_comb, double *rho_0, double bin_width, int &m_max_rate, double *energies_rate, double fragmentation_energy, double max_rate);

// MAIN
int main()
{
  int m_max;
  int m_max_rate;
  double energy_max;
  double energy_max_rate;
  int num_oscillators_0;
  int num_oscillators_1;
  int num_oscillators_2;
  int num_oscillators_comb;
  double bin_width;
  double *rho_0;
  double *rho_1;
  double *rho_2;
  double *rho_comb;
  double *energies;
  double *frequencies_0;
  double *frequencies_1;
  double *frequencies_2;
  double *rotations_0;
  double *rotations_1;
  double *rotations_2;
  double *frequencies_comb;
  double *k0;
  double *k_rate;
  double *energies_rate;
  double inertia_moment_0;
  double inertia_moment_1;
  double inertia_moment_2;
  double radius_0;
  double radius_1;
  double radius_2;
  double mass_0;
  double mass_1;
  double mass_2;
  double amu_0;
  double amu_1;
  double amu_2;
  double electronic_energy_0;
  double electronic_energy_1;
  double electronic_energy_2;
  double fragmentation_energy;
  double coll_freq;
  double P1;
  double T;
  double R_tot;
  double R_gas;
  double m_gas;
  double max_rate;
  char file_frequencies_0[150];
  char file_frequencies_1[150];
  char file_frequencies_2[150];
  char file_density_0[150];
  char file_density_1[150];
  char file_density_2[150];
  char file_density_comb[150];
  char file_rotations_0[150];
  char file_rotations_1[150];
  char file_rotations_2[150];
  char file_electronic_energy_0[150];
  char file_electronic_energy_1[150];
  char file_electronic_energy_2[150];
  char file_rate_constant[150];

  // Use read_config to read all input fields
  read_config(
    std::cin,
    nullptr, // title
    nullptr, // cluster_charge_sign
    &amu_0,
    &amu_1,
    &amu_2,
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
    file_frequencies_0,
    file_frequencies_1,
    file_frequencies_2,
    file_rotations_0,
    file_rotations_1,
    file_rotations_2,
    file_electronic_energy_0,
    file_electronic_energy_1,
    file_electronic_energy_2,
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

  rotations_0 = new double[3];
  rotations_1 = new double[3];
  rotations_2 = new double[3];


  // printf("]\033[F\033[J%s:%3lld%% [",text,c);
  cout << "###" << endl;
  cout << "Reading inputs..." << endl;
  // Read frequencies
  read_frequencies(file_frequencies_0, num_oscillators_0, frequencies_0);
  read_frequencies(file_frequencies_1, num_oscillators_1, frequencies_1);
  read_frequencies(file_frequencies_2, num_oscillators_2, frequencies_2);

  // Read rotational constants
  read_rotations(file_rotations_0, rotations_0);
  read_rotations(file_rotations_1, rotations_1);
  read_rotations(file_rotations_2, rotations_2);

  // for(int i=0;i<3;i++)  rotations_2[i]*=1.0e2;

  // Read electronic energies
  read_electronic_energy(file_electronic_energy_0, electronic_energy_0);
  read_electronic_energy(file_electronic_energy_1, electronic_energy_1);
  read_electronic_energy(file_electronic_energy_2, electronic_energy_2);

  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == 0)
  {
    fragmentation_energy = (electronic_energy_1 + electronic_energy_2 - electronic_energy_0) * hartK;
  }
  // cout << fragmentation_energy << endl;

  // Combine the frequencies of two products
  combine_frequencies(frequencies_comb, frequencies_1, frequencies_2, num_oscillators_1, num_oscillators_2, num_oscillators_comb);

  if (num_oscillators_1 > 0 && num_oscillators_2 > 0)
  {
    if (num_oscillators_0 - num_oscillators_comb != 6)
    {
      cout << "Number of frequencies wrong!!!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if (num_oscillators_0 - num_oscillators_comb != 3)
    {
      cout << "Number of frequencies wrong!!!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  // cout << "]\033[F\033[J";
  cout << endl;
  cout << "Cluster vibrational modes: " << num_oscillators_0 << endl;
  cout << "First product vibrational modes: " << num_oscillators_1 << endl;
  cout << "Second product vibrational modes: " << num_oscillators_2 << endl;
  cout << "Combined vibrational modes of two products: " << num_oscillators_comb << " (+ " << num_oscillators_0 - num_oscillators_comb << " degrees of freedom, translational and rotational)" << endl;
  cout << "Fragmentation energy: " << std::scientific << fragmentation_energy << " K (" << fragmentation_energy * kelvinkcal << " kcal/mol)" << endl;
  cout << "Energy resolution: " << std::scientific << bin_width << " K" << endl;

  // Compute moments of inertia
  compute_inertia(rotations_0, inertia_moment_0);
  compute_inertia(rotations_1, inertia_moment_1);
  compute_inertia(rotations_2, inertia_moment_2);

  compute_mass_and_radius(inertia_moment_0, amu_0, mass_0, radius_0);
  compute_mass_and_radius(inertia_moment_1, amu_1, mass_1, radius_1);
  compute_mass_and_radius(inertia_moment_2, amu_2, mass_2, radius_2);

  // TO BE DELETED ###############
  //  radius_0=4.675e-10;
  //  radius_1=3.396e-10;
  //  radius_2=2.621e-10;
  //  inertia_moment_0=0.4*mass_0*radius_0*radius_0;
  //  inertia_moment_1=0.4*mass_1*radius_1*radius_1;
  //  inertia_moment_2=0.4*mass_2*radius_2*radius_2;
  // #############################

  cout << "Inertia moment of cluster : " << inertia_moment_0 << " Kg m^2" << endl;
  cout << "Inertia moment of first product: " << inertia_moment_1 << " Kg m^2" << endl;
  cout << "Inertia moment of second product: " << inertia_moment_2 << " Kg m^2" << endl;

  cout << "Mass of cluster: " << mass_0 << " Kg" << endl;
  cout << "Mass of first product: " << mass_1 << " Kg" << endl;
  cout << "Mass of second product: " << mass_2 << " Kg" << endl;

  cout << "Radius of cluster: " << radius_0 << " m" << endl;
  cout << "Radius of first product: " << radius_1 << " m" << endl;
  cout << "Radius of second product: " << radius_2 << " m" << endl;

  R_tot = radius_0 + R_gas;
  coll_freq = P1 * R_tot * R_tot * sqrt(8.0 * M_PI / (kb * T * m_gas));

  cout << "Collision frequency: " << coll_freq << " 1/s" << endl;

  // energy_max_rate = 1.0e6;
  max_rate = coll_freq * 1.0e3; // rate constant evaluated up to this value
  // max_rate=1.0e-1; // rate constant evaluated up to this value
  m_max_rate = int(energy_max_rate / bin_width);

  k0 = new double[m_max_rate];
  k_rate = new double[m_max_rate];
  energies_rate = new double[m_max_rate];

  // Compute density of states neglecting zero level energy
  cout << endl
       << "Computing density of states of cluster..." << endl;
  compute_density_of_states_noE0(frequencies_0, energies, rho_0, num_oscillators_0, energy_max, bin_width);
  cout << endl
       << "Computing density of states of 1st product..." << endl;
  compute_density_of_states_noE0(frequencies_1, energies, rho_1, num_oscillators_1, energy_max, bin_width);
  cout << endl
       << "Computing density of states of 2nd product..." << endl;
  compute_density_of_states_noE0(frequencies_2, energies, rho_2, num_oscillators_2, energy_max, bin_width);
  cout << endl
       << "Computing density of states of combined products..." << endl;
  compute_density_of_states_noE0(frequencies_comb, energies, rho_comb, num_oscillators_comb, energy_max, bin_width);


  // Compute fragmentation rate at zero rotation
  // cout << endl << "Computing fragmentation rate constant at zero angular momentum..."<<endl;
  // compute_k0(k0,inertia_moment_1,inertia_moment_2,rotations_1,rotations_2,rho_comb,rho_0,bin_width,m_max_rate,energies_rate,fragmentation_energy);
  // Compute total fragmentation rate constant
  // cout << endl << "Computing total fragmentation rate constant..."<<endl;
  // k0_integral(k_rate,k0,rho_0,m_max_rate,bin_width, fragmentation_energy);

  cout << endl
       << "Computing total fragmentation rate constant..." << endl;
  if (num_oscillators_1 > 0 && num_oscillators_2 > 0)
  {
    cout << "Generic products" << endl;
    compute_k_total(k0, k_rate, inertia_moment_1, inertia_moment_2, rotations_1, rotations_2, rho_comb, rho_0, bin_width, m_max_rate, energies_rate, fragmentation_energy, max_rate);
  }
  else
  {
    cout << "Atom-like product" << endl;
    compute_k_total_atom(k0, k_rate, inertia_moment_1, rho_comb, rho_0, bin_width, m_max_rate, energies_rate, fragmentation_energy, max_rate);
  }

  // compute_k_rate(k_rate,k0,inertia_moment_1,inertia_moment_2,rotations_1,rotations_2,rho_comb,rho_0,bin_width,m_max_rate,energies_rate,fragmentation_energy);
  // cout << "]\033[F\033[J";
  cout << endl
       << "END OF COMPUTATION" << endl;
  // Write density of states on files
  cout << endl;
  cout << "OUTPUTS" << endl;
  write_on_file(file_density_0, energies, rho_0, m_max);
  write_on_file(file_density_1, energies, rho_1, m_max);
  write_on_file(file_density_2, energies, rho_2, m_max);
  write_on_file(file_density_comb, energies, rho_comb, m_max);
  write_on_file(file_rate_constant, energies_rate, k_rate, m_max_rate);
  cout << "###" << endl;
  // Free memory
  delete[] rho_0;
  delete[] rho_1;
  delete[] rho_2;
  delete[] rho_comb;
  delete[] k0;
  delete[] k_rate;
  delete[] energies;
  delete[] energies_rate;
  delete[] frequencies_0;
  delete[] frequencies_1;
  delete[] frequencies_2;
  delete[] rotations_0;
  delete[] rotations_1;
  delete[] rotations_2;
  delete[] frequencies_comb;
  return 0;
}

// FUNCTIONS


void compute_k_total(double *k0, double *k_rate, double inertia_moment_1, double inertia_moment_2, double *rotations_1, double *rotations_2, double *rho_comb, double *rho_0, double bin_width, int &m_max_rate, double *energies_rate, double fragmentation_energy, double max_rate)
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
    energies_rate[m] = bin_width * (m + 0.5);
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

void compute_k_total_atom(double *k0, double *k_rate, double inertia_moment_1, double *rho_comb, double *rho_0, double bin_width, int &m_max_rate, double *energies_rate, double fragmentation_energy, double max_rate)
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
    energies_rate[m] = bin_width * (m + 0.5);
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


void compute_k_rate(double *k_rate, double *k0, double inertia_moment_1, double inertia_moment_2, double *rotations_1, double *rotations_2, double *rho_comb, double *rho_0, double bin_width, int m_max_rate, double *energies_rate, double fragmentation_energy)
{

  compute_k0(k0, inertia_moment_1, inertia_moment_2, rotations_1, rotations_2, rho_comb, rho_0, bin_width, m_max_rate, energies_rate, fragmentation_energy);
  k0_integral(k_rate, k0, rho_0, m_max_rate, bin_width, fragmentation_energy);
}

void k0_integral(double *k_rate, double *k0, double *rho_0, int m_max_rate, double bin_width, double fragmentation_energy)
{
  double integral;
  double rot_energy;
  double normalization;
  int n_fragmentation;

  n_fragmentation = int(fragmentation_energy / bin_width);
  // Cycle over the energy at which we calculate the rate constant
  for (int m = 0; m < m_max_rate; m++)
  {
    // if(100*m%m_max_rate==0) cout << "]\033[F\033[J  "<< int(100.0*m/m_max_rate) << "%"<<endl;
    normalization = 0.0;
    for (int i = 0; i <= n_fragmentation + m; i++)
    {
      rot_energy = bin_width * (i + 0.5);
      normalization += rho_0[n_fragmentation + m - i] * sqrt(rot_energy);
    }

    integral = 0.0;
    // Cycle over integral differential
    for (int i = 0; i <= m; i++)
    {
      rot_energy = bin_width * (i + 0.5);
      integral += rho_0[n_fragmentation + m - i] * sqrt(rot_energy) * k0[m - i];
    }
    k_rate[m] = integral / normalization;
  }
  cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 << "% " << string(50, '*') << " (" << bin_width * m_max_rate << " K)" << endl;
  // cout << "100%" << endl;
  // cout <<"]\033[F\033[J";
}


void compute_k0(double *k0, double inertia_moment_1, double inertia_moment_2, double *rotations_1, double *rotations_2, double *rho_comb, double *rho_0, double bin_width, int m_max_rate, double *energies_rate, double fragmentation_energy)
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

  rotations_product_1 = rotations_1[0] * rotations_1[1] * rotations_1[2];
  rotations_product_2 = rotations_2[0] * rotations_2[1] * rotations_2[2];

  prefactor = 2.0 * kb * kb * (inertia_moment_1 + inertia_moment_2) / (M_PI * hbar * hbar * hbar * pow(pow(rotations_product_1, 1.0 / 3) + pow(rotations_product_2, 1.0 / 3), 1.5));
  n_fragmentation = int(fragmentation_energy / bin_width);
  for (int m = 0; m < m_max_rate; m++)
  {
    density_cluster = rho_0[n_fragmentation + m];
    energies_rate[m] = bin_width * (m + 0.5);
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

    if ((m + 1) % progress == 0)
    {
      cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 * (m + 1) / m_max_rate << "% " << string((int)50.0 * (m + 1) / m_max_rate, '*') << string((int)50.0 * (m_max_rate - m - 1) / m_max_rate, '-') << " (" << bin_width * (m + 1) << " K)" << endl;
    }
    // if(m%1000==0 and m>0) cout << std::defaultfloat << 100.0*m/m_max_rate << "%" << endl;
  }
  // cout << "100%" << endl;
  if ((m_max_rate) % progress != 0)
    cout << defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 << "% " << string(50, '*') << " (" << bin_width * m_max_rate << " K)" << endl;
  // cout <<"]\033[F\033[J";
}


// Geometrical mean of moment of inertia
void compute_inertia(double *rotations, double &inertia_moment)
{
  inertia_moment = 0.5 * hbar * hbar / (kb * pow(rotations[0] * rotations[1] * rotations[2], 1.0 / 3));
}

// Compute radius of cluster
void compute_mass_and_radius(double inertia, double amu, double &mass, double &radius)
{
  mass = 1.6726219e-27 * amu; // proton mass * nucleons
  radius = sqrt(2.5 * inertia / mass);
}


// Read electronic energy
void read_electronic_energy(char *filename, double &electronic_energy)
{
  ifstream file;

  file.open(filename);

  file >> electronic_energy;
}


void read_rotations(char *filename, double *rotations)
{
  ifstream file;

  file.open(filename);

  for (int i = 0; i < 3; i++)
  {
    file >> rotations[i];
  }

  file.close();
}


// Compute density of states from vector of frequencies neglecting the zero level energy
void compute_density_of_states_noE0(double *frequencies, double *&energies, double *&rho, int num_oscillators, double energy_max, double bin_width)
{
  int k_max;
  int i = 0;
  int m;
  int m_max;
  int progress = 10;
  double *rho_new;
  double delta_energy;
  double energy;
  double E_m;


  m_max = int(energy_max / bin_width);
  rho = new double[m_max];
  energies = new double[m_max];
  rho_new = new double[m_max];

  for (m = 0; m < m_max; m++)
  {
    rho[m] = 0.0;
  }

  k_max = int(energy_max / frequencies[0]) + 1;

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
    energies[m] = bin_width * (m + 0.5);
  }

  delete[] rho_new;
}

void read_frequencies(char *filename, int &num_oscillators, double *&frequencies)
{
  ifstream file;
  char garb[150];
  file.open(filename);

  // Count the number of frequencies
  num_oscillators = 0;
  while (file >> garb)
  {
    num_oscillators++;
  }
  file.close();
  frequencies = new double[num_oscillators];

  // Save the frequencies on a vector
  file.open(filename);
  for (int i = 0; i < num_oscillators; i++)
  {
    file >> frequencies[i];
  }
  file.close();
}

// Combined frequencies of two products
void combine_frequencies(double *&frequencies_comb, double *frequencies_1, double *frequencies_2, int num_oscillators_1, int num_oscillators_2, int &num_oscillators_comb)
{
  num_oscillators_comb = num_oscillators_1 + num_oscillators_2;

  frequencies_comb = new double[num_oscillators_comb];

  for (int i = 0; i < num_oscillators_1; i++)
  {
    frequencies_comb[i] = frequencies_1[i];
  }
  for (int i = 0; i < num_oscillators_2; i++)
  {
    frequencies_comb[i + num_oscillators_1] = frequencies_2[i];
  }
}

void write_on_file(char *filename, double *x, double *y, int m_max)
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
