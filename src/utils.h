#pragma once

#include <iostream>
#include <cstring>

void check_field_name(const char *buffer, const char *expected)
{
  if (strcmp(buffer, expected) != 0)
  {
    std::cerr << "Error while reading input configuration: Expected to read field name '" << expected << "' but got field name '" << buffer << "'" << std::endl;
    exit(EXIT_FAILURE);
  }
}

template <typename T>
void read_field(std::istream &config_in, T *variable, char *buffer, const char *expected)
{
  if (variable)
  {
    config_in >> *variable >> buffer;
  }
  else
  {
    config_in >> buffer >> buffer;
  }
  check_field_name(buffer, expected);
}


template <>
void read_field(std::istream &config_in, char *variable, char *buffer, const char *expected)
{
  if (variable)
  {
    config_in >> variable >> buffer;
  }
  else
  {
    config_in >> buffer >> buffer;
  }
  check_field_name(buffer, expected);
}

template <typename AmuT, typename FloatT>
void read_config(
  std::istream &config_in,
  char *title,
  int *cluster_charge_sign,
  AmuT *amu_0,
  AmuT *amu_1,
  AmuT *amu_2,
  FloatT *T,
  FloatT *pressure_first,
  FloatT *pressure_second,
  float *L0,
  FloatT *Lsk,
  float *L1,
  float *L2,
  float *L3,
  float *V0,
  float *V1,
  float *V2,
  float *V3,
  float *V4,
  int *N,
  double *dc,
  double *alpha_factor,
  double *bonding_energy,
  double *energy_max,
  double *energy_max_rate,
  double *bin_width,
  double *R_gas,
  double *m_gas,
  double *ga,
  double *dc_field,
  double *ac_field,
  double *radiofrequency,
  double *r_quadrupole,
  char *file_skimmer,
  char *file_frequencies_0,
  char *file_frequencies_1,
  char *file_frequencies_2,
  char *file_rotations_0,
  char *file_rotations_1,
  char *file_rotations_2,
  char *file_electronic_energy_0,
  char *file_electronic_energy_1,
  char *file_electronic_energy_2,
  char *file_density_0,
  char *file_density_1,
  char *file_density_2,
  char *file_density_comb,
  char *file_rate_constant,
  char *file_probabilities,
  int *N_iter,
  int *M_iter,
  int *resolution,
  double *tolerance)
{
  char buffer[256];
  if (title)
  {
    config_in >> title; // Title line
  }
  else
  {
    config_in >> buffer; // Skip title line
  }
  read_field(config_in, cluster_charge_sign, buffer, "Cluster_charge_sign");
  read_field(config_in, amu_0, buffer, "Atomic_mass_cluster");
  read_field(config_in, amu_1, buffer, "Atomic_mass_first_product");
  read_field(config_in, amu_2, buffer, "Atomic_mass_second_product");
  read_field(config_in, T, buffer, "Temperature_(K)");
  read_field(config_in, pressure_first, buffer, "Pressure_first_chamber(Pa)");
  read_field(config_in, pressure_second, buffer, "Pressure_second_chamber(Pa)");
  read_field(config_in, L0, buffer, "Length_of_1st_chamber_(meters)");
  read_field(config_in, Lsk, buffer, "Length_of_skimmer_(meters)");
  read_field(config_in, L1, buffer, "Length_between_skimmer_and_front_quadrupole_(meters)");
  read_field(config_in, L2, buffer, "Length_between_front_quadrupole_and_back_quadrupole_(meters)");
  read_field(config_in, L3, buffer, "Length_between_back_quadrupole_and_2nd_skimmer_(meters)");
  read_field(config_in, V0, buffer, "Voltage0_(Volt)");
  read_field(config_in, V1, buffer, "Voltage1_(Volt)");
  read_field(config_in, V2, buffer, "Voltage2_(Volt)");
  read_field(config_in, V3, buffer, "Voltage3_(Volt)");
  read_field(config_in, V4, buffer, "Voltage4_(Volt)");
  read_field(config_in, N, buffer, "Number_of_realizations");
  read_field(config_in, dc, buffer, "Radius_at_smallest_cross_section_skimmer_(m)");
  read_field(config_in, alpha_factor, buffer, "Angle_of_skimmer_(multiple_of_PI)");
  read_field(config_in, bonding_energy, buffer, "Fragmentation_energy_(Kelvin)");
  read_field(config_in, energy_max, buffer, "Energy_max_for_density_of_states_(Kelvin)");
  read_field(config_in, energy_max_rate, buffer, "Energy_max_for_rate_constant_(Kelvin)");
  read_field(config_in, bin_width, buffer, "Energy_resolution_(Kelvin)");
  read_field(config_in, R_gas, buffer, "Gas_molecule_radius_(meters)");
  read_field(config_in, m_gas, buffer, "Gas_molecule_mass_(kg)");
  read_field(config_in, ga, buffer, "Adiabatic_index");
  read_field(config_in, dc_field, buffer, "DC_quadrupole");
  read_field(config_in, ac_field, buffer, "AC_quadrupole");
  read_field(config_in, radiofrequency, buffer, "Radiofrequency_quadrupole");
  read_field(config_in, r_quadrupole, buffer, "Half-distance_between_quadrupole_rods");
  read_field(config_in, file_skimmer, buffer, "Output_file_skimmer");
  read_field(config_in, file_frequencies_0, buffer, "file_vibrational_temperatures_cluster");
  read_field(config_in, file_frequencies_1, buffer, "file_vibrational_temperatures_first_product");
  read_field(config_in, file_frequencies_2, buffer, "file_vibrational_temperatures_second_product");
  read_field(config_in, file_rotations_0, buffer, "file_rotational_temperatures_cluster");
  read_field(config_in, file_rotations_1, buffer, "file_rotational_temperatures_first_product");
  read_field(config_in, file_rotations_2, buffer, "file_rotational_temperatures_second_product");
  read_field(config_in, file_electronic_energy_0, buffer, "file_electronic_energy_cluster");
  read_field(config_in, file_electronic_energy_1, buffer, "file_electronic_energy_first_product");
  read_field(config_in, file_electronic_energy_2, buffer, "file_electronic_energy_second_product");
  read_field(config_in, file_density_0, buffer, "output_file_density_cluster");
  read_field(config_in, file_density_1, buffer, "output_file_density_first_product");
  read_field(config_in, file_density_2, buffer, "output_file_density_second_product");
  read_field(config_in, file_density_comb, buffer, "output_file_density_combined_products");
  read_field(config_in, file_rate_constant, buffer, "output_file_rate_constant");
  read_field(config_in, file_probabilities, buffer, "output_file_probabilities");
  read_field(config_in, N_iter, buffer, "Number_of_iterations_in_solving_equation");
  read_field(config_in, M_iter, buffer, "Number_of_iterations_in_solving_equation2");
  read_field(config_in, resolution, buffer, "Number_of_solved_points");
  read_field(config_in, tolerance, buffer, "Tolerance_in_solving_equation");
}

const int LOGLEVEL_NONE = 0,
          LOGLEVEL_MIN = 1,
          LOGLEVEL_NORMAL = 2,
          LOGLEVEL_EXTRA = 3;
const int LOGLEVEL = LOGLEVEL_MIN;

namespace Filenames
{
const char *const SKIMMER_WARNINGS = "work/log/warnings_skimmer.dat";

const char *const COLLISIONS = "work/log/collisions.dat";
const char *const INTENERGY = "work/log/intenergy.dat";
const char *const WARNINGS = "work/log/warnings.dat";
const char *const FRAGMENTS = "work/log/fragments.dat";
const char *const TMP = "work/log/tmp.dat";
const char *const TMP_EVOLUTION = "work/log/tmp_evolution.dat";
const char *const ENERGY_DISTRIBUTION = "work/log/energy_distribution.dat";
const char *const FINAL_POSITION = "work/log/final_position.dat";
const char *const PINHOLE = "work/log/pinhole.dat";
} // namespace Filenames

template <typename CallbackT>
void warn_omp(int &nwarnings, CallbackT callback)
{
#pragma omp atomic
  nwarnings++;
#pragma omp critical
  {
    callback();
  }
}
