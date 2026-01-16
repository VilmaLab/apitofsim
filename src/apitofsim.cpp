#include "apitofsim.h"
#include "consts.h"

#include <iostream>

ClusterData::ClusterData()
{
}

ClusterData::ClusterData(int atomic_mass, double electronic_energy, Eigen::Vector3d rotations, Eigen::ArrayXd frequencies)
    : atomic_mass(atomic_mass), electronic_energy(electronic_energy), rotations(rotations), frequencies(frequencies)
{
}

void ClusterData::validate()
{
  if (this->is_atom_like_product() && !this->rotations.isZero(0))
  {
    std::cout << "Atom-like products must have 0 rotations" << std::endl;
    exit(EXIT_FAILURE);
  }
}

int ClusterData::num_oscillators()
{
  return this->frequencies.rows();
}

bool ClusterData::is_atom_like_product()
{
  return this->num_oscillators() == 0;
}

void ClusterData::compute_derived()
{
  using consts::pmass;
  if (this->is_atom_like_product())
  {
    // No rotations, so can't calculate inertia moment/radius
    inertia_moment = 0;
    radius = 0;
    mass = pmass * this->atomic_mass; // proton mass * nucleons
  }
  else
  {
    inertia_moment = compute_inertia(rotations);
    compute_mass_and_radius(inertia_moment, atomic_mass, mass, radius);
  }
}

double FragmentationPathway::fragmentation_energy_kelvin()
{
  return (this->product1.electronic_energy + this->product2.electronic_energy - this->parent.electronic_energy) * consts::hartK;
}

double compute_inertia(const Eigen::Vector3d &rotations)
{
  using consts::hbar, consts::boltzmann;
  return 0.5 * hbar * hbar / (boltzmann * pow(rotations[0] * rotations[1] * rotations[2], 1.0 / 3));
}

void compute_mass_and_radius(double inertia, double amu, double &mass, double &radius)
{
  using consts::pmass;
  mass = pmass * amu; // proton mass * nucleons
  radius = sqrt(2.5 * inertia / mass);
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

void debug_info()
{
#ifdef _OPENMP
  std::cout << "OpenMP version: " << _OPENMP << "\n";
#else
  std::cout << "OpenMP not enabled\n";
#endif
  std::cout << "Num threads: " << omp_get_max_threads() << "\n";
}

void debug_info_on_env()
{
  const char *debug_info_env = getenv("DEBUG_INFO");
  if (debug_info_env != nullptr)
  {
    debug_info();
  }
}
