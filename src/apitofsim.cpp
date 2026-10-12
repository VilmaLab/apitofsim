#include "apitofsim.h"
#include "consts.h"

#include <oneapi/tbb/info.h>
#include <oneapi/tbb/version.h>

#include <iostream>

ClusterSpec::ClusterSpec(int nucleon_count, double electronic_energy, Eigen::Vector3d rotations, Eigen::ArrayXd frequencies, int charge)
    : nucleon_count(nucleon_count), electronic_energy(electronic_energy), rotations(rotations), frequencies(frequencies), charge(charge)
{
}

int ClusterSpec::num_oscillators() const
{
  return this->frequencies.rows();
}

bool ClusterSpec::is_atom_like_product() const
{
  return this->num_oscillators() == 0;
}

std::optional<std::string> ClusterSpec::validation_error() const
{
  if (this->is_atom_like_product() && !this->rotations.isZero(0))
  {
    return "Atom-like products must have 0 rotations";
  }
  return std::nullopt;
}

ClusterDerived compute_derived(const ClusterSpec &spec)
{
  using consts::pmass;
  ClusterDerived derived;
  if (spec.is_atom_like_product())
  {
    // No rotations, so can't calculate inertia moment/radius
    derived.inertia_moment = 0;
    derived.radius = 0;
    derived.mass = pmass * spec.nucleon_count; // proton mass * nucleons
  }
  else
  {
    derived.inertia_moment = compute_inertia(spec.rotations);
    std::tie(derived.mass, derived.radius) = compute_mass_and_radius(derived.inertia_moment, spec.nucleon_count);
  }
  return derived;
}

double FragmentationPathway::fragmentation_energy_kelvin() const
{
  return (this->product1.electronic_energy + this->product2.electronic_energy - this->parent.electronic_energy) * consts::hartK;
}

double compute_inertia(const Eigen::Vector3d &rotations)
{
  using consts::hbar, consts::boltzmann;
  return 0.5 * hbar * hbar / (boltzmann * pow(rotations[0] * rotations[1] * rotations[2], 1.0 / 3));
}

std::tuple<double, double> compute_mass_and_radius(double inertia, double nucleon_count)
{
  using consts::pmass;
  double mass = pmass * nucleon_count; // proton mass * nucleons
  double radius = sqrt(2.5 * inertia / mass);
  return std::make_tuple(mass, radius);
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
  std::cout << "oneTBB version: " << TBB_VERSION_MAJOR << '.' << TBB_VERSION_MINOR << '.' << TBB_VERSION_PATCH << '\n';
  std::cout << "Default concurrency: " << oneapi::tbb::info::default_concurrency() << '\n';
}

void debug_info_on_env()
{
  const char *debug_info_env = getenv("DEBUG_INFO");
  if (debug_info_env != nullptr)
  {
    debug_info();
  }
}
