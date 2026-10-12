#include "apitofsim.h"
#include "consts.h"

#include <oneapi/tbb/info.h>
#include <oneapi/tbb/version.h>

#include <iostream>
#include <stdexcept>

AtomLikeCluster::AtomLikeCluster(int nucleon_count, double electronic_energy, int charge)
    : nucleon_count(nucleon_count), electronic_energy(electronic_energy), charge(charge)
{
}

MolecularCluster::MolecularCluster(int nucleon_count, double electronic_energy, Eigen::Vector3d rotations, Eigen::ArrayXd frequencies, int charge)
    : nucleon_count(nucleon_count), electronic_energy(electronic_energy), rotations(rotations), frequencies(frequencies), charge(charge)
{
  if (this->frequencies.rows() == 0)
  {
    throw std::invalid_argument("A molecular cluster must have at least one vibrational mode");
  }
}

int MolecularCluster::num_oscillators() const
{
  return this->frequencies.rows();
}

int num_oscillators(const ClusterSpec &spec)
{
  if (const auto *molecular = std::get_if<MolecularCluster>(&spec))
  {
    return molecular->num_oscillators();
  }
  return 0;
}

double electronic_energy(const ClusterSpec &spec)
{
  return std::visit([](const auto &cluster)
  {
    return cluster.electronic_energy;
  }, spec);
}

namespace
{
Products make_products(const MolecularCluster &first, const AtomLikeCluster &second)
{
  return AtomLikeProducts{first, second};
}

Products make_products(const MolecularCluster &first, const MolecularCluster &second)
{
  return MolecularProducts{first, second};
}

void fill_derived(const AtomLikeCluster &cluster, ClusterDerived &derived)
{
  // Atom-like products have no rotational structure
  derived.inertia_moment = 0;
  derived.radius = 0;
  derived.mass = consts::pmass * cluster.nucleon_count; // proton mass * nucleons
}

void fill_derived(const MolecularCluster &cluster, ClusterDerived &derived)
{
  derived.inertia_moment = compute_inertia(cluster.rotations);
  std::tie(derived.mass, derived.radius) = compute_mass_and_radius(derived.inertia_moment, cluster.nucleon_count);
}
} // namespace

Products make_products(const MolecularCluster &first, const ClusterSpec &second)
{
  return std::visit([&first](const auto &cluster)
  {
    return make_products(first, cluster);
  }, second);
}

ClusterDerived compute_derived(const ClusterSpec &spec)
{
  ClusterDerived derived;
  std::visit([&derived](const auto &cluster)
  {
    fill_derived(cluster, derived);
  }, spec);
  return derived;
}

FragmentationPathway::FragmentationPathway(const MolecularCluster &parent, const MolecularCluster &product1, const ClusterSpec &product2)
    : parent(parent), products(make_products(product1, product2))
{
}

double FragmentationPathway::fragmentation_energy_kelvin() const
{
  const double products_energy = std::visit([](const auto &products)
  {
    return products.first.electronic_energy + products.second.electronic_energy;
  }, this->products);
  return (products_energy - this->parent.electronic_energy) * consts::hartK;
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
