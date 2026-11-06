#pragma once

#include "apitofsim.h"
#include "operation_context.h"
#include <memory>
#include <optional>
#include <random>

enum class FragmentationStateMethod
{
  zero,
  phase_space
};

struct FragmentationState
{
  Eigen::Vector3d velocity, omega, neutral_velocity, neutral_omega, orbital_momentum;
  double vibrational_energy{}, neutral_vibrational_energy{};
};

// One immutable ground-inclusive log-count table, shared by product pairs.
class FragmentationDos
{
  Eigen::ArrayXd counts_;

public:
  const double bin_width, energy_max;
  FragmentationDos(const Eigen::ArrayXd &frequencies, double bin_width, double energy_max,
                   std::optional<Eigen::ArrayXd> log_counts = std::nullopt);
  const Eigen::ArrayXd &counts() const
  {
    return counts_;
  }
  std::shared_ptr<const FragmentationDos> extend(const Eigen::ArrayXd &frequencies) const;
};
using FragmentationDosPtr = std::shared_ptr<const FragmentationDos>;

// Immutable, shared across realizations. DOS bins represent excitation i * dE,
// including the ground state at zero. Runtime energies are in joules.
class FragmentationProducts
{
  double bin_width;
  FragmentationDosPtr daughter_dos_, neutral_dos_, combined_dos_;

public:
  const ClusterDerived daughter, neutral;
  const FragmentationDosPtr &daughter_dos() const
  {
    return daughter_dos_;
  }
  const FragmentationDosPtr &neutral_dos() const
  {
    return neutral_dos_;
  }
  const FragmentationDosPtr &combined_dos() const
  {
    return combined_dos_;
  }

  FragmentationProducts(const MolecularCluster &daughter, const ClusterSpec &neutral,
                        double bin_width, double energy_max, std::optional<Eigen::ArrayXXd> log_dos = std::nullopt);
  FragmentationProducts(const MolecularCluster &daughter, const ClusterSpec &neutral,
                        FragmentationDosPtr daughter_dos, FragmentationDosPtr neutral_dos,
                        FragmentationDosPtr combined_dos);

  // Cache snapshot: daughter, neutral, combined log counts. Atom-like neutral
  // columns are -infinity; these are counts, not the rate DOS densities.
  Eigen::ArrayXXd log_dos() const;

  // Fixed-J spherical phase space, with the paper's unlimited capture parameter.
  // Returns nullopt on cancellation. Does not change the fragmentation hazard.
  std::optional<FragmentationState> sample(std::mt19937 &gen, double available_energy,
                                           const Eigen::Vector3d &momentum, const Eigen::Vector3d &velocity,
                                           OperationContext &operation) const;
};
using FragmentationProductsPtr = std::shared_ptr<const FragmentationProducts>;
