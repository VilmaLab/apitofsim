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

// Immutable, shared across realizations. DOS bins represent excitation i * dE,
// including the ground state at zero. Runtime energies are in joules.
class FragmentationProducts
{
  Eigen::ArrayXd daughter_log_dos, neutral_log_dos, combined_log_dos;
  double bin_width;

public:
  const ClusterDerived daughter, neutral;

  FragmentationProducts(const MolecularCluster &daughter, const ClusterSpec &neutral,
                        double bin_width, double energy_max, std::optional<Eigen::ArrayXXd> log_dos = std::nullopt);

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
