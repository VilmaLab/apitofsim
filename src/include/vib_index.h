#pragma once

#include "exceptions.h"
#include <Eigen/Dense>
#include <bit>
#include <memory>

enum class VibEnergyMode
{
  direct,
  dos_only,
  group_dos
};

// Arrays use the original DOS units. Both rejection laws cancel a common DOS scale.
struct VibEnergyIndex
{
  const VibEnergyMode mode;
  const Eigen::ArrayXd data;

  VibEnergyIndex(VibEnergyMode mode, Eigen::ArrayXd values);
  void check_length(int bins) const;
};
using VibIndexPtr = std::shared_ptr<const VibEnergyIndex>;
