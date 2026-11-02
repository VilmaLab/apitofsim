#pragma once

#include <Eigen/Dense>

struct ParticleState
{
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double time = 0.0;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  double vibrational_energy = 0.0;

  double rotational_energy(double inertia) const
  {
    return 0.5 * inertia * omega.squaredNorm();
  }

  double internal_energy(double inertia) const
  {
    return rotational_energy(inertia) + vibrational_energy;
  }
};
