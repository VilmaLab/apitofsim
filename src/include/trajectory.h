#pragma once

#include "mass_spec.h"

// Internal trajectory sampler: callers draw the collision quantile, while tests
// can supply explicit event inputs without depending on random distributions.
enum struct TimeNextCollOutcome
{
  fragmentation,
  gas_collision,
  escape
};

TimeNextCollOutcome time_next_coll_quadrupole(double collision_quantile, ParticleState &particle, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double t_fragmentation, const Eigen::Array4d &acc, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole);

int zone(double z, const CumulativeLengths &clens);

// Columns are collision-frame axes expressed in lab coordinates.
Eigen::Matrix3d collision_frame(const Eigen::Vector3d &velocity, double theta, double phi, double alpha);
