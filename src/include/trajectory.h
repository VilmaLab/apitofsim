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

TimeNextCollOutcome next_coll_old(double collision_quantile, ParticleState &particle, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double t_fragmentation, const Eigen::Array4d &acc, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole);
TimeNextCollOutcome next_coll(double collision_quantile, ParticleState &particle, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double t_fragmentation, const Eigen::Array4d &acc, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole, const CollisionTimeOptions &options = {}, const RFBasis *rf_basis = nullptr);

int zone(double z, const CumulativeLengths &clens);

// Columns are collision-frame axes expressed in lab coordinates.
Eigen::Matrix3d collision_frame(const Eigen::Vector3d &v_cluster, double theta, double phi, double alpha);

std::optional<std::array<double, 2>> rf_parameters(const std::optional<Quadrupole> &quadrupole);
void validate_rf_basis(const std::optional<Quadrupole> &quadrupole, const CollisionTimeOptions &options, const RFBasis *basis);
