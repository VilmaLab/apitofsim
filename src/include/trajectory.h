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

TimeNextCollOutcome time_next_coll_quadrupole(double collision_quantile, Eigen::Vector3d &v_cluster, double &v_cluster_norm, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double &z, double &x, double &y, double &t_fragmentation, const Eigen::Array4d &acc, double &t, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole);

int zone(double z, const CumulativeLengths &clens);
