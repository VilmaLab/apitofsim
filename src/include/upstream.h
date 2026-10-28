#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include "exceptions.h"

namespace upstream
{
// Earliest strictly future crossing, selected by the velocity at the boundary.
inline double crossing_time(double z, double velocity, double acceleration, bool inward)
{
  double result = std::numeric_limits<double>::infinity();
  auto consider = [&](double dt)
  {
    double final_velocity = velocity + acceleration * dt;
    if (dt > 0 && std::isfinite(dt) && (inward ? final_velocity > 0 : final_velocity < 0))
      result = std::min(result, dt);
  };
  if (acceleration == 0)
  {
    if (velocity != 0)
      consider(-z / velocity);
  }
  else
  {
    double discriminant = std::fma(-2 * acceleration, z, velocity * velocity);
    if (discriminant > 0)
    {
      double q = -0.5 * (velocity + std::copysign(std::sqrt(discriminant), velocity));
      consider(q / (0.5 * acceleration));
      if (q != 0)
        consider(z / q);
    }
  }
  return result;
}

// Return true when fragmentation wins, including a tie with return.
inline bool advance(Eigen::Vector3d &velocity, double &z, double &x, double &y,
                    double acceleration, double remaining_fragmentation, double &time, double &elapsed)
{
  double return_time = crossing_time(z, velocity[2], acceleration, true);
  if (!std::isfinite(return_time))
    throw ApiTofParticleEscapeError([&](auto &msg)
    {
      msg << "Particle cannot return to z = 0: z = " << z
          << " m, axial velocity = " << velocity[2]
          << " m/s, acceleration = " << acceleration << " m/s^2";
    });
  bool fragmented = remaining_fragmentation <= return_time;
  double dt = fragmented ? remaining_fragmentation : return_time;
  x += velocity[0] * dt;
  y += velocity[1] * dt;
  z += (velocity[2] + 0.5 * acceleration * dt) * dt;
  velocity[2] += acceleration * dt;
  time += dt;
  elapsed += dt;
  if (!fragmented || remaining_fragmentation == return_time)
    z = 0;
  return fragmented;
}
} // namespace upstream
