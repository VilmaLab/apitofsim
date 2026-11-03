#include "trajectory.h"
#include "samplers.h"
#include "upstream.h"
#include <array>
#include <numeric>

namespace
{
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double hazard_rtol = 1e-9;
constexpr double hazard_atol = 1e-12;

struct Estimate
{
  double value;
  double error;
};

// Embedded Gauss(7)/Kronrod(15) rule. Odd-indexed nodes and the center
// belong to both rules, so the error estimate requires no extra rate calls.
template <typename Rate>
Estimate panel(const Rate &rate, double lo, double hi)
{
  constexpr std::array nodes{0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
                             0.7415311855993944, 0.5860872354676911, 0.4058451513773972, 0.2077849550078985};
  constexpr std::array kronrod{0.02293532201052922, 0.06309209262997855, 0.1047900103222502,
                               0.1406532597155259, 0.1690047266392679, 0.1903505780647854, 0.2044329400752989};
  constexpr std::array gauss{0.1294849661688697, 0.2797053914892767, 0.3818300505051189};
  double mid = std::midpoint(lo, hi);
  double half = (hi - lo) / 2;
  double center = rate(mid);
  double k = 0.2094821410847278 * center;
  double g = 0.4179591836734694 * center;
  for (size_t i = 0; i < nodes.size(); ++i)
  {
    double pair = rate(mid - half * nodes[i]) + rate(mid + half * nodes[i]);
    k += kronrod[i] * pair;
    if (i % 2 == 1)
      g += gauss[i / 2] * pair;
  }
  return {half * k, half * std::abs(k - g)};
}

// Consume accepted panels from the left. Only subdivide an accurate panel
// further if it contains the collision, or its error might change that decision.
template <typename Rate>
std::optional<double> consume(const Rate &rate, double lo, double hi, double &remaining, double atol, double event_tol, int depth = 0)
{
  auto [value, error] = panel(rate, lo, hi);
  double roundoff = 16 * std::numeric_limits<double>::epsilon() * value;
  if (!std::isfinite(value) || !std::isfinite(error))
    throw ApiTofUnexpectedNumericalError("Nonfinite collision hazard integral");
  if (error <= atol + hazard_rtol * value)
  {
    if (value <= remaining && (remaining - value >= error || error <= roundoff))
    {
      remaining -= value;
      return std::nullopt;
    }
    // Localize within the final panel when its hazard, or the residual at
    // the right endpoint, is already within the event tolerance.
    if (value > remaining && (value <= event_tol || (value - remaining <= event_tol && error < value - remaining)) && error <= atol + roundoff)
      return lo + (hi - lo) * remaining / value;
  }
  double mid = std::midpoint(lo, hi);
  if (error <= atol + hazard_rtol * value && value > remaining)
    mid = lo + (hi - lo) * std::clamp(remaining / value, 0.1, 0.9);
  if (depth >= 100 || mid == lo || mid == hi)
    throw ApiTofUnexpectedNumericalError("Collision hazard refinement failed to converge");
  if (auto event = consume(rate, lo, mid, remaining, atol / 2, event_tol, depth + 1))
    return event;
  return consume(rate, mid, hi, remaining, atol / 2, event_tol, depth + 1);
}

ParticleState drift(const ParticleState &start, double dt, double az)
{
  ParticleState end = start;
  end.position += dt * start.velocity;
  end.position.z() += 0.5 * az * dt * dt;
  end.velocity.z() += az * dt;
  end.time += dt;
  return end;
}

ParticleState static_field(const ParticleState &start, double s, double az, double k)
{
  ParticleState end = drift(start, s, az);
  // In a static quadrupole each transverse axis satisfies x'' = k*x.
  if (k != 0)
  {
    double w = std::sqrt(std::abs(k));
    for (int axis = 0; axis < 2; ++axis)
    {
      double coeff = axis == 0 ? k : -k;
      double c = coeff > 0 ? std::cosh(w * s) : std::cos(w * s);
      double b = (coeff > 0 ? std::sinh(w * s) : std::sin(w * s)) / w;
      end.position[axis] = c * start.position[axis] + b * start.velocity[axis];
      end.velocity[axis] = coeff * b * start.position[axis] + c * start.velocity[axis];
    }
  }
  return end;
}

double rf_factor(const Quadrupole &q, double time)
{
  return q.mathieu_factor * (q.ac_field * std::cos(q.angular_velocity * time) - q.dc_field);
}

ParticleState verlet(const ParticleState &start, double dt, double az, const Quadrupole &q)
{
  ParticleState half = start;
  auto kick = [](ParticleState &p, double scale)
  {
    p.velocity.x() += scale * p.position.x();
    p.velocity.y() -= scale * p.position.y();
  };
  kick(half, dt / 2 * rf_factor(q, start.time));
  ParticleState end = drift(half, dt, az);
  kick(end, dt / 2 * rf_factor(q, end.time));
  return end;
}

struct RFPanel
{
  ParticleState end;
  double hazard;
  double error;
};

template <typename Rate>
bool rf_step(ParticleState &particle, double &dt, double &remaining, double az, const Quadrupole &q, double max_step, const Rate &rate)
{
  double field_scale = std::sqrt(std::abs(q.mathieu_factor) * (std::abs(q.dc_field) + std::abs(q.ac_field)));
  dt = std::min({dt, 2 * max_step, 0.02 / field_scale, 0.02 / std::abs(q.angular_velocity)});
  double initial_rate = rate(particle);
  auto propagate = [&](double duration)
  {
    auto mid = verlet(particle, duration / 2, az, q);
    auto end = verlet(mid, duration / 2, az, q);
    double midpoint_rate = rate(mid);
    double end_rate = rate(end);
    double simpson = duration / 6 * (initial_rate + 4 * midpoint_rate + end_rate);
    double trapezoid = duration / 2 * (initial_rate + end_rate);
    return RFPanel{end, simpson, std::abs(simpson - trapezoid)};
  };
  RFPanel panel = propagate(dt);
  // Near a collision/deadline decision, shorten the interval if even the
  // embedded trapezoid disagrees about which side of the threshold we are on.
  while (panel.error > 16 * std::numeric_limits<double>::epsilon() * panel.hazard &&
         std::abs(panel.hazard - remaining) <= panel.error)
  {
    dt /= 2;
    panel = propagate(dt);
  }
  if (!std::isfinite(panel.hazard))
    throw ApiTofUnexpectedNumericalError("Nonfinite RF collision hazard");
  if (remaining < panel.hazard)
  {
    double lo = 0;
    double hi = dt;
    for (int i = 0; i < 64; ++i)
    {
      double mid = std::midpoint(lo, hi);
      double trial = propagate(mid).hazard;
      if (std::abs(trial - remaining) <= 1e-10 * remaining || mid == lo || mid == hi)
      {
        lo = hi = mid;
        break;
      }
      if (trial < remaining)
        lo = mid;
      else
        hi = mid;
    }
    dt = std::midpoint(lo, hi);
    particle = propagate(dt).end;
    return true;
  }
  remaining -= panel.hazard;
  particle = panel.end;
  return false;
}

struct Flight
{
  double end;
  double az;
  double boundary;
  int region;
};

// Use strictly future crossings. Include only skimmer mesh knots that are
// present in the gas table (the interpolator is constant past the last row).
Flight flight(const ParticleState &p, const CumulativeLengths &l, const Eigen::Array4d &acc,
              double mesh, Eigen::Index rows)
{
  const std::array edges{0.0, l.first_chamber_end, l.sk_end, l.quadrupole_start, l.quadrupole_end, l.second_chamber_end};
  const std::array acceleration{acc[0], 0.0, acc[1], acc[2], acc[3]};
  double z = p.position.z();
  const auto *it = std::ranges::upper_bound(edges, z);
  int region = static_cast<int>(it - edges.begin()) - 1;
  bool at_rest = p.velocity.z() == 0;
  double az = acceleration[region];
  if (z == edges[region] && (p.velocity.z() < 0 || (at_rest && az < 0)))
  {
    int left = static_cast<int>(std::ranges::lower_bound(edges, z) - edges.begin()) - 1;
    if (left >= 0 && (!at_rest || acceleration[left] < 0))
    {
      region = left;
      az = acceleration[left];
    }
    else
    {
      // At rest at a potential minimum: neither side accelerates away.
      az = 0;
    }
  }
  double low = edges[region];
  double high = edges[region + 1];
  if (region == 1)
  {
    auto cell = static_cast<Eigen::Index>((z - low) / mesh);
    for (Eigen::Index i = std::max<Eigen::Index>(1, cell - 1); i <= std::min(rows - 1, cell + 2); ++i)
    {
      double knot = edges[1] + static_cast<double>(i) * mesh;
      if (knot < z)
        low = std::max(low, knot);
      if (knot > z)
        high = std::min(high, knot);
    }
  }
  double back = upstream::crossing_time(z - low, p.velocity.z(), az, false);
  double forward = upstream::crossing_time(z - high, p.velocity.z(), az, true);
  return back < forward ? Flight{back, az, low, region} : Flight{forward, az, high, region};
}
} // namespace

TimeNextCollOutcome next_coll(double collision_quantile, ParticleState &particle, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double t_fragmentation, const Eigen::Array4d &acc, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole)
{
  double remaining = collision_quantile == 0 ? infinity : -std::log(collision_quantile);
  const double event_tol = hazard_atol + hazard_rtol * remaining;
  double elapsed = 0;
  while (true)
  {
    double z = particle.position.z();
    if (z >= chamber.clens.second_chamber_end)
      return TimeNextCollOutcome::escape;
    if (elapsed >= t_fragmentation)
      return TimeNextCollOutcome::fragmentation;
    if (z < 0 || (z == 0 && (particle.velocity.z() < 0 || (particle.velocity.z() == 0 && acc[0] < 0))))
    {
      if (upstream::advance(particle.velocity, particle.position.z(), particle.position.x(), particle.position.y(),
                            acc[0], t_fragmentation - elapsed, particle.time, elapsed))
        return TimeNextCollOutcome::fragmentation;
      continue;
    }
    auto [crossing, az, boundary, region] = flight(particle, chamber.clens, acc, mesh_skimmer, skimmer.rows());
    double deadline = t_fragmentation - elapsed;
    double dt = std::min(crossing, deadline);
    // Choose the one-sided rate at an interface from the region traversed.
    auto rate = [&](const ParticleState &p)
    {
      if (region == 1)
        return collision_frequency(std::clamp(p.position.z(), chamber.clens.first_chamber_end, chamber.clens.sk_end),
                                   p.velocity, chamber, m_gas, R, skimmer, mesh_skimmer);
      return coll_freq(chamber.pressures.n[region == 0 ? 0 : 1], chamber.mobility_gas, chamber.mobility_gas_inv, R, p.velocity.norm());
    };
    if (!std::isfinite(dt) && (!std::isfinite(remaining) || rate(particle) == 0))
      throw ApiTofUnexpectedNumericalError("Flight has no reachable event");
    if (remaining == 0 && rate(particle) > 0)
      return TimeNextCollOutcome::gas_collision;
    bool collision = false;
    bool field = region == 3 && quadrupole && quadrupole->mathieu_factor != 0 && (quadrupole->ac_field != 0 || quadrupole->dc_field != 0);
    bool rf = field && quadrupole->ac_field != 0 && quadrupole->angular_velocity != 0;
    if (rf)
    {
      collision = rf_step(particle, dt, remaining, az, *quadrupole, dts[1], rate);
    }
    else
    {
      double k = field ? rf_factor(*quadrupole, particle.time) : 0;
      if (k != 0)
        dt = std::min(dt, 1 / std::sqrt(std::abs(k)));
      auto advance = [&](double s)
      { return static_field(particle, s, az, k); };
      bool constant = k == 0 && az == 0 && (region != 1 || particle.velocity.z() == 0);
      if (constant)
      {
        double h = rate(particle);
        double wait = h > 0 ? remaining / h : infinity;
        if (wait < dt)
        {
          dt = wait;
          collision = true;
        }
        else if (std::isfinite(dt))
        {
          remaining = std::max(0.0, remaining - h * dt);
        }
      }
      else if (std::isfinite(remaining))
      {
        auto h = [&](double s)
        { return rate(advance(s)); };
        // A local horizon avoids integrating far beyond the likely event.
        double h0 = h(0);
        if (h0 > 0)
          dt = std::min(dt, 2 * std::max(remaining, hazard_atol) / h0);
        if (auto event = consume(h, 0, dt, remaining, hazard_atol, event_tol))
        {
          dt = *event;
          collision = true;
        }
      }
      if (!std::isfinite(dt))
        throw ApiTofUnexpectedNumericalError("Flight has no reachable event");
      particle = advance(dt);
    }
    if (collision)
      return TimeNextCollOutcome::gas_collision;
    if (dt == crossing)
      particle.position.z() = boundary;
    if (particle.position.z() >= chamber.clens.second_chamber_end)
      return TimeNextCollOutcome::escape;
    if (dt == deadline)
      return TimeNextCollOutcome::fragmentation;
    if (remaining == 0)
      return TimeNextCollOutcome::gas_collision;
    if (!(elapsed + dt > elapsed))
      throw ApiTofUnexpectedNumericalError("Collision flight made no time progress");
    elapsed += dt;
  }
}
