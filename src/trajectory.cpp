#include "trajectory.h"
#include "samplers.h"
#include "upstream.h"
#include <array>
#include <numeric>

namespace
{
constexpr double infinity = std::numeric_limits<double>::infinity();
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
    double pair = rate(mid - (half * nodes[i])) + rate(mid + (half * nodes[i]));
    k += kronrod[i] * pair;
    if (i % 2 == 1)
      g += gauss[i / 2] * pair;
  }
  return {half * k, half * std::abs(k - g)};
}

// Consume accepted panels from the left. Only subdivide an accurate panel
// further if it contains the collision, or its error might change that decision.
template <typename Rate>
std::optional<double> consume(const Rate &rate, double lo, double hi, double &remaining, double atol, double rtol, double event_tol, int depth = 0)
{
  auto [value, error] = panel(rate, lo, hi);
  double roundoff = 16 * std::numeric_limits<double>::epsilon() * value;
  if (!std::isfinite(value) || !std::isfinite(error))
    throw ApiTofUnexpectedNumericalError("Nonfinite collision hazard integral");
  if (error <= atol + (rtol * value))
  {
    if (value <= remaining && (remaining - value >= error || error <= roundoff))
    {
      remaining -= value;
      return std::nullopt;
    }
    // Localize within the final panel when its hazard, or the residual at
    // the right endpoint, is already within the event tolerance.
    if (value > remaining && (value <= event_tol || (value - remaining <= event_tol && error < value - remaining)) && error <= atol + roundoff)
      return lo + ((hi - lo) * remaining / value);
  }
  double mid = std::midpoint(lo, hi);
  if (error <= atol + (rtol * value) && value > remaining)
    mid = lo + ((hi - lo) * std::clamp(remaining / value, 0.1, 0.9));
  if (depth >= 100 || mid == lo || mid == hi)
    throw ApiTofUnexpectedNumericalError("Collision hazard refinement failed to converge");
  if (auto event = consume(rate, lo, mid, remaining, atol / 2, rtol, event_tol, depth + 1))
    return event;
  return consume(rate, mid, hi, remaining, atol / 2, rtol, event_tol, depth + 1);
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
      end.position[axis] = (c * start.position[axis]) + (b * start.velocity[axis]);
      end.velocity[axis] = (coeff * b * start.position[axis]) + (c * start.velocity[axis]);
    }
  }
  return end;
}

double rf_factor(const Quadrupole &q, double time)
{
  return q.mathieu_factor * ((q.ac_field * std::cos(q.angular_velocity * time)) - q.dc_field);
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
  int region = static_cast<int>(std::ranges::upper_bound(edges, z) - edges.begin()) - 1;
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
      double knot = edges[1] + (static_cast<double>(i) * mesh);
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

CollisionTimeOptions::CollisionTimeOptions(CollisionTimeMethod method, double hazard_atol, double hazard_rtol)
    : method(method), hazard_atol(hazard_atol), hazard_rtol(hazard_rtol)
{
  if (!std::isfinite(hazard_atol) || hazard_atol <= 0)
    throw std::invalid_argument("hazard_atol must be finite and positive");
  if (!std::isfinite(hazard_rtol) || hazard_rtol <= 0)
    throw std::invalid_argument("hazard_rtol must be finite and positive");
}

std::optional<std::array<double, 2>> rf_parameters(const std::optional<Quadrupole> &quadrupole)
{
  if (!quadrupole || quadrupole->mathieu_factor == 0 || quadrupole->ac_field == 0 || quadrupole->angular_velocity == 0)
    return std::nullopt;
  const auto &q = *quadrupole;
  const double omega2 = q.angular_velocity * q.angular_velocity;
  return std::array{q.mathieu_factor * q.ac_field / omega2, q.mathieu_factor * q.dc_field / omega2};
}

void validate_rf_basis(const std::optional<Quadrupole> &quadrupole, const CollisionTimeOptions &options, const RFBasis *basis)
{
  if (options.method == CollisionTimeMethod::legacy)
    return;
  if (auto parameters = rf_parameters(quadrupole))
    if (basis == nullptr || !basis->matches((*parameters)[0], (*parameters)[1]))
      throw std::invalid_argument("integrated_hazard requires a matching prepared RF basis for each particle");
}

TimeNextCollOutcome next_coll(double collision_quantile, ParticleState &particle, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double t_fragmentation, const Eigen::Array4d &acc, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole, const CollisionTimeOptions &options, const RFBasis *rf_basis, const std::function<double()> &uniform)
{
  validate_rf_basis(quadrupole, options, rf_basis);
  if (options.method == CollisionTimeMethod::legacy)
    return next_coll_old(collision_quantile, particle, chamber, R, dts, t_fragmentation, acc, m_gas, skimmer, mesh_skimmer, quadrupole);
  auto draw = [&]
  {
    if (!uniform)
      throw std::invalid_argument("RF collision timing requires a realization-local RNG");
    const double u = uniform();
    if (!std::isfinite(u) || u < 0 || u >= 1)
      throw std::invalid_argument("Thinning RNG must return values in [0, 1)");
    return u;
  };
  double remaining = collision_quantile == 0 ? infinity : -std::log(collision_quantile);
  const double event_tol = options.hazard_atol + (options.hazard_rtol * remaining);
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
    bool collision = false;
    bool field = region == 3 && quadrupole && quadrupole->mathieu_factor != 0 && (quadrupole->ac_field != 0 || quadrupole->dc_field != 0);
    bool rf = field && quadrupole->ac_field != 0 && quadrupole->angular_velocity != 0;
    if (remaining == 0 && !rf && rate(particle) > 0)
      return TimeNextCollOutcome::gas_collision;
    if (rf)
    {
      const auto &q = *quadrupole;
      const auto anchor = rf_basis->anchor(q.angular_velocity * particle.time,
                                           {{{particle.position.x(), particle.velocity.x() / q.angular_velocity},
                                             {particle.position.y(), particle.velocity.y() / q.angular_velocity}}});
      auto advance = [&](double duration)
      {
        auto end = drift(particle, duration, az);
        auto transverse = anchor.evaluate(q.angular_velocity * duration);
        for (int axis = 0; axis < 2; ++axis)
        {
          end.position[axis] = transverse[axis][0];
          end.velocity[axis] = q.angular_velocity * transverse[axis][1];
        }
        return end;
      };
      // Choose the window/bound independently of the exponential proposal.
      dt = std::min(dt, rf_detail::period / std::abs(q.angular_velocity));
      const auto transverse = anchor.speed_bound(q.angular_velocity * dt);
      const double vz = std::max(std::abs(particle.velocity.z()), std::abs(particle.velocity.z() + (az * dt)));
      const double speed = std::hypot(std::abs(q.angular_velocity) * std::hypot(transverse[0], transverse[1]), vz);
      double bound = coll_freq(chamber.pressures.n[1], chamber.mobility_gas, chamber.mobility_gas_inv, R, speed);
      bound *= 1 + 1e-12;
      if (!std::isfinite(bound) || bound < 0)
        throw ApiTofUnexpectedNumericalError("Invalid RF thinning envelope");

      double offset = 0;
      while (bound > 0 && std::isfinite(remaining))
      {
        const double wait = remaining / bound;
        if (!(wait < dt - offset))
        {
          remaining = std::max(0.0, remaining - (bound * (dt - offset)));
          break;
        }
        const double proposal = offset + wait;
        if (remaining > 0 && !(proposal > offset))
          throw ApiTofUnexpectedNumericalError("RF thinning proposal made no progress");
        offset = proposal;
        const auto candidate = advance(offset);
        const double hazard = rate(candidate);

        if (!std::isfinite(hazard) || hazard < 0 || hazard > bound)
          throw ApiTofUnexpectedNumericalError("RF thinning envelope violated");

        if (draw() * bound < hazard)
        {
          dt = offset;
          particle = candidate;
          collision = true;
          break;
        }

        const double u = draw();
        remaining = u == 0 ? infinity : -std::log(u);
      }
      // The proposal residual is exponential conditional on survival at the
      // window boundary; it may be carried into either solver/next region.
      if (!collision)
        particle = advance(dt);
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
          remaining = std::max(0.0, remaining - (h * dt));
        }
      }
      else if (std::isfinite(remaining))
      {
        auto h = [&](double s)
        { return rate(advance(s)); };
        // A local horizon avoids integrating far beyond the likely event.
        double h0 = h(0);
        if (h0 > 0)
          dt = std::min(dt, 2 * std::max(remaining, options.hazard_atol) / h0);
        if (auto event = consume(h, 0, dt, remaining, options.hazard_atol, options.hazard_rtol, event_tol))
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
    if (!(elapsed + dt > elapsed))
      throw ApiTofUnexpectedNumericalError("Collision flight made no time progress");
    elapsed += dt;
  }
}
