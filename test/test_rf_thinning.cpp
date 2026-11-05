#include <doctest.h>
#include "trajectory.h"
#include "samplers.h"
#include <random>
#include <limits>

namespace
{
RFBasis rf_fixture_basis(const Quadrupole &q)
{
  const auto parameters = rf_parameters(q);
  if (!parameters)
    throw std::logic_error("RF test fixture must have nonzero AC and frequency");
  return {(*parameters)[0], (*parameters)[1]};
}
} // namespace

TEST_CASE("RF thinning bounds cover the actual Hermite velocity interpolant")
{
  std::mt19937 gen(971);
  std::uniform_real_distribution<double> unif(0, 1);
  for (double ac : {0.0, 0.01, 0.1, 0.35})
  {
    RFBasis built(ac, 0.02);
    RFBasis basis(ac, 0.02, built.data());
    for (int trial = 0; trial < 80; ++trial)
    {
      const double phase = ((40 * unif(gen)) - 20) * rf_detail::period;
      std::array<rf_detail::State, 2> initial;
      for (auto &axis : initial)
        axis = {(20 * unif(gen)) - 10, (20 * unif(gen)) - 10};
      const auto anchor = basis.anchor(phase, initial);
      for (double fraction : {-1.0, -0.5, -0.125, 0.125, 0.5, 1.0})
      {
        const double delta = fraction * rf_detail::period;
        const auto bound = anchor.speed_bound(delta);
        for (int sample = 0; sample <= 200; ++sample)
        {
          const auto point = anchor.evaluate(delta * sample / 200);
          for (int axis = 0; axis < 2; ++axis)
            REQUIRE(std::abs(point[axis][1]) <= bound[axis]);
        }
      }
    }
    CHECK_THROWS_AS(basis.anchor(0, {{{1, 0}, {1, 0}}}).speed_bound(7), std::invalid_argument);
  }
}

TEST_CASE("RF thinning survival matches independently integrated hazard")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 100, 0, 300, 100;
  Quadrupole q{0, 3, 1, 0.01};
  q.mathieu_factor = 0.5;
  MassSpectrometer ms{skimmer, 5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Constant(1e-5), q};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  const auto basis = rf_fixture_basis(q);
  const double radius = 7e-10;
  const double deadline = 0.7;
  const std::array cuts{0.1, 0.3, deadline};
  for (double phase : {0.01, 1.3, 3.2, 6.2})
  {
    ParticleState initial{.position = {0.01, -0.03, 35}, .time = phase / q.angular_velocity, .velocity = {100, -200, 0.01}};
    auto anchor = basis.anchor(phase, {{{initial.position.x(), initial.velocity.x() / q.angular_velocity}, {initial.position.y(), initial.velocity.y() / q.angular_velocity}}});
    auto rate = [&](double t)
    {
      auto state = anchor.evaluate(t * q.angular_velocity);
      const double speed = std::hypot(q.angular_velocity * std::hypot(state[0][1], state[1][1]), initial.velocity.z());
      return coll_freq(chamber.pressures.n[1], chamber.mobility_gas, chamber.mobility_gas_inv, radius, speed);
    };
    std::array<double, 3> expected;
    for (int cut = 0; cut < 3; ++cut)
    {
      const int panels = 4096;
      const double h = cuts[cut] / panels;
      double integral = rate(0) + rate(cuts[cut]);
      for (int i = 1; i < panels; ++i)
        integral += (((i % 2) != 0) ? 4 : 2) * rate(i * h);
      expected[cut] = -std::expm1(-integral * h / 3);
    }
    const CollisionTimeOptions options;
    std::mt19937 gen(314);
    std::uniform_real_distribution<double> unif(0, 1);
    const std::function<double()> uniform = [&]
    { return unif(gen); };
    std::array<int, 3> counts{};
    const int n = 6000;
    for (int i = 0; i < n; ++i)
    {
      auto particle = initial;
      const auto result = next_coll(uniform(), particle, chamber, radius, Eigen::Array2d::Ones(), deadline, Eigen::Array4d::Zero(), gas.mass, skimmer, 5, q, options, &basis, uniform);
      const double elapsed = particle.time - initial.time;
      if (result == TimeNextCollOutcome::gas_collision)
      {
        for (int cut = 0; cut < 3; ++cut)
          counts[cut] += static_cast<int>(elapsed <= cuts[cut]);
      }
      else
      {
        REQUIRE(result == TimeNextCollOutcome::fragmentation);
        REQUIRE(elapsed == doctest::Approx(deadline));
      }
    }
    for (int cut = 0; cut < 3; ++cut)
    {
      const double sigma = std::sqrt(expected[cut] * (1 - expected[cut]) / n);
      CHECK(std::abs((double(counts[cut]) / n) - expected[cut]) < (6 * sigma) + 0.002);
    }
  }
}

TEST_CASE("RF thinning preserves zero-hazard boundaries and deadlines")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 0, 0, 300, 0;
  Quadrupole q{0, 3, 1, 0.01};
  q.mathieu_factor = 0.5;
  MassSpectrometer ms{skimmer, 5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Zero(), q};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  const auto basis = rf_fixture_basis(q);
  for (double vz : {-1.0, 0.0, 1.0})
    for (double deadline : {0.1, 0.5, 2.0})
    {
      ParticleState initial{.position = {0.01, -0.03, 39.9}, .time = 0.99, .velocity = {100, -200, vz}};
      auto expected = initial;
      auto actual = initial;
      const Eigen::Array4d acceleration{0, 0, 1, 0};
      const auto outcome = next_coll(0, expected, chamber, 7e-10, Eigen::Array2d::Ones(), deadline, acceleration, gas.mass, skimmer, 5, q, {}, &basis);
      const std::function<double()> unused = []() -> double
      { throw std::runtime_error("Zero hazard consumed RNG"); };
      CHECK(next_coll(0.5, actual, chamber, 7e-10, Eigen::Array2d::Ones(), deadline, acceleration, gas.mass, skimmer, 5, q,
                      {}, &basis, unused) == outcome);
      CHECK(actual.time == doctest::Approx(expected.time));
      CHECK(actual.position.isApprox(expected.position, 1e-9));
      CHECK(actual.velocity.isApprox(expected.velocity, 1e-9));
    }
}

TEST_CASE("RF thinning consumes realization-local proposal and acceptance draws")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 100, 0, 300, 100;
  Quadrupole q{0, 3, 1, 0.01};
  q.mathieu_factor = 0.5;
  MassSpectrometer ms{skimmer, 5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Constant(1e-5), q};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  const auto basis = rf_fixture_basis(q);
  auto run = [&](const std::function<double()> &uniform)
  {
    ParticleState particle{.position = {1, 0, 35}, .velocity = {0, 0, 0.01}};
    const auto outcome = next_coll(1, particle, chamber, 7e-10, Eigen::Array2d::Ones(), 0.5, Eigen::Array4d::Zero(), gas.mass, skimmer, 5, q, {}, &basis, uniform);
    return std::pair{outcome, particle.time};
  };
  CHECK_THROWS_AS(run({}), std::invalid_argument);
  for (double invalid : {-1.0, 1.0, std::numeric_limits<double>::quiet_NaN()})
    CHECK_THROWS_AS(run([=]
    { return invalid; }),
                    std::invalid_argument);
  const auto accepted = run([]
  { return 0.0; });
  CHECK(accepted.first == TimeNextCollOutcome::gas_collision);
  CHECK(accepted.second == 0);
  int draws = 0;
  const auto rejected = run([&]
  { return ++draws == 1 ? std::nextafter(1.0, 0.0) : 0.0; });
  CHECK(draws == 2);
  CHECK(rejected.first == TimeNextCollOutcome::fragmentation);
  CHECK(rejected.second == doctest::Approx(0.5));
}
