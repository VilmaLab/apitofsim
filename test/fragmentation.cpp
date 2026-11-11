#include <doctest/doctest.h>
#include "fragmentation.h"
#include "consts.h"
#include <limits>

namespace
{
MolecularCluster molecule(int mass, int charge, double frequency = 1.)
{
  return {mass, 0., Eigen::Vector3d::Constant(1.), Eigen::ArrayXd::Constant(1, frequency), charge};
}
} // namespace

TEST_CASE("fragmentation phase space conserves energy and both momenta")
{
  for (const bool atomic : {false, true})
  {
    const auto daughter = molecule(100, -1);
    const ClusterSpec neutral = atomic ? ClusterSpec(AtomLikeCluster{20, 0., 0}) : ClusterSpec(molecule(20, 0));
    const FragmentationProducts products(daughter, neutral, consts::boltzmann, 100 * consts::boltzmann);
    const FragmentationProducts restored(daughter, neutral, consts::boltzmann, 100 * consts::boltzmann, products.log_dos());
    const auto daughter_dos = std::make_shared<FragmentationDos>(daughter.frequencies, consts::boltzmann, 100 * consts::boltzmann);
    const auto neutral_dos = atomic ? nullptr : std::make_shared<FragmentationDos>(std::get<MolecularCluster>(neutral).frequencies, consts::boltzmann, 100 * consts::boltzmann);
    const auto combined_dos = atomic ? daughter_dos : daughter_dos->extend(std::get<MolecularCluster>(neutral).frequencies);
    const FragmentationProducts shared(daughter, neutral, daughter_dos, neutral_dos, combined_dos);
    CHECK((products.log_dos() == shared.log_dos()).all());
    const double id = products.daughter.inertia_moment, in = products.neutral.inertia_moment;
    const double md = products.daughter.mass, mn = products.neutral.mass;
    const double q = 70 * consts::boltzmann;
    const Eigen::Vector3d velocity{30., -20., 1000.};
    OperationContext operation;
    std::mt19937 gen(42), repeat(42), shared_gen(42);
    for (double angular_scale : {0., 1., 100.})
    {
      const Eigen::Vector3d angular_momentum = Eigen::Vector3d{1., 2., 3.}.normalized() * angular_scale * std::sqrt(2 * id * q);
      for (int i = 0; i < 200; ++i)
      {
        const auto sampled = products.sample(gen, q, angular_momentum, velocity, operation);
        REQUIRE(sampled.has_value());
        const auto identical = restored.sample(repeat, q, angular_momentum, velocity, operation);
        const auto shared_state = shared.sample(shared_gen, q, angular_momentum, velocity, operation);
        REQUIRE(identical.has_value());
        REQUIRE(shared_state.has_value());
        CHECK(shared_state->velocity == identical->velocity);
        CHECK(shared_state->omega == identical->omega);
        CHECK(shared_state->neutral_velocity == identical->neutral_velocity);
        CHECK(shared_state->neutral_omega == identical->neutral_omega);
        CHECK(shared_state->orbital_momentum == identical->orbital_momentum);
        CHECK(shared_state->vibrational_energy == identical->vibrational_energy);
        CHECK(shared_state->neutral_vibrational_energy == identical->neutral_vibrational_energy);
        const auto &state = *sampled;
        CHECK(state.velocity == identical->velocity);
        CHECK(state.omega == identical->omega);
        CHECK(state.vibrational_energy == identical->vibrational_energy);
        CHECK(state.neutral_velocity == identical->neutral_velocity);
        CHECK(state.neutral_omega == identical->neutral_omega);
        CHECK(state.neutral_vibrational_energy == identical->neutral_vibrational_energy);
        CHECK(state.orbital_momentum == identical->orbital_momentum);
        const Eigen::Vector3d relative = state.velocity - state.neutral_velocity;
        const double energy = state.vibrational_energy + state.neutral_vibrational_energy + (0.5 * id * state.omega.squaredNorm()) + (0.5 * in * state.neutral_omega.squaredNorm()) + (0.5 * md * mn / (md + mn) * relative.squaredNorm());
        CHECK(energy / q == doctest::Approx(1.).epsilon(1e-10));
        CHECK(((md * state.velocity) + (mn * state.neutral_velocity) - ((md + mn) * velocity)).norm() / ((md + mn) * velocity.norm()) < 1e-12);
        CHECK(((id * state.omega) + (in * state.neutral_omega) + state.orbital_momentum - angular_momentum).norm() / std::sqrt(id * q) < 1e-10);
        CHECK(std::abs(state.orbital_momentum.dot(relative)) / (std::sqrt(id * q) * relative.norm()) < 1e-10);
        CHECK(state.vibrational_energy >= 0);
        CHECK(state.neutral_vibrational_energy >= 0);
        if (atomic)
        {
          CHECK(state.neutral_vibrational_energy == 0);
          CHECK(state.neutral_omega.isZero());
        }
      }
    }
    CHECK_THROWS_AS(products.sample(gen, 0., Eigen::Vector3d::Zero(), velocity, operation), ApiTofArgumentError);
    CHECK_THROWS_AS(products.sample(gen, 101 * consts::boltzmann, Eigen::Vector3d::Zero(), velocity, operation), ApiTofDosOverflow);
    // The ground-state bin remains accessible below one DOS bin width.
    const auto cold = products.sample(gen, 0.01 * consts::boltzmann, Eigen::Vector3d::Zero(), velocity, operation);
    REQUIRE(cold.has_value());
    CHECK(cold->vibrational_energy == 0);
    CHECK(cold->neutral_vibrational_energy == 0);
    operation.request_cancel();
    CHECK_FALSE(products.sample(gen, q, Eigen::Vector3d::Zero(), velocity, operation));
  }
}

TEST_CASE("fragmentation sampling reproduces the zero-J flux distribution")
{
  const double q = 20 * consts::boltzmann;
  const auto daughter = molecule(100, -1);
  for (bool atomic : {false, true})
  {
    const ClusterSpec neutral = atomic ? ClusterSpec(AtomLikeCluster{20, 0., 0}) : ClusterSpec(molecule(20, 0));
    const FragmentationProducts products(daughter, neutral, consts::boltzmann, 30 * consts::boltzmann);
    const double md = products.daughter.mass, mn = products.neutral.mass;
    std::mt19937 gen(123);
    OperationContext operation;
    Eigen::ArrayXd observed = Eigen::ArrayXd::Zero(20);
    double recoil_share = 0, daughter_fraction = 0;
    Eigen::Vector3d mean_direction = Eigen::Vector3d::Zero();
    constexpr int draws = 20000;
    for (int i = 0; i < draws; ++i)
    {
      const auto state = products.sample(gen, q, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), operation);
      REQUIRE(state.has_value());
      const double vibration = state->vibrational_energy + state->neutral_vibrational_energy;
      const double kinetic = 0.5 * md * mn / (md + mn) * (state->velocity - state->neutral_velocity).squaredNorm();
      observed[static_cast<int>(std::round(vibration / consts::boltzmann))] += 1;
      recoil_share += kinetic / (q - vibration);
      daughter_fraction += vibration > 0 ? state->vibrational_energy / vibration : (atomic ? 1. : 0.5);
      mean_direction += state->velocity.normalized();
    }
    Eigen::ArrayXd expected(20);
    for (int i = 0; i < 20; ++i)
      expected[i] = (atomic ? 1. : i + 1.) * std::pow(20. - i, atomic ? 1. : 2.5);
    expected /= expected.sum();
    // Independent marginal from the analytic flux integral, not sampler output.
    CHECK(((observed / draws) - expected).abs().maxCoeff() < 0.012);
    CHECK(recoil_share / draws == doctest::Approx(atomic ? 0.5 : 2. / 7.).epsilon(0.025));
    CHECK(daughter_fraction / draws == doctest::Approx(atomic ? 1. : 0.5).epsilon(0.025));
    CHECK(mean_direction.norm() / draws < 0.025);
  }
}

TEST_CASE("fragmentation product data reject invalid grids and unresolved modes")
{
  const auto daughter = molecule(100, -1), neutral = molecule(20, 0);
  CHECK_THROWS_AS(FragmentationProducts(daughter, neutral, 0., 1.), ApiTofArgumentError);
  CHECK_THROWS_AS(FragmentationProducts(daughter, neutral, consts::boltzmann, 0.), ApiTofArgumentError);
  CHECK_THROWS_AS(FragmentationProducts(daughter, neutral, 10 * consts::boltzmann, 100 * consts::boltzmann), ApiTofArgumentError);
  CHECK_THROWS_AS(FragmentationProducts(neutral, daughter, consts::boltzmann, 10 * consts::boltzmann), ApiTofArgumentError);
}

TEST_CASE("fragmentation directions follow the fixed-J flux law")
{
  OperationContext operation;
  std::mt19937 gen(47);
  const double q = 10 * consts::boltzmann;
  const auto daughter = molecule(100, -1, 1000.);
  for (bool atomic : {false, true})
  {
    const ClusterSpec neutral = atomic ? ClusterSpec(AtomLikeCluster{20, 0., 0}) : ClusterSpec(molecule(20, 0, 1000.));
    const FragmentationProducts products(daughter, neutral, consts::boltzmann, 100 * consts::boltzmann);
    const double sum = products.daughter.inertia_moment + products.neutral.inertia_moment;
    const Eigen::Vector3d angular_momentum = Eigen::Vector3d::UnitZ() * std::sqrt(8 * sum * q);
    double moment = 0;
    for (int i = 0; i < 10000; ++i)
    {
      const auto state = products.sample(gen, q, angular_momentum, Eigen::Vector3d::Zero(), operation);
      REQUIRE(state.has_value());
      const double scaled_cosine = 2 * state->velocity.normalized().z();
      CHECK(std::abs(scaled_cosine) <= 1.);
      moment += scaled_cosine * scaled_cosine;
    }
    // With ground-only vibration, p(c) proportional to (1-c*c)^a on [-1,1].
    CHECK(moment / 10000 == doctest::Approx(atomic ? 1. / 5. : 1. / 8.).epsilon(0.04));
  }
}

TEST_CASE("fragmentation state counts can exceed floating point density range")
{
  auto daughter = molecule(100, -1), neutral = molecule(20, 0);
  daughter.frequencies = Eigen::ArrayXd::Ones(200);
  neutral.frequencies = Eigen::ArrayXd::Ones(200);
  const FragmentationProducts products(daughter, neutral, consts::boltzmann, 10000 * consts::boltzmann);
  OperationContext operation;
  std::mt19937 gen(62);
  const auto state = products.sample(gen, 5000 * consts::boltzmann, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), operation);
  REQUIRE(state.has_value());
  CHECK(state->velocity.allFinite());
  CHECK(state->omega.allFinite());
  CHECK(state->vibrational_energy > 0);
  CHECK(state->neutral_vibrational_energy > 0);
}

TEST_CASE("fragmentation DOS matches enumerated oscillator occupations")
{
  constexpr int bins = 64;
  const double width = consts::boltzmann;
  Eigen::ArrayXd frequencies(3);
  frequencies << 2.49, 5.01, 7.7; // Rounded shifts 2, 5, 8.
  Eigen::ArrayXi expected = Eigen::ArrayXi::Zero(bins);
  for (int a = 0; 2 * a < bins; ++a)
    for (int b = 0; 2 * a + 5 * b < bins; ++b)
      for (int c = 0; 2 * a + 5 * b + 8 * c < bins; ++c)
        ++expected[2 * a + 5 * b + 8 * c];
  const FragmentationDos direct(frequencies, width, bins * width);
  const FragmentationDos marginal(frequencies.head(2), width, bins * width);
  const auto extended = marginal.extend(frequencies.tail(1));
  for (int i = 0; i < bins; ++i)
  {
    if (expected[i] == 0)
    {
      CHECK(direct.counts()[i] == -std::numeric_limits<double>::infinity());
      CHECK(extended->counts()[i] == -std::numeric_limits<double>::infinity());
    }
    else
    {
      CHECK(direct.counts()[i] == doctest::Approx(std::log(expected[i])).epsilon(1e-13));
      CHECK(extended->counts()[i] == doctest::Approx(std::log(expected[i])).epsilon(1e-13));
    }
  }
  CHECK((direct.extend(Eigen::ArrayXd{})->counts() == direct.counts()).all());
  CHECK((direct.extend(Eigen::ArrayXd::Constant(1, bins - 0.1))->counts() == direct.counts()).all());
  for (double frequency : {0., -1., 0.49, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    CHECK_THROWS_AS(direct.extend(Eigen::ArrayXd::Constant(1, frequency)), ApiTofArgumentError);
}

TEST_CASE("fragmentation DOS switches to logs before overflow without losing low bins")
{
  constexpr int bins = 10000;
  const double width = consts::boltzmann;
  const FragmentationDos marginal(Eigen::ArrayXd::Ones(200), width, bins * width);
  const auto combined = marginal.extend(Eigen::ArrayXd::Ones(200));
  for (const auto &[modes, counts] : {std::pair{200, &marginal.counts()}, std::pair{400, &combined->counts()}})
  {
    CHECK(counts->isFinite().all());
    CHECK((*counts)[0] == 0.);
    CHECK((*counts)[bins - 1] > std::log(std::numeric_limits<double>::max()));
    for (int energy : {1, 9, 100, 1000, bins - 1})
    {
      // Identical oscillators have C(energy + modes - 1, modes - 1) states.
      long double expected = 0.;
      for (int k = 1; k < modes; ++k)
        expected += std::log1p(static_cast<long double>(energy) / k);
      CHECK(std::abs((*counts)[energy] - static_cast<double>(expected)) < 1e-10);
    }
  }
}

TEST_CASE("fragmentation DOS preserves fractional and enormous prepared counts")
{
  const double width = consts::boltzmann;
  for (double value : {-1000., 1000.})
  {
    Eigen::ArrayXd logs(4);
    logs << 0., value, -std::numeric_limits<double>::infinity(), value - 200.;
    const FragmentationDos prepared(Eigen::ArrayXd{}, width, 4 * width, logs);
    const auto extended = prepared.extend(Eigen::ArrayXd::Constant(1, 2.));
    CHECK(extended->counts()[0] == 0.);
    CHECK(extended->counts()[1] == value);
    CHECK(extended->counts()[2] == 0.);
    CHECK(extended->counts()[3] == doctest::Approx(value + std::log1p(std::exp(-200.))));
    CHECK((prepared.counts() == logs).all());
  }
}
