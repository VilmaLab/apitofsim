#include <doctest.h>
#include "fragmentation.h"
#include "consts.h"

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
    const double id = products.daughter.inertia_moment, in = products.neutral.inertia_moment;
    const double md = products.daughter.mass, mn = products.neutral.mass;
    const double q = 70 * consts::boltzmann;
    const Eigen::Vector3d velocity{30., -20., 1000.};
    OperationContext operation;
    std::mt19937 gen(42), repeat(42);
    for (double angular_scale : {0., 1., 100.})
    {
      const Eigen::Vector3d angular_momentum = Eigen::Vector3d{1., 2., 3.}.normalized() * angular_scale * std::sqrt(2 * id * q);
      for (int i = 0; i < 200; ++i)
      {
        const auto sampled = products.sample(gen, q, angular_momentum, velocity, operation);
        REQUIRE(sampled.has_value());
        const auto identical = restored.sample(repeat, q, angular_momentum, velocity, operation);
        REQUIRE(identical.has_value());
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
