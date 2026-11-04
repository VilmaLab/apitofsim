#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "skimmer.h"
#include "densityandrate_smoke.h"
#include "cli/mass_spec_io.h"
#include "operation_context.h"
#include "samplers.h"
#include "upstream.h"
#include "trajectory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>

int main(int argc, char **argv)
{
  doctest::Context context(argc, argv);
  if (const char *value = std::getenv("APITOFSIM_TEST_CONCURRENCY"))
  {
    oneapi::tbb::task_arena arena(std::stoul(value));
    return arena.execute([&]
    {
      return context.run();
    });
  }
  return context.run();
}

namespace
{
void test_signal_handler(int)
{
}
} // namespace

TEST_CASE_TEMPLATE("vibrational energy sampling handles cold clusters", Sampler,
                   VibEnergyNormSampler, VibEnergyUnnormSampler)
{
  Eigen::ArrayXd x(4), y(4);
  x << 2, 6, 10, 14;
  x *= consts::boltzmann;
  y << 0, 0, 1, 1;
  Histogram density(x, y);
  Sampler sampler(density);
  std::mt19937 gen(1);

  for (double energy : {0., 1., 2., 3., 6., 9., 10.})
  {
    CAPTURE(energy);
    CHECK(sampler.sample(gen, energy * consts::boltzmann) == 0);
  }

  // With an accessible populated bin, ordinary sampling still works.
  CHECK(sampler.sample(gen, 11 * consts::boltzmann) == x[2]);
  for (double energy : {-1., std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()})
  {
    CAPTURE(energy);
    CHECK_THROWS_AS(sampler.sample(gen, energy), ApiTofArgumentError);
  }
  CHECK_THROWS_AS(sampler.sample(gen, 20 * consts::boltzmann), ApiTofDosOverflow);
}

TEST_CASE_TEMPLATE("vibrational sampler assignment rebinds density and resets distribution", Sampler,
                   VibEnergyNormSampler, VibEnergyUnnormSampler)
{
  Eigen::ArrayXd energies(4);
  energies << 2, 6, 10, 14;
  energies *= consts::boltzmann;
  Histogram first(energies, Eigen::Array4d{0, 0, 1, 0});
  Histogram second(energies, Eigen::Array4d{0, 1, 0, 0});
  Sampler sampler(first);
  std::mt19937 gen(1);
  CHECK(sampler.sample(gen, 11 * consts::boltzmann) == energies[2]);

  sampler.unif = std::uniform_real_distribution<double>{0.25, 0.75};
  sampler = Sampler(second);
  CHECK(&sampler.density() == &second);
  CHECK(sampler.unif.a() == 0.0);
  CHECK(sampler.unif.b() == 1.0);
  CHECK(sampler.sample(gen, 11 * consts::boltzmann) == energies[1]);
}

static RFBasisPtr test_rf_basis(std::optional<Quadrupole> quadrupole, std::optional<double> mass = std::nullopt)
{
  if (quadrupole && mass)
    quadrupole->compute_mathieu_factor(*mass);
  auto parameters = rf_parameters(quadrupole);
  return parameters ? std::make_shared<RFBasis>((*parameters)[0], (*parameters)[1]) : nullptr;
}

TEST_CASE("operation context restores handlers across nested scopes")
{
  const auto previous_int = std::signal(SIGINT, test_signal_handler);
  const auto previous_term = std::signal(SIGTERM, test_signal_handler);
  const auto previous_abrt = std::signal(SIGABRT, test_signal_handler);
  {
    OperationContext outer;
    {
      OperationContext inner;
    }
  }

  CHECK(std::signal(SIGINT, previous_int) == test_signal_handler);
  CHECK(std::signal(SIGTERM, previous_term) == test_signal_handler);
  CHECK(std::signal(SIGABRT, previous_abrt) == test_signal_handler);
}

TEST_CASE("operation context cancels on the first cooperative signal")
{
  OperationContext operation;
  std::raise(SIGTERM);
  std::raise(SIGINT);

  CHECK_FALSE(operation.checkpoint());
  CHECK(operation.tbb_context().is_group_execution_cancelled());
  CHECK_THROWS_WITH_AS(
    operation.rethrow_pending_signal(true),
    "Signal-as-exception",
    SignalError);
  try
  {
    operation.rethrow_pending_signal(true);
  }
  catch (const SignalError &error)
  {
    CHECK(error.signum == SIGTERM);
  }
}

TEST_CASE("programmatic cancellation is idempotent and isolated")
{
  OperationContext cancelled;
  OperationContext independent;
  std::thread requester([&]
  {
    cancelled.request_cancel();
    cancelled.request_cancel();
  });
  requester.join();

  CHECK_FALSE(cancelled.checkpoint());
  CHECK(independent.checkpoint());
  CHECK_NOTHROW(cancelled.rethrow_pending_signal(true));
}

TEST_CASE("exception transport rethrows a background exception")
{
  ExceptionTransport transport;
  std::thread worker([&]
  {
    transport.guard([]
    {
      throw std::runtime_error("background failure");
    });
  });
  worker.join();

  CHECK_FALSE(transport.should_continue());
  CHECK_THROWS_WITH_AS(transport.rethrow(), "background failure", std::runtime_error);
}

TEST_CASE("oneTBB task exceptions cancel their operation and rethrow")
{
  OperationContext operation;
  std::atomic<int> visited = 0;
  CHECK_THROWS_WITH_AS(
    oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<int>(0, 128),
      [&](const oneapi::tbb::blocked_range<int> &range)
  {
    for (int i = range.begin(); i != range.end(); ++i)
    {
      ++visited;
      if (i == 64)
      {
        throw std::runtime_error("task failure");
      }
    }
  },
      operation.tbb_context()),
    "task failure",
    std::runtime_error);
  CHECK(operation.tbb_context().is_group_execution_cancelled());
  CHECK(visited > 0);
}

TEST_CASE("a pending signal takes precedence over an application exception")
{
  OperationContext operation;
  std::raise(SIGTERM);

  try
  {
    operation.run([]
    {
      throw std::runtime_error("application failure");
    }, true);
    FAIL("expected the pending signal to be rethrown");
  }
  catch (const SignalError &error)
  {
    CHECK(error.signum == SIGTERM);
  }
}

TEST_CASE("skimmer smoke tests")
{
  const double m = 4.8506e-26;
  double T0 = 300;
  double P0 = 182.0;
  const double ga = 1.4;
  const double dc = 5.0e-4;
  const double alpha_factor = 0.25;
  const double rmax = 5.0e-4;
  const int N = 100;
  const int M = 100;
  const int resolution = 100;
  const double tolerance = 1.0e-8;

  int nwarnings = 0;
  std::stringstream warnings;

  auto skimmer = Skimmer(
    T0,
    P0,
    rmax,
    dc,
    alpha_factor,
    m,
    ga,
    N,
    M,
    resolution,
    tolerance,
    nwarnings,
    warnings);

  SkimmerRow fr;
  while (true)
  {
    skimmer.next();
    auto r = skimmer.get();
    if (r.has_value())
    {
      fr = *r;
    }
    else
    {
      break;
    }
  }

  CHECK(fr.r == doctest::Approx(0.000495).epsilon(0.01));
  CHECK(fr.vel == doctest::Approx(614.574).epsilon(0.01));
  CHECK(fr.T == doctest::Approx(110.433).epsilon(0.01));
  CHECK(fr.P == doctest::Approx(5.508).epsilon(0.01));
  CHECK(fr.rho == doctest::Approx(0.000175229).epsilon(0.01));
  CHECK(fr.speed_of_sound == doctest::Approx(209.777).epsilon(0.01));
}

bool is_increasing(const Eigen::ArrayXd &arr)
{
  for (int i = 1; i < arr.size(); i++)
  {
    if (arr[i] < arr[i - 1])
    {
      return false;
    }
  }
  return true;
}

TEST_CASE("dos smoke tests")
{
  auto ds = dos_smoke();
  CHECK(ds[ds.size() - 1] > ds[0]);
  CHECK(ds[ds.size() - 1] > ds[ds.size() / 10]);
}

TEST_CASE("k total smoke tests")
{
  auto [k_rate, k0] = k_total_smoke();
  CHECK(is_increasing(k_rate));
  CHECK(is_increasing(k0));
}

TEST_CASE("all mesh implementations match the serial reference")
{
  constexpr double energy_max_rate = 64.0;
  constexpr double bin_width = 0.5;

  const auto serial = precompute_mesh(
    energy_max_rate,
    bin_width,
    MeshMode::compute_mesh_single_threaded);
  const auto parallel = precompute_mesh(
    energy_max_rate,
    bin_width,
    MeshMode::compute_mesh_multithreaded);
  const auto diagonal_serial = precompute_mesh(
    energy_max_rate,
    bin_width,
    MeshMode::compute_mesh_diagonal_single_threaded);
  const auto diagonal_parallel = precompute_mesh(
    energy_max_rate,
    bin_width,
    MeshMode::compute_mesh_diagonal_multithreaded);

  CHECK(parallel.isApprox(serial, 1.0e-12));
  CHECK(diagonal_serial.isApprox(serial, 1.0e-12));
  CHECK(diagonal_parallel.isApprox(serial, 1.0e-12));
}

TEST_CASE("linear mesh preserves midpoint weights and is the default")
{
  for (double bin_width : {0.125, 1.0, 4.0, 20.0})
  {
    for (int size : {0, 1, 7, 128, 4096})
    {
      CAPTURE(bin_width);
      CAPTURE(size);
      const auto expected = precompute_mesh(size * bin_width, bin_width, MeshMode::compute_mesh_single_threaded);
      const auto actual = precompute_mesh(size * bin_width, bin_width);
      CHECK(actual.size() == size);
      CHECK(((actual - expected).abs() <= 1e-12 * expected.abs()).all());
    }
  }
}

TEST_CASE("rate entry points preserve the direct midpoint sums")
{
  const Eigen::Vector3d rotations(1.0, 2.0, 3.0);
  const Eigen::ArrayXd frequencies = Eigen::ArrayXd::Constant(1, 2.0);
  const MolecularCluster molecular(10, 0.0, rotations, frequencies, 0);
  const AtomLikeCluster atom(1, 0.0, 0);
  const ClusterDerived derived = compute_derived(molecular);
  for (double bin_width : {0.5, 4.0})
  {
    constexpr int size = 32;
    constexpr int shift = 7;
    const double fragmentation_energy = (shift + 0.25) * bin_width;
    DensityResult rhos = DensityResult::Ones(size + shift, 4);
    rhos.col(C0_ROW) = Eigen::ArrayXd::LinSpaced(size + shift, 1.0, 100.0).square();
    rhos.col(COMB_ROW) = Eigen::ArrayXd::LinSpaced(size + shift, 0.0, 30.0).square();
    rhos.col(COMB_ROW).head(3).setZero();
    for (const ClusterSpec &second : {ClusterSpec(molecular), ClusterSpec(atom)})
    {
      const auto products = make_products(molecular, second);
      const auto derived_second = compute_derived(second);
      Eigen::ArrayXd k0(size), expected(size);
      if (std::holds_alternative<MolecularCluster>(second))
      {
        compute_k_total(k0, expected, derived.inertia_moment, derived_second.inertia_moment, rotations, rotations, rhos.col(COMB_ROW), rhos.col(C0_ROW), bin_width, size, fragmentation_energy);
      }
      else
      {
        const double prefactor = consts::boltzmann * consts::boltzmann * derived.inertia_moment / (consts::pi * pow(consts::hbar, 3));
        for (int m = 0; m < size; ++m)
        {
          double integral = 0.0;
          for (int i = 0; i <= m; ++i)
          {
            integral += bin_width * (i + 0.5) * rhos(m - i, COMB_ROW);
          }
          k0[m] = prefactor * integral * bin_width / rhos(shift + m, C0_ROW);
          double numerator = 0.0;
          double denominator = 0.0;
          for (int i = 0; i <= shift + m; ++i)
          {
            const double weight = sqrt(bin_width * (i + 0.5)) * rhos(shift + m - i, C0_ROW);
            denominator += weight;
            if (i <= m)
            {
              numerator += weight * k0[m - i];
            }
          }
          expected[m] = numerator / denominator;
        }
      }
      const auto full = compute_k_total_full(molecular, products, derived, derived_second, rhos, fragmentation_energy, size * bin_width, bin_width);
      CHECK(((full - expected).abs() <= 1e-12 * expected.abs()).all());
      const std::vector<KTotalInput> inputs{{molecular, second, fragmentation_energy, rhos.col(C0_ROW), rhos.col(COMB_ROW)}};
      for (auto mode : {MeshMode::no_mesh, MeshMode::compute_mesh_linear, MeshMode::compute_mesh_single_threaded, MeshMode::compute_mesh_diagonal_single_threaded, MeshMode::compute_mesh_multithreaded, MeshMode::compute_mesh_diagonal_multithreaded})
      {
        const auto batch = compute_k_total_batch(inputs, size * bin_width, bin_width, mode);
        CHECK(((batch.col(0) - expected).abs() <= 1e-12 * expected.abs()).all());
      }
      const auto batch = compute_k_total_batch(inputs, size * bin_width, bin_width);
      CHECK(((batch.col(0) - expected).abs() <= 1e-12 * expected.abs()).all());
    }
  }
  const auto empty = compute_k_total_batch({}, 8.0, 1.0);
  CHECK(empty.rows() == 8);
  CHECK(empty.cols() == 0);
}

TEST_CASE("parallel APIs support concurrent callers")
{
  constexpr double energy_max_rate = 64.0;
  constexpr double bin_width = 0.5;
  const auto expected = precompute_mesh(
    energy_max_rate,
    bin_width,
    MeshMode::compute_mesh_single_threaded);
  const auto diagonal_expected = precompute_mesh(
    energy_max_rate,
    bin_width,
    MeshMode::compute_mesh_diagonal_single_threaded);
  Eigen::ArrayXd first;
  Eigen::ArrayXd second;
  std::thread first_caller([&]
  {
    first = precompute_mesh(
      energy_max_rate,
      bin_width,
      MeshMode::compute_mesh_multithreaded);
  });
  std::thread second_caller([&]
  {
    second = precompute_mesh(
      energy_max_rate,
      bin_width,
      MeshMode::compute_mesh_diagonal_multithreaded);
  });
  first_caller.join();
  second_caller.join();

  CHECK(first.isApprox(expected, 1.0e-12));
  CHECK(second.isApprox(diagonal_expected, 1.0e-12));
}

TEST_CASE("density recurrence agrees with state enumeration across SIMD shifts")
{
  constexpr int bins = 64;
  for (double bin_width : {1.0, 4.0})
  {
    for (int shift : {1, 2, 3, 4, 5, 7, 8})
    {
      CAPTURE(bin_width);
      CAPTURE(shift);
      Eigen::ArrayXd frequencies(2);
      frequencies << 3 * bin_width, shift * bin_width;
      Eigen::ArrayXd expected = Eigen::ArrayXd::Zero(bins);
      for (int first = 0; 3 * first < bins; ++first)
      {
        for (int second = 0; 3 * first + shift * second < bins; ++second)
        {
          expected[3 * first + shift * second] += 1;
        }
      }
      // The current DOS API reports positive-energy states only.
      expected[0] = 0;
      expected /= bin_width;

      Eigen::ArrayXd actual(bins);
      compute_density_of_states(frequencies, actual, bins * bin_width, bin_width);
      CHECK(actual.isApprox(expected, 1.0e-12));
    }
  }
}

TEST_CASE("density rejects modes outside the histogram range")
{
  for (double bin_width : {1.0, 4.0})
  {
    // The allocated range excludes the fractional remainder of energy_max.
    const double energy_max = 10.5 * bin_width;
    Eigen::ArrayXd rho(10);
    for (double frequency : {10 * bin_width, 10.25 * bin_width, 11 * bin_width, 1e100})
    {
      CAPTURE(bin_width);
      CAPTURE(frequency);
      Eigen::ArrayXd frequencies(2);
      frequencies << 2 * bin_width, frequency;
      CHECK_THROWS_AS(compute_density_of_states(frequencies, rho, energy_max, bin_width), ApiTofArgumentError);
      CHECK_THROWS_AS(compute_density_of_states_batch({frequencies}, energy_max, bin_width), ApiTofArgumentError);
    }

    Eigen::ArrayXd frequencies(1);
    frequencies << 9.75 * bin_width;
    CHECK_NOTHROW(compute_density_of_states(frequencies, rho, energy_max, bin_width));
    CHECK(rho[9] == doctest::Approx(1 / bin_width));
  }
}

TEST_CASE("density batch handles empty and single-item inputs")
{
  const auto empty = compute_density_of_states_batch({}, 16.0, 1.0);
  CHECK(empty.rows() == 16);
  CHECK(empty.cols() == 0);

  std::vector<Eigen::ArrayXd> batch{Eigen::ArrayXd(2)};
  batch[0] << 2.0, 3.0;
  Eigen::ArrayXd expected(16);
  compute_density_of_states(batch[0], expected, 16.0, 1.0);

  const auto actual = compute_density_of_states_batch(batch, 16.0, 1.0);
  REQUIRE(actual.cols() == 1);
  CHECK(actual.col(0).isApprox(expected, 1.0e-12));
}

TEST_CASE("k total batch reports every completed item in order")
{
  constexpr size_t batch_size = 32;
  Eigen::ArrayXd rho_parent = Eigen::ArrayXd::Ones(32);
  Eigen::ArrayXd rho_comb = Eigen::ArrayXd::Ones(32);
  Eigen::Vector3d rotations = Eigen::Vector3d::Ones();
  Eigen::ArrayXd frequencies(1);
  frequencies << 1.0;
  MolecularCluster product_1(1, 0.0, rotations, frequencies, 0);
  MolecularCluster product_2(1, 0.0, rotations, frequencies, 0);
  std::vector<KTotalInput> batch;
  batch.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i)
  {
    batch.push_back(KTotalInput{
      product_1,
      product_2,
      1.0,
      rho_parent,
      rho_comb,
    });
  }

  std::mutex callback_mutex;
  std::vector<size_t> completed;
  const auto result = compute_k_total_batch(
    batch,
    8.0,
    1.0,
    MeshMode::compute_mesh_diagonal_multithreaded,
    [&](size_t count)
  {
    const std::lock_guard<std::mutex> lock(callback_mutex);
    completed.push_back(count);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });

  CHECK(result.rows() == 8);
  CHECK(result.cols() == batch_size);
  REQUIRE(completed.size() == batch_size);
  for (size_t i = 0; i < completed.size(); ++i)
  {
    CHECK(completed[i] == i + 1);
  }
}

TEST_CASE("particle snapshots own physical state and capture the current inertia")
{
  ParticleState particle{
    .position = {1, 2, 3},
    .time = 4,
    .velocity = {3, 4, 0},
    .omega = {1, 2, 2},
    .vibrational_energy = 5,
  };
  CHECK(particle.rotational_energy(2) == 9);
  CHECK(particle.internal_energy(2) == 14);

  StreamingResultQueue queue;
  queue.enqueue(InitEvent{snapshot(7, particle, 2, 11)});
  CHECK(snapshot(7, particle, 4).rot_energy == 18);
  CHECK(snapshot(7, particle, 4).particle_index == 0);

  particle.position.setZero();
  particle.time = 0;
  particle.velocity.setZero();
  particle.omega.setZero();
  particle.vibrational_energy = 0;

  StreamingResultElement result;
  REQUIRE(queue.try_dequeue(result));
  const auto &state = std::get<InitEvent>(std::get<EventMessage>(result)).state;
  CHECK(state.realization == 7);
  CHECK(state.particle_index == 11);
  CHECK((state.postime() == Eigen::Array4d{1, 2, 3, 4}).all());
  CHECK(state.particle.velocity == Eigen::Vector3d{3, 4, 0});
  CHECK(state.particle.omega == Eigen::Vector3d{1, 2, 2});
  CHECK(state.particle.vibrational_energy == 5);
  CHECK(state.rot_energy == 9);
}

TEST_CASE("collision frames preserve orientation and vector norms")
{
  const double pi = consts::pi;
  for (const Eigen::Vector3d &velocity : {Eigen::Vector3d{0, 0, 0}, Eigen::Vector3d{1, 0, 0}, Eigen::Vector3d{-1, 0, 0}, Eigen::Vector3d{0, 0, 1}, Eigen::Vector3d{3, -4, 2}})
  {
    for (double theta : {0.0, 0.3, pi / 2, pi})
    {
      for (double phi : {0.0, 0.7})
      {
        for (double alpha : {0.0, 0.8})
        {
          CAPTURE(velocity);
          CAPTURE(theta);
          CAPTURE(phi);
          CAPTURE(alpha);
          const Eigen::Matrix3d frame = collision_frame(velocity, theta, phi, alpha);
          CHECK((frame.transpose() * frame).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
          CHECK(frame.determinant() == doctest::Approx(1).epsilon(1e-12));
          const Eigen::Vector3d vector{2, -3, 5};
          const Eigen::Vector3d local = frame.transpose() * vector;
          CHECK(local.squaredNorm() == doctest::Approx(vector.squaredNorm()).epsilon(1e-12));
          CHECK((frame * local).isApprox(vector, 1e-12));
          if (theta == 0.0 || theta == pi)
          {
            Eigen::Vector3d expected = Eigen::Vector3d::UnitZ();
            if (velocity.norm() > 0)
              expected = velocity.normalized();
            if (theta == pi)
              expected = -expected;
            CHECK(frame.col(2).isApprox(expected, 1e-12));
          }
        }
      }
    }
  }

  // For an axial incident velocity, check the original collision-axis convention.
  const double theta = 0.3, phi = 0.7, alpha = 0.8;
  const Eigen::Matrix3d frame = collision_frame(Eigen::Vector3d::UnitZ(), theta, phi, alpha);
  const Eigen::Vector3d x2{sin(phi), -cos(phi), 0};
  const Eigen::Vector3d y2{cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)};
  const Eigen::Vector3d z2{sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)};
  CHECK(frame.col(0).isApprox(cos(alpha) * x2 + sin(alpha) * y2, 1e-12));
  CHECK(frame.col(1).isApprox(-sin(alpha) * x2 + cos(alpha) * y2, 1e-12));
  CHECK(frame.col(2).isApprox(z2, 1e-12));
  CHECK_THROWS_AS(collision_frame(Eigen::Vector3d::UnitZ(), -0.1, phi, alpha), ApiTofUnexpectedNumericalError);
  CHECK_THROWS_AS(collision_frame(Eigen::Vector3d::UnitZ(), pi + 0.1, phi, alpha), ApiTofUnexpectedNumericalError);
}

TEST_CASE("apitof pinhole smoke tests")
{
  namespace fs = std::filesystem;
  const char *data_dir_env = getenv("DATA_DIR");
  REQUIRE_MESSAGE(data_dir_env != nullptr, "DATA_DIR environment variable not set");
  auto density_cluster = scaled_density(read_histogram(string(data_dir_env) + "/ready/density_cluster.out"));
  auto rate_const = scaled_rate_const(read_histogram(string(data_dir_env) + "/ready/rate_constant.out"));
  auto [skimmer, mesh_skimmer] = read_skimmer(string(data_dir_env) + "/ready/skimmer.dat");
  StreamingResultQueue result_queue;
  Eigen::Vector3d rotations_0 = Eigen::Vector3d(0.0197112, 0.0229917, 0.0591769);
  auto inertia = compute_inertia(rotations_0);
  double m_ion;
  double R_cluster;
  std::tie(m_ion, R_cluster) = compute_mass_and_radius(inertia, 216);
  MassSpectrometer ms{
    skimmer,
    mesh_skimmer,
    InstrumentDims(
      1.0e-3,
      2.44e-3,
      1.0e-3, // Keep RF smoke coverage affordable under Valgrind.
      4.48e-3,
      5.0e-4),
    InstrumentVoltages(
      -19.0,
      -9.0,
      -7.0,
      -6.0,
      11.0),
    300.0,
    InstrumentPressures{182.0, 3.53},
    Quadrupole(
      0.0,
      200.0,
      1.3e6,
      6.0e-3),
  };
  bool test_rejections = false;
  SUBCASE("standard pinhole")
  {
  }
  SUBCASE("large pinhole rejects collisions")
  {
    // Put the pinhole region immediately after a uniform skimmer flow.
    ms.skimmer = SkimmerData(2, 3);
    ms.skimmer << 0, 300, 100, 0, 300, 100;
    ms.mesh_skimmer = 0.0005;
    ms.radius_pinhole = 1.0;
    ms.lengths = InstrumentDims{0.001, 0.000001, 0.000001, 0.05, 0.001};
    ms.voltages = InstrumentVoltages{0, 10, 10, 10, 20};
    // Give the pinhole section enough gas to exercise rejection with five ions.
    ms.pressures = InstrumentPressures{100, 2};
    ms.quadrupole = std::nullopt;
    test_rejections = true;
  }
  auto subs = MassSpecSubstanceSingleInput(
    -1,
    m_ion,
    R_cluster,
    density_cluster,
    std::vector({MassSpecInputFragmentationPathway(rate_const, 23420.7)}),
    Gas{
      2.46e-10,
      4.8506e-26,
      1.4},
    {}, test_rf_basis(ms.quadrupole, m_ion));
  auto counters = std::get<0>(apitof_mass_spec(ms, subs, 5, 42, result_queue, SampleMode::dss_normalized, true, MassSpecLogConf{DEFAULT_LOGLEVEL, true}));
  result_queue.enqueue(std::monostate{});
  CHECK(counters[Counter::nwarnings] == 0);
  CHECK(counters[Counter::n_fragmented_total] + counters[Counter::n_escaped_total] == 5);
  CHECK(counters[Counter::ncoll_total] >= 0);
  CHECK(counters[Counter::counter_collision_rejections] >= 0);
  bool exiting = false;
  auto num_partial_results = 0;
  std::vector<bool> initialized(5, false);
  std::vector<std::optional<ParticleStateMsg>> pending_collisions(5);
  int post_collisions = 0;
  Eigen::ArrayXi streamed_counters = Eigen::ArrayXi::Zero(counters.size());
  while (true)
  {
    StreamingResultElement result;
    if (exiting)
    {
      bool got = result_queue.try_dequeue(result);
      if (!got)
      {
        break;
      }
    }
    else
    {
      result_queue.wait_dequeue(result);
    }
    if (std::holds_alternative<std::monostate>(result))
    {
      // Still need to pump out any pending messages
      exiting = true;
    }
    else if (std::holds_alternative<PartialResult>(result))
    {
      const PartialResult &partial_result = std::get<PartialResult>(result);
      CHECK(partial_result.counters[Counter::n_realizations] == 1);
      CHECK((partial_result.counters >= 0).all());
      streamed_counters += partial_result.counters;
      CHECK((streamed_counters <= counters).all());
      num_partial_results++;
    }
    else if (std::holds_alternative<EventMessage>(result))
    {
      std::visit([&](const auto &event)
      {
        const auto &state = event.state;
        REQUIRE(state.realization >= 0);
        REQUIRE(state.realization < 5);
        auto &pending = pending_collisions[state.realization];
        if constexpr (std::is_same_v<std::decay_t<decltype(event)>, InitEvent>)
          CHECK(state.particle.velocity[2] >= 0);
        constexpr bool is_post_collision = std::is_same_v<std::decay_t<decltype(event)>, PostCollisionEvent>;
        CHECK(pending.has_value() == is_post_collision);
        if constexpr (is_post_collision)
        {
          REQUIRE(pending.has_value());
          CHECK((state.postime() == pending->postime()).all());
          CHECK(state.particle_index == pending->particle_index);
          CHECK(state.particle.velocity.allFinite());
          CHECK(state.particle.omega.allFinite());
          CHECK(state.rot_energy == doctest::Approx(0.5 * inertia * state.particle.omega.squaredNorm()).epsilon(1e-12));
          pending.reset();
          post_collisions++;
        }
        else if constexpr (std::is_same_v<std::decay_t<decltype(event)>, CollisionEvent>)
        {
          CHECK(state.rot_energy == doctest::Approx(0.5 * inertia * state.particle.omega.squaredNorm()).epsilon(1e-12));
          if (event.accepted)
          {
            pending = state;
          }
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(event)>, InitEvent>)
        {
          CHECK_FALSE(initialized[state.realization]);
          initialized[state.realization] = true;
          CHECK(state.postime().isZero());
          CHECK(state.particle.velocity.allFinite());
          CHECK(state.particle.omega.allFinite());
          CHECK(state.rot_energy >= 0.0);
          CHECK(state.particle.vibrational_energy >= 0.0);
          CHECK(state.particle_index == 0);
        }
        else
        {
          CHECK(initialized[state.realization]);
        }
      }, std::get<EventMessage>(result));
    }
    else if (std::holds_alternative<LogMessage>(result))
    {
      const LogMessage &msg = std::get<LogMessage>(result);
      {
        INFO("Unexpected log message type: ", msg.type, " content: ", msg.message);
        CHECK_UNARY(
          msg.type == LogMessage::initial_trace ||
          msg.type == LogMessage::fragments ||
          msg.type == LogMessage::probabilities ||
          msg.type == LogMessage::tmp ||
          msg.type == LogMessage::tmp_evolution ||
          msg.type == LogMessage::final_position ||
          msg.type == LogMessage::pinhole);
      }
    }
  }
  CHECK(post_collisions == counters[Counter::ncoll_total] - counters[Counter::counter_collision_rejections]);
  CHECK(std::none_of(pending_collisions.begin(), pending_collisions.end(), [](const auto &state)
  { return state.has_value(); }));
  CHECK(num_partial_results == 5);
  CHECK(std::all_of(initialized.begin(), initialized.end(), [](bool value)
  { return value; }));
  CHECK((streamed_counters == counters).all());
  if (test_rejections)
  {
    CHECK(counters[Counter::counter_collision_rejections] > 0);
  }
  else
  {
    // Exact trajectories vary with floating-point code generation. The same
    // seed must still reproduce counters when run with a single worker.
    oneapi::tbb::task_arena serial(1);
    auto repeated_counters = serial.execute([&]
    {
      StreamingResultQueue repeated_queue;
      return std::get<0>(apitof_mass_spec(ms, subs, 5, 42, repeated_queue, SampleMode::dss_normalized, true, MassSpecLogConf{DEFAULT_LOGLEVEL, false}));
    });
    CHECK((counters == repeated_counters).all());
  }
}

TEST_CASE("tree fragmentation events keep pathway and particle indices separate")
{
  SkimmerData skimmer(2, 3);
  skimmer << 100, 300, 100, 100, 300, 100;
  MassSpectrometer ms{skimmer, 0.0005, InstrumentDims::Constant(0.001), InstrumentVoltages::Zero(), 300, InstrumentPressures{100, 2}};
  Histogram density(1e-21, 32, Eigen::ArrayXd::Ones(32));
  // Fast, barrierless fragmentation also fires after the first transition resets energy.
  Histogram rate(1e-21, 32, Eigen::ArrayXd::Constant(32, 1e30));
  Histogram inactive_rate(1e-21, 32, Eigen::ArrayXd::Zero(32));
  MassSpecSubstanceTreeInput subs{
    -1,
    Gas{2e-10, 4.8e-26, 1.4},
    {MSSubstanceTreeCluster{1e-25, 5e-10, density}},
    {MassSpecInputFragmentationPathway{rate, -1e-21}, MassSpecInputFragmentationPathway{inactive_rate, -1e-21}},
    {{0, {1, 2}}, {0, {0}}},
    {{0, std::nullopt}, {1, std::nullopt}, {0, 1}},
  };
  // Local pathway 1 selects tree pathway 2 to node 1, then pathway 0 terminates.
  StreamingResultQueue queue;
  apitof_mass_spec(ms, subs, 1, 42, queue, SampleMode::dss_normalized, true, MassSpecLogConf{0, true});
  std::vector<FragmentationEvent> events;
  std::vector<PostFragmentationEvent> after_events;
  StreamingResultElement result;
  while (queue.try_dequeue(result))
  {
    if (auto *event = std::get_if<EventMessage>(&result))
    {
      if (auto *fragmentation = std::get_if<FragmentationEvent>(event))
      {
        events.push_back(*fragmentation);
      }
      if (auto *after = std::get_if<PostFragmentationEvent>(event))
      {
        after_events.push_back(*after);
      }
    }
  }
  REQUIRE(events.size() == 2);
  CHECK(events[0].state.particle_index == 0);
  CHECK(events[0].pathway_index == 2);
  CHECK(events[0].next_particle_index == 1);
  CHECK(events[1].state.particle_index == 1);
  CHECK(events[1].pathway_index == 0);
  CHECK(events[1].next_particle_index == -1);
  REQUIRE(after_events.size() == 1);
  const auto &after = after_events[0].state;
  CHECK(after.realization == events[0].state.realization);
  CHECK((after.postime() == events[0].state.postime()).all());
  CHECK(after.particle_index == events[0].next_particle_index);
  CHECK(after.particle.velocity.isZero());
  CHECK(after.particle.omega.isZero());
  CHECK(after.rot_energy == 0);
  CHECK(after.particle.vibrational_energy == 0);
  CHECK((events[1].state.particle.velocity.array() == after.particle.velocity.array()).all());
  CHECK(events[1].state.rot_energy == after.rot_energy);
  CHECK(events[1].state.particle.vibrational_energy == after.particle.vibrational_energy);
}

TEST_CASE("collision profile shares instantaneous simulator frequency")
{
  SkimmerData skimmer(2, 3);
  skimmer << 100, 250, 80, 300, 150, 20;
  MassSpectrometer ms{skimmer, 0.0005, InstrumentDims::Constant(0.001), InstrumentVoltages::Zero(), 300, InstrumentPressures{100, 2}};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  CollisionProfile profile(ms, gas, 5e-10);
  Eigen::ArrayXd z(6);
  z << 0, 0.001, 0.00125, 0.00175, 0.002, 0.005;
  for (const Eigen::Vector3d &velocity : {Eigen::Vector3d::Zero().eval(), Eigen::Vector3d{200, 0, 0}, Eigen::Vector3d{0, 0, 200}, Eigen::Vector3d{0, 0, -200}})
  {
    Eigen::Matrix<double, Eigen::Dynamic, 3> velocities = velocity.transpose().replicate(z.size(), 1);
    auto frequencies = profile.frequency(z, velocities);
    for (Eigen::Index i = 0; i < z.size(); ++i)
    {
      double temperature = 300;
      double pressure = z[i] < 0.001 ? 100 : 2;
      Eigen::Vector3d relative = velocity;
      if (z[i] >= 0.001 && z[i] <= 0.002)
      {
        double fraction = std::min((z[i] - 0.001) / 0.0005, 1.0);
        relative[2] -= 100 + 200 * fraction;
        temperature = 250 - 100 * fraction;
        pressure = 80 - 60 * fraction;
      }
      double kT = consts::boltzmann * temperature;
      CHECK(frequencies[i] == doctest::Approx(coll_freq(particle_density(pressure, kT), kT / gas.mass, gas.mass / kT, 7e-10, relative.norm())));
      CHECK(frequencies[i] == doctest::Approx(collision_frequency(z[i], velocity, chamber, gas.mass, 7e-10, skimmer, ms.mesh_skimmer)));
    }
  }
}

TEST_CASE("upstream crossing selects the first crossing into the instrument")
{
  CHECK(upstream::crossing_time(-3, -2, 2, true) == doctest::Approx(3));
  CHECK(upstream::crossing_time(-3, 2, 0, true) == doctest::Approx(1.5));
  CHECK(upstream::crossing_time(-3, 4, -2, true) == doctest::Approx(1));
  CHECK(upstream::crossing_time(0, -2, 2, true) == doctest::Approx(2));
  CHECK(upstream::crossing_time(0, 2, -2, false) == doctest::Approx(2));
  // An excursion can cross zero twice even when the step's endpoint is positive.
  double outward = upstream::crossing_time(1, -3, 2, false);
  CHECK(outward == doctest::Approx((3 - std::sqrt(5.0)) / 2));
  CHECK(1 - 3 * 3 + 3 * 3 > 0);
  CHECK(std::isinf(upstream::crossing_time(-1, 2, -2, true))); // tangency
  CHECK(std::isinf(upstream::crossing_time(-3, 1, -2, true)));
  // The stable root avoids subtracting nearly equal quantities.
  CHECK(upstream::crossing_time(-1, 1e12, 1, true) == doctest::Approx(1e-12).epsilon(1e-12));
}

TEST_CASE("upstream motion advances exactly to fragmentation or return")
{
  Eigen::Vector3d velocity{3, 4, -2};
  double z = -3, x = 1, y = 2, time = 10, elapsed = 0.5;
  double remaining = std::numeric_limits<double>::infinity();
  SUBCASE("return without fragmentation")
  {
  }
  SUBCASE("fragment before return")
  {
    remaining = 1;
  }
  SUBCASE("fragment at return")
  {
    remaining = 3;
  }
  bool fragmented = upstream::advance(velocity, z, x, y, 2, remaining, time, elapsed);
  double dt = std::min(remaining, 3.0);
  CHECK(fragmented == (remaining <= 3));
  CHECK(z == doctest::Approx(-3 - 2 * dt + dt * dt));
  CHECK(velocity[2] == doctest::Approx(-2 + 2 * dt));
  CHECK(x == doctest::Approx(1 + 3 * dt));
  CHECK(y == doctest::Approx(2 + 4 * dt));
  CHECK(time == doctest::Approx(10 + dt));
  CHECK(elapsed == doctest::Approx(0.5 + dt));
}

TEST_CASE("nonreturning upstream particles invalidate the run before fragmentation")
{
  for (auto [z0, vz, acceleration] : std::vector<std::tuple<double, double, double>>{
         {-1, -1, 0}, {-1, 0, 0}, {-1, -1, -1}, {-3, 1, -2}, {0, -1, 0}, {0, 0, -1}})
  {
    Eigen::Vector3d velocity{3, 4, vz};
    double z = z0, x = 0, y = 0, time = 0, elapsed = 0;
    CHECK_THROWS_AS(upstream::advance(velocity, z, x, y, acceleration, 0.01, time, elapsed), ApiTofParticleEscapeError);
    CHECK(z == z0);
    CHECK(time == 0);
    CHECK(elapsed == 0);
  }
}

TEST_CASE("upstream collision frequency is zero")
{
  SkimmerData skimmer(2, 3);
  skimmer << 100, 300, 100, 100, 300, 100;
  MassSpectrometer ms{skimmer, 0.0005, InstrumentDims::Constant(0.001), InstrumentVoltages::Zero(), 300, InstrumentPressures{100, 2}};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  CHECK(collision_frequency(-1, Eigen::Vector3d{0, 0, -200}, chamber, gas.mass, 7e-10, skimmer, ms.mesh_skimmer) == 0);
  CollisionProfile profile(ms, gas, 5e-10);
  Eigen::Matrix<double, 1, 3> velocity = Eigen::Matrix<double, 1, 3>::Zero();
  CHECK_THROWS_AS(profile.frequency(Eigen::ArrayXd::Constant(1, -1), velocity), std::invalid_argument);
}

TEST_CASE("upstream fragmentation products and repeated excursions return")
{
  Eigen::Vector3d velocity{3, 4, -2};
  double z = -3, x = 0, y = 0, time = 0, elapsed = 0;
  REQUIRE(upstream::advance(velocity, z, x, y, 2, 1, time, elapsed));
  CHECK(z == doctest::Approx(-4));
  // Tree products start at rest, with acceleration determined by their new mass.
  velocity.setZero();
  CHECK_FALSE(upstream::advance(velocity, z, x, y, 8, INFINITY, time, elapsed));
  CHECK(z == 0);
  CHECK(time == doctest::Approx(2));
  CHECK(velocity[2] == doctest::Approx(8));
  for (int i = 0; i < 3; ++i)
  {
    velocity[2] = -8;
    CHECK_FALSE(upstream::advance(velocity, z, x, y, 8, INFINITY, time, elapsed));
    CHECK(z == 0);
    CHECK(velocity[2] == doctest::Approx(8));
  }
  CHECK(time == doctest::Approx(8));
  CHECK(elapsed == time);
  CHECK(x == 3);
  CHECK(y == 4);
}

TEST_CASE("trajectory sampler preserves fragmentation time across upstream product return")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 100, 0, 300, 100;
  // Keep in-instrument collisions negligible; the event thresholds are explicit.
  MassSpectrometer ms{skimmer, 0.5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Constant(1e-12)};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  Eigen::Array2d dts = Eigen::Array2d::Constant(0.125);
  Eigen::Array4d acc = Eigen::Array4d::Constant(2);
  ParticleState particle{
    .position = {0, 0, -3},
    .velocity = {3, 4, -2},
  };
  double fragmentation = 1;
  auto next_event = [&]
  {
    return next_coll(0.5, particle, chamber, 7e-10, dts,
                     fragmentation, acc, gas.mass, skimmer, ms.mesh_skimmer, ms.quadrupole);
  };
  REQUIRE(next_event() == TimeNextCollOutcome::fragmentation);
  REQUIRE(particle.position.z() == doctest::Approx(-4));
  CHECK(zone(particle.position.z(), chamber.clens) == 0);
  CHECK(particle.time == doctest::Approx(1));
  CHECK(particle.velocity.norm() == doctest::Approx(5));
  CHECK(particle.position.x() == doctest::Approx(3));
  CHECK(particle.position.y() == doctest::Approx(4));

  // Tree products start at rest. Changing mass changes acceleration, while the
  // same explicit fragmentation deadline falls before or after return.
  particle.velocity.setZero();
  fragmentation = 1.5;
  SUBCASE("heavier product fragments upstream")
  {
    acc[0] = 2; // return takes 2 seconds
    REQUIRE(next_event() == TimeNextCollOutcome::fragmentation);
    CHECK(particle.position.z() == doctest::Approx(-1.75));
    CHECK(zone(particle.position.z(), chamber.clens) == 0);
    CHECK(particle.velocity[2] == doctest::Approx(3));
  }
  SUBCASE("lighter product returns before fragmentation")
  {
    acc[0] = 8; // return takes 1 second
    REQUIRE(next_event() == TimeNextCollOutcome::fragmentation);
    CHECK(particle.position.z() == doctest::Approx(5));
    CHECK(zone(particle.position.z(), chamber.clens) == 1);
    CHECK(particle.velocity[2] == doctest::Approx(12));
  }
  CHECK(particle.time == doctest::Approx(2.5));
  CHECK(particle.position.x() == doctest::Approx(3));
  CHECK(particle.position.y() == doctest::Approx(4));
}

TEST_CASE("trajectory collision timing uses number density in both chambers")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 100, 0, 300, 100;
  MassSpectrometer ms{skimmer, 5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures{100, 2}};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  for (double z : {5.0, 25.0, 35.0, 45.0})
  {
    CAPTURE(z);
    ParticleState particle{.position = {0, 0, z}, .velocity = Eigen::Vector3d::Zero()};
    const double frequency = collision_frequency(z, particle.velocity, chamber, gas.mass, 7e-10, skimmer, ms.mesh_skimmer);
    const double dt = 0.1 / frequency;
    // A stationary particle has constant hazard: 0.25 is reached at exactly 2.5 nominal steps.
    auto outcome = next_coll(std::exp(-0.25), particle, chamber, 7e-10,
                             Eigen::Array2d::Constant(dt), 10 * dt, Eigen::Array4d::Zero(),
                             gas.mass, skimmer, ms.mesh_skimmer, ms.quadrupole);
    CHECK(outcome == TimeNextCollOutcome::gas_collision);
    CHECK(particle.time / dt == doctest::Approx(2.5));
    particle.time = 0;
    CHECK(next_coll(std::exp(-0.25), particle, chamber, 7e-10, Eigen::Array2d::Constant(dt), 10 * dt,
                    Eigen::Array4d::Zero(), gas.mass, skimmer, ms.mesh_skimmer, ms.quadrupole,
                    {CollisionTimeMethod::legacy, 0.1, 0.1}) == TimeNextCollOutcome::gas_collision);
    CHECK(particle.time / dt == doctest::Approx(3));
  }
}

TEST_CASE("trajectory steps retain regional integration rules")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0.5, 300, 100, 0.5, 300, 100;
  Quadrupole quadrupole{2, 3, 1, 0.01};
  quadrupole.mathieu_factor = 0.5;
  MassSpectrometer ms{skimmer, 5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures{100, 2}, quadrupole};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  const Eigen::Array2d dts{0.1, 0.2};
  const Eigen::Array4d acceleration{1, 2, 3, 4};
  struct Step
  {
    double z;
    Eigen::Vector3d velocity;
    Eigen::Vector3d position;
    double dt;
  };
  for (const auto &step : {
         Step{5, {2, 3, 4.1}, {1.2, 2.3, 5.405}, 0.1},
         Step{25, {2, 3, 4.4}, {1.4, 2.6, 25.84}, 0.2},
         Step{35, {2.12, 2.77, 4.6}, {1.412, 2.577, 35.86}, 0.2},
         Step{45, {2, 3, 4.8}, {1.4, 2.6, 45.88}, 0.2},
       })
  {
    CAPTURE(step.z);
    ParticleState particle{.position = {1, 2, step.z}, .velocity = {2, 3, 4}};
    CHECK(next_coll_old(0, particle, chamber, 7e-10, dts, step.dt, acceleration, gas.mass, skimmer, ms.mesh_skimmer, ms.quadrupole) == TimeNextCollOutcome::fragmentation);
    CHECK(particle.time == doctest::Approx(step.dt));
    CHECK(particle.velocity.isApprox(step.velocity, 1e-12));
    CHECK(particle.position.isApprox(step.position, 1e-12));
  }

  // The exact skimmer exit retains the chamber kick but uses skimmer timing and Euler position updates.
  for (double z : {15.0, 20.0})
  {
    CAPTURE(z);
    ParticleState particle{.position = {1, 2, z}, .velocity = {2, 3, 4}};
    Eigen::Vector3d expected_velocity{2, 3, z == 20 ? 4.4 : 4};
    double dt = 1e-3 / collision_frequency(z, expected_velocity, chamber, gas.mass, 7e-10, skimmer, ms.mesh_skimmer);
    Eigen::Vector3d expected_position = particle.position + particle.velocity * dt;
    CHECK(next_coll_old(0, particle, chamber, 7e-10, dts, 1e-100, acceleration, gas.mass, skimmer, ms.mesh_skimmer, ms.quadrupole) == TimeNextCollOutcome::fragmentation);
    CHECK(particle.time == doctest::Approx(dt).epsilon(1e-12));
    CHECK(particle.velocity.isApprox(expected_velocity, 1e-12));
    CHECK(particle.position.isApprox(expected_position, 1e-12));
  }
}

TEST_CASE("trajectory sampler suppresses gas collisions upstream")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 100, 0, 300, 100;
  MassSpectrometer ms{skimmer, 0.5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Constant(100)};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  ParticleState particle{
    .position = {0, 0, -3},
    .velocity = {0, 0, -2},
  };
  double fragmentation = 1;
  // At this pressure, ordinary chamber motion would produce a collision almost
  // immediately at this threshold. Upstream motion must fragment instead.
  auto outcome = next_coll(std::nextafter(1.0, 0.0), particle, chamber, 7e-10,
                           Eigen::Array2d::Constant(0.125), fragmentation,
                           Eigen::Array4d::Constant(2), gas.mass, skimmer, ms.mesh_skimmer, ms.quadrupole);
  CHECK(outcome == TimeNextCollOutcome::fragmentation);
  CHECK(particle.position.z() == doctest::Approx(-4));
  CHECK(particle.time == doctest::Approx(1));
}

TEST_CASE("next_coll integrates the skimmer gas profile across knots")
{
  SkimmerData skimmer(3, 3);
  skimmer << 0, 300, 1, 0, 300, 3, 0, 300, 2;
  MassSpectrometer ms{skimmer, 1, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Constant(1)};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  ParticleState p{.position = {0, 0, 10}, .velocity = {0, 0, 1}};
  // T and relative velocity are constant, so the rate is linear in pressure.
  // Integrated pressure to z=11.5 is 2 + 0.5*(3+2.5)/2 = 3.375 Pa s.
  double unit = collision_frequency(10, p.velocity, chamber, gas.mass, 2e-12, skimmer, 1);
  auto collision_time = [&](const CollisionTimeOptions &options)
  {
    ParticleState ion = p;
    REQUIRE(next_coll(std::exp(-3.375 * unit), ion, chamber, 2e-12, Eigen::Array2d::Ones(), 10,
                      Eigen::Array4d::Zero(), gas.mass, skimmer, 1, ms.quadrupole, options) == TimeNextCollOutcome::gas_collision);
    CHECK(ion.position.z() == doctest::Approx(10 + ion.time));
    return ion.time;
  };
  CHECK(collision_time({}) == doctest::Approx(1.5).epsilon(1e-9));
  for (const auto &loose : {CollisionTimeOptions{CollisionTimeMethod::integrated_hazard, 1.0, 1e-9},
                            CollisionTimeOptions{CollisionTimeMethod::integrated_hazard, 1e-12, 1.0}})
    CHECK(std::abs(collision_time(loose) - 1.5) > 1e-5);
}

TEST_CASE("next_coll resolves interfaces in either direction and escape at a deadline")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 1, 0, 300, 1;
  MassSpectrometer ms{skimmer, 5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Constant(1)};
  Gas gas{2e-10, 4.8e-26, 1.4};
  ChamberQuantities chamber(ms, gas);
  ParticleState p{.position = {0, 0, 19}, .velocity = {0, 0, 1}};
  auto run = [&](double dt)
  {
    return next_coll(0, p, chamber, 2e-10, Eigen::Array2d::Ones(), dt,
                     Eigen::Array4d{0, 2, 0, 0}, gas.mass, skimmer, 5, ms.quadrupole);
  };
  REQUIRE(run(2) == TimeNextCollOutcome::fragmentation);
  CHECK(p.position.z() == doctest::Approx(22));
  CHECK(p.velocity.z() == doctest::Approx(3));
  p.velocity.z() = -3;
  REQUIRE(run(2) == TimeNextCollOutcome::fragmentation);
  CHECK(p.position.z() == doctest::Approx(19));
  CHECK(p.velocity.z() == doctest::Approx(-1));
  p.position.z() = 49;
  p.velocity.z() = 1;
  REQUIRE(run(1) == TimeNextCollOutcome::escape);
  CHECK(p.position.z() == 50);
}

TEST_CASE("RF flight converges to independent trajectory at nonzero entry phase")
{
  SkimmerData skimmer(2, 3);
  skimmer << 0, 300, 1, 0, 300, 1;
  Quadrupole q{2, 3, 1, 0.01};
  q.mathieu_factor = 0.5;
  MassSpectrometer ms{skimmer, 5, InstrumentDims::Constant(10), InstrumentVoltages::Zero(), 300, InstrumentPressures::Ones(), q};
  Gas gas{2e-10, 4.8e-22, 1.4};
  ChamberQuantities chamber(ms, gas);
  // Independent DOP853 reference, rtol=3e-14, from t=0.13 to 0.37.
  Eigen::Vector4d expected{1.4569258813237311, 2.7641639237064104, 1.6840371968120658, 3.601275068708212};
  const auto basis = test_rf_basis(q);
  auto error = [&](double step)
  {
    ParticleState p{.position = {1, 2, 35}, .time = 0.13, .velocity = {2, 3, 4}};
    REQUIRE(next_coll(0, p, chamber, 2e-10, Eigen::Array2d::Constant(step), 0.24, Eigen::Array4d::Zero(),
                      gas.mass, skimmer, 5, q, {}, basis.get()) == TimeNextCollOutcome::fragmentation);
    CHECK(p.time == doctest::Approx(0.37));
    CHECK(p.position.z() == doctest::Approx(35.96));
    return (Eigen::Vector4d{p.position.x(), p.position.y(), p.velocity.x(), p.velocity.y()} - expected).norm();
  };
  double coarse = error(0.0015);
  double fine = error(0.00075);
  CHECK(fine < 1e-11);
  CHECK(coarse < 1e-11);

  // Integrating H'=collision_frequency along the same reference flight gives
  // exp(-H(0.24)) below. The deadline deliberately falls after the collision.
  ParticleState p{.position = {1, 2, 35}, .time = 0.13, .velocity = {2, 3, 4}};
  auto collision_time = [&](const CollisionTimeOptions &options)
  {
    ParticleState ion = p;
    REQUIRE(next_coll(0.6001412074736865, ion, chamber, 2e-11, Eigen::Array2d::Constant(0.00075), 0.3,
                      Eigen::Array4d::Zero(), gas.mass, skimmer, 5, q, options, basis.get()) == TimeNextCollOutcome::gas_collision);
    return ion.time;
  };
  CHECK(collision_time({}) == doctest::Approx(0.37).epsilon(1e-7));
  CHECK(collision_time({CollisionTimeMethod::integrated_hazard, 1e-14, 1e-11}) == doctest::Approx(0.37).epsilon(1e-11));
}

TEST_CASE("simulation quadrupole forces follow the current cluster mass")
{
  bool tree = false;
  bool fragment = false;
  SUBCASE("single cluster")
  {
  }
  SUBCASE("tree root")
  {
    tree = true;
  }
  SUBCASE("tree product after fragmentation")
  {
    tree = fragment = true;
  }

  for (auto method : {CollisionTimeMethod::integrated_hazard, CollisionTimeMethod::legacy})
  {
    CAPTURE(method);
    // Low pressure keeps this seeded flight collision-free. The first chamber
    // and skimmer still need finite steps with the legacy collision clock.
    SkimmerData skimmer(2, 3);
    skimmer << 0, 300, 1, 0, 300, 1;
    MassSpectrometer ms{skimmer, 5e-5, InstrumentDims{1e-4, 1e-4, 1e-4, 1e-3, 1e-4},
                        InstrumentVoltages{0, 0.01, 0.02, 0.03, 0.04}, 300, InstrumentPressures{1, 1e-12},
                        Quadrupole{0, 1, 1e5, 0.01}};
    Gas gas{2e-10, 4.8e-26, 1.4};
    const double parent_mass = 2e-25, product_mass = 1e-25, radius = 5e-10;
    Histogram density(1e-21, 256, Eigen::ArrayXd::Ones(256));
    Histogram inactive_rate(1e-21, 256, Eigen::ArrayXd::Zero(256));
    MassSpecInputFragmentationPathway inactive{inactive_rate, -1e-21};
    CollisionTimeOptions options{method};
    StreamingResultQueue queue;
    auto run = [&](const auto &subs)
    {
      auto counters = std::get<0>(apitof_mass_spec(ms, subs, 1, 42, queue, SampleMode::rejection,
                                                   true, MassSpecLogConf{0, true}, false, options));
      CHECK(counters[Counter::ncoll_total] == 0);
      CHECK(counters[Counter::n_escaped_total] == (fragment ? 0 : 1));
      CHECK(counters[Counter::nwarnings] == 0);
    };
    if (tree)
    {
      // Include both masses even for the nonfragmenting root: preparing a later
      // payload must not overwrite the instrument or another cluster's field.
      Histogram rate(1e-21, 256, Eigen::ArrayXd::Constant(256, fragment ? 1e9 : 0));
      run(MassSpecSubstanceTreeInput{
        -1, gas, {MSSubstanceTreeCluster{parent_mass, radius, density, {}, test_rf_basis(ms.quadrupole, parent_mass)}, MSSubstanceTreeCluster{product_mass, radius, density, {}, test_rf_basis(ms.quadrupole, product_mass)}}, {MassSpecInputFragmentationPathway{rate, -1e-21}, inactive}, {{0, {0}}, {1, {1}}}, {{0, 1}, {1, std::nullopt}}});
    }
    else
    {
      run(MassSpecSubstanceSingleInput{-1, parent_mass, radius, density, {inactive}, gas, {}, test_rf_basis(ms.quadrupole, parent_mass)});
    }

    std::optional<ParticleStateMsg> start, end;
    StreamingResultElement item;
    while (queue.try_dequeue(item))
    {
      if (auto *event = std::get_if<EventMessage>(&item))
      {
        if (auto *initial = std::get_if<InitEvent>(event); initial && !fragment)
          start = initial->state;
        if (auto *product = std::get_if<PostFragmentationEvent>(event); product && fragment)
          start = product->state;
        if (auto *escaped = std::get_if<EscapeEvent>(event))
          end = escaped->state;
      }
    }
    REQUIRE(start.has_value());
    REQUIRE(end.has_value());
    CHECK(start->particle_index == (fragment ? 1 : 0));
    CHECK(end->particle_index == start->particle_index);
    if (fragment)
    {
      CHECK(start->particle.velocity.isZero());
      CHECK(start->particle.position.z() < ms.lengths[0]);
    }

    ChamberQuantities chamber(ms, gas);
    double mass = fragment ? product_mass : parent_mass;
    SubstanceQuantities quantities(ms, chamber, gas, -1, mass, radius);
    auto propagate = [&](double factor)
    {
      auto quadrupole = *ms.quadrupole;
      quadrupole.mathieu_factor = factor;
      ParticleState particle = start->particle;
      REQUIRE(next_coll(0, particle, chamber, gas.radius + radius, quantities.dts,
                        std::numeric_limits<double>::infinity(), quantities.acc, gas.mass,
                        ms.skimmer, ms.mesh_skimmer, quadrupole, options, test_rf_basis(quadrupole).get()) == TimeNextCollOutcome::escape);
      return particle;
    };
    // The reference supplies the physical factor explicitly, independently of
    // the simulation's preparation and tree payload selection.
    const double factor = consts::eV / (mass * std::pow(ms.quadrupole->r_quadrupole, 2));
    auto expected = propagate(factor);
    CHECK(end->particle.position.isApprox(expected.position, 1e-10));
    CHECK(end->particle.velocity.isApprox(expected.velocity, 1e-10));
    CHECK(end->particle.time == doctest::Approx(expected.time).epsilon(1e-10));
    CHECK((end->particle.velocity - propagate(0).velocity).head<2>().norm() > 1e-6);
    if (fragment)
      CHECK((end->particle.velocity - propagate(factor / 2).velocity).head<2>().norm() > 1e-6);
    CHECK(ms.quadrupole->mathieu_factor == 0);
  }
}

TEST_CASE("Prepared vibrational samplers share the discrete energy law")
{
  Eigen::ArrayXd x(5), y(5);
  x << 1, 3, 5, 7, 9;
  y << 0, 2, 0, 4, 1;
  Histogram density(x, y);
  Eigen::ArrayXd cdf(5), tree(16);
  cdf << 0, 2, 2, 6, 7;
  tree << 0, 7, 6, 1, 2, 4, 1, 0, 0, 2, 0, 4, 1, 0, 0, 0;
  // Zero-mass nodes (including padded leaves) are represented by -infinity.
  for (auto &v : tree)
    v = v > 0 ? std::log(v) : -std::numeric_limits<double>::infinity();
  auto check = [&](auto &sampler)
  {
    std::mt19937 gen(15);
    CHECK(sampler.sample(gen, 0) == 0);
    CHECK(sampler.sample(gen, 3) == 0);
    CHECK(sampler.sample(gen, 3.01) == 3);
    CHECK_THROWS_AS(sampler.sample(gen, 11), ApiTofDosOverflow);
    CHECK_THROWS_AS(sampler.sample(gen, -1), ApiTofArgumentError);
    // The upper edge is above the last bin center. Include every accessible bin.
    Eigen::ArrayXi counts = Eigen::ArrayXi::Zero(5);
    for (int i = 0; i < 20000; ++i)
    {
      auto sample = sampler.sample(gen, 10);
      ++counts[static_cast<int>((sample - 1) / 2)];
    }
    Eigen::ArrayXd weights = y * (10 - x).sqrt();
    CHECK((counts.cast<double>() / counts.sum() - weights / weights.sum()).abs().maxCoeff() < 0.015);
  };
  auto cdf_index = std::make_shared<VibEnergyIndex>(VibEnergyMode::dos_only, cdf);
  auto tree_index = std::make_shared<VibEnergyIndex>(VibEnergyMode::group_dos, tree);
  VibEnergyDosSampler dos_sampler(density, cdf_index);
  VibEnergyGroupSampler group_sampler(density, tree_index);
  check(dos_sampler);
  check(group_sampler);
}

TEST_CASE("Prepared RF basis matches independent trajectories across phase wraps")
{
  // DOP853, rtol=2.3e-14, atol=2e-15; state (x, dx/dphase, y, dy/dphase).
  struct Reference
  {
    double ac, dc, phase, duration;
    std::array<rf_detail::State, 2> expected;
  };
  const std::array references{
    Reference{0.1, 0, 6.2, 20, {{{24.411238076194028, 2.399226585097148}, {65.10898235619696, -5.175517903461773}}}},
    Reference{0.35, 0.02, -1.2, 3, {{{10.696777544588402, 4.367547130126496}, {8.18348420769359, -0.05303333569744881}}}},
    Reference{0.1, 0, 6283.315307179586, 0.24, {{{1.483259499400257, 2.0287328191170912}, {3.950675798925281, 3.919453938678203}}}},
    Reference{0.1, 0, 0.13, 628.3185307179587, {{{17.18056979886848, 1.5587014389627596}, {52.277265079224165, 2.028658481706738}}}},
    Reference{0.2, -0.25, 9.7, 20, {{{46752.892827541255, 17270.156116869777}, {-50.815206705040815, -9.06548003877477}}}},
  };
  for (const auto &r : references)
  {
    CAPTURE(r.ac);
    CAPTURE(r.dc);
    CAPTURE(r.phase);
    CAPTURE(r.duration);
    RFBasis built(r.ac, r.dc);
    RFBasis restored(r.ac, r.dc, built.data());
    auto result = restored.anchor(r.phase, {{{1, 2}, {3, 4}}}).evaluate(r.duration);
    for (int axis = 0; axis < 2; ++axis)
      for (int j = 0; j < 2; ++j)
        CHECK(result[axis][j] == doctest::Approx(r.expected[axis][j]).epsilon(2e-10));
  }
  RFBasis free(0, 0);
  auto state = free.anchor(-2, {{{1, 2}, {3, 4}}}).evaluate(100);
  CHECK(state[0][0] == doctest::Approx(201));
  CHECK(state[1][0] == doctest::Approx(403));
  auto reverse = free.anchor(98, state).evaluate(-100);
  CHECK(reverse[0][0] == doctest::Approx(1));
  CHECK(reverse[1][0] == doctest::Approx(3));
}

TEST_CASE("RF cache is required and matched before simulation")
{
  Quadrupole q{2, 3, 1, 0.01};
  q.mathieu_factor = 0.5;
  auto basis = test_rf_basis(q);
  CHECK_NOTHROW(validate_rf_basis(q, {}, basis.get()));
  CHECK_THROWS_AS(validate_rf_basis(q, {}, nullptr), std::invalid_argument);
  q.mathieu_factor *= 2;
  CHECK_THROWS_AS(validate_rf_basis(q, {}, basis.get()), std::invalid_argument);
  CHECK_NOTHROW(validate_rf_basis(q, {CollisionTimeMethod::legacy}, nullptr));
  q.ac_field = 0;
  CHECK_NOTHROW(validate_rf_basis(q, {}, nullptr));
  q.ac_field = 3;
  q.angular_velocity = 0;
  CHECK_NOTHROW(validate_rf_basis(q, {}, nullptr));
}

TEST_CASE("RF preparation rejects unsupported or corrupt tables")
{
  CHECK_THROWS_AS(RFBasis(0, 0, 0), std::invalid_argument);
  CHECK_THROWS_AS(RFBasis(0, 0, -1), std::invalid_argument);
  CHECK_THROWS_AS(RFBasis(0, -4), std::invalid_argument);
  CHECK_THROWS_AS(RFBasis(1, 0, 16), std::invalid_argument);
  RFBasis basis(0.1, 0.03);
  Eigen::ArrayXd data = basis.data();
  data[32] += 0.01;
  CHECK_THROWS_AS(RFBasis(0.1, 0.03, data), std::invalid_argument);
  CHECK_THROWS_AS(RFBasis(0, 0, Eigen::ArrayXd::Zero(16)), std::invalid_argument);
  CHECK_THROWS_AS(basis.anchor(std::numeric_limits<double>::infinity(), {{{1, 2}, {3, 4}}}), std::invalid_argument);
}
