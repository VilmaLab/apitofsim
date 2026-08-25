#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "skimmer.h"
#include "densityandrate_smoke.h"
#include "cli/mass_spec_io.h"
#include "operation_context.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>

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

TEST_CASE("parallel mesh implementations match serial implementations")
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
  CHECK(diagonal_parallel.isApprox(diagonal_serial, 1.0e-12));
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
  ClusterData product_1(1, 0.0, rotations, frequencies, 0);
  ClusterData product_2(1, 0.0, rotations, frequencies, 0);
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

TEST_CASE("apitof pinhole smoke tests")
{
  namespace fs = std::filesystem;
  const char *data_dir_env = getenv("DATA_DIR");
  REQUIRE_MESSAGE(data_dir_env != nullptr, "DATA_DIR environment variable not set");
  auto density_cluster = scaled_density(read_histogram((string(data_dir_env) + "/ready/density_cluster.out").c_str()));
  auto rate_const = scaled_rate_const(read_histogram((string(data_dir_env) + "/ready/rate_constant.out").c_str()));
  SkimmerData skimmer;
  double mesh_skimmer;
  std::tie(skimmer, mesh_skimmer) = read_skimmer((string(data_dir_env) + "/ready/skimmer.dat").c_str());
  StreamingResultQueue result_queue;
  Eigen::Vector3d rotations_0 = Eigen::Vector3d(0.0197112, 0.0229917, 0.0591769);
  auto inertia = compute_inertia(rotations_0);
  double m_ion;
  double R_cluster;
  compute_mass_and_radius(inertia, 216, m_ion, R_cluster);
  MassSpectrometer ms{
    skimmer,
    mesh_skimmer,
    InstrumentDims(
      1.0e-3,
      2.44e-3,
      0.101,
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
  auto subs = MassSpecSubstanceSingleInput(
    -1,
    m_ion,
    R_cluster,
    density_cluster,
    std::vector({MassSpecInputFragmentationPathway(rate_const, 23420.7)}),
    Gas{
      2.46e-10,
      4.8506e-26,
      1.4});
  auto counters = std::get<0>(apitof_mass_spec(ms, subs, 5, 42, result_queue, SampleMode::dss_normalized));
  result_queue.enqueue(std::monostate{});
  CHECK(counters[Counter::nwarnings] == 0);
  CHECK(counters[Counter::n_fragmented_total] + counters[Counter::n_escaped_total] == 5);
  CHECK(counters[Counter::ncoll_total] >= 0);
  CHECK(counters[Counter::counter_collision_rejections] >= 0);
  bool exiting = false;
  auto num_partial_results = 0;
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
  CHECK(num_partial_results == 5);
  CHECK((streamed_counters == counters).all());
  Eigen::ArrayXi expected_counters(6);
  expected_counters << 0, 15, 0, 5, 5, 0;
  CHECK((counters == expected_counters).all());
}
