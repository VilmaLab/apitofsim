#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "skimmer_lib.h"
#include "densityandrate_smoke.h"
#include "apitof_pinhole_io.h"

using namespace Catch::Matchers;

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

  CHECK_THAT(fr.r, WithinRel(0.000495, 0.01));
  CHECK_THAT(fr.vel, WithinRel(614.574, 0.01));
  CHECK_THAT(fr.T, WithinRel(110.433, 0.01));
  CHECK_THAT(fr.P, WithinRel(5.508, 0.01));
  CHECK_THAT(fr.rho, WithinRel(0.000175229, 0.01));
  CHECK_THAT(fr.speed_of_sound, WithinRel(209.777, 0.01));
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

TEST_CASE("apitof pinhole smoke tests")
{
  namespace fs = std::filesystem;
  const char *data_dir_env = getenv("DATA_DIR");
  UNSCOPED_INFO("DATA_DIR environment variable not set");
  REQUIRE(data_dir_env != nullptr);
  auto density_cluster = read_histogram((string(data_dir_env) + "/density_cluster.out").c_str());
  auto rate_const = read_histogram((string(data_dir_env) + "/rate_constant.out").c_str());
  SkimmerData skimmer;
  double mesh_skimmer;
  std::tie(skimmer, mesh_skimmer) = read_skimmer((string(data_dir_env) + "/skimmer.dat").c_str());
  rescale_density(density_cluster);
  rescale_energies(density_cluster);
  rescale_energies(rate_const);
  StreamingResultQueue result_queue;
  Eigen::Vector3d rotations_0 = Eigen::Vector3d(0.0197112, 0.0229917, 0.0591769);
  auto inertia = compute_inertia(rotations_0);
  double m_ion;
  double R_cluster;
  compute_mass_and_radius(inertia, 216, m_ion, R_cluster);
  auto counters = apitof_pinhole(
    -1,
    300.0,
    182.0,
    3.53,
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
    5,
    23420.7,
    Gas{
      2.46e-10,
      4.8506e-26,
      1.4},
    Quadrupole(
      0.0,
      200.0,
      1.3e6,
      6.0e-3),
    m_ion,
    R_cluster,
    density_cluster,
    rate_const,
    skimmer,
    mesh_skimmer,
    42,
    result_queue,
    0);
  result_queue.enqueue(std::monostate{});
  CHECK(counters[Counter::nwarnings] == 0);
  CHECK(counters[Counter::n_fragmented_total] + counters[Counter::n_escaped_total] == 5);
  CHECK(counters[Counter::ncoll_total] >= 0);
  CHECK(counters[Counter::counter_collision_rejections] >= 0);
  bool exiting = false;
  auto num_partial_results = 0;
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
      // const PartialResult &partial_result = std::get<PartialResult>(result);
      num_partial_results++;
    }
    else if (std::holds_alternative<LogMessage>(result))
    {
      const LogMessage &msg = std::get<LogMessage>(result);
      {
        INFO("Unexpected log message type: " << msg.type << " content: " << msg.message);
        CHECK((
          msg.type == LogMessage::fragments ||
          msg.type == LogMessage::probabilities ||
          msg.type == LogMessage::tmp ||
          msg.type == LogMessage::tmp_evolution ||
          msg.type == LogMessage::final_position ||
          msg.type == LogMessage::pinhole));
      }
    }
  }
  CHECK(num_partial_results == 5);
}
