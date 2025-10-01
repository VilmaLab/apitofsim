#include <iostream>
#include <stdlib.h>

#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/optional.h>

#include "skimmer_lib.h"
#include "densityandrate_lib.h"
#include "apitof_pinhole_lib.h"

namespace nb = nanobind;
using namespace nb::literals;

typedef Eigen::Array<double, Eigen::Dynamic, 6> SkimmerResult;

SkimmerResult skimmer(
  double T0,
  double P0,
  double rmax,
  double dc,
  double alpha_factor,
  double m,
  double ga,
  int N,
  int M,
  int resolution,
  double tolerance)
{
  int nwarnings = 0;
  std::ofstream warnings;
  warnings.open("warnings_skimmer.dat");
  warnings << std::scientific;
  Skimmer s = {
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
    warnings,
  };

  SkimmerResult result(resolution, 6);

  int i = 0;
  while (true)
  {
    s.next();
    auto r = s.get();
    if (r.has_value())
    {
      result(i, 0) = r->r;
      result(i, 1) = r->vel;
      result(i, 2) = r->T;
      result(i, 3) = r->P;
      result(i, 4) = r->rho;
      result(i, 5) = r->speed_of_sound;
      i++;
    }
    else
    {
      break;
    }
  }
  return result;
}

nb::tuple densityandrate(
  ClusterData &cluster_0,
  ClusterData &cluster_1,
  ClusterData &cluster_2,
  double energy_max,
  double energy_max_rate,
  double bin_width,
  double fragmentation_energy)
{
  // TODO:: The individual product densities are not used -- should be possible to not compute them
  DensityResult rhos = compute_density_of_states_all(cluster_0, cluster_1, cluster_2, energy_max, bin_width);
  const Eigen::ArrayXd k_rate = compute_k_total_full(
    cluster_0,
    cluster_1,
    cluster_2,
    rhos,
    fragmentation_energy,
    energy_max_rate,
    bin_width);
  return nb::make_tuple(rhos, k_rate);
}

Counters pinhole(
  ClusterData &cluster_0,
  ClusterData &cluster_1,
  ClusterData &cluster_2,
  Gas gas,
  double fragmentation_energy,
  Histogram &density_cluster,
  Histogram &rate_const,
  SkimmerData skimmer,
  double mesh_skimmer,
  int cluster_charge_sign,
  double L0,
  double Lsk,
  double L1,
  double L2,
  double L3,
  double V0,
  double V1,
  double V2,
  double V3,
  double V4,
  double T,
  double pressure_first,
  double pressure_second,
  double r_quadrupole,
  double radiofrequency,
  double dc_field,
  double ac_field,
  int N,
  unsigned long long seed,
  std::optional<std::function<void(std::string_view, std::string)>> log_callback,
  std::optional<std::function<void(Counters)>> result_callback)
{
  using magic_enum::enum_name;
  using consts::hartK;

  mt19937 root_gen = mt19937(seed);
  unsigned long long root_seed = root_gen();

  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == 0)
  {
    fragmentation_energy = (cluster_1.electronic_energy + cluster_2.electronic_energy - cluster_0.electronic_energy) * hartK;
  }

  auto inertia = compute_inertia(cluster_0.rotations);
  double m_ion;
  double R_cluster;
  compute_mass_and_radius(inertia, cluster_0.atomic_mass, m_ion, R_cluster);

  rescale_density(density_cluster);
  rescale_energies(density_cluster);
  rescale_energies(rate_const);

  StreamingResultQueue result_queue;
  Counters counters;
  OMPExceptionHelper exception_helper;
  std::thread execution_thread = std::thread([&]
  {
    // TODO: Probably want to switch to jthread when possible
    exception_helper.guard([&]
    {
      counters = apitof_pinhole(
        cluster_charge_sign,
        T,
        pressure_first,
        pressure_second,
        L0,
        Lsk,
        L1,
        L2,
        L3,
        V0,
        V1,
        V2,
        V3,
        V4,
        N,
        fragmentation_energy,
        gas.radius,
        gas.mass,
        dc_field,
        ac_field,
        radiofrequency,
        r_quadrupole,
        m_ion,
        R_cluster,
        density_cluster,
        rate_const,
        skimmer,
        mesh_skimmer,
        root_seed,
        result_queue);
    });
    result_queue.enqueue(std::monostate{});
  });

  Eigen::Array<int, Eigen::Dynamic, n_counters> partial_counters = Eigen::Array<int, Eigen::Dynamic, n_counters>::Zero(omp_get_max_threads(), n_counters);
  bool exiting = false;
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
      partial_counters.row(partial_result.thread_id) = partial_result.counters.transpose();
      Counters cur_counters = partial_counters.colwise().sum();
      if (result_callback)
      {
        (*result_callback)(cur_counters);
      }
    }
    else if (std::holds_alternative<LogMessage>(result))
    {
      const LogMessage &msg = std::get<LogMessage>(result);
      if (log_callback)
      {
        (*log_callback)(enum_name(msg.type), msg.message);
      }
    }
  }
  execution_thread.join();
  exception_helper.rethrow();

  std::cout << setprecision(3);

  return counters;
}

NB_MODULE(_apitofsim, m)
{
  m.doc() = "APi-TOF-MS simulation module";
  m.def("skimmer", &skimmer);

  nb::class_<ClusterData>(m, "ClusterData")
    .def(nb::init<int, double, Eigen::Vector3d, Eigen::ArrayXd>(),
         "atomic_mass"_a,
         "electronic_energy"_a,
         "rotations"_a,
         "frequencies"_a)
    .def_ro("atomic_mass", &ClusterData::atomic_mass)
    .def_ro("electronic_energy", &ClusterData::electronic_energy)
    .def_ro("rotations", &ClusterData::rotations)
    .def_ro("frequencies", &ClusterData::frequencies)
    .def_ro("inertia_moment", &ClusterData::inertia_moment)
    .def_ro("radius", &ClusterData::radius)
    .def_ro("mass", &ClusterData::mass)
    .def("num_oscillators", &ClusterData::num_oscillators)
    .def("is_atom_like_product", &ClusterData::is_atom_like_product)
    .def("compute_derived", &ClusterData::compute_derived);

  nb::class_<Gas>(m, "Gas")
    .def(nb::init<double, double>(),
         nb::arg("radius"),
         nb::arg("mass"))
    .def_ro("radius", &Gas::radius)
    .def_ro("mass", &Gas::mass);

  nb::class_<Histogram>(m, "Histogram")
    .def(nb::init<Eigen::ArrayXd, Eigen::ArrayXd>(),
         nb::arg("x"),
         nb::arg("y"))
    .def_ro("x", &Histogram::x)
    .def_ro("y", &Histogram::y);

  m.def("densityandrate", &densityandrate,
        "cluster_0"_a,
        "cluster_1"_a,
        "cluster_2"_a,
        "energy_max"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "fragmentation_energy"_a);

  m.def("pinhole", &pinhole,
        "cluster_0"_a,
        "cluster_1"_a,
        "cluster_2"_a,
        "gas"_a,
        "fragmentation_energy"_a,
        "density_cluster"_a,
        "rate_const"_a,
        "skimmer"_a,
        "mesh_skimmer"_a,
        "cluster_charge_sign"_a,
        "L0"_a,
        "Lsk"_a,
        "L1"_a,
        "L2"_a,
        "L3"_a,
        "V0"_a,
        "V1"_a,
        "V2"_a,
        "V3"_a,
        "V4"_a,
        "T"_a,
        "pressure_first"_a,
        "pressure_second"_a,
        "r_quadrupole"_a,
        "radiofrequency"_a,
        "dc_field"_a,
        "ac_field"_a,
        "N"_a,
        "seed"_a,
        "log_callback"_a = std::nullopt,
        "result_callback"_a = std::nullopt);
}
