#include <iostream>
#include <optional>
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
  Gas gas,
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
    gas.mass,
    gas.adiabatic_index,
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

nb::typed<nb::tuple, Histogram, Histogram> densityandrate(
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
  int m_max_rate = int(energy_max_rate / bin_width);
  int m_max = int(energy_max / bin_width);
  auto energies = prepare_energies(bin_width, m_max);
  auto energies_rate = prepare_energies(bin_width, m_max_rate);
  return nb::make_tuple(Histogram(energies, rhos.col(COMB_ROW)), Histogram(energies_rate, k_rate));
}

Counters pinhole(
  ClusterData &cluster_0,
  ClusterData &cluster_1,
  ClusterData &cluster_2,
  Gas gas,
  Histogram &density_cluster,
  Histogram &rate_const,
  SkimmerData skimmer,
  double mesh_skimmer,
  InstrumentDims lengths,
  InstrumentVoltages voltages,
  double T,
  double pressure_first,
  double pressure_second,
  int N,
  std::optional<double> fragmentation_energy = nullopt,
  std::optional<Quadrupole> quadrupole = nullopt,
  int cluster_charge_sign = 1,
  unsigned long long seed = 42ull,
  std::optional<std::function<void(std::string_view, std::string)>> log_callback = nullopt,
  std::optional<std::function<void(Counters)>> result_callback = nullopt)
{
  using magic_enum::enum_name;
  using consts::hartK;
  mt19937 root_gen = mt19937(seed);
  unsigned long long root_seed = root_gen();

  double computed_fragmentation_energy;
  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == nullopt)
  {
    computed_fragmentation_energy = (cluster_1.electronic_energy + cluster_2.electronic_energy - cluster_0.electronic_energy) * hartK;
  }
  else
  {
    computed_fragmentation_energy = *fragmentation_energy;
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
        lengths,
        voltages,
        N,
        computed_fragmentation_energy,
        gas,
        quadrupole,
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
    .def(nb::init<double, double, double>(),
         nb::arg("radius"),
         nb::arg("mass"),
         nb::arg("adiabatic_index"))
    .def_ro("radius", &Gas::radius)
    .def_ro("mass", &Gas::mass)
    .def_ro("adiabatic_index", &Gas::adiabatic_index);

  nb::class_<Histogram>(m, "Histogram")
    .def(nb::init<Eigen::ArrayXd, Eigen::ArrayXd>(),
         nb::arg("x"),
         nb::arg("y"))
    .def_ro("x", &Histogram::x)
    .def_ro("y", &Histogram::y);

  nb::class_<Quadrupole>(m, "Quadrupole")
    .def(nb::init<double, double, double, double>(),
         nb::arg("dc_field"),
         nb::arg("ac_field"),
         nb::arg("radiofrequency"),
         nb::arg("r_quadrupole"))
    .def_ro("dc_field", &Quadrupole::dc_field)
    .def_ro("ac_field", &Quadrupole::dc_field)
    .def_ro("radiofrequency", &Quadrupole::dc_field)
    .def_ro("r_quadrupole", &Quadrupole::dc_field);

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
        "density_cluster"_a,
        "rate_const"_a,
        "skimmer"_a,
        "mesh_skimmer"_a,
        "lengths"_a,
        "voltages"_a,
        "T"_a,
        "pressure_first"_a,
        "pressure_second"_a,
        "N"_a,
        "fragmentation_energy"_a = std::nullopt,
        "quadrupole"_a = std::nullopt,
        "cluster_charge_sign"_a = 1,
        "seed"_a = 42ull,
        "log_callback"_a = std::nullopt,
        "result_callback"_a = std::nullopt);
}
