#include <iostream>
#include <optional>
#include <stdlib.h>
#include <random>

#include <Eigen/Dense>
#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/tuple.h>

#include "skimmer.h"
#include "densityandrate.h"
#include "mass_spec.h"
#include "consts.h"
#include "warnlogcount.h"
#include "openmp_helper.h"

using namespace std;

namespace nb = nanobind;
using namespace nb::literals;

typedef Eigen::Array<double, Eigen::Dynamic, 6> SkimmerResult;

struct PythonWarningHelper
{
  template <typename Arg>
  void operator()(Arg msg)
  {
    PyErr_WarnEx(PyExc_Warning, prepare_message(msg).c_str(), 1);
  }
};

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

SimulationResult mass_spec(
  const MassSpectrometer &ms,
  ClusterData &cluster_0,
  ClusterData &cluster_1,
  ClusterData &cluster_2,
  Gas gas,
  Histogram &density_cluster,
  Histogram &rate_const,
  int N,
  std::optional<double> fragmentation_energy = nullopt,
  int cluster_charge_sign = 1,
  unsigned long long seed = 42ull,
  std::optional<std::function<void(std::string_view, std::string)>> log_callback = nullopt,
  std::optional<std::function<void(Counters)>> result_callback = nullopt,
  SampleMode sample_mode = SampleMode::rejection,
  int loglevel = DEFAULT_LOGLEVEL)
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

  OMPExceptionHelper exception_helper;
  SimulationResult result;
  std::thread execution_thread = std::thread([&]
  {
    // TODO: Probably want to switch to jthread when possible
    exception_helper.guard([&]
    {
      result = apitof_mass_spec(
        ms,
        cluster_charge_sign,
        N,
        computed_fragmentation_energy,
        gas,
        m_ion,
        R_cluster,
        density_cluster,
        rate_const,
        root_seed,
        result_queue,
        sample_mode,
        loglevel);
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

  return result;
}

template <typename SamplerT, typename GenT>
Eigen::ArrayX2d dispatch_sample_collision(
  SamplerT sampler,
  GenT gen,
  int num_samples,
  double v_rel_norm,
  double mobility_gas,
  double mobility_gas_inv,
  double R_tot,
  double n)
{
  auto warn = PythonWarningHelper();
  Eigen::ArrayX2d samples(num_samples, 2);
  for (int i = 0; i < num_samples; i++)
  {
    std::tie(samples(i, 0), samples(i, 1)) = sampler.sample(gen, n, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, warn);
  }
  return samples;
}

Eigen::ArrayX2d sample_collision(
  SampleMode sample_mode,
  int num_samples,
  double v_rel_norm,
  Gas gas,
  double R_cluster,
  double P,
  double T,
  unsigned long long seed,
  double dtheta,
  std::optional<double> du)
{
  mt19937 gen = mt19937(seed);
  double kT = consts::boltzmann * T;
  double mobility_gas = kT / gas.mass;
  double mobility_gas_inv = gas.mass / kT;
  double boundary_u = 5.0 * sqrt(mobility_gas);
  double R_tot = gas.radius + R_cluster;
  double n = particle_density(P, T);
  double du_val;
  if (du.has_value())
  {
    du_val = *du;
  }
  else
  {
    du_val = 1.0e-4 * sqrt(mobility_gas);
  }
  if (sample_mode == SampleMode::dss_normalized)
  {
    auto sampler = GasCollCondNormHistDSSSampler(dtheta, du_val, boundary_u);
    return dispatch_sample_collision(sampler, gen, num_samples, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, n);
  }
  else if (sample_mode == SampleMode::dss_unnormalized)
  {
    auto sampler = GasCollCondUnnormHistDSSSampler(dtheta, du_val, boundary_u);
    return dispatch_sample_collision(sampler, gen, num_samples, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, n);
  }
  else if (sample_mode == SampleMode::rejection)
  {
    auto sampler = GasCollRejectionSampler(boundary_u);
    return dispatch_sample_collision(sampler, gen, num_samples, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, n);
  }
  else
  {
    throw ApiTofArgumentError([&](auto &msg)
    {
      msg << "Unknown sampling mode: " << static_cast<int>(sample_mode) << std::endl;
    });
  }
}

template <typename EnumT>
void nb_magic_enum(nanobind::handle scope, const char *name)
{
  auto enum_wrap = nb::enum_<EnumT>(scope, name);
  for (auto entry : magic_enum::enum_entries<EnumT>())
  {
    enum_wrap.value(entry.second.data(), entry.first);
  }
  enum_wrap.export_values();
}

template <typename CppExceptionT>
void register_overflow_translator(nb::exception<CppExceptionT> nb_py_exception)
{
  nb::register_exception_translator(
    [](const std::exception_ptr &exc, void *payload)
  {
    try
    {
      std::rethrow_exception(exc);
    }
    catch (const CppExceptionT &err)
    {
      auto c_py_exc = (PyObject *)payload;
      auto py_exc = nb::borrow(c_py_exc)(err.what());
      py_exc.attr("max") = err.max;
      py_exc.attr("current") = err.current;
      PyErr_SetObject(c_py_exc, py_exc.ptr());
    }
  }, nb_py_exception.ptr());
}

NB_MODULE(apitofsimraw, m)
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
    .def(nb::init<double, int, Eigen::ArrayXd>(),
         nb::arg("bin_width"),
         nb::arg("m_max"),
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
    .def_ro("ac_field", &Quadrupole::ac_field)
    .def_ro("radiofrequency", &Quadrupole::radiofrequency)
    .def_ro("r_quadrupole", &Quadrupole::r_quadrupole);

  nb::class_<MassSpectrometer>(m, "MassSpectrometer")
    .def(nb::init<SkimmerData, double, InstrumentDims, InstrumentVoltages, double, InstrumentPressures, std::optional<Quadrupole>>(),
         "skimmer"_a,
         "mesh_skimmer"_a,
         "lengths"_a,
         "voltages"_a,
         "T"_a,
         "pressures"_a,
         "quadrupole"_a = std::nullopt)
    .def_ro("skimmer", &MassSpectrometer::skimmer)
    .def_ro("mesh_skimmer", &MassSpectrometer::mesh_skimmer)
    .def_ro("lengths", &MassSpectrometer::lengths)
    .def_ro("voltages", &MassSpectrometer::voltages)
    .def_ro("T", &MassSpectrometer::T)
    .def_ro("pressures", &MassSpectrometer::pressures)
    .def_ro("quadrupole", &MassSpectrometer::quadrupole);

  m.def("validate_max_energies", static_cast<void (*)(double, double, double, double)>(validate_max_energies),
        "fragmentation_energy"_a,
        "energy_max"_a,
        "energy_max_rate"_a,
        "bin_width"_a);

  m.def("densityandrate", &densityandrate,
        "cluster_0"_a,
        "cluster_1"_a,
        "cluster_2"_a,
        "energy_max"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "fragmentation_energy"_a);

  m.def("compute_density_of_states_batch", &compute_density_of_states_batch,
        "batch_frequencies"_a,
        "energy_max"_a,
        "bin_width"_a,
        "use_old_impl"_a = false);

  nb::class_<KTotalInput>(m, "KTotalInput")
    .def(nb::init<ClusterData &, ClusterData &, double, Eigen::Ref<Eigen::ArrayXd>, Eigen::Ref<Eigen::ArrayXd>>(),
         nb::arg("cluster_1"),
         nb::arg("cluster_2"),
         nb::arg("fragmentation_energy"),
         nb::arg("rho_parent"),
         nb::arg("rho_comb"))
    .def_ro("cluster_1", &KTotalInput::cluster_1)
    .def_ro("cluster_2", &KTotalInput::cluster_2)
    .def_ro("fragmentation_energy", &KTotalInput::fragmentation_energy)
    .def_ro("rho_parent", &KTotalInput::rho_parent)
    .def_ro("rho_comb", &KTotalInput::rho_comb);

  m.def("precompute_mesh", &precompute_mesh,
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh_mode"_a);

  m.def("compute_k_total_batch", static_cast<Eigen::ArrayXXd (*)(std::vector<KTotalInput>, double, double, MeshMode)>(compute_k_total_batch),
        "batch_input"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh_mode"_a);

  m.def("compute_k_total_batch", static_cast<Eigen::ArrayXXd (*)(std::vector<KTotalInput>, double, double, std::optional<Eigen::ArrayXd>)>(compute_k_total_batch),
        "batch_input"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh"_a = std::nullopt);

  nb_magic_enum<SampleMode>(m, "SampleMode");

  m.def("mass_spec", &mass_spec,
        "ms"_a,
        "cluster_0"_a,
        "cluster_1"_a,
        "cluster_2"_a,
        "gas"_a,
        "density_cluster"_a,
        "rate_const"_a,
        "N"_a,
        "fragmentation_energy"_a = std::nullopt,
        "cluster_charge_sign"_a = 1,
        "seed"_a = 42ull,
        "log_callback"_a = std::nullopt,
        "result_callback"_a = std::nullopt,
        "sample_mode"_a = SampleMode::rejection,
        "loglevel"_a = DEFAULT_LOGLEVEL);

  nb::class_<FragmentationPathway>(m, "FragmentationPathway")
    .def(nb::init<ClusterData, ClusterData, ClusterData>(),
         nb::arg("parent"),
         nb::arg("product1"),
         nb::arg("product2"))
    .def("fragmentation_energy_kelvin", &FragmentationPathway::fragmentation_energy_kelvin);

  nb_magic_enum<Counter::Counter>(m, "Counter");
  nb_magic_enum<MeshMode>(m, "MeshMode");

  nb::exception<ApiTofError>(m, "ApiTofError");
  nb::exception<ApiTofArgumentError>(m, "ApiTofArgumentError", m.attr("ApiTofError"));
  nb::exception<ApiTofOverflowError>(m, "ApiTofOverflowError", m.attr("ApiTofError"));
  nb::exception<ApiTofDosOverflow> PyApiTofDosOverflow(m, "ApiTofDosOverflow", m.attr("ApiTofOverflowError"));
  nb::exception<ApiTofRateConstantOverflow> PyApiTofRateConstantOverflow(m, "ApiTofRateConstantOverflow", m.attr("ApiTofOverflowError"));
  nb::exception<ApiTofMaxCollisions> PyApiTofMaxCollisions(m, "ApiTofMaxCollisions", m.attr("ApiTofOverflowError"));
  nb::exception<ApiTofUnexpectedNumericalError>(m, "ApiTofUnexpectedNumericalError", m.attr("ApiTofError"));

  register_overflow_translator<ApiTofDosOverflow>(PyApiTofDosOverflow);
  register_overflow_translator<ApiTofRateConstantOverflow>(PyApiTofRateConstantOverflow);
  register_overflow_translator<ApiTofMaxCollisions>(PyApiTofMaxCollisions);

  m.def("sample_collision", &sample_collision,
        "sample_mode"_a,
        "num_samples"_a,
        "v_rel_norm"_a,
        "gas"_a,
        "R_cluster"_a,
        "P"_a,
        "T"_a,
        "seed"_a = 42,
        "dtheta"_a = 1.0e-3,
        "du"_a = std::nullopt);
}
