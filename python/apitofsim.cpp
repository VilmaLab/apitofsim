#include <cassert>
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
#include <nanobind/stl/variant.h>

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

const unsigned long long DEFAULT_SEED = 42ull;

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

struct MassSpecCleanup
{
  std::thread execution_thread;

  ~MassSpecCleanup()
  {
    execution_thread.join();
  }
};

unsigned long long root_seed(unsigned long long seed)
{
  mt19937 root_gen = mt19937(seed);
  return root_gen();
}

typedef Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> PartialCounters;

PartialCounters mk_partial_counters(const MassSpecSubstanceInput &subs)
{
  int total_counters = n_counters - 1 + subs.pathways.size();
  return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(omp_get_max_threads(), total_counters);
}

/* Caller must ensure that all parameters passed as reference outlive thread */
std::thread run_mass_spec_in_thread(
  SimulationResult &result,
  OMPExceptionHelper &exception_helper,
  const MassSpectrometer &ms,
  const MassSpecSubstanceInput &subs,
  int N,
  unsigned long long seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict,
  MassSpecLogConf logconf)
{
  return std::thread([&, N, seed, sample_mode, strict, logconf]
  {
    // TODO: Probably want to switch to jthread when possible
    exception_helper.guard([&, N, seed, sample_mode, strict, logconf]
    {
      result = apitof_mass_spec(
        ms,
        subs,
        N,
        root_seed(seed),
        result_queue,
        sample_mode,
        strict,
        logconf);
    });
    result_queue.enqueue(std::monostate{});
  });
}

std::variant<std::tuple<const std::string, const std::string>, Eigen::ArrayXi, EventMessage, std::monostate> pump_mass_spec_queue(
  StreamingResultQueue &result_queue,
  PartialCounters &partial_counters,
  OMPExceptionHelper &exception_helper)
{
  using magic_enum::enum_name;

  try
  {
    StreamingResultElement result;
    result_queue.wait_dequeue(result);
    if (std::holds_alternative<std::monostate>(result))
    {
      // Still need to pump out any pending messages
      while (true)
      {
        bool got = result_queue.try_dequeue(result);
        if (!got)
        {
          break;
        }
      }
    }
    else if (std::holds_alternative<PartialResult>(result))
    {
      const PartialResult &partial_result = std::get<PartialResult>(result);
      partial_counters.row(partial_result.thread_id) = partial_result.counters.transpose();
      Eigen::ArrayXi cur_counters = partial_counters.colwise().sum();
      return cur_counters;
    }
    else if (std::holds_alternative<LogMessage>(result))
    {
      const LogMessage &msg = std::get<LogMessage>(result);
      const std::string name = std::string(enum_name(msg.type));
      return std::tuple<const std::string, const std::string>(name, msg.message);
    }
    else if (std::holds_alternative<EventMessage>(result))
    {
      return std::get<EventMessage>(result);
    }
    else if (std::holds_alternative<std::exception>(result))
    {
      const std::exception &exc = std::get<std::exception>(result);
      throw exc;
    }
  }
  catch (...)
  {
    try
    {
      exception_helper.rethrow(true);
    }
    catch (const std::exception &exc)
    {
      std::cerr << "\nGot multiple exceptions! I had to ignore the following while propagating the another: " << exc.what() << "\n\n"
                << std::flush;
    }
    catch (...)
    {
      std::cerr << "\nGot multiple exceptions! I had to ignore one while propagating the another, but it is not a std::exception!\n\n"
                << std::flush;
    }
    throw;
  }
  return std::monostate{};
}

SimulationResult
mass_spec(
  const MassSpectrometer &ms,
  const MassSpecSubstanceInput &subs,
  int N,
  unsigned long long seed = DEFAULT_SEED,
  std::optional<std::function<void(std::string_view, std::string)>> log_callback = nullopt,
  std::optional<std::function<void(Eigen::ArrayXi)>> result_callback = nullopt,
  std::optional<std::function<void(EventMessage)>> event_callback = nullopt,
  SampleMode sample_mode = SampleMode::rejection,
  bool strict = true,
  MassSpecLogConf logconf = MassSpecLogConf{})
{
  StreamingResultQueue result_queue;
  OMPExceptionHelper exception_helper;
  SimulationResult result;
  PartialCounters partial_counters = mk_partial_counters(subs);
  auto cleanup = MassSpecCleanup{run_mass_spec_in_thread(result, exception_helper, ms, subs, N, seed, result_queue, sample_mode, strict, logconf)};
  while (true)
  {
    auto result = pump_mass_spec_queue(result_queue, partial_counters, exception_helper);
    if (std::holds_alternative<std::tuple<const std::string, const std::string>>(result))
    {
      auto tpl = std::get<std::tuple<const std::string, const std::string>>(result);
      if (log_callback)
      {
        (*log_callback)(std::get<0>(tpl), std::get<1>(tpl));
      }
    }
    else if (std::holds_alternative<Eigen::ArrayXi>(result))
    {
      if (result_callback)
      {
        (*result_callback)(std::get<Eigen::ArrayXi>(result));
      }
    }
    else if (std::holds_alternative<EventMessage>(result))
    {
      if (event_callback)
      {
        (*event_callback)(std::get<EventMessage>(result));
      }
    }
    else
    {
      assert(std::holds_alternative<std::monostate>(result));
      break;
    }
  }
  exception_helper.rethrow();
  return result;
}

struct MassSpecIterator
{
  StreamingResultQueue result_queue;
  PartialCounters partial_counters;
  OMPExceptionHelper exception_helper;
  std::thread execution_thread;
  SimulationResult final_result{};
  bool finished;

  MassSpecIterator(
    const MassSpectrometer &ms,
    const MassSpecSubstanceInput &subs,
    int N,
    unsigned long long seed = DEFAULT_SEED,
    SampleMode sample_mode = SampleMode::rejection,
    bool strict = true,
    MassSpecLogConf logconf = MassSpecLogConf{}) : result_queue(),
                                       partial_counters(mk_partial_counters(subs)),
                                       exception_helper(),
                                       execution_thread(run_mass_spec_in_thread(final_result, exception_helper, ms, subs, N, seed, result_queue, sample_mode, strict, logconf)),
                                       finished(false)
  {
  }

  std::variant<std::tuple<const std::string, const std::string>, Eigen::ArrayXi, EventMessage, SimulationResult> __next__()
  {
    if (finished)
    {
      throw nb::stop_iteration();
    }
    auto result = pump_mass_spec_queue(result_queue, partial_counters, exception_helper);
    if (std::holds_alternative<std::monostate>(result))
    {
      finished = true;
      execution_thread.join();
      return final_result;
    }
    else if (std::holds_alternative<Eigen::ArrayXi>(result))
    {
      return std::get<Eigen::ArrayXi>(result);
    }
    else if (std::holds_alternative<EventMessage>(result))
    {
      return std::get<EventMessage>(result);
    }
    else
    {
      assert((std::holds_alternative<std::tuple<const std::string, const std::string>>(result)));
      return std::get<std::tuple<const std::string, const std::string>>(result);
    }
  }

  void join_if_joinable()
  {
    if (execution_thread.joinable())
    {
      execution_thread.join();
    }
  }
};

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
    .def(nb::init<SkimmerData, double, InstrumentDims, InstrumentVoltages, double, InstrumentPressures, std::optional<Quadrupole>, std::optional<double>>(),
         "skimmer"_a,
         "mesh_skimmer"_a,
         "lengths"_a,
         "voltages"_a,
         "T"_a,
         "pressures"_a,
         "quadrupole"_a = std::nullopt,
         "radius_pinhole"_a = 0.001)
    .def_ro("skimmer", &MassSpectrometer::skimmer)
    .def_ro("mesh_skimmer", &MassSpectrometer::mesh_skimmer)
    .def_ro("lengths", &MassSpectrometer::lengths)
    .def_ro("voltages", &MassSpectrometer::voltages)
    .def_ro("T", &MassSpectrometer::T)
    .def_ro("pressures", &MassSpectrometer::pressures)
    .def_ro("quadrupole", &MassSpectrometer::quadrupole)
    .def_ro("radius_pinhole", &MassSpectrometer::radius_pinhole);

  nb::class_<MassSpecInputFragmentationPathway>(m, "MassSpecInputFragmentationPathway")
    .def(nb::init<ClusterData &, ClusterData &, ClusterData &, const Histogram &, std::optional<double>>(),
         "cluster_0"_a,
         "cluster_1"_a,
         "cluster_2"_a,
         "rate_const"_a,
         "bonding_energy"_a = std::nullopt)
    .def(nb::init<Histogram, double>(), "rate_const"_a, "bonding_energy"_a)
    .def_ro("rate_const", &MassSpecInputFragmentationPathway::rate_const)
    .def_ro("bonding_energy", &MassSpecInputFragmentationPathway::bonding_energy);

  nb::class_<MassSpecSubstanceInput>(m, "MassSpecSubstanceInput")
    .def(nb::init<ClusterData &, ClusterData &, ClusterData &, Gas, const Histogram &, const Histogram &, std::optional<double>, int>(),
         "cluster_0"_a,
         "cluster_1"_a,
         "cluster_2"_a,
         "gas"_a,
         "density_cluster"_a,
         "rate_const"_a,
         "fragmentation_energy"_a = std::nullopt,
         "cluster_charge_sign"_a = defaults::cluster_charge_sign)
    .def(nb::init<ClusterData &, std::vector<MassSpecInputFragmentationPathway>, Gas, const Histogram &, int>(),
         "cluster_0"_a,
         "pathways"_a,
         "gas"_a,
         "density_cluster"_a,
         "cluster_charge_sign"_a = defaults::cluster_charge_sign)
    .def(nb::init<int, double, double, const Histogram, const std::vector<MassSpecInputFragmentationPathway>, const Gas>(),
         "cluster_charge_sign"_a,
         "m_ion"_a,
         "R_cluster"_a,
         "density_cluster"_a,
         "pathways"_a,
         "gas"_a)
    .def_ro("cluster_charge_sign", &MassSpecSubstanceInput::cluster_charge_sign)
    .def_ro("m_ion", &MassSpecSubstanceInput::m_ion)
    .def_ro("R_cluster", &MassSpecSubstanceInput::R_cluster)
    .def_ro("density_cluster", &MassSpecSubstanceInput::density_cluster)
    .def_ro("pathways", &MassSpecSubstanceInput::pathways)
    .def_ro("gas", &MassSpecSubstanceInput::gas);

  m.def("validate_max_energies", static_cast<void (*)(double, double, double, double)>(validate_max_energies),
        "fragmentation_energy"_a,
        "energy_max"_a,
        "energy_max_rate"_a,
        "bin_width"_a);

  m.def("densityandrate",
        &densityandrate,
        nb::call_guard<nb::gil_scoped_release>(),
        "cluster_0"_a,
        "cluster_1"_a,
        "cluster_2"_a,
        "energy_max"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "fragmentation_energy"_a);

  m.def("compute_density_of_states_batch",
        &compute_density_of_states_batch,
        nb::call_guard<nb::gil_scoped_release>(),
        "batch_frequencies"_a,
        "energy_max"_a,
        "bin_width"_a,
        "use_old_impl"_a = false);

  nb::class_<KTotalInput>(m, "KTotalInput")
    .def(nb::init<ClusterData &, ClusterData &, double, Eigen::Ref<const Eigen::ArrayXd>, Eigen::Ref<const Eigen::ArrayXd>>(),
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

  m.def("precompute_mesh",
        &precompute_mesh,
        nb::call_guard<nb::gil_scoped_release>(),
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh_mode"_a);

  m.def("compute_k_total_batch",
        static_cast<Eigen::ArrayXXd (*)(std::vector<KTotalInput>, double, double, MeshMode, std::optional<std::function<void(size_t)>>)>(compute_k_total_batch),
        nb::call_guard<nb::gil_scoped_release>(),
        "batch_input"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh_mode"_a,
        "progress_callback"_a = std::nullopt);

  m.def("compute_k_total_batch",
        static_cast<Eigen::ArrayXXd (*)(std::vector<KTotalInput>, double, double, std::optional<const Eigen::ArrayXd>, std::optional<std::function<void(size_t)>>)>(compute_k_total_batch),
        nb::call_guard<nb::gil_scoped_release>(),
        "batch_input"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh"_a = std::nullopt,
        "progress_callback"_a = std::nullopt);

  nb_magic_enum<SampleMode>(m, "SampleMode");

  nb::class_<MassSpecLogConf>(m, "MassSpecLogConf")
    .def(nb::init<>())
    .def(nb::init<int, bool>(), "level"_a, "log_events"_a)
    .def_ro("level", &MassSpecLogConf::level)
    .def_ro("log_events", &MassSpecLogConf::log_events);

  m.def("mass_spec",
        &mass_spec,
        nb::call_guard<nb::gil_scoped_release>(),
        "ms"_a,
        "subs"_a,
        "N"_a,
        "seed"_a = DEFAULT_SEED,
        "log_callback"_a = std::nullopt,
        "result_callback"_a = std::nullopt,
        "event_callback"_a = std::nullopt,
        "sample_mode"_a = SampleMode::rejection,
        "strict"_a = true,
        "logconf"_a = DEFAULT_LOGCONF);

  nb::class_<ParticleStateMsg>(m, "ParticleState")
    .def_ro("realization", &ParticleStateMsg::realization)
    .def_ro("postime", &ParticleStateMsg::postime)
    .def_ro("velocity", &ParticleStateMsg::velocity)
    .def_ro("omega", &ParticleStateMsg::omega)
    .def_ro("rot_energy", &ParticleStateMsg::rot_energy)
    .def_ro("internal_energy", &ParticleStateMsg::internal_energy);

  nb::class_<CollisionEvent>(m, "CollisionEvent")
    .def_ro("state", &CollisionEvent::state)
    .def_ro("theta", &CollisionEvent::theta)
    .def_ro("u_norm", &CollisionEvent::u_norm)
    .def_ro("accepted", &CollisionEvent::accepted);

  nb::class_<FragmentationEvent>(m, "FragmentationEvent")
    .def_ro("state", &FragmentationEvent::state)
    .def_ro("pathway_index", &FragmentationEvent::pathway_index);

  nb::class_<EscapeEvent>(m, "EscapeEvent")
    .def_ro("state", &EscapeEvent::state);

  nb::class_<MassSpecIterator>(m, "MassSpecIterator")
    .def(nb::init<const MassSpectrometer &, const MassSpecSubstanceInput &, int, unsigned long long, SampleMode, bool, MassSpecLogConf>(),
         nb::call_guard<nb::gil_scoped_release>(),
         "ms"_a,
         "subs"_a,
         "N"_a,
         "seed"_a = DEFAULT_SEED,
         "sample_mode"_a = SampleMode::rejection,
         "strict"_a = true,
         "logconf"_a = MassSpecLogConf{})
    .def("__next__", &MassSpecIterator::__next__)
    .def("join_if_joinable", &MassSpecIterator::join_if_joinable);

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

  m.def("debug_info", &debug_info);

  nb::module_ m_defaults = m.def_submodule("defaults", "Default parameter values");

  m_defaults.attr("cluster_charge_sign") = defaults::cluster_charge_sign;
}
