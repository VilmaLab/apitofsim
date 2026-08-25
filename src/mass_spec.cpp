#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <variant>
#include <vector>
#include "apitofsim.h"
#include <magic_enum/magic_enum.hpp>
#include "consts.h"
#include "warnlogcount.h"
#include "exceptions.h"
#include "samplers.h"
#include "mass_spec.h"
#include "operation_context.h"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

using namespace std;
using magic_enum::enum_count;
using moodycamel::BlockingConcurrentQueue;
using consts::boltzmann;

const double DT_MULTIPLIER = 1.0e-3;
const int MAX_COLL = 1e6;

Quadrupole::Quadrupole(
  double dc_field,
  double ac_field,
  double radiofrequency,
  double r_quadrupole)
    : dc_field(dc_field), ac_field(ac_field), radiofrequency(radiofrequency), r_quadrupole(r_quadrupole)
{
  angular_velocity = 2.0 * consts::pi * radiofrequency;
}

double compute_mathieu_factor(double m_ion, double r_quadrupole)
{
  return consts::eV / (m_ion * r_quadrupole * r_quadrupole);
}

void rescale_density(Histogram &density)
{
  using consts::boltzmann;
  for (int m = 0; m < density.length(); m++)
  {
    density.y[m] = density.y[m] / boltzmann;
  }
}

void rescale_energies(Histogram &energies)
{
  using consts::boltzmann;
  for (int m = 0; m < energies.length(); m++)
  {
    energies.x[m] = energies.x[m] * boltzmann;
  }
  energies.x_max *= boltzmann;
  energies.bin_width *= boltzmann;
}

Histogram scaled_density(const Histogram &density_cluster)
{
  Histogram my_density_cluster(density_cluster);
  rescale_density(my_density_cluster);
  rescale_energies(my_density_cluster);
  return my_density_cluster;
}

Histogram scaled_rate_const(const Histogram &rate_const)
{
  Histogram my_rate_const(rate_const);
  rescale_energies(my_rate_const);
  return my_rate_const;
}

MassSpecInputFragmentationPathway::MassSpecInputFragmentationPathway(
  const ClusterData &cluster_0,
  const ClusterData &cluster_1,
  const ClusterData &cluster_2,
  const Histogram &rate_const,
  std::optional<double> fragmentation_energy) : rate_const(rate_const)
{
  using consts::hartK;
  double computed_fragmentation_energy;
  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == std::nullopt)
  {
    computed_fragmentation_energy = (cluster_1.electronic_energy + cluster_2.electronic_energy - cluster_0.electronic_energy) * hartK;
  }
  else
  {
    computed_fragmentation_energy = *fragmentation_energy;
  }
  this->bonding_energy = computed_fragmentation_energy * boltzmann;
}

MassSpecInputFragmentationPathway::MassSpecInputFragmentationPathway(
  const Histogram rate_const,
  double bonding_energy) : rate_const(rate_const), bonding_energy(bonding_energy * boltzmann)
{
}

MassSpecSubstanceSingleInput::MassSpecSubstanceSingleInput(
  const ClusterData &cluster_0,
  const ClusterData &cluster_1,
  const ClusterData &cluster_2,
  Gas gas,
  const Histogram &density_cluster,
  const Histogram &rate_const,
  std::optional<double> fragmentation_energy,
  int cluster_charge_sign) : cluster_charge_sign(cluster_charge_sign),
                             density_cluster(density_cluster),
                             pathways({MassSpecInputFragmentationPathway(cluster_0, cluster_1, cluster_2, rate_const, fragmentation_energy)}),
                             gas(gas)
{
  compute_mass_and_radius(compute_inertia(cluster_0.rotations), cluster_0.atomic_mass, this->m_ion, this->R_cluster);
}

MassSpecSubstanceSingleInput::MassSpecSubstanceSingleInput(
  int cluster_charge_sign,
  double m_ion,
  double R_cluster,
  const Histogram density_cluster,
  const std::vector<MassSpecInputFragmentationPathway> pathways,
  const Gas gas) : cluster_charge_sign(cluster_charge_sign),
                   m_ion(m_ion),
                   R_cluster(R_cluster),
                   density_cluster(density_cluster),
                   pathways(pathways),
                   gas(gas)
{
}

MassSpecSubstanceSingleInput::MassSpecSubstanceSingleInput(
  const ClusterData &cluster_0,
  const std::vector<MassSpecInputFragmentationPathway> pathways,
  Gas gas,
  const Histogram &density_cluster,
  int cluster_charge_sign) : cluster_charge_sign(cluster_charge_sign),
                             density_cluster(density_cluster),
                             pathways(pathways),
                             gas(gas)
{
  compute_mass_and_radius(compute_inertia(cluster_0.rotations), cluster_0.atomic_mass, this->m_ion, this->R_cluster);
}

MSSubstanceTreeCluster::MSSubstanceTreeCluster(
  double m_ion,
  double R_cluster,
  const Histogram density_cluster) : m_ion(m_ion),
                                     R_cluster(R_cluster),
                                     density_cluster(density_cluster)
{
}

MSSubstanceTreeCluster::MSSubstanceTreeCluster(
  const ClusterData &cluster_0,
  const Histogram density_cluster) : density_cluster(density_cluster)
{
  compute_mass_and_radius(compute_inertia(cluster_0.rotations), cluster_0.atomic_mass, this->m_ion, this->R_cluster);
}

MassSpecSubstanceTreeInput::MassSpecSubstanceTreeInput(
  int cluster_charge_sign,
  Gas gas,
  std::vector<MSSubstanceTreeCluster> cluster_payloads,
  std::vector<MassSpecInputFragmentationPathway> pathway_payloads,
  std::vector<MSSubstanceTreeNode> tree_nodes,
  std::vector<MSSubstanceTreePathway> tree_pathways) : cluster_charge_sign(cluster_charge_sign),
                                                       gas(gas),
                                                       cluster_payloads(cluster_payloads),
                                                       pathway_payloads(pathway_payloads),
                                                       tree_nodes(tree_nodes),
                                                       tree_pathways(tree_pathways)
{
}

enum struct TimeNextCollOutcome
{
  fragmentation,
  gas_collision,
  escape
};

// LIST OF FUNCTIONS
// Here we are
template <typename GenT>
Eigen::Vector3d init_vel(GenT &gen, normal_distribution<double> &gauss, double m, double kT);
template <typename GenT>
Eigen::Vector3d init_ang_vel(GenT &gen, normal_distribution<double> &gauss, double m, double kT, double R);
template <typename GenT>
double init_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double kT, const Histogram &density_cluster);
double evaluate_rotational_energy(const Eigen::Vector3d &omega, double inertia);
double evaluate_internal_energy(double vib_energy, double rot_energy);
double evaluate_rate_const(const Histogram &rate_const, double energy);
template <typename GenT>
TimeNextCollOutcome time_next_coll_quadrupole(GenT &gen, uniform_real_distribution<double> &unif, Eigen::Vector3d &v_cluster, double &v_cluster_norm, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double &z, double &x, double &y, double &t_fragmentation, const Eigen::Array4d &acc, double &t, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole);
std::tuple<double, Eigen::Vector3d, double, double, double> get_quantities_for_collision(double z, const ChamberQuantities &chamber, double m_gas, const Eigen::Vector3d &v_cluster, double v_gas, double pressure, double temperature);
void update_physical_quantities(double z, const SkimmerData &skimmer, double mesh_skimmer, double &v_gas, double &temperature, double &pressure, double &density, const ChamberQuantities &chamber, double T);
// void evaluate_relative_velocity(double z, double *v_cluster, double &v_rel_norm, double v_gas, double *v_rel, double first_chamber_end, double sk_end);
void update_velocities(Eigen::Vector3d &v_cluster, double &v_cluster_norm, const Eigen::Vector3d &v_rel, double v_gas);
void update_rot_vel(Eigen::Vector3d &omega, double rot_energy_old, double rot_energy);
double boundary_vib_energy(double vib_energy_old, double reduced_mass, double u_norm, double v_cluster_norm, double theta);
template <typename GenT, typename VibEnergySamplerT>
std::tuple<double, double> redistribute_internal_energy(GenT &gen, VibEnergySamplerT &sampler, double vib_energy, double rot_energy);
void eval_velocities(Eigen::Vector3d &v, Eigen::Vector3d &omega, const Eigen::Vector2d &u, double vib_energy, double vib_energy_old, double M, double m, double R_cluster);
void change_coord(const Eigen::Vector3d &v_cluster, double theta, double phi, double alpha, Eigen::Vector3d &x3, Eigen::Vector3d &y3, Eigen::Vector3d &z3);
template <typename GenT>
bool eval_collision(GenT &gen, uniform_real_distribution<double> &unif, double gas_mean_free_path, double x, double y, double z, double L, std::optional<double> pinhole, double quadrupole_end, Eigen::Vector3d &v_cluster, Eigen::Vector3d &omega, double u_norm, double theta, double R_cluster, double vib_energy, double vib_energy_old, double m_ion, double m_gas, double temperature, LogHelper pinhole_logger, int loglevel);
template <typename GenT>
double onedimMaxwell(GenT &gen, normal_distribution<double> &gauss, double m, double kT);
double mean_free_path(double R, double kT, double pressure);
double eval_solid_angle_stokes(double R, double L, double xx, double yy, double zz);
int zone(double z, const CumulativeLengths &clens);

template <typename GasCollSamplerT, typename VibEnergySamplerT>
SimulationResult apitof_mass_spec(
  const MassSpectrometer &ms,
  const MassSpecSubstanceSingleInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  GasCollSamplerT gas_coll_sampler,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation);

template <typename GasCollSamplerT, typename VibEnergySamplerT>
SimulationResult apitof_mass_spec(
  const MassSpectrometer &ms,
  const MassSpecSubstanceTreeInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  GasCollSamplerT gas_coll_sampler,
  // VibEnergySamplerT vib_energy_sampler,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation);

template <typename MassSpecSubstanceT>
SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceT &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation)
{
  using consts::boltzmann;
  double m_gas = subs.gas.mass;
  double kT = boltzmann * mass_spec.T;
  double mobility_gas = kT / m_gas; // thermal agitation
  double boundary_u = 5.0 * sqrt(mobility_gas);
  const double du = 1.0e-4 * sqrt(mobility_gas);
  const double dtheta = 1.0e-3;
  if (sample_mode == SampleMode::dss_normalized)
  {
    return apitof_mass_spec<GasCollCondNormHistDSSSampler, VibEnergyNormSampler>(
      mass_spec,
      subs,
      N,
      root_seed,
      result_queue,
      GasCollCondNormHistDSSSampler(dtheta, du, boundary_u),
      // VibEnergyNormSampler(subs.density_cluster),
      strict,
      logconf,
      operation);
  }
  else if (sample_mode == SampleMode::dss_unnormalized)
  {
    return apitof_mass_spec<GasCollCondUnnormHistDSSSampler, VibEnergyUnnormSampler>(
      mass_spec,
      subs,
      N,
      root_seed,
      result_queue,
      GasCollCondUnnormHistDSSSampler(dtheta, du, boundary_u),
      // VibEnergyUnnormSampler(subs.density_cluster),
      strict,
      logconf,
      operation);
  }
  else if (sample_mode == SampleMode::rejection)
  {
    return apitof_mass_spec<GasCollRejectionSampler, VibEnergyNormSampler>(
      mass_spec,
      subs,
      N,
      root_seed,
      result_queue,
      GasCollRejectionSampler(boundary_u),
      // VibEnergyNormSampler(subs.density_cluster),
      strict,
      logconf,
      operation);
  }
  else
  {
    throw ApiTofArgumentError([&](auto &msg)
    {
      msg << "Unknown sampling mode: " << static_cast<int>(sample_mode) << std::endl;
    });
  }
}

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceSingleInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  bool on_main_thread)
{
  OperationContext operation;
  return operation.run([&]
  {
    return apitof_mass_spec<MassSpecSubstanceSingleInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation);
  }, !on_main_thread);
}

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  bool on_main_thread)
{
  OperationContext operation;
  return operation.run([&]
  {
    return apitof_mass_spec<MassSpecSubstanceTreeInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation);
  }, !on_main_thread);
}

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceSingleInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation)
{
  return apitof_mass_spec<MassSpecSubstanceSingleInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation);
}

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation)
{
  return apitof_mass_spec<MassSpecSubstanceTreeInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation);
}

template SimulationResult apitof_mass_spec<MassSpecSubstanceSingleInput>(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceSingleInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation);

template SimulationResult apitof_mass_spec<MassSpecSubstanceTreeInput>(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation);

template <typename GenT>
std::tuple<double, std::optional<ApiTofRateConstantOverflow>> next_fragmentation_time(
  GenT &gen, uniform_real_distribution<double> &unif, const MassSpecInputFragmentationPathway &pathway, double internal_energy, bool strict = true)
{
  using consts::boltzmann;

  double rate_constant;
  std::optional<ApiTofRateConstantOverflow> exception = std::nullopt;
  double delta_en = internal_energy - pathway.bonding_energy;

  if (delta_en > 0.0)
  {
    auto energy_max_rate = pathway.rate_const.x_max;
    if (delta_en > energy_max_rate)
    {
      exception = ApiTofRateConstantOverflow(energy_max_rate / boltzmann, delta_en / boltzmann);
      if (strict)
      {
        throw *exception;
      }
      delta_en = energy_max_rate;
      rate_constant = pathway.rate_const.last_y();
    }
    else
    {
      rate_constant = evaluate_rate_const(pathway.rate_const, delta_en);
    }
  }
  else
  {
    rate_constant = 0.0;
  }

  double r = unif(gen);
  double t_fragmentation;
  if (rate_constant > 0)
  {
    t_fragmentation = -log(r) / rate_constant;
  }
  else
  {
    t_fragmentation = std::numeric_limits<double>::infinity();
  }

  return std::make_tuple(t_fragmentation, exception);
}

template <typename GenT, typename PathwaysT>
std::tuple<int, double, std::optional<ApiTofRateConstantOverflow>> next_fragmentation_time_multi(
  GenT &gen, uniform_real_distribution<double> &unif, PathwaysT pathways, double internal_energy, bool strict = true)
{
  int effective_pathway_index = 0;
  double t_next_fragmentation = std::numeric_limits<double>::infinity();
  std::optional<ApiTofRateConstantOverflow> effective_exception = std::nullopt;
  int pathway_index = 0;
  for (const MassSpecInputFragmentationPathway &pathway : pathways)
  {
    double t_fragmentation;
    std::optional<ApiTofRateConstantOverflow> exception = std::nullopt;
    std::tie(t_fragmentation, exception) = next_fragmentation_time<GenT>(gen, unif, pathway, internal_energy, strict);
    if (t_fragmentation < t_next_fragmentation)
    {
      effective_pathway_index = pathway_index;
      t_next_fragmentation = t_fragmentation;
      effective_exception = exception;
    }
    pathway_index++;
  }
  return std::make_tuple(effective_pathway_index, t_next_fragmentation, effective_exception);
}

Pressures::Pressures(const InstrumentPressures &pressures, double kT) : P(pressures),
                                                                        n(particle_density(pressures[0], kT), particle_density(pressures[1], kT))
{
}

Eigen::Array2d Pressures::histogram_dts(double R_tot, double mobility_gas, double mobility_gas_inv, double multiplier, std::optional<Quadrupole> quadrupole) const
{
  double dt1 = multiplier / coll_freq(this->n[0], mobility_gas, mobility_gas_inv, R_tot, 0.0);
  double dt2 = multiplier / coll_freq(this->n[1], mobility_gas, mobility_gas_inv, R_tot, 0.0);
  if (quadrupole && dt2 > 1.0 / quadrupole->radiofrequency / 1000.0)
  {
    dt2 = 1.0 / quadrupole->radiofrequency / 1000.0;
  }
  return Eigen::Array2d(dt1, dt2);
}

CumulativeLengths::CumulativeLengths(const InstrumentDims &lengths)
{
  first_chamber_end = lengths[0];
  sk_end = first_chamber_end + lengths[SKIMMER_LENGTH];
  quadrupole_start = sk_end + lengths[1];
  quadrupole_end = quadrupole_start + lengths[2];
  second_chamber_end = quadrupole_end + lengths[3];
  total_length = second_chamber_end;
}

void CumulativeLengths::info(std::ostream &out) const
{
  out << "Physical quantities:" << endl;
  out << "L1: " << first_chamber_end << " m" << endl;
  out << "L2: " << sk_end << " m" << endl;
  out << "L3: " << quadrupole_start << " m" << endl;
  out << "L4: " << quadrupole_end << " m" << endl;
  out << "L5: " << second_chamber_end << " m" << endl;
}

ChamberQuantities::ChamberQuantities(const MassSpectrometer &ms, const Gas &gas) : kT(boltzmann * ms.T),
                                                                                   pressures(ms.pressures, kT),
                                                                                   gas_mean_free_paths(ms.pressures.unaryExpr([&](double P)
{ return mean_free_path(gas.radius, kT, P); })),
                                                                                   mobility_gas(kT / gas.mass), // thermal agitation
                                                                                   mobility_gas_inv(gas.mass / kT),
                                                                                   clens(ms.lengths),
                                                                                   E(-(
                                                                                     (ms.voltages(Eigen::seq(1, 4)) - ms.voltages(Eigen::seq(0, 3))) /
                                                                                     ms.lengths(Eigen::seq(0, 3))))
{
}

SubstanceQuantities::SubstanceQuantities(
  const MassSpectrometer &ms,
  const ChamberQuantities &chamber,
  const MassSpecSubstanceSingleInput &subs) : SubstanceQuantities(ms,
                                                                  chamber,
                                                                  subs.gas,
                                                                  subs.cluster_charge_sign,
                                                                  subs.m_ion,
                                                                  subs.R_cluster)
{
}

SubstanceQuantities::SubstanceQuantities(
  const MassSpectrometer &ms,
  const ChamberQuantities &chamber,
  const Gas &gas,
  const int cluster_charge_sign,
  const MSSubstanceTreeCluster &cluster) : SubstanceQuantities(ms,
                                                               chamber,
                                                               gas,
                                                               cluster_charge_sign,
                                                               cluster.m_ion,
                                                               cluster.R_cluster)
{
}

SubstanceQuantities::SubstanceQuantities(
  const MassSpectrometer &ms,
  const ChamberQuantities &chamber,
  const Gas &gas,
  const int cluster_charge_sign,
  const double m_ion,
  const double R_cluster) : reduced_mass(1. / (1. / m_ion + 1. / gas.mass)),
                            inertia(0.4 * m_ion * R_cluster * R_cluster),
                            acc(chamber.E * consts::eV * cluster_charge_sign / m_ion),
                            dts(chamber.pressures.histogram_dts(R_cluster + gas.radius, chamber.mobility_gas, chamber.mobility_gas_inv, DT_MULTIPLIER, ms.quadrupole))
{
  if (ms.quadrupole)
  {
    mathieu_factor = compute_mathieu_factor(m_ion, ms.quadrupole->r_quadrupole);
  }
}

void print_substance(LogHelper initial_trace, const MassSpecSubstanceSingleInput &subs, const SubstanceQuantities &subquants)
{
  initial_trace([&](auto &initial_trace)
  {
    initial_trace << "Cluster charge sign: " << subs.cluster_charge_sign << endl;
    for (size_t i = 0; i < subs.pathways.size(); i++)
    {
      initial_trace << "Pathway #" << (i + 1) << " fragmentation energy: " << subs.pathways[i].bonding_energy / boltzmann << " K (" << subs.pathways[i].bonding_energy * consts::kcal << " kcal/mol)" << endl;
    }
    initial_trace << "Cluster mass: " << subs.m_ion << " Kg" << endl;
    initial_trace << "Inertia momentum: " << subquants.inertia << " kg*m^2" << endl;
    initial_trace << "Cluster radius: " << subs.R_cluster << " m" << endl;
  });
}

void print_initial_trace(
  StreamingResultQueue &result_queue,
  LogHelper initial_trace,
  const MassSpectrometer &ms,
  const MassSpecSubstanceSingleInput &subs,
  const ChamberQuantities &chamber,
  const SubstanceQuantities &subquants)
{
  using namespace consts;

  result_queue.enqueue(LogMessage{LogMessage::probabilities, "#1_FragmentationEnergy 2_SurvivalProbability 3_Error\n"});
  result_queue.enqueue(LogMessage{LogMessage::fragments, "#1_Realization 2_Time 3_Position 4_FragmentationZone 5_PositionOfCollision 6_CollisionZone 7_VelocityAtCollision\n"});


  print_substance(initial_trace, subs, subquants);
  initial_trace([&](auto &initial_trace)
  {
    chamber.clens.info(initial_trace);
    initial_trace << "Pressure 1st chamber: " << chamber.pressures.P[0] << " Pa" << endl;
    initial_trace << "Pressure 2nd chamber: " << chamber.pressures.P[1] << " Pa" << endl;
    for (int i = 0; i < 4; i++)
    {
      initial_trace << "E" << (i + 1) << ": " << chamber.E[i] << " V/m, Acceleration: " << subquants.acc[i] << " m/s^2" << endl;
    }
    initial_trace << "Particle density 1st chamber: " << chamber.pressures.n[0] << " 1/m^3" << endl;
    initial_trace << "Particle density 2nd chamber: " << chamber.pressures.n[1] << " 1/m^3" << endl;
    initial_trace << "Cluster mean free path 1st chamber: " << mean_free_path(subs.R_cluster + subs.gas.radius, chamber.kT, chamber.pressures.P[0]) << " m" << endl;
    initial_trace << "Cluster mean free path 2nd chamber: " << mean_free_path(subs.R_cluster + subs.gas.radius, chamber.kT, chamber.pressures.P[1]) << " m" << endl;
    initial_trace << "Gas mean free path 1st chamber: " << mean_free_path(subs.gas.radius, chamber.kT, chamber.pressures.P[0]) << " m" << endl;
    initial_trace << "Gas mean free path 2nd chamber: " << mean_free_path(subs.gas.radius, chamber.kT, chamber.pressures.P[1]) << " m" << endl;
    initial_trace << "Gas density 1st chamber: " << chamber.pressures.n[0] << " 1/m^3" << endl;
    initial_trace << "Gas density 2nd chamber: " << chamber.pressures.n[1] << " 1/m^3" << endl;
    initial_trace << "Collision frequency 1st chamber (at v=0): " << coll_freq(chamber.pressures.n[0], chamber.mobility_gas, chamber.mobility_gas_inv, subs.R_cluster + subs.gas.radius, 0.0) << " 1/s" << endl;
    initial_trace << "Collision frequency 2nd chamber (at v=0): " << coll_freq(chamber.pressures.n[1], chamber.mobility_gas, chamber.mobility_gas_inv, subs.R_cluster + subs.gas.radius, 0.0) << " 1/s" << endl;
    initial_trace << "Standard deviation velocity_x: " << sqrt(boltzmann * ms.T / subs.m_ion) << " m/s" << endl;
    initial_trace << "R_tot: " << subs.R_cluster + subs.gas.radius << " m" << endl;
    initial_trace << "Time step t1: " << subquants.dts[0] << " s" << endl;
    initial_trace << "Time step t2: " << subquants.dts[1] << " s" << endl
                  << endl;
    initial_trace << "Simulating dynamics... (Fragments *, Intacts -)" << endl;
  });
}

template <typename GasCollSamplerT, typename VibEnergySamplerT>
SimulationResult apitof_mass_spec(
  const MassSpectrometer &ms,
  const MassSpecSubstanceSingleInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  GasCollSamplerT gas_coll_sampler,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation)
{
  VibEnergySamplerT vib_energy_sampler = VibEnergySamplerT(subs.density_cluster);
  const ChamberQuantities chamber(ms, subs.gas);
  const SubstanceQuantities subquants(ms, chamber, subs);

  LogHelper initial_trace = LogHelper{result_queue, LogMessage::initial_trace, &operation};
  if (logconf.level >= LOGLEVEL_MIN)
  {
    print_initial_trace(result_queue, initial_trace, ms, subs, chamber, subquants);
  }

  auto start = std::chrono::high_resolution_clock::now();

  Eigen::ArrayXi identity = Eigen::ArrayXi::Zero(n_counters - 1 + subs.pathways.size());

  // All firstprivate variables *should* be constant within the loop
  // Truly private variables are declared in the loop
  auto loop_start = std::chrono::high_resolution_clock::now();
  Eigen::ArrayXi counters = oneapi::tbb::parallel_reduce(
    oneapi::tbb::blocked_range<int>(0, N),
    identity,
    [&, gas_coll_sampler, vib_energy_sampler](const oneapi::tbb::blocked_range<int> &range, Eigen::ArrayXi counters)
  {
    auto local_vib_energy_sampler = vib_energy_sampler;
    auto local_gas_coll_sampler = gas_coll_sampler;
    for (int j = range.begin(); j != range.end() && operation.checkpoint(); ++j)
    {
      using consts::pi, consts::boltzmann;
      Eigen::ArrayXi realization_counters = Eigen::ArrayXi::Zero(identity.size());
      WarningHelper warn{realization_counters, result_queue, &operation};
      LogHelper fragments{result_queue, LogMessage::fragments, &operation};
      LogHelper final_position{result_queue, LogMessage::final_position, &operation};
      mt19937 gen = mt19937(root_seed ^ j);
      // Define uniform distribution from 0 to 1
      uniform_real_distribution<double> unif = uniform_real_distribution<>(0.0, 1.0);
      // Define normal (gaussian) distribution with 0 mean and 1 standard deviation
      normal_distribution<double> gauss = normal_distribution<>(0.0, 1.0);

      double t = 0.0;
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      int ncoll = 0;
      double coll_z = 0.0;

      // Draw initial random velocity from Maxwell-Boltzmann distribution
      Eigen::Vector3d v_cluster = init_vel(gen, gauss, subs.m_ion, chamber.kT);
      Eigen::Vector3d omega = init_ang_vel(gen, gauss, subs.m_ion, chamber.kT, subs.R_cluster);
      double vib_energy = init_vib_energy(gen, unif, chamber.kT, subs.density_cluster);

      while (z < chamber.clens.total_length && operation.checkpoint()) // single realization // TO BE CHANGED IN SECOND CHAMBER!!!!!!!!!!!
      {
        double v_cluster_norm = v_cluster.norm();
        double rot_energy = evaluate_rotational_energy(omega, subquants.inertia);
        double internal_energy = evaluate_internal_energy(vib_energy, rot_energy);

        int effective_pathway_index;
        double t_fragmentation;
        std::optional<ApiTofRateConstantOverflow> overflow_exception = std::nullopt;
        std::tie(effective_pathway_index, t_fragmentation, overflow_exception) = next_fragmentation_time_multi(gen, unif, subs.pathways, internal_energy, strict);

        double old_t = t;
        TimeNextCollOutcome outcome = time_next_coll_quadrupole(gen, unif, v_cluster, v_cluster_norm, chamber, subs.gas.radius + subs.R_cluster, subquants.dts, z, x, y, t_fragmentation, subquants.acc, t, subs.gas.mass, ms.skimmer, ms.mesh_skimmer, ms.quadrupole);

        if (logconf.level >= LOGLEVEL_NORMAL)
        {
          if (z < chamber.clens.first_chamber_end)
          {
            LogHelper tmp_evolution = LogHelper{result_queue, LogMessage::tmp_evolution, &operation};
            tmp_evolution([&](auto &tmp_evolution)
            {
              tmp_evolution << z << " " << t - old_t << " " << v_cluster_norm << " " << endl;
            });
          }
        }

        if (outcome == TimeNextCollOutcome::fragmentation)
        {
          realization_counters[Counter::n_fragmented_total + effective_pathway_index]++;
          if (logconf.level >= LOGLEVEL_NORMAL)
          {
            fragments([&](auto &fragments)
            {
              fragments << j + 1 << "\t" << t << "\t" << z << "\t" << zone(z, chamber.clens) << "\t" << coll_z << "\t" << zone(coll_z, chamber.clens) << endl;
            });
          }
          if (logconf.log_events && operation.should_continue())
          {
            result_queue.enqueue(FragmentationEvent{ParticleStateMsg{j, {x, y, z, t}, v_cluster, omega, rot_energy, vib_energy}, effective_pathway_index});
          }
          break;
        }
        else
        {
          if (!strict && overflow_exception.has_value())
          {
            // We didn't fragment, which means it's particularly bad that that the rate constant was out of range => rethrow
            throw *overflow_exception;
          }
          if (outcome == TimeNextCollOutcome::gas_collision)
          {
            // Keep track on number of collisions per realization
            ncoll++;
            if (ncoll > MAX_COLL)
            {
              throw ApiTofMaxCollisions(MAX_COLL, ncoll);
            }

            double v_gas;
            double temperature;
            double pressure;
            double density;
            update_physical_quantities(z, ms.skimmer, ms.mesh_skimmer, v_gas, temperature, pressure, density, chamber, ms.T);

            double effective_n;
            Eigen::Vector3d v_rel;
            double v_rel_norm;
            double effective_mobility_gas;
            double effective_mobility_gas_inv;
            std::tie(effective_n, v_rel, v_rel_norm, effective_mobility_gas, effective_mobility_gas_inv) = get_quantities_for_collision(z, chamber, subs.gas.mass, v_cluster, v_gas, pressure, temperature);
            double theta;
            double u_norm; // normal velocity of colliding gas molecule
            std::tie(theta, u_norm) = local_gas_coll_sampler.sample(gen, effective_n, v_rel_norm, effective_mobility_gas, effective_mobility_gas_inv, subs.R_cluster + subs.gas.radius, warn);

            // Evaluate the dissipated energy in the collision (energy that goes to vibrational modes)
            double vib_energy_new = local_vib_energy_sampler.sample(gen, boundary_vib_energy(vib_energy, subquants.reduced_mass, u_norm, v_rel_norm, theta));

            bool collision_accepted = eval_collision(gen, unif, chamber.gas_mean_free_paths[1], x, y, z, chamber.clens.total_length, ms.radius_pinhole, chamber.clens.quadrupole_end, v_rel, omega, u_norm, theta, subs.R_cluster, vib_energy_new, vib_energy, subs.m_ion, subs.gas.mass, temperature, LogHelper{result_queue, LogMessage::pinhole, &operation}, logconf.level);

            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(CollisionEvent{ParticleStateMsg{j, {x, y, z, t}, v_cluster, omega, rot_energy, vib_energy}, theta, u_norm, collision_accepted});
            }

            if (collision_accepted)
            {
              vib_energy = vib_energy_new;
              update_velocities(v_cluster, v_cluster_norm, v_rel, v_gas);
              // tmp << kin_energy << endl;

              rot_energy = evaluate_rotational_energy(omega, subquants.inertia);
              double rot_energy_old = rot_energy;
              std::tie(vib_energy, rot_energy) = redistribute_internal_energy(gen, local_vib_energy_sampler, vib_energy, rot_energy);
              update_rot_vel(omega, rot_energy_old, rot_energy);
            }
            else
            {
              realization_counters[Counter::counter_collision_rejections]++;
            }
          }
          else // outcome == TimeNextCollOutcome::escape
          {
            realization_counters[Counter::n_escaped_total]++;
            if (logconf.level >= LOGLEVEL_NORMAL)
            {
              final_position([&](auto &final_position)
              {
                final_position << x << "\t" << y << endl;
              });
            }
            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(EscapeEvent{ParticleStateMsg{j, {x, y, z, t}, v_cluster, omega, rot_energy, vib_energy}});
            }
          }
        }
      }

      if (operation.checkpoint())
      {
        realization_counters[Counter::ncoll_total] += ncoll;
        realization_counters[Counter::n_realizations]++;
        counters += realization_counters;
        result_queue.enqueue(PartialResult(realization_counters));
      }
    }
    return counters;
  },
    [](Eigen::ArrayXi left, const Eigen::ArrayXi &right)
  {
    return (left + right).eval();
  },
    operation.tbb_context());
  // End of parallel loop

  auto end = std::chrono::high_resolution_clock::now();

  RuntimeDuration loop_time = end - loop_start;
  RuntimeDuration total_time = end - start;

  return std::tuple(counters, loop_time, total_time);
}

void print_initial_trace(
  StreamingResultQueue &result_queue,
  LogHelper initial_trace,
  const MassSpectrometer &ms,
  const MassSpecSubstanceTreeInput &subs,
  const ChamberQuantities &chamber,
  const std::vector<SubstanceQuantities> &all_subquants)
{
  using namespace consts;

  (void)ms;
  (void)subs;
  (void)chamber;
  (void)all_subquants;
  result_queue.enqueue(LogMessage{LogMessage::probabilities, "#1_FragmentationEnergy 2_SurvivalProbability 3_Error\n"});
  result_queue.enqueue(LogMessage{LogMessage::fragments, "#1_Realization 2_Time 3_Position 4_FragmentationZone 5_PositionOfCollision 6_CollisionZone 7_VelocityAtCollision\n"});

  initial_trace([&](auto &initial_trace)
  {
    initial_trace << "TODO" << endl;
  });
}

void prepare_pathways_from_tree(
  std::vector<std::reference_wrapper<const MassSpecInputFragmentationPathway>> &pathways,
  const MassSpecSubstanceTreeInput &subs,
  int subnode_index)
{
  pathways.clear();
  for (auto pathway_index : subs.tree_nodes[subnode_index].pathway_indices)
  {
    pathways.push_back(std::cref(
      subs.pathway_payloads[subs.tree_pathways[pathway_index].payload_idx]));
  }
}

template <typename GasCollSamplerT, typename VibEnergySamplerT>
SimulationResult apitof_mass_spec(
  const MassSpectrometer &ms,
  const MassSpecSubstanceTreeInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  GasCollSamplerT gas_coll_sampler,
  // VibEnergySamplerT vib_energy_sampler,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation)
{
  const ChamberQuantities chamber(ms, subs.gas);
  std::vector<SubstanceQuantities> all_subquants;
  all_subquants.reserve(subs.cluster_payloads.size());
  for (const auto &cluster : subs.cluster_payloads)
  {
    all_subquants.push_back(SubstanceQuantities(ms, chamber, subs.gas, subs.cluster_charge_sign, cluster));
  }

  LogHelper initial_trace = LogHelper{result_queue, LogMessage::initial_trace, &operation};
  if (logconf.level >= LOGLEVEL_MIN)
  {
    print_initial_trace(result_queue, initial_trace, ms, subs, chamber, all_subquants);
  }

  auto start = std::chrono::high_resolution_clock::now();
  Eigen::ArrayXi identity = Eigen::ArrayXi::Zero(n_counters - 1 + subs.tree_pathways.size());

  // All firstprivate variables *should* be constant within the loop
  // Truly private variables are declared in the loop
  auto loop_start = std::chrono::high_resolution_clock::now();
  Eigen::ArrayXi counters = oneapi::tbb::parallel_reduce(
    oneapi::tbb::blocked_range<int>(0, N),
    identity,
    [&, gas_coll_sampler](const oneapi::tbb::blocked_range<int> &range, Eigen::ArrayXi counters)
  {
    auto local_gas_coll_sampler = gas_coll_sampler;
    for (int j = range.begin(); j != range.end() && operation.checkpoint(); ++j)
    {
      using consts::pi, consts::boltzmann;
      Eigen::ArrayXi realization_counters = Eigen::ArrayXi::Zero(identity.size());
      int subnode_index = 0;
      int subpayload_index = 0;
      const MSSubstanceTreeCluster &subpayload = subs.cluster_payloads[subpayload_index];
      std::vector<std::reference_wrapper<const MassSpecInputFragmentationPathway>> pathways;
      prepare_pathways_from_tree(pathways, subs, subnode_index);

      WarningHelper warn{realization_counters, result_queue, &operation};
      LogHelper fragments{result_queue, LogMessage::fragments, &operation};
      LogHelper final_position{result_queue, LogMessage::final_position, &operation};
      mt19937 gen = mt19937(root_seed ^ j);
      // Define uniform distribution from 0 to 1
      uniform_real_distribution<double> unif = uniform_real_distribution<>(0.0, 1.0);
      // Define normal (gaussian) distribution with 0 mean and 1 standard deviation
      normal_distribution<double> gauss = normal_distribution<>(0.0, 1.0);

      double t = 0.0;
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      int ncoll = 0;
      double coll_z = 0.0;

      // Draw initial random velocity from Maxwell-Boltzmann distribution
      Eigen::Vector3d v_cluster = init_vel(gen, gauss, subpayload.m_ion, chamber.kT);
      Eigen::Vector3d omega = init_ang_vel(gen, gauss, subpayload.m_ion, chamber.kT, subpayload.R_cluster);
      double vib_energy = init_vib_energy(gen, unif, chamber.kT, subpayload.density_cluster);
      auto vib_energy_sampler = std::unique_ptr<VibEnergySamplerT>(new VibEnergySamplerT(subpayload.density_cluster));
      int last_pathway_index = -1;

      while (z < chamber.clens.total_length && operation.checkpoint()) // single realization // TO BE CHANGED IN SECOND CHAMBER!!!!!!!!!!!
      {
        const MSSubstanceTreeCluster &subpayload = subs.cluster_payloads[subpayload_index];
        const SubstanceQuantities &subquants = all_subquants[subpayload_index];

        double v_cluster_norm = v_cluster.norm();
        double rot_energy = evaluate_rotational_energy(omega, subquants.inertia);
        double internal_energy = evaluate_internal_energy(vib_energy, rot_energy);

        int effective_pathway_index;
        double t_fragmentation;
        std::optional<ApiTofRateConstantOverflow> overflow_exception = std::nullopt;
        std::tie(effective_pathway_index, t_fragmentation, overflow_exception) = next_fragmentation_time_multi(gen, unif, pathways, internal_energy, strict);

        double old_t = t;
        TimeNextCollOutcome outcome = time_next_coll_quadrupole(gen, unif, v_cluster, v_cluster_norm, chamber, subs.gas.radius + subpayload.R_cluster, subquants.dts, z, x, y, t_fragmentation, subquants.acc, t, subs.gas.mass, ms.skimmer, ms.mesh_skimmer, ms.quadrupole);

        if (logconf.level >= LOGLEVEL_NORMAL)
        {
          if (z < chamber.clens.first_chamber_end)
          {
            LogHelper tmp_evolution = LogHelper{result_queue, LogMessage::tmp_evolution, &operation};
            tmp_evolution([&](auto &tmp_evolution)
            {
              tmp_evolution << z << " " << t - old_t << " " << v_cluster_norm << " " << endl;
            });
          }
        }

        if (outcome == TimeNextCollOutcome::fragmentation)
        {
          size_t tree_pathway_idx = subs.tree_nodes[subnode_index].pathway_indices[effective_pathway_index];
          std::optional<size_t> product_idx = subs.tree_pathways[tree_pathway_idx].product_idx;
          last_pathway_index = tree_pathway_idx;
          if (logconf.level >= LOGLEVEL_NORMAL)
          {
            fragments([&](auto &fragments)
            {
              fragments << j + 1 << "\t" << t << "\t" << z << "\t" << zone(z, chamber.clens) << "\t" << coll_z << "\t" << zone(coll_z, chamber.clens) << endl;
            });
          }
          if (logconf.log_events && operation.should_continue())
          {
            int next_particle_index = product_idx ? *product_idx : -1;
            result_queue.enqueue(FragmentationEvent{ParticleStateMsg{j, {x, y, z, t}, v_cluster, omega, rot_energy, vib_energy, subnode_index}, next_particle_index, next_particle_index});
          }
          if (product_idx)
          {
            subnode_index = *product_idx;
            subpayload_index = subs.tree_nodes[subnode_index].payload_idx;
            prepare_pathways_from_tree(pathways, subs, subnode_index);

            // Assumption: All energy was used up in the fragmentation so we zero everything out.
            v_cluster = Eigen::Vector3d::Zero();
            omega = Eigen::Vector3d::Zero();
            vib_energy = 0;
            vib_energy_sampler = std::unique_ptr<VibEnergySamplerT>(new VibEnergySamplerT(subs.cluster_payloads[subpayload_index].density_cluster));
          }
          else
          {
            break;
          }
        }
        else
        {
          if (!strict && overflow_exception.has_value())
          {
            // We didn't fragment, which means it's particularly bad that that the rate constant was out of range => rethrow
            throw *overflow_exception;
          }
          if (outcome == TimeNextCollOutcome::gas_collision)
          {
            // Keep track on number of collisions per realization
            ncoll++;
            if (ncoll > MAX_COLL)
            {
              throw ApiTofMaxCollisions(MAX_COLL, ncoll);
            }

            double v_gas;
            double temperature;
            double pressure;
            double density;
            update_physical_quantities(z, ms.skimmer, ms.mesh_skimmer, v_gas, temperature, pressure, density, chamber, ms.T);

            double effective_n;
            Eigen::Vector3d v_rel;
            double v_rel_norm;
            double effective_mobility_gas;
            double effective_mobility_gas_inv;
            std::tie(effective_n, v_rel, v_rel_norm, effective_mobility_gas, effective_mobility_gas_inv) = get_quantities_for_collision(z, chamber, subs.gas.mass, v_cluster, v_gas, pressure, temperature);
            double theta;
            double u_norm; // normal velocity of colliding gas molecule
            std::tie(theta, u_norm) = local_gas_coll_sampler.sample(gen, effective_n, v_rel_norm, effective_mobility_gas, effective_mobility_gas_inv, subpayload.R_cluster + subs.gas.radius, warn);

            // Evaluate the dissipated energy in the collision (energy that goes to vibrational modes)
            double vib_energy_new = vib_energy_sampler->sample(gen, boundary_vib_energy(vib_energy, subquants.reduced_mass, u_norm, v_rel_norm, theta));

            bool collision_accepted = eval_collision(gen, unif, chamber.gas_mean_free_paths[1], x, y, z, chamber.clens.total_length, ms.radius_pinhole, chamber.clens.quadrupole_end, v_rel, omega, u_norm, theta, subpayload.R_cluster, vib_energy_new, vib_energy, subpayload.m_ion, subs.gas.mass, temperature, LogHelper{result_queue, LogMessage::pinhole, &operation}, logconf.level);

            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(CollisionEvent{ParticleStateMsg{j, {x, y, z, t}, v_cluster, omega, rot_energy, vib_energy, subnode_index}, theta, u_norm, collision_accepted});
            }

            if (collision_accepted)
            {
              vib_energy = vib_energy_new;
              update_velocities(v_cluster, v_cluster_norm, v_rel, v_gas);
              // tmp << kin_energy << endl;

              rot_energy = evaluate_rotational_energy(omega, subquants.inertia);
              double rot_energy_old = rot_energy;
              std::tie(vib_energy, rot_energy) = redistribute_internal_energy(gen, *vib_energy_sampler, vib_energy, rot_energy);
              update_rot_vel(omega, rot_energy_old, rot_energy);
            }
            else
            {
              realization_counters[Counter::counter_collision_rejections]++;
            }
          }
          else // outcome == TimeNextCollOutcome::escape
          {
            if (logconf.level >= LOGLEVEL_NORMAL)
            {
              final_position([&](auto &final_position)
              {
                final_position << x << "\t" << y << endl;
              });
            }
            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(EscapeEvent{ParticleStateMsg{j, {x, y, z, t}, v_cluster, omega, rot_energy, vib_energy, subnode_index}});
            }
          }
        }
      }

      if (operation.checkpoint())
      {
        realization_counters[Counter::ncoll_total] += ncoll;
        realization_counters[Counter::n_realizations]++;
        realization_counters[Counter::n_fragmented_total + last_pathway_index]++;
        counters += realization_counters;
        result_queue.enqueue(PartialResult(realization_counters));
      }
    }
    return counters;
  },
    [](Eigen::ArrayXi left, const Eigen::ArrayXi &right)
  {
    return (left + right).eval();
  },
    operation.tbb_context());
  // End of parallel loop

  auto end = std::chrono::high_resolution_clock::now();

  RuntimeDuration loop_time = end - loop_start;
  RuntimeDuration total_time = end - start;

  return std::tuple(counters, loop_time, total_time);
}

double evaluate_error(int n, int k)
{
  return sqrt((6.0 * k * k - k * (6.0 + k) * n + (2.0 + k) * n * n) / (n * n * (3.0 + n) * (2.0 + n)));
}

// Compute normalized cross product of vectors
Eigen::Vector3d cross_norm(const Eigen::Vector3d &in1, const Eigen::Vector3d &in2)
{
  Eigen::Vector3d out = in1.cross(in2);
  double norm = out.norm();
  if (norm > 0)
  {
    return out / norm;
  }
  else
  {
    throw ApiTofUnexpectedNumericalError("Zero result in evaluating the cross product");
  }
}


double particle_density(double pressure, double kT)
{
  using namespace consts;
  return pressure / kT;
}


// Distribution of 1-dim Maxwell velocity
template <typename GenT>
double onedimMaxwell(GenT &gen, normal_distribution<double> &gauss, double m, double kT)
{
  return sqrt(kT / m) * gauss(gen);
}


// Distribution of 2-dim Maxwell velocity
template <typename GenT>
double twodimMaxwell(GenT &gen, uniform_real_distribution<double> &unif, double m, double kT)
{
  double r = 0.0;
  while (r == 0.0)
  {
    r = unif(gen);
  }
  return sqrt(-2.0 * kT * log(r) / m);
}


// Distribution of 1-dim Maxwell angular velocity
template <typename GenT>
double onedimMaxwell_angular(GenT &gen, normal_distribution<double> &gauss, double m, double R, double kT)
{
  return sqrt(2.5 * kT / (m * R * R)) * gauss(gen);
}


// Inizialize the cluster velocity
template <typename GenT>
Eigen::Vector3d init_vel(GenT &gen, normal_distribution<double> &gauss, double m, double kT)
{
  return Eigen::Vector3d(
    onedimMaxwell(gen, gauss, m, kT),
    onedimMaxwell(gen, gauss, m, kT),
    onedimMaxwell(gen, gauss, m, kT));
}

// Inizialize the cluster angular velocity
template <typename GenT>
Eigen::Vector3d init_ang_vel(GenT &gen, normal_distribution<double> &gauss, double m, double kT, double R)
{
  return Eigen::Vector3d(
    onedimMaxwell_angular(gen, gauss, m, R, kT),
    onedimMaxwell_angular(gen, gauss, m, R, kT),
    onedimMaxwell_angular(gen, gauss, m, R, kT));
}


double evaluate_rate_const(const Histogram &rate_const, double energy)
{
  using namespace consts;
  auto result = rate_const.get_lerp(energy);
  if (std::holds_alternative<double>(result))
  {
    return std::get<double>(result);
  }
  else
  {
    if (std::get<Histogram::OutOfBounds>(result) == Histogram::OutOfBounds::underflow)
    {
      throw ApiTofUnexpectedNumericalError([&energy](auto &msg)
      {
        msg << "Rate constant evaluation failed underflow: delta_energy= " << energy << endl;
      });
    }
    else
    {
      throw ApiTofRateConstantOverflow(rate_const.x_max / boltzmann, energy / boltzmann);
    }
  }
}


void update_skimmer_quantities(const SkimmerData &skimmer, double z, double first_chamber_end, double mesh_skimmer, double &v_gas, double &temp, double &pressure)
{
  int m;
  double coeff1;
  double coeff2;
  double position;
  position = z - first_chamber_end;
  m = int(position / mesh_skimmer);
  if (m == skimmer.rows() - 1)
  {
    v_gas = skimmer(m, VEL_SKIMMER);
    temp = skimmer(m, TEMP_SKIMMER);
    pressure = skimmer(m, PRESSURE_SKIMMER);
  }
  else
  {
    coeff1 = (position - m * mesh_skimmer) / mesh_skimmer;
    coeff2 = 1.0 - coeff1;
    v_gas = coeff2 * skimmer(m, VEL_SKIMMER) + coeff1 * skimmer(m + 1, VEL_SKIMMER);
    temp = coeff2 * skimmer(m, TEMP_SKIMMER) + coeff1 * skimmer(m + 1, TEMP_SKIMMER);
    pressure = coeff2 * skimmer(m, PRESSURE_SKIMMER) + coeff1 * skimmer(m + 1, PRESSURE_SKIMMER);
  }
  // density=coeff2*density_skimmer[m]+coeff1*density_skimmer[m+1];
}

std::tuple<double, Eigen::Vector3d, double, double, double> get_quantities_for_collision(double z, const ChamberQuantities &chamber, double m_gas, const Eigen::Vector3d &v_cluster, double v_gas, double pressure, double temperature)
{
  using consts::boltzmann;
  double n;
  double v_rel_norm;
  Eigen::Vector3d v_rel = v_cluster;
  double mobility_gas = chamber.mobility_gas;
  double mobility_gas_inv = chamber.mobility_gas_inv;
  if (z < chamber.clens.first_chamber_end)
  {
    n = chamber.pressures.n[0];
  }
  else if (z < chamber.clens.second_chamber_end)
  {
    v_rel[2] = v_rel[2] - v_gas;
    double kT = boltzmann * temperature;
    mobility_gas = kT / m_gas;
    mobility_gas_inv = m_gas / kT;
    n = particle_density(pressure, kT);
  }
  else
  {
    n = chamber.pressures.n[1];
  }
  v_rel_norm = v_rel.norm();
  return std::make_tuple(n, v_rel, v_rel_norm, mobility_gas, mobility_gas_inv);
}

void update_physical_quantities(double z, const SkimmerData &skimmer, double mesh_skimmer, double &v_gas, double &temperature, double &pressure, double &density, const ChamberQuantities &chamber, double T)
{
  int m;
  double coeff1;
  double coeff2;
  double position;

  if (z < chamber.clens.first_chamber_end)
  {
    density = chamber.pressures.n[0];
    pressure = chamber.pressures.P[0];
    temperature = T;
    v_gas = 0;
  }
  else if (z < chamber.clens.sk_end)
  {
    position = z - chamber.clens.first_chamber_end;
    m = int(position / mesh_skimmer);
    if (m == skimmer.rows() - 1)
    {
      v_gas = skimmer(m, VEL_SKIMMER);
      temperature = skimmer(m, TEMP_SKIMMER);
      pressure = skimmer(m, PRESSURE_SKIMMER);
    }
    else
    {
      coeff1 = (position - m * mesh_skimmer) / mesh_skimmer;
      coeff2 = 1.0 - coeff1;
      v_gas = coeff2 * skimmer(m, VEL_SKIMMER) + coeff1 * skimmer(m + 1, VEL_SKIMMER);
      temperature = coeff2 * skimmer(m, TEMP_SKIMMER) + coeff1 * skimmer(m + 1, TEMP_SKIMMER);
      pressure = coeff2 * skimmer(m, PRESSURE_SKIMMER) + coeff1 * skimmer(m + 1, PRESSURE_SKIMMER);
    }
  }
  else
  {
    density = chamber.pressures.n[1];
    pressure = chamber.pressures.P[1];
    temperature = T;
    v_gas = 0;
  }
}

// Draw initial vibrational energy
template <typename GenT>
double init_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double kT, const Histogram &density_cluster)
{
  double sum1 = 0.0;
  double sum2 = 0.0;
  double r = unif(gen);
  int m;

  for (m = 0; m < density_cluster.length(); m++)
  {
    sum1 += density_cluster.y[m] * exp(-density_cluster.x[m] / kT);
  }

  if (sum1 <= 0.0)
  {
    throw ApiTofUnexpectedNumericalError([&kT](auto &msg)
    {
      msg << "Boltzmann weights of the cluster density of states sum to zero for kT = " << kT
          << " J. Are the histogram energies scaled to Joules?" << endl;
    });
  }

  m = 0;
  while (sum2 < r && m < density_cluster.length())
  {
    sum2 += density_cluster.y[m] * exp(-density_cluster.x[m] / kT) / sum1;
    m++;
  }
  return density_cluster.x[m - 1];
}


// Evaluate time to next collision
template <typename GenT>
TimeNextCollOutcome time_next_coll_quadrupole(GenT &gen, uniform_real_distribution<double> &unif, Eigen::Vector3d &v_cluster, double &v_cluster_norm, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double &z, double &x, double &y, double &t_fragmentation, const Eigen::Array4d &acc, double &t, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole)
{
  using namespace consts;
  double integral = 0.0;
  double P = 1.0;
  double c1;
  double c2;
  double v1;
  double v_cluster_norm_xy = v_cluster[0] * v_cluster[0] + v_cluster[1] * v_cluster[1];
  double r = unif(gen);
  double mobility_gas_skimmer;
  double mobility_gas_inv_skimmer;
  double T_skimmer;
  double kT_skimmer;
  double P_skimmer;
  double n_skimmer = NAN;
  double v_gas = NAN;
  double v_rel_norm;
  double v1x;
  double v1y;
  double accx;
  double accy;
  double delta_t = 0.0;
  v_cluster_norm = v_cluster.norm();

  if (z < chamber.clens.first_chamber_end) // In first chamber
  {
    c1 = coll_freq(chamber.pressures.n[0], chamber.mobility_gas, chamber.mobility_gas_inv, R, v_cluster_norm);
  }
  else if (z > chamber.clens.sk_end) // In the second chamber
  {
    c1 = coll_freq(chamber.pressures.n[1], chamber.mobility_gas, chamber.mobility_gas_inv, R, v_cluster_norm);
  }
  else // In the skimmer
  {
    update_skimmer_quantities(skimmer, z, chamber.clens.first_chamber_end, mesh_skimmer, v_gas, T_skimmer, P_skimmer);
    kT_skimmer = boltzmann * T_skimmer;
    mobility_gas_skimmer = boltzmann * T_skimmer / m_gas;
    mobility_gas_inv_skimmer = 1.0 / mobility_gas_skimmer;
    n_skimmer = particle_density(P_skimmer, kT_skimmer);
    v_rel_norm = sqrt(v_cluster_norm_xy + pow(v_cluster[2] - v_gas, 2));
    c1 = coll_freq(n_skimmer, mobility_gas_skimmer, mobility_gas_inv_skimmer, R, v_rel_norm);
  }

  // tmp_evolution << z << " " << c1 << endl;
  // if(z<first_chamber_end) tmp_evolution << z << " " << c1 << endl;

  while (true)
  {
    if (z >= chamber.clens.second_chamber_end)
    {
      return TimeNextCollOutcome::escape;
    }
    if (delta_t >= t_fragmentation)
    {
      return TimeNextCollOutcome::fragmentation;
    }
    if (r >= P)
    {
      return TimeNextCollOutcome::gas_collision;
    }
    v1 = v_cluster[2];
    v1x = v_cluster[0];
    v1y = v_cluster[1];

    if (z < chamber.clens.first_chamber_end)
    {
      v_cluster[2] += acc[0] * dts[0];
    }

    else if (z >= chamber.clens.sk_end and z < chamber.clens.quadrupole_start)
    {
      v_cluster[2] += acc[1] * dts[1];
    }

    else if (z >= chamber.clens.quadrupole_start and z < chamber.clens.quadrupole_end)
    {
      if (quadrupole)
      {
        accx = quadrupole->mathieu_factor * (-quadrupole->dc_field + quadrupole->ac_field * cos(quadrupole->angular_velocity * t)) * (x + v_cluster[0] * dts[1] / 2.0);
        accy = quadrupole->mathieu_factor * (quadrupole->dc_field - quadrupole->ac_field * cos(quadrupole->angular_velocity * t)) * (y + v_cluster[1] * dts[1] / 2.0);
        v_cluster[0] += accx * dts[1];
        v_cluster[1] += accy * dts[1];
      }
      v_cluster[2] += acc[2] * dts[1];
    }

    else if (z >= chamber.clens.quadrupole_end)
    {
      v_cluster[2] += acc[3] * dts[1];
    }

    // XXX: This takes a bunch of time
    v_cluster_norm = v_cluster.norm();

    if (z < chamber.clens.first_chamber_end) // Dynamics in the 1st chamber
    {
      c2 = coll_freq(chamber.pressures.P[0], chamber.mobility_gas, chamber.mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dts[0] / 2.0;
      P = exp(-integral);
      delta_t += dts[0];
      x += v1x * dts[0];
      y += v1y * dts[0];
      z += (v1 + v_cluster[2]) * dts[0] / 2.0;
      t += dts[0];
    }

    else if (z > chamber.clens.sk_end and z < chamber.clens.quadrupole_start) // Dynamics in the 2nd chamber
    {
      c2 = coll_freq(chamber.pressures.P[1], chamber.mobility_gas, chamber.mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dts[1] / 2.0;
      P = exp(-integral);
      delta_t += dts[1];
      x += v1x * dts[1];
      y += v1y * dts[1];
      z += (v1 + v_cluster[2]) * dts[1] / 2.0;
      t += dts[1];
    }

    else if (z >= chamber.clens.quadrupole_start and z < chamber.clens.quadrupole_end) // Dynamics in the 2nd chamber
    {
      c2 = coll_freq(chamber.pressures.P[1], chamber.mobility_gas, chamber.mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dts[1] / 2.0;
      P = exp(-integral);
      delta_t += dts[1];
      x += (v1x + v_cluster[0]) * dts[1] / 2.0;
      y += (v1y + v_cluster[1]) * dts[1] / 2.0;
      z += (v1 + v_cluster[2]) * dts[1] / 2.0;
      t += dts[1];
    }
    else if (z >= chamber.clens.quadrupole_end) // Dynamics in the 2nd chamber
    {
      c2 = coll_freq(chamber.pressures.P[1], chamber.mobility_gas, chamber.mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dts[1] / 2.0;
      P = exp(-integral);
      delta_t += dts[1];
      x += v1x * dts[1];
      y += v1y * dts[1];
      z += (v1 + v_cluster[2]) * dts[1] / 2.0;
      t += dts[1];
    }

    else // Dynamics in the skimmer
    {
      double dt;
      update_skimmer_quantities(skimmer, z, chamber.clens.first_chamber_end, mesh_skimmer, v_gas, T_skimmer, P_skimmer);
      kT_skimmer = boltzmann * T_skimmer;
      mobility_gas_skimmer = boltzmann * T_skimmer / m_gas;
      mobility_gas_inv_skimmer = 1.0 / mobility_gas_skimmer;
      n_skimmer = particle_density(P_skimmer, kT_skimmer);
      v_rel_norm = sqrt(v_cluster_norm_xy + pow(v_cluster[2] - v_gas, 2));
      c2 = coll_freq(n_skimmer, mobility_gas_skimmer, mobility_gas_inv_skimmer, R, v_rel_norm);
      dt = 1.0e-3 / c2;
      integral += (c1 + c2) * dt / 2.0;
      P = exp(-integral);
      delta_t += dt;
      x += v1x * dt;
      y += v1y * dt;
      z += v1 * dt;
      t += dt;
    }
    c1 = c2;

    // if(z>quadrupole_start and z<quadrupole_end) tmp_evolution << t << "\t" << x << "\t" << y << "\t" << z << "\t" << v_cluster[0] << "\t" << v_cluster[1] << "\t" << v_cluster[2] << endl;
    // positionz << t << " " << z << " " << c1 << " " << c2 << " " << v1 << " " << v_cluster[2] << " " << P << " " << r << endl;
  }
  // if(z<first_chamber_end) tmp_evolution << z << " " << c1 << " " << n_skimmer << " " << mobility_gas_skimmer << " " << mobility_gas_inv_skimmer << " " << R << " " << v_rel_norm << endl;
}

double boundary_vib_energy(double vib_energy_old, double reduced_mass, double u_norm, double v_cluster_norm, double theta)
{
  double relative_speed = u_norm + v_cluster_norm * cos(theta);
  return vib_energy_old + reduced_mass * 0.5 * relative_speed * relative_speed;
}

// Redistribution of internal energy (between vibrational and rotational modes)
template <typename GenT, typename VibEnergySamplerT>
std::tuple<double, double> redistribute_internal_energy(GenT &gen, VibEnergySamplerT &sampler, double vib_energy, double rot_energy)
{
  double E = vib_energy + rot_energy;

  vib_energy = sampler.sample(gen, E);
  rot_energy = E - vib_energy;
  return std::make_tuple(vib_energy, rot_energy);
}


// Update angular velocity after redistribution of vibrational and rotational energy
void update_rot_vel(Eigen::Vector3d &omega, double rot_energy_old, double rot_energy)
{
  omega[0] = omega[0] * sqrt(rot_energy / rot_energy_old);
  omega[1] = omega[1] * sqrt(rot_energy / rot_energy_old);
  omega[2] = omega[2] * sqrt(rot_energy / rot_energy_old);
}

// Evaluate internal energy (rotational+vibrational)
double evaluate_internal_energy(double vib_energy, double rot_energy)
{
  return rot_energy + vib_energy;
}

// Evaluate rotational energy
double evaluate_rotational_energy(const Eigen::Vector3d &omega, double inertia)
{
  return 0.5 * inertia * omega.squaredNorm();
}

// Mean free path
double mean_free_path(double R, double kT, double pressure)
{
  using consts::pi;
  return kT / (sqrt(2.0) * pi * 4.0 * R * R * pressure);
}


void evaluate_relative_velocity(double z, const Eigen::Vector3d &v_cluster, double &v_rel_norm, double v_gas, Eigen::Vector3d &v_rel, double first_chamber_end, double sk_end)
{
  if (z > first_chamber_end and z < sk_end)
  {
    v_rel[0] = v_cluster[0];
    v_rel[1] = v_cluster[1];
    v_rel[2] = v_cluster[2] - v_gas;
  }
  else
  {
    v_rel[0] = v_cluster[0];
    v_rel[1] = v_cluster[1];
    v_rel[2] = v_cluster[2];
  }
  v_rel_norm = v_rel.norm();
}

void update_velocities(Eigen::Vector3d &v_cluster, double &v_cluster_norm, const Eigen::Vector3d &v_rel, double v_gas)
{
  v_cluster[0] = v_rel[0];
  v_cluster[1] = v_rel[1];
  v_cluster[2] = v_rel[2] + v_gas;
  v_cluster_norm = v_cluster.norm();
}


// Evaluate the velocities after collision in the rotated reference system
void eval_velocities(Eigen::Vector3d &v, Eigen::Vector3d &omega, const Eigen::Vector2d &u, double vib_energy, double vib_energy_old, double M, double m, double R_cluster)
{
  double vx;
  double vy;
  double vz;
  double omegax;
  double omegay;
  double m_reduced = m / (m + M);
  double M_reduced = M / (m + M);
  double radicand;
  double ratio_masses = M / m;

  // cout << v[0] << endl<<endl;
  // cout << v[1] << endl<<endl;
  // cout << v[2] << endl<<endl;


  // cout << v[0]<< " " << v[1]<< " " << v[2]<<endl<<endl;
  vy = (4.0 * omega[0] * R_cluster + 4.0 * u[1] + (3.0 + 2.0 * ratio_masses) * v[1]) / (7.0 + 2.0 * ratio_masses);

  // cout << u[1]-v[1] << endl<<endl;
  //  In case of anelastic collision, part of the energy (vib_energy) is absorbed by the cluster into vibrational modes, and the y-velocity becomes
  radicand = m_reduced * m_reduced * pow(u[0] - v[2], 2) - 2.0 * (vib_energy - vib_energy_old) * m_reduced / M;
  // cout << radicand << endl;
  if (radicand < 0)
  {
    throw ApiTofUnexpectedNumericalError([&](auto &msg)
    {
      msg << "sqrt of negative number in evaluation of velocities after collision! radicand: " << radicand << endl;
    });
  }
  vz = m_reduced * u[0] + M_reduced * v[2] - sqrt(radicand);

  vx = (-4.0 * omega[1] * R_cluster + (3.0 + 2.0 * ratio_masses) * v[0]) / (7.0 + 2.0 * ratio_masses);
  omegay = ((2.0 * ratio_masses - 3.0) * omega[1] - 10.0 * (v[0] / R_cluster)) / (7.0 + 2.0 * ratio_masses);
  omegax = ((-3.0 + 2.0 * ratio_masses) * omega[0] + (10.0 * (v[1] - u[1])) / R_cluster) / (7.0 + 2.0 * ratio_masses);

  v[0] = vx;
  v[1] = vy;
  v[2] = vz;
  omega[0] = omegax;
  omega[1] = omegay;
  // omega[2]=omegaz;
  // cout << v[0]<< " " << v[1]<< " " << v[2]<<endl<<endl;
}


// Change of coordinates routine
void change_coord(const Eigen::Vector3d &v_cluster, double theta, double phi, double alpha, Eigen::Vector3d &x3, Eigen::Vector3d &y3, Eigen::Vector3d &z3)
{
  using consts::pi;
  auto x = Eigen::Vector3d(1.0, 0.0, 0.0);
  auto y = Eigen::Vector3d(0.0, 1.0, 0.0);
  Eigen::Vector3d x1;
  Eigen::Vector3d y1;
  auto z1 = Eigen::Vector3d(0.0, 0.0, 1.0);
  Eigen::Vector3d x2;
  Eigen::Vector3d y2;
  Eigen::Vector3d z2;
  Eigen::Vector3d foo;

  // check if v_cluster is null
  double v_cluster_norm = v_cluster.norm();
  if (v_cluster_norm > 0)
  {
    z1 = v_cluster / v_cluster_norm;
  }

  // build reference system with v_cluster aligned to z1 versor
  foo = z1.cross(x);
  if (foo.norm() != 0.0)
  {
    y1 = cross_norm(z1, x);
    x1 = cross_norm(y1, z1);
  }
  else
  {
    x1 = cross_norm(y, z1);
    y1 = cross_norm(z1, x1);
  }

  // build reference of system centered in point of collision (x2,y2,z2)
  if (theta > 0 and theta < pi)
  {
    for (int i = 0; i < 3; i++)
    {
      z2[i] = sin(theta) * cos(phi) * x1[i] + sin(theta) * sin(phi) * y1[i] + cos(theta) * z1[i];
    }
    x2 = cross_norm(z2, z1);
    y2 = cross_norm(z2, x2);
  }
  else if (theta == 0.0)
  {
    for (int i = 0; i < 3; i++)
    {
      z2[i] = z1[i];
    }
    y2 = cross_norm(z2, x1);
    x2 = cross_norm(y2, z2);
  }
  else if (theta == pi)
  {
    for (int i = 0; i < 3; i++)
    {
      z2[i] = -z1[i];
    }
    y2 = cross_norm(z2, x1);
    x2 = cross_norm(y2, z2);
  }
  else
  {
    throw ApiTofUnexpectedNumericalError([&](auto &msg)
    {
      msg << "ERROR in defining reference system at theta: " << theta << endl;
    });
  }

  // find versor of tangential velocity
  for (int i = 0; i < 3; i++)
  {
    z3[i] = z2[i];
    x3[i] = cos(alpha) * x2[i] + sin(alpha) * y2[i];
    y3[i] = -sin(alpha) * x2[i] + cos(alpha) * y2[i];
  }
}

// Evaluate solid angle using Stokes theorem (1d integral) (REF: Eq 32, Conway, Nuclear Instruments and Methods in Physics Research A 614, 2010)
double eval_solid_angle_stokes(double R, double L, double xx, double yy, double z)
{
  using consts::pi;
  int N = 1000;
  double dphi;
  double sum = 0.0;
  double integrand;
  double c;
  double phi;
  double xphi;
  double yphi;
  double zz = L - z;

  dphi = 2.0 * pi / N;

  phi = 0.0;
  xphi = R * xx * cos(phi);
  yphi = R * yy * sin(phi);
  c = R * R + xx * xx + yy * yy - 2.0 * xphi - 2.0 * yphi;
  integrand = (1.0 - zz / sqrt(c + zz * zz)) * (R * R - xphi - yphi) / c;
  sum += 0.5 * integrand;

  for (int i = 1; i < N; i++)
  {
    phi = dphi * i;
    xphi = R * xx * cos(phi);
    yphi = R * yy * sin(phi);
    c = R * R + xx * xx + yy * yy - 2.0 * xphi - 2.0 * yphi;
    integrand = (1.0 - zz / sqrt(c + zz * zz)) * (R * R - xphi - yphi) / c;
    sum += integrand;
  }

  phi = 2.0 * pi;
  xphi = R * xx * cos(phi);
  yphi = R * yy * sin(phi);
  c = R * R + xx * xx + yy * yy - 2.0 * xphi - 2.0 * yphi;
  integrand = (1.0 - zz / sqrt(c + zz * zz)) * (R * R - xphi - yphi) / c;
  sum += 0.5 * integrand;

  return sum * dphi;
}

//
template <typename GenT>
bool eval_collision(GenT &gen, uniform_real_distribution<double> &unif, double gas_mean_free_path, double x, double y, double z, double L, std::optional<double> pinhole, double quadrupole_end, Eigen::Vector3d &v_cluster, Eigen::Vector3d &omega, double u_norm, double theta, double R_cluster, double vib_energy, double vib_energy_old, double m_ion, double m_gas, double temperature, LogHelper pinhole_logger, int loglevel)
{
  using namespace consts;
  Eigen::Vector3d x3;
  Eigen::Vector3d y3;
  Eigen::Vector3d z3;
  Eigen::Vector3d v2;
  Eigen::Vector3d omega2;
  double phi = 2.0 * pi * unif(gen);
  double alpha = 2.0 * pi * unif(gen);
  double kT = boltzmann * temperature;
  Eigen::Vector2d u;
  Eigen::Vector3d velocity_gas;
  Eigen::Vector2d target;
  bool inside_target = false;
  double prob_coll = 1.0;
  double distance;

  bool collision_accepted = true;
  change_coord(v_cluster, theta, phi, alpha, x3, y3, z3);


  v2[0] = v_cluster.dot(x3);
  v2[1] = v_cluster.dot(y3);
  v2[2] = v_cluster.dot(z3);


  omega2[0] = omega.dot(x3);
  omega2[1] = omega.dot(y3);
  omega2[2] = omega.dot(z3);


  // Normal component of air molecule velocity
  u[0] = -u_norm;
  // Tangential component of air molecule velocity
  u[1] = twodimMaxwell(gen, unif, m_gas, kT);
  // cout << kT << endl;
  if (u[0] > v2[2])
  {
    throw ApiTofUnexpectedNumericalError([&](auto &msg)
    {
      msg << "ERROR: relative velocities prevent collision! " << u[0] << " > " << v2[2] << endl;
    });
  }

  if (pinhole)
  {
    double radius_pinhole = *pinhole;
    // Check if the gas particle comes from the pinhole
    if (z > quadrupole_end and z < L)
    {
      // Evaluate gas molecule velocity
      for (int i = 0; i < 3; i++)
      {
        velocity_gas[i] = u[1] * y3[i] + u[0] * z3[i];
      }
      // Check if the gas molecule comes from the pinhole
      if (velocity_gas[2] < 0.0)
      {
        target[0] = velocity_gas[0] * (L - z) / velocity_gas[2] + x;
        target[1] = velocity_gas[1] * (L - z) / velocity_gas[2] + y;
        if (target[0] * target[0] + target[1] * target[1] < radius_pinhole * radius_pinhole)
          inside_target = true;
      }
      else
      {
        inside_target = false;
      }
      if (loglevel >= LOGLEVEL_MIN)
      {
        pinhole_logger([&](auto &pinhole_logger)
        {
          pinhole_logger << x << " " << y << " " << z << " " << velocity_gas[0] << " " << velocity_gas[1] << " " << velocity_gas[2] << " " << inside_target << endl;
        });
      }
      if (inside_target)
      {
        double r = unif(gen);
        distance = sqrt(x * x + y * y + (L - z) * (L - z));
        // Probability to accept the collision prob_coll
        prob_coll = (1.0 - exp(-distance / gas_mean_free_path)) * (1.0 - eval_solid_angle_stokes(radius_pinhole, L, x, y, z) / (2.0 * pi));

        // prob_coll=1.0-eval_solid_angle(radius_pinhole, L, x, y, z)/(2.0*pi);
        // prob_coll=1.0;
        // prob_coll=0.0;
        if (r > prob_coll)
        {
          collision_accepted = false;
          // cout << "Rejected collision close to pinhole" << endl;
        }
      }
    }
  }

  if (collision_accepted) // Normal procedure
  {
    eval_velocities(v2, omega2, u, vib_energy, vib_energy_old, m_ion, m_gas, R_cluster);
    // Express new velocities in lab reference system
    for (int i = 0; i < 3; i++)
    {
      v_cluster[i] = v2[0] * x3[i] + v2[1] * y3[i] + v2[2] * z3[i];
      omega[i] = omega2[0] * x3[i] + omega2[1] * y3[i] + omega2[2] * z3[i];
    }
  }
  return collision_accepted;
}

int zone(double z, const CumulativeLengths &clens)
{
  if (z < clens.first_chamber_end)
    return 1;
  else if (z < clens.sk_end)
    return 2;
  else if (z < clens.quadrupole_start)
    return 3;
  else if (z < clens.quadrupole_end)
    return 4;
  else if (z <= clens.second_chamber_end)
    return 5;
  else
    return 9999999;
}
