#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <numbers>
#include <optional>
#include <variant>
#include <vector>
#include <utility>
#include "apitofsim.h"
#include <magic_enum/magic_enum.hpp>
#include "consts.h"
#include "warnlogcount.h"
#include "exceptions.h"
#include "samplers.h"
#include "mass_spec.h"
#include "operation_context.h"
#include "upstream.h"
#include "trajectory.h"

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

void Quadrupole::compute_mathieu_factor(double m_ion)
{
  mathieu_factor = consts::eV / (m_ion * r_quadrupole * r_quadrupole);
}

Histogram scaled_density(const Histogram &density_cluster)
{
  using consts::boltzmann;
  Histogram my_density_cluster(density_cluster);
  my_density_cluster.rescale_y(boltzmann);
  my_density_cluster.rescale_x(boltzmann);
  return my_density_cluster;
}

Histogram scaled_rate_const(const Histogram &rate_const)
{
  using consts::boltzmann;
  Histogram my_rate_const(rate_const);
  my_rate_const.rescale_x(boltzmann);
  return my_rate_const;
}

MassSpecInputFragmentationPathway::MassSpecInputFragmentationPathway(
  const MolecularCluster &cluster_0,
  const ClusterSpec &cluster_1,
  const ClusterSpec &cluster_2,
  const Histogram &rate_const,
  std::optional<double> fragmentation_energy, FragmentationProductsPtr products) : rate_const(rate_const), products(std::move(products))
{
  using consts::hartK;
  double computed_fragmentation_energy;
  // Compute fragmentation energy in Kelvin
  if (fragmentation_energy == std::nullopt)
  {
    computed_fragmentation_energy = (electronic_energy(cluster_1) + electronic_energy(cluster_2) - cluster_0.electronic_energy) * hartK;
  }
  else
  {
    computed_fragmentation_energy = *fragmentation_energy;
  }
  this->bonding_energy = computed_fragmentation_energy * boltzmann;
}

MassSpecInputFragmentationPathway::MassSpecInputFragmentationPathway(
  const Histogram rate_const,
  double bonding_energy, FragmentationProductsPtr products) : rate_const(rate_const), bonding_energy(bonding_energy * boltzmann), products(std::move(products))
{
}

MassSpecSubstanceSingleInput::MassSpecSubstanceSingleInput(
  const MolecularCluster &cluster_0,
  const ClusterSpec &cluster_1,
  const ClusterSpec &cluster_2,
  Gas gas,
  const Histogram &density_cluster,
  const Histogram &rate_const,
  std::optional<double> fragmentation_energy,
  int cluster_charge_sign, VibIndexPtr vib_index, RFBasisPtr rf_basis) : MassSpecSubstanceSingleInput(compute_derived(cluster_0),
                                                                                                      cluster_charge_sign,
                                                                                                      density_cluster,
                                                                                                      {MassSpecInputFragmentationPathway(cluster_0, cluster_1, cluster_2, rate_const, fragmentation_energy)},
                                                                                                      gas, std::move(vib_index), std::move(rf_basis))
{
}

MassSpecSubstanceSingleInput::MassSpecSubstanceSingleInput(
  const ClusterDerived &derived,
  int cluster_charge_sign,
  const Histogram &density_cluster,
  std::vector<MassSpecInputFragmentationPathway> pathways,
  Gas gas, VibIndexPtr vib_index, RFBasisPtr rf_basis) : cluster_charge_sign(cluster_charge_sign),
                                                         m_ion(derived.mass),
                                                         R_cluster(derived.radius),
                                                         density_cluster(density_cluster),
                                                         vib_index(std::move(vib_index)), rf_basis(std::move(rf_basis)),
                                                         pathways(pathways),
                                                         gas(gas)
{
}

MassSpecSubstanceSingleInput::MassSpecSubstanceSingleInput(
  int cluster_charge_sign,
  double m_ion,
  double R_cluster,
  const Histogram density_cluster,
  const std::vector<MassSpecInputFragmentationPathway> pathways,
  const Gas gas, VibIndexPtr vib_index, RFBasisPtr rf_basis) : cluster_charge_sign(cluster_charge_sign),
                                                               m_ion(m_ion),
                                                               R_cluster(R_cluster),
                                                               density_cluster(density_cluster),
                                                               vib_index(std::move(vib_index)), rf_basis(std::move(rf_basis)),
                                                               pathways(pathways),
                                                               gas(gas)
{
}

MassSpecSubstanceSingleInput::MassSpecSubstanceSingleInput(
  const MolecularCluster &cluster_0,
  const std::vector<MassSpecInputFragmentationPathway> pathways,
  Gas gas,
  const Histogram &density_cluster,
  int cluster_charge_sign, VibIndexPtr vib_index, RFBasisPtr rf_basis) : MassSpecSubstanceSingleInput(compute_derived(cluster_0),
                                                                                                      cluster_charge_sign,
                                                                                                      density_cluster,
                                                                                                      pathways,
                                                                                                      gas, std::move(vib_index), std::move(rf_basis))
{
}

MSSubstanceTreeCluster::MSSubstanceTreeCluster(
  double m_ion,
  double R_cluster,
  Histogram density_cluster, VibIndexPtr vib_index, RFBasisPtr rf_basis) : m_ion(m_ion),
                                                                           R_cluster(R_cluster),
                                                                           density_cluster(std::move(density_cluster)), vib_index(std::move(vib_index)), rf_basis(std::move(rf_basis))
{
}

MSSubstanceTreeCluster::MSSubstanceTreeCluster(
  const MolecularCluster &cluster_0,
  Histogram density_cluster, VibIndexPtr vib_index, RFBasisPtr rf_basis) : MSSubstanceTreeCluster(compute_derived(cluster_0), std::move(density_cluster), std::move(vib_index), std::move(rf_basis))
{
}

MSSubstanceTreeCluster::MSSubstanceTreeCluster(
  const ClusterDerived &derived,
  Histogram density_cluster, VibIndexPtr vib_index, RFBasisPtr rf_basis) : m_ion(derived.mass),
                                                                           R_cluster(derived.radius),
                                                                           density_cluster(std::move(density_cluster)), vib_index(std::move(vib_index)), rf_basis(std::move(rf_basis))
{
}

MassSpecSubstanceTreeInput::MassSpecSubstanceTreeInput(
  int cluster_charge_sign,
  Gas gas,
  std::vector<MSSubstanceTreeCluster> cluster_payloads,
  std::vector<MassSpecInputFragmentationPathway> pathway_payloads,
  std::vector<MSSubstanceTreeNode> tree_nodes,
  std::vector<MSSubstanceTreePathway> tree_pathways,
  FragmentationStateMethod fragmentation_state_method) : cluster_charge_sign(cluster_charge_sign),
                                                         gas(gas),
                                                         fragmentation_state_method(fragmentation_state_method),
                                                         cluster_payloads(std::move(cluster_payloads)),
                                                         pathway_payloads(std::move(pathway_payloads)),
                                                         tree_nodes(std::move(tree_nodes)),
                                                         tree_pathways(std::move(tree_pathways))
{
  if (fragmentation_state_method == FragmentationStateMethod::phase_space)
  {
    for (const auto &node : this->tree_nodes)
      for (const auto index : node.pathway_indices)
      {
        const auto &transition = this->tree_pathways.at(index);
        if (!transition.product_idx)
          continue;
        const auto &pathway = this->pathway_payloads.at(transition.payload_idx);
        if (!pathway.products)
          throw ApiTofArgumentError("phase-space fragmentation requires product data for every continuing transition");
        const auto &parent = this->cluster_payloads.at(node.payload_idx);
        const auto &child = this->cluster_payloads.at(this->tree_nodes.at(*transition.product_idx).payload_idx);
        const auto &products = *pathway.products;
        const auto matches = [](double a, double b)
        { return std::abs(a - b) <= 1e-10 * std::abs(b); };
        if (!matches(parent.m_ion, products.daughter.mass + products.neutral.mass) || !matches(child.m_ion, products.daughter.mass) || !matches(0.4 * child.m_ion * child.R_cluster * child.R_cluster, products.daughter.inertia_moment))
          throw ApiTofArgumentError("Fragmentation product masses and daughter inertia must match the tree transition");
      }
  }
}

// LIST OF FUNCTIONS
// Here we are
template <typename GenT>
Eigen::Vector3d init_vel(GenT &gen, normal_distribution<double> &gauss, double m, double kT);
template <typename GenT>
Eigen::Vector3d init_ang_vel(GenT &gen, normal_distribution<double> &gauss, double m, double kT, double R);
template <typename GenT>
double init_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double kT, const Histogram &density_cluster);
double evaluate_rate_const(const Histogram &rate_const, double energy);
struct GasState
{
  double axial_velocity;
  double temperature;
  double pressure;
};

struct CollisionQuantities
{
  double number_density;
  Eigen::Vector3d relative_velocity;
  double relative_speed;
  double mobility;
  double inverse_mobility;
};

CollisionQuantities get_quantities_for_collision(const ParticleState &particle, const ChamberQuantities &chamber, double m_gas, const GasState &gas);
GasState update_physical_quantities(double z, const SkimmerData &skimmer, double mesh_skimmer, const ChamberQuantities &chamber, double T);
void update_rot_vel(Eigen::Vector3d &omega, double rot_energy_old, double rot_energy);
double boundary_vib_energy(double vib_energy_old, double reduced_mass, double u_norm, double v_cluster_norm, double theta);
template <typename GenT, typename VibEnergySamplerT>
std::tuple<double, double> redistribute_internal_energy(GenT &gen, VibEnergySamplerT &sampler, double vib_energy, double rot_energy);
std::tuple<Eigen::Vector3d, Eigen::Vector3d> eval_velocities(const Eigen::Vector3d &v, const Eigen::Vector3d &omega, const Eigen::Vector2d &u, double vib_energy, double vib_energy_old, double M, double m, double R_cluster);
template <typename GenT>
bool eval_collision(GenT &gen, uniform_real_distribution<double> &unif, double gas_mean_free_path, double x, double y, double z, double L, std::optional<double> pinhole, double quadrupole_end, Eigen::Vector3d &v_cluster, Eigen::Vector3d &omega, double u_norm, double theta, double R_cluster, double vib_energy, double vib_energy_old, double m_ion, double m_gas, double temperature, LogHelper pinhole_logger, int loglevel);
template <typename GenT>
double onedimMaxwell(GenT &gen, normal_distribution<double> &gauss, double m, double kT);
double mean_free_path(double R, double kT, double pressure);
double eval_solid_angle_stokes(double R, double L, double xx, double yy, double zz);

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
  OperationContext &operation,
  const CollisionTimeOptions &collision_time);

template <typename GasCollSamplerT, typename VibEnergySamplerT>
SimulationResult apitof_mass_spec(
  const MassSpectrometer &ms,
  const MassSpecSubstanceTreeInput &subs,
  const int N,
  const unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  GasCollSamplerT gas_coll_sampler,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation,
  const CollisionTimeOptions &collision_time);

template <typename Cluster>
static VibEnergyMode cluster_vib_mode(const Cluster &cluster)
{
  if (!cluster.vib_index)
    return VibEnergyMode::direct;
  cluster.vib_index->check_length(cluster.density_cluster.length());
  return cluster.vib_index->mode;
}

static VibEnergyMode vibrational_mode(const MassSpecSubstanceSingleInput &subs)
{
  return cluster_vib_mode(subs);
}

static VibEnergyMode vibrational_mode(const MassSpecSubstanceTreeInput &subs)
{
  if (subs.cluster_payloads.empty())
    throw ApiTofArgumentError("Missing tree cluster payloads");
  const auto mode = cluster_vib_mode(subs.cluster_payloads.front());
  for (const auto &cluster : subs.cluster_payloads)
    if (cluster_vib_mode(cluster) != mode)
      throw ApiTofArgumentError("Tree cluster payloads must use the same vibrational sampler");
  return mode;
}

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
  OperationContext &operation,
  const CollisionTimeOptions &collision_time)
{
  using consts::boltzmann;
  double m_gas = subs.gas.mass;
  double kT = boltzmann * mass_spec.T;
  double mobility_gas = kT / m_gas; // thermal agitation
  double boundary_u = 5.0 * sqrt(mobility_gas);
  const double du = 1.0e-4 * sqrt(mobility_gas);
  const double dtheta = 1.0e-3;
  const auto vib_mode = vibrational_mode(subs);
  auto dispatch = [&]<typename DirectSampler, typename GasSampler>(GasSampler gas_sampler)
  {
    auto run = [&]<typename VibSampler>()
    {
      return apitof_mass_spec<GasSampler, VibSampler>(mass_spec, subs, N, root_seed, result_queue, gas_sampler, strict, logconf, operation, collision_time);
    };
    switch (vib_mode)
    {
    case VibEnergyMode::dos_only:
      return run.template operator()<VibEnergyDosSampler>();
    case VibEnergyMode::group_dos:
      return run.template operator()<VibEnergyGroupSampler>();
    default:
      return run.template operator()<DirectSampler>();
    }
  };
  if (sample_mode == SampleMode::dss_normalized)
    return dispatch.template operator()<VibEnergyNormSampler>(GasCollCondNormHistDSSSampler(dtheta, du, boundary_u));
  if (sample_mode == SampleMode::dss_unnormalized)
    return dispatch.template operator()<VibEnergyUnnormSampler>(GasCollCondUnnormHistDSSSampler(dtheta, du, boundary_u));
  if (sample_mode == SampleMode::rejection)
    return dispatch.template operator()<VibEnergyNormSampler>(GasCollRejectionSampler(boundary_u));
  throw ApiTofArgumentError([&](auto &msg)
  {
    msg << "Unknown sampling mode: " << static_cast<int>(sample_mode) << std::endl;
  });
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
  bool on_main_thread,
  const CollisionTimeOptions &collision_time)
{
  OperationContext operation;
  return operation.run([&]
  {
    return apitof_mass_spec<MassSpecSubstanceSingleInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation, collision_time);
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
  bool on_main_thread,
  const CollisionTimeOptions &collision_time)
{
  OperationContext operation;
  return operation.run([&]
  {
    return apitof_mass_spec<MassSpecSubstanceTreeInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation, collision_time);
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
  OperationContext &operation,
  const CollisionTimeOptions &collision_time)
{
  return apitof_mass_spec<MassSpecSubstanceSingleInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation, collision_time);
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
  OperationContext &operation,
  const CollisionTimeOptions &collision_time)
{
  return apitof_mass_spec<MassSpecSubstanceTreeInput>(mass_spec, subs, N, root_seed, result_queue, sample_mode, strict, logconf, operation, collision_time);
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
  OperationContext &operation,
  const CollisionTimeOptions &collision_time);

template SimulationResult apitof_mass_spec<MassSpecSubstanceTreeInput>(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation,
  const CollisionTimeOptions &collision_time);

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
    auto energy_max_rate = pathway.rate_const.x_max();
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
  GenT &gen, uniform_real_distribution<double> &unif, const PathwaysT &pathways, double internal_energy, bool strict = true)
{
  int effective_pathway_index = 0;
  double t_next_fragmentation = std::numeric_limits<double>::infinity();
  std::optional<ApiTofRateConstantOverflow> effective_exception = std::nullopt;
  int pathway_index = 0;
  for (const MassSpecInputFragmentationPathway &pathway : pathways)
  {
    auto [t_fragmentation, exception] = next_fragmentation_time<GenT>(gen, unif, pathway, internal_energy, strict);
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
  return {dt1, dt2};
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
  const double R_cluster) : reduced_mass(1. / ((1. / m_ion) + (1. / gas.mass))),
                            inertia(0.4 * m_ion * R_cluster * R_cluster),
                            acc(chamber.E * consts::eV * cluster_charge_sign / m_ion),
                            dts(chamber.pressures.histogram_dts(R_cluster + gas.radius, chamber.mobility_gas, chamber.mobility_gas_inv, DT_MULTIPLIER, ms.quadrupole)),
                            quadrupole(ms.quadrupole)
{
  if (quadrupole)
  {
    quadrupole->compute_mathieu_factor(m_ion);
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

// The callback observes the particle before accepted collision updates are applied.
template <typename GenT, typename GasCollSamplerT, typename VibEnergySamplerT, typename ClusterT, typename OnCollisionT>
bool process_collision(
  GenT &gen,
  uniform_real_distribution<double> &unif,
  GasCollSamplerT &gas_coll_sampler,
  VibEnergySamplerT &vib_energy_sampler,
  const MassSpectrometer &ms,
  const ChamberQuantities &chamber,
  const Gas &gas,
  const ClusterT &cluster,
  const SubstanceQuantities &subquants,
  ParticleState &particle,
  WarningHelper &warn,
  LogHelper pinhole_logger,
  int loglevel,
  const OnCollisionT &on_collision)
{
  const auto local_gas = update_physical_quantities(particle.position.z(), ms.skimmer, ms.mesh_skimmer, chamber, ms.T);
  auto collision = get_quantities_for_collision(particle, chamber, gas.mass, local_gas);
  auto [theta, u_norm] = gas_coll_sampler.sample(gen, collision.number_density, collision.relative_speed, collision.mobility, collision.inverse_mobility, cluster.R_cluster + gas.radius, warn);

  double vib_energy_new = vib_energy_sampler.sample(gen, boundary_vib_energy(particle.vibrational_energy, subquants.reduced_mass, u_norm, collision.relative_speed, theta));
  Eigen::Vector3d omega = particle.omega;
  bool collision_accepted = eval_collision(gen, unif, chamber.gas_mean_free_paths[1], particle.position.x(), particle.position.y(), particle.position.z(), chamber.clens.total_length, ms.radius_pinhole, chamber.clens.quadrupole_end, collision.relative_velocity, omega, u_norm, theta, cluster.R_cluster, vib_energy_new, particle.vibrational_energy, cluster.m_ion, gas.mass, local_gas.temperature, pinhole_logger, loglevel);

  on_collision(theta, u_norm, collision_accepted, particle);

  if (collision_accepted)
  {
    particle.vibrational_energy = vib_energy_new;
    particle.velocity = collision.relative_velocity;
    particle.velocity.z() += local_gas.axial_velocity;
    particle.omega = omega;
    double rot_energy_old = particle.rotational_energy(subquants.inertia);
    double rot_energy;
    std::tie(particle.vibrational_energy, rot_energy) = redistribute_internal_energy(gen, vib_energy_sampler, particle.vibrational_energy, rot_energy_old);
    update_rot_vel(particle.omega, rot_energy_old, rot_energy);
  }
  return collision_accepted;
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
  OperationContext &operation,
  const CollisionTimeOptions &collision_time)
{
  VibEnergySamplerT vib_energy_sampler = VibEnergySamplerT(subs.density_cluster, subs.vib_index);
  const ChamberQuantities chamber(ms, subs.gas);
  const SubstanceQuantities subquants(ms, chamber, subs);
  validate_rf_basis(subquants.quadrupole, collision_time, subs.rf_basis.get());

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

      // Draw initial random velocity from Maxwell-Boltzmann distribution
      ParticleState particle{
        .velocity = init_vel(gen, gauss, subs.m_ion, chamber.kT),
        .omega = init_ang_vel(gen, gauss, subs.m_ion, chamber.kT, subs.R_cluster),
        .vibrational_energy = init_vib_energy(gen, unif, chamber.kT, subs.density_cluster),
      };
      int ncoll = 0;
      double coll_z = 0.0;

      if (logconf.log_events && operation.should_continue())
      {
        result_queue.enqueue(InitEvent{snapshot(j, particle, subquants.inertia)});
      }

      while (particle.position.z() < chamber.clens.total_length && operation.checkpoint()) // single realization // TO BE CHANGED IN SECOND CHAMBER!!!!!!!!!!!
      {
        double internal_energy = particle.internal_energy(subquants.inertia);

        auto [effective_pathway_index, t_fragmentation, overflow_exception] = next_fragmentation_time_multi(gen, unif, subs.pathways, internal_energy, strict);

        double old_t = particle.time;
        TimeNextCollOutcome outcome = next_coll(unif(gen), particle, chamber, subs.gas.radius + subs.R_cluster, subquants.dts, t_fragmentation, subquants.acc, subs.gas.mass, ms.skimmer, ms.mesh_skimmer, subquants.quadrupole, collision_time, subs.rf_basis.get());

        if (logconf.level >= LOGLEVEL_NORMAL)
        {
          if (particle.position.z() < chamber.clens.first_chamber_end)
          {
            LogHelper tmp_evolution = LogHelper{result_queue, LogMessage::tmp_evolution, &operation};
            tmp_evolution([&](auto &tmp_evolution)
            {
              tmp_evolution << particle.position.z() << " " << particle.time - old_t << " " << particle.velocity.norm() << " " << endl;
            });
          }
        }

        if (outcome != TimeNextCollOutcome::fragmentation && !strict && overflow_exception.has_value())
        {
          // An out-of-range rate is only tolerated when fragmentation actually occurs.
          throw *overflow_exception;
        }
        switch (outcome)
        {
        case TimeNextCollOutcome::fragmentation:
        {
          realization_counters[Counter::n_fragmented_total + effective_pathway_index]++;
          if (logconf.level >= LOGLEVEL_NORMAL)
          {
            fragments([&](auto &fragments)
            {
              fragments << j + 1 << "\t" << particle.time << "\t" << particle.position.z() << "\t" << zone(particle.position.z(), chamber.clens) << "\t" << coll_z << "\t" << zone(coll_z, chamber.clens) << endl;
            });
          }
          if (logconf.log_events && operation.should_continue())
          {
            result_queue.enqueue(FragmentationEvent{snapshot(j, particle, subquants.inertia), effective_pathway_index});
          }
          break;
        }
        case TimeNextCollOutcome::gas_collision:
        {
          // Keep track on number of collisions per realization
          ncoll++;
          if (ncoll > MAX_COLL)
          {
            throw ApiTofMaxCollisions(MAX_COLL, ncoll);
          }

          bool collision_accepted = process_collision(
            gen, unif, local_gas_coll_sampler, local_vib_energy_sampler, ms, chamber, subs.gas, subs, subquants,
            particle, warn,
            LogHelper{result_queue, LogMessage::pinhole, &operation}, logconf.level,
            [&](double theta, double u_norm, bool accepted, const ParticleState &before_collision)
          {
            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(CollisionEvent{snapshot(j, before_collision, subquants.inertia), theta, u_norm, accepted});
            }
          });

          if (collision_accepted)
          {
            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(PostCollisionEvent{snapshot(j, particle, subquants.inertia)});
            }
          }
          else
          {
            realization_counters[Counter::counter_collision_rejections]++;
          }
          continue;
        }
        case TimeNextCollOutcome::escape:
        {
          realization_counters[Counter::n_escaped_total]++;
          if (logconf.level >= LOGLEVEL_NORMAL)
          {
            final_position([&](auto &final_position)
            {
              final_position << particle.position.x() << "\t" << particle.position.y() << endl;
            });
          }
          if (logconf.log_events && operation.should_continue())
          {
            result_queue.enqueue(EscapeEvent{snapshot(j, particle, subquants.inertia)});
          }
          break;
        }
        }
        // Terminal outcomes finish this realization.
        break;
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

  return {counters, loop_time, total_time};
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
  const bool strict,
  const MassSpecLogConf logconf,
  OperationContext &operation,
  const CollisionTimeOptions &collision_time)
{
  const ChamberQuantities chamber(ms, subs.gas);
  std::vector<SubstanceQuantities> all_subquants;
  all_subquants.reserve(subs.cluster_payloads.size());
  for (const auto &cluster : subs.cluster_payloads)
  {
    all_subquants.emplace_back(ms, chamber, subs.gas, subs.cluster_charge_sign, cluster);
    validate_rf_basis(all_subquants.back().quadrupole, collision_time, cluster.rf_basis.get());
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

      // Draw initial random velocity from Maxwell-Boltzmann distribution
      ParticleState particle{
        .velocity = init_vel(gen, gauss, subpayload.m_ion, chamber.kT),
        .omega = init_ang_vel(gen, gauss, subpayload.m_ion, chamber.kT, subpayload.R_cluster),
        .vibrational_energy = init_vib_energy(gen, unif, chamber.kT, subpayload.density_cluster),
      };
      int ncoll = 0;
      double coll_z = 0.0;
      VibEnergySamplerT vib_energy_sampler{subpayload.density_cluster, subpayload.vib_index};
      int last_pathway_index = -1;

      if (logconf.log_events && operation.should_continue())
      {
        result_queue.enqueue(InitEvent{snapshot(j, particle, all_subquants[subpayload_index].inertia, subnode_index)});
      }

      while (particle.position.z() < chamber.clens.total_length && operation.checkpoint()) // single realization // TO BE CHANGED IN SECOND CHAMBER!!!!!!!!!!!
      {
        const MSSubstanceTreeCluster &subpayload = subs.cluster_payloads[subpayload_index];
        const SubstanceQuantities &subquants = all_subquants[subpayload_index];

        double internal_energy = particle.internal_energy(subquants.inertia);

        auto [effective_pathway_index, t_fragmentation, overflow_exception] = next_fragmentation_time_multi(gen, unif, pathways, internal_energy, strict);

        double old_t = particle.time;
        TimeNextCollOutcome outcome = next_coll(unif(gen), particle, chamber, subs.gas.radius + subpayload.R_cluster, subquants.dts, t_fragmentation, subquants.acc, subs.gas.mass, ms.skimmer, ms.mesh_skimmer, subquants.quadrupole, collision_time, subpayload.rf_basis.get());

        if (logconf.level >= LOGLEVEL_NORMAL)
        {
          if (particle.position.z() < chamber.clens.first_chamber_end)
          {
            LogHelper tmp_evolution = LogHelper{result_queue, LogMessage::tmp_evolution, &operation};
            tmp_evolution([&](auto &tmp_evolution)
            {
              tmp_evolution << particle.position.z() << " " << particle.time - old_t << " " << particle.velocity.norm() << " " << endl;
            });
          }
        }

        if (outcome != TimeNextCollOutcome::fragmentation && !strict && overflow_exception.has_value())
        {
          // An out-of-range rate is only tolerated when fragmentation actually occurs.
          throw *overflow_exception;
        }
        switch (outcome)
        {
        case TimeNextCollOutcome::fragmentation:
        {
          size_t tree_pathway_idx = subs.tree_nodes[subnode_index].pathway_indices[effective_pathway_index];
          std::optional<size_t> product_idx = subs.tree_pathways[tree_pathway_idx].product_idx;
          last_pathway_index = tree_pathway_idx;
          if (logconf.level >= LOGLEVEL_NORMAL)
          {
            fragments([&](auto &fragments)
            {
              fragments << j + 1 << "\t" << particle.time << "\t" << particle.position.z() << "\t" << zone(particle.position.z(), chamber.clens) << "\t" << coll_z << "\t" << zone(coll_z, chamber.clens) << endl;
            });
          }
          if (logconf.log_events && operation.should_continue())
          {
            int next_particle_index = product_idx ? *product_idx : -1;
            result_queue.enqueue(FragmentationEvent{snapshot(j, particle, subquants.inertia, subnode_index), static_cast<int>(tree_pathway_idx), next_particle_index});
          }
          if (product_idx)
          {
            if (subs.fragmentation_state_method == FragmentationStateMethod::phase_space)
            {
              const auto &pathway = subs.pathway_payloads[subs.tree_pathways[tree_pathway_idx].payload_idx];
              auto state = pathway.products->sample(gen, internal_energy - pathway.bonding_energy,
                                                    subquants.inertia * particle.omega, particle.velocity, operation);
              if (!state)
                break;
              particle.velocity = state->velocity;
              particle.omega = state->omega;
              particle.vibrational_energy = state->vibrational_energy;
            }
            else
            {
              // Legacy continuation: all daughter motion and excitation are reset.
              particle.velocity = Eigen::Vector3d::Zero();
              particle.omega = Eigen::Vector3d::Zero();
              particle.vibrational_energy = 0;
            }
            subnode_index = *product_idx;
            subpayload_index = subs.tree_nodes[subnode_index].payload_idx;
            prepare_pathways_from_tree(pathways, subs, subnode_index);

            vib_energy_sampler = VibEnergySamplerT(subs.cluster_payloads[subpayload_index].density_cluster, subs.cluster_payloads[subpayload_index].vib_index);
            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(PostFragmentationEvent{snapshot(j, particle, all_subquants[subpayload_index].inertia, subnode_index)});
            }
            continue;
          }
          break;
        }
        case TimeNextCollOutcome::gas_collision:
        {
          // Keep track on number of collisions per realization
          ncoll++;
          if (ncoll > MAX_COLL)
          {
            throw ApiTofMaxCollisions(MAX_COLL, ncoll);
          }

          bool collision_accepted = process_collision(
            gen, unif, local_gas_coll_sampler, vib_energy_sampler, ms, chamber, subs.gas, subpayload, subquants,
            particle, warn,
            LogHelper{result_queue, LogMessage::pinhole, &operation}, logconf.level,
            [&](double theta, double u_norm, bool accepted, const ParticleState &before_collision)
          {
            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(CollisionEvent{snapshot(j, before_collision, subquants.inertia, subnode_index), theta, u_norm, accepted});
            }
          });

          if (collision_accepted)
          {
            if (logconf.log_events && operation.should_continue())
            {
              result_queue.enqueue(PostCollisionEvent{snapshot(j, particle, subquants.inertia, subnode_index)});
            }
          }
          else
          {
            realization_counters[Counter::counter_collision_rejections]++;
          }
          continue;
        }
        case TimeNextCollOutcome::escape:
        {
          if (logconf.level >= LOGLEVEL_NORMAL)
          {
            final_position([&](auto &final_position)
            {
              final_position << particle.position.x() << "\t" << particle.position.y() << endl;
            });
          }
          if (logconf.log_events && operation.should_continue())
          {
            result_queue.enqueue(EscapeEvent{snapshot(j, particle, subquants.inertia, subnode_index)});
          }
          break;
        }
        }
        // Terminal outcomes finish this realization.
        break;
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

  return {counters, loop_time, total_time};
}

double evaluate_error(int n, int k)
{
  return sqrt(((6.0 * k * k) - (k * (6.0 + k) * n) + ((2.0 + k) * n * n)) / (n * n * (3.0 + n) * (2.0 + n)));
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
  Eigen::Vector3d velocity(
    onedimMaxwell(gen, gauss, m, kT),
    onedimMaxwell(gen, gauss, m, kT),
    onedimMaxwell(gen, gauss, m, kT));
  velocity[2] = std::abs(velocity[2]);
  return velocity;
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
      throw ApiTofRateConstantOverflow(rate_const.x_max() / boltzmann, energy / boltzmann);
    }
  }
}


GasState update_skimmer_quantities(const SkimmerData &skimmer, double z, double first_chamber_end, double mesh_skimmer)
{
  int m;
  double coeff1;
  double coeff2;
  double position;
  double v_gas;
  double temp;
  double pressure;
  position = z - first_chamber_end;
  m = static_cast<int>(position / mesh_skimmer);
  if (m >= skimmer.rows() - 1)
  {
    m = static_cast<int>(skimmer.rows() - 1);
    v_gas = skimmer(m, VEL_SKIMMER);
    temp = skimmer(m, TEMP_SKIMMER);
    pressure = skimmer(m, PRESSURE_SKIMMER);
  }
  else
  {
    coeff1 = (position - (m * mesh_skimmer)) / mesh_skimmer;
    coeff2 = 1.0 - coeff1;
    v_gas = (coeff2 * skimmer(m, VEL_SKIMMER)) + (coeff1 * skimmer(m + 1, VEL_SKIMMER));
    temp = (coeff2 * skimmer(m, TEMP_SKIMMER)) + (coeff1 * skimmer(m + 1, TEMP_SKIMMER));
    pressure = (coeff2 * skimmer(m, PRESSURE_SKIMMER)) + (coeff1 * skimmer(m + 1, PRESSURE_SKIMMER));
  }
  // density=coeff2*density_skimmer[m]+coeff1*density_skimmer[m+1];
  return {v_gas, temp, pressure};
}

CollisionQuantities get_quantities_for_collision(const ParticleState &particle, const ChamberQuantities &chamber, double m_gas, const GasState &gas)
{
  double n;
  Eigen::Vector3d v_rel = particle.velocity;
  double mobility_gas = chamber.mobility_gas;
  double mobility_gas_inv = chamber.mobility_gas_inv;
  if (particle.position.z() < chamber.clens.first_chamber_end)
  {
    n = chamber.pressures.n[0];
  }
  else if (particle.position.z() < chamber.clens.second_chamber_end)
  {
    v_rel.z() -= gas.axial_velocity;
    double kT = boltzmann * gas.temperature;
    mobility_gas = kT / m_gas;
    mobility_gas_inv = m_gas / kT;
    n = particle_density(gas.pressure, kT);
  }
  else
  {
    n = chamber.pressures.n[1];
  }
  return {n, v_rel, v_rel.norm(), mobility_gas, mobility_gas_inv};
}

GasState update_physical_quantities(double z, const SkimmerData &skimmer, double mesh_skimmer, const ChamberQuantities &chamber, double T)
{
  if (z < chamber.clens.first_chamber_end)
  {
    return {0.0, T, chamber.pressures.P[0]};
  }
  if (z < chamber.clens.sk_end)
  {
    return update_skimmer_quantities(skimmer, z, chamber.clens.first_chamber_end, mesh_skimmer);
  }
  return {0.0, T, chamber.pressures.P[1]};
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
    sum1 += density_cluster.y()[m] * exp(-density_cluster.x()[m] / kT);
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
    sum2 += density_cluster.y()[m] * exp(-density_cluster.x()[m] / kT) / sum1;
    m++;
  }
  return density_cluster.x()[m - 1];
}


double collision_frequency(double z, const Eigen::Vector3d &velocity, const ChamberQuantities &chamber, double m_gas, double R, const SkimmerData &skimmer, double mesh_skimmer)
{
  if (z < 0)
    return 0;
  if (z < chamber.clens.first_chamber_end)
    return coll_freq(chamber.pressures.n[0], chamber.mobility_gas, chamber.mobility_gas_inv, R, velocity.norm());
  if (z > chamber.clens.sk_end)
    return coll_freq(chamber.pressures.n[1], chamber.mobility_gas, chamber.mobility_gas_inv, R, velocity.norm());
  const auto gas = update_skimmer_quantities(skimmer, z, chamber.clens.first_chamber_end, mesh_skimmer);
  double kT = consts::boltzmann * gas.temperature;
  Eigen::Vector3d relative_velocity = velocity;
  relative_velocity[2] -= gas.axial_velocity;
  return coll_freq(particle_density(gas.pressure, kT), kT / m_gas, m_gas / kT, R, relative_velocity.norm());
}

CollisionProfile::CollisionProfile(const MassSpectrometer &ms, const Gas &gas, double cluster_radius)
    : ms(ms), gas(gas), chamber(ms, gas), radius(cluster_radius + gas.radius)
{
}

Eigen::ArrayXd CollisionProfile::frequency(const Eigen::ArrayXd &z, const Eigen::Matrix<double, Eigen::Dynamic, 3> &velocity) const
{
  if (velocity.rows() != z.size())
    throw std::invalid_argument("velocity must have one row per z");
  Eigen::ArrayXd result(z.size());
  for (Eigen::Index i = 0; i < z.size(); ++i)
  {
    if (!std::isfinite(z[i]) || z[i] < 0 || z[i] > chamber.clens.total_length || !velocity.row(i).allFinite())
      throw std::invalid_argument("z must be within the instrument and z/velocity must be finite");
    result[i] = collision_frequency(z[i], velocity.row(i).transpose(), chamber, gas.mass, radius, ms.skimmer, ms.mesh_skimmer);
  }
  return result;
}

// Evaluate time to next collision
TimeNextCollOutcome next_coll_old(double collision_quantile, ParticleState &particle, const ChamberQuantities &chamber, double R, Eigen::Array2d dts, double t_fragmentation, const Eigen::Array4d &acc, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, const std::optional<Quadrupole> quadrupole)
{
  using namespace consts;
  double integral = 0.0;
  double P = 1.0;
  const double v_cluster_norm_xy = (particle.velocity[0] * particle.velocity[0]) + (particle.velocity[1] * particle.velocity[1]);
  const double r = collision_quantile;
  double delta_t = 0.0;
  double c1 = collision_frequency(particle.position.z(), particle.velocity, chamber, m_gas, R, skimmer, mesh_skimmer);

  while (true)
  {
    if (particle.position.z() < 0 || (particle.position.z() == 0 && (particle.velocity[2] < 0 || (particle.velocity[2] == 0 && acc[0] < 0))))
    {
      bool fragmented = upstream::advance(particle.velocity, particle.position.z(), particle.position.x(), particle.position.y(), acc[0],
                                          std::max(0.0, t_fragmentation - delta_t), particle.time, delta_t);
      if (fragmented)
        return TimeNextCollOutcome::fragmentation;
      c1 = collision_frequency(particle.position.z(), particle.velocity, chamber, m_gas, R, skimmer, mesh_skimmer);
    }
    if (particle.position.z() >= chamber.clens.second_chamber_end)
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
    const Eigen::Vector3d old_velocity = particle.velocity;
    const double z = particle.position.z();
    const bool first_chamber = z < chamber.clens.first_chamber_end;
    const bool in_quadrupole = z >= chamber.clens.quadrupole_start && z < chamber.clens.quadrupole_end;
    // Preserve skimmer integration at z == sk_end, including its chamber velocity kick.
    const bool second_chamber = (z > chamber.clens.sk_end && z < chamber.clens.quadrupole_start) ||
                                in_quadrupole || z >= chamber.clens.quadrupole_end;
    const bool skimmer_step = !first_chamber && !second_chamber;
    double dt = dts[first_chamber ? 0 : 1];
    double outward_time = std::numeric_limits<double>::infinity();

    if (first_chamber)
    {
      outward_time = upstream::crossing_time(z, old_velocity[2], acc[0], false);
      dt = std::min({dt, outward_time, t_fragmentation - delta_t});
      particle.velocity[2] += acc[0] * dt;
    }
    else if (z >= chamber.clens.sk_end && z < chamber.clens.quadrupole_start)
    {
      particle.velocity[2] += acc[1] * dt;
    }
    else if (in_quadrupole)
    {
      if (quadrupole)
      {
        double accx = quadrupole->mathieu_factor * (-quadrupole->dc_field + (quadrupole->ac_field * cos(quadrupole->angular_velocity * particle.time))) * (particle.position.x() + (particle.velocity[0] * dt / 2.0));
        double accy = quadrupole->mathieu_factor * (quadrupole->dc_field - (quadrupole->ac_field * cos(quadrupole->angular_velocity * particle.time))) * (particle.position.y() + (particle.velocity[1] * dt / 2.0));
        particle.velocity[0] += accx * dt;
        particle.velocity[1] += accy * dt;
      }
      particle.velocity[2] += acc[2] * dt;
    }
    else if (z >= chamber.clens.quadrupole_end)
    {
      particle.velocity[2] += acc[3] * dt;
    }

    double c2;
    if (skimmer_step)
    {
      const auto gas = update_skimmer_quantities(skimmer, z, chamber.clens.first_chamber_end, mesh_skimmer);
      double kT = boltzmann * gas.temperature;
      double mobility_gas = boltzmann * gas.temperature / m_gas;
      double mobility_gas_inv = 1.0 / mobility_gas;
      double n = particle_density(gas.pressure, kT);
      double v_rel_norm = sqrt(v_cluster_norm_xy + pow(particle.velocity[2] - gas.axial_velocity, 2));
      c2 = coll_freq(n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
      dt = DT_MULTIPLIER / c2;
    }
    else
    {
      c2 = coll_freq(chamber.pressures.n[first_chamber ? 0 : 1], chamber.mobility_gas, chamber.mobility_gas_inv, R, particle.velocity.norm());
    }

    integral += (c1 + c2) * dt / 2.0;
    P = exp(-integral);
    delta_t += dt;
    if (!first_chamber && in_quadrupole)
    {
      particle.position.x() += (old_velocity[0] + particle.velocity[0]) * dt / 2.0;
      particle.position.y() += (old_velocity[1] + particle.velocity[1]) * dt / 2.0;
    }
    else
    {
      particle.position.x() += old_velocity[0] * dt;
      particle.position.y() += old_velocity[1] * dt;
    }
    particle.position.z() += skimmer_step ? old_velocity[2] * dt : (old_velocity[2] + particle.velocity[2]) * dt / 2.0;
    particle.time += dt;

    if (first_chamber && dt == outward_time)
    {
      particle.position.z() = 0;
      // Resolve a collision accumulated inside the chamber before entering vacuum.
      if (r >= P && delta_t < t_fragmentation)
        return TimeNextCollOutcome::gas_collision;
    }
    c1 = c2;
  }
}

double boundary_vib_energy(double vib_energy_old, double reduced_mass, double u_norm, double v_cluster_norm, double theta)
{
  double relative_speed = u_norm + (v_cluster_norm * cos(theta));
  return vib_energy_old + (reduced_mass * 0.5 * relative_speed * relative_speed);
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
  omega *= sqrt(rot_energy / rot_energy_old);
}

// Mean free path
double mean_free_path(double R, double kT, double pressure)
{
  using consts::pi;
  return kT / (std::numbers::sqrt2 * pi * 4.0 * R * R * pressure);
}


// Evaluate the velocities after collision in the rotated reference system
std::tuple<Eigen::Vector3d, Eigen::Vector3d> eval_velocities(const Eigen::Vector3d &v, const Eigen::Vector3d &omega, const Eigen::Vector2d &u, double vib_energy, double vib_energy_old, double M, double m, double R_cluster)
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
  vy = ((4.0 * omega[0] * R_cluster) + (4.0 * u[1]) + ((3.0 + (2.0 * ratio_masses)) * v[1])) / (7.0 + (2.0 * ratio_masses));

  // cout << u[1]-v[1] << endl<<endl;
  //  In case of anelastic collision, part of the energy (vib_energy) is absorbed by the cluster into vibrational modes, and the y-velocity becomes
  radicand = (m_reduced * m_reduced * pow(u[0] - v[2], 2)) - (2.0 * (vib_energy - vib_energy_old) * m_reduced / M);
  // cout << radicand << endl;
  if (radicand < 0)
  {
    throw ApiTofUnexpectedNumericalError([&](auto &msg)
    {
      msg << "sqrt of negative number in evaluation of velocities after collision! radicand: " << radicand << endl;
    });
  }
  vz = (m_reduced * u[0]) + (M_reduced * v[2]) - sqrt(radicand);

  vx = ((-4.0 * omega[1] * R_cluster) + ((3.0 + (2.0 * ratio_masses)) * v[0])) / (7.0 + (2.0 * ratio_masses));
  omegay = ((((2.0 * ratio_masses) - 3.0) * omega[1]) - (10.0 * (v[0] / R_cluster))) / (7.0 + (2.0 * ratio_masses));
  omegax = (((-3.0 + (2.0 * ratio_masses)) * omega[0]) + ((10.0 * (v[1] - u[1])) / R_cluster)) / (7.0 + (2.0 * ratio_masses));

  Eigen::Vector3d v_new;
  v_new[0] = vx;
  v_new[1] = vy;
  v_new[2] = vz;
  Eigen::Vector3d omega_new = omega;
  omega_new[0] = omegax;
  omega_new[1] = omegay;
  // omega[2] is intentionally unchanged
  return std::make_tuple(v_new, omega_new);
}


// Collision-frame axes expressed in lab coordinates.
Eigen::Matrix3d collision_frame(const Eigen::Vector3d &v_cluster, double theta, double phi, double alpha)
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
    z2 = (sin(theta) * cos(phi) * x1) + (sin(theta) * sin(phi) * y1) + (cos(theta) * z1);
    x2 = cross_norm(z2, z1);
    y2 = cross_norm(z2, x2);
  }
  else if (theta == 0.0)
  {
    z2 = z1;
    y2 = cross_norm(z2, x1);
    x2 = cross_norm(y2, z2);
  }
  else if (theta == pi)
  {
    z2 = -z1;
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

  Eigen::Matrix3d frame;
  // Find the direction of tangential velocity.
  frame.col(0) = (cos(alpha) * x2) + (sin(alpha) * y2);
  frame.col(1) = (-sin(alpha) * x2) + (cos(alpha) * y2);
  frame.col(2) = z2;
  return frame;
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
  c = (R * R) + (xx * xx) + (yy * yy) - (2.0 * xphi) - (2.0 * yphi);
  integrand = (1.0 - (zz / sqrt(c + (zz * zz)))) * ((R * R) - xphi - yphi) / c;
  sum += 0.5 * integrand;

  for (int i = 1; i < N; i++)
  {
    phi = dphi * i;
    xphi = R * xx * cos(phi);
    yphi = R * yy * sin(phi);
    c = (R * R) + (xx * xx) + (yy * yy) - (2.0 * xphi) - (2.0 * yphi);
    integrand = (1.0 - (zz / sqrt(c + (zz * zz)))) * ((R * R) - xphi - yphi) / c;
    sum += integrand;
  }

  phi = 2.0 * pi;
  xphi = R * xx * cos(phi);
  yphi = R * yy * sin(phi);
  c = (R * R) + (xx * xx) + (yy * yy) - (2.0 * xphi) - (2.0 * yphi);
  integrand = (1.0 - (zz / sqrt(c + (zz * zz)))) * ((R * R) - xphi - yphi) / c;
  sum += 0.5 * integrand;

  return sum * dphi;
}

bool gas_from_pinhole(const Eigen::Vector3d &velocity_gas, double x, double y, double z, double L, double radius_pinhole)
{
  if (velocity_gas[2] >= 0.0)
  {
    return false;
  }
  Eigen::Vector2d target;
  target[0] = (velocity_gas[0] * (L - z) / velocity_gas[2]) + x;
  target[1] = (velocity_gas[1] * (L - z) / velocity_gas[2]) + y;
  return (target[0] * target[0]) + (target[1] * target[1]) < radius_pinhole * radius_pinhole;
}

double pinhole_collision_probability(double gas_mean_free_path, double x, double y, double z, double L, double radius_pinhole)
{
  double distance = sqrt((x * x) + (y * y) + ((L - z) * (L - z)));
  return (1.0 - exp(-distance / gas_mean_free_path)) * (1.0 - (eval_solid_angle_stokes(radius_pinhole, L, x, y, z) / (2.0 * consts::pi)));
}

template <typename GenT>
bool accept_pinhole_collision(GenT &gen, uniform_real_distribution<double> &unif, double gas_mean_free_path, double x, double y, double z, double L, double radius_pinhole, const Eigen::Vector3d &velocity_gas, LogHelper pinhole_logger, int loglevel)
{
  bool inside_target = gas_from_pinhole(velocity_gas, x, y, z, L, radius_pinhole);
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
    double prob_coll = pinhole_collision_probability(gas_mean_free_path, x, y, z, L, radius_pinhole);
    if (r > prob_coll)
    {
      return false;
    }
  }
  return true;
}

template <typename GenT>
bool eval_collision(GenT &gen, uniform_real_distribution<double> &unif, double gas_mean_free_path, double x, double y, double z, double L, std::optional<double> pinhole, double quadrupole_end, Eigen::Vector3d &v_cluster, Eigen::Vector3d &omega, double u_norm, double theta, double R_cluster, double vib_energy, double vib_energy_old, double m_ion, double m_gas, double temperature, LogHelper pinhole_logger, int loglevel)
{
  using namespace consts;
  double phi = 2.0 * pi * unif(gen);
  double alpha = 2.0 * pi * unif(gen);
  double kT = boltzmann * temperature;
  Eigen::Vector2d u;

  bool collision_accepted = true;
  const Eigen::Matrix3d frame = collision_frame(v_cluster, theta, phi, alpha);
  Eigen::Vector3d v2 = frame.transpose() * v_cluster;
  Eigen::Vector3d omega2 = frame.transpose() * omega;

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

  // Check whether the gas molecule comes from the pinhole.
  if (pinhole && z > quadrupole_end && z < L)
  {
    Eigen::Vector3d velocity_gas = (u[1] * frame.col(1)) + (u[0] * frame.col(2));
    collision_accepted = accept_pinhole_collision(gen, unif, gas_mean_free_path, x, y, z, L, *pinhole, velocity_gas, pinhole_logger, loglevel);
  }

  if (collision_accepted) // Normal procedure
  {
    std::tie(v2, omega2) = eval_velocities(v2, omega2, u, vib_energy, vib_energy_old, m_ion, m_gas, R_cluster);
    // Express new velocities in lab reference system
    v_cluster = frame * v2;
    omega = frame * omega2;
  }
  return collision_accepted;
}

int zone(double z, const CumulativeLengths &clens)
{
  if (z < 0)
    return 0;
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
