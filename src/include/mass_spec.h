#pragma once

#include "consts.h"
#include "vib_index.h"
#include "rf_basis.h"
#include "warnlogcount.h"
#include "apitofsim.h"
#include "exceptions.h"
#include "fragmentation.h"

#include <Eigen/Dense>
#include <optional>

class OperationContext;

using SkimmerData = Eigen::Array<double, Eigen::Dynamic, 3>;
const int VEL_SKIMMER = 0;
const int TEMP_SKIMMER = 1;
const int PRESSURE_SKIMMER = 2;
using InstrumentDims = Eigen::Array<double, 5, 1>;
const int SKIMMER_LENGTH = 4;
using InstrumentVoltages = Eigen::Array<double, 5, 1>;
using InstrumentPressures = Eigen::Array<double, 2, 1>;

struct Quadrupole
{
  double dc_field;
  double ac_field;
  double radiofrequency;
  double r_quadrupole;

  double mathieu_factor{};
  double angular_velocity;

  Quadrupole(
    double dc_field,
    double ac_field,
    double radiofrequency,
    double r_quadrupole);

  void compute_mathieu_factor(double m_ion);
};

struct MassSpectrometer
{
  SkimmerData skimmer;
  double mesh_skimmer;
  InstrumentDims lengths;
  InstrumentVoltages voltages;
  double T;
  InstrumentPressures pressures;
  std::optional<Quadrupole> quadrupole = std::nullopt;
  std::optional<double> radius_pinhole = 1.0e-3;
};

struct MassSpecInputFragmentationPathway
{
  const Histogram rate_const;
  double bonding_energy;
  const FragmentationProductsPtr products;

  MassSpecInputFragmentationPathway(
    const MolecularCluster &cluster_0,
    const ClusterSpec &cluster_1,
    const ClusterSpec &cluster_2,
    const Histogram &rate_const,
    std::optional<double> fragmentation_energy = std::nullopt, FragmentationProductsPtr products = {});

  MassSpecInputFragmentationPathway(
    const Histogram rate_const,
    double bonding_energy, FragmentationProductsPtr products = {});
};

struct MassSpecSubstanceSingleInput
{
  int cluster_charge_sign;
  double m_ion;
  double R_cluster;
  const Histogram density_cluster;
  const VibIndexPtr vib_index;
  const RFBasisPtr rf_basis;
  std::vector<MassSpecInputFragmentationPathway> pathways;
  const Gas gas;

  MassSpecSubstanceSingleInput(
    const MolecularCluster &cluster_0,
    const ClusterSpec &cluster_1,
    const ClusterSpec &cluster_2,
    Gas gas,
    const Histogram &density_cluster,
    const Histogram &rate_const,
    std::optional<double> fragmentation_energy = std::nullopt,
    int cluster_charge_sign = defaults::cluster_charge_sign, VibIndexPtr vib_index = {}, RFBasisPtr rf_basis = {});

  MassSpecSubstanceSingleInput(
    int cluster_charge_sign,
    double m_ion,
    double R_cluster,
    const Histogram density_cluster,
    std::vector<MassSpecInputFragmentationPathway> pathways,
    const Gas gas, VibIndexPtr vib_index = {}, RFBasisPtr rf_basis = {});

  MassSpecSubstanceSingleInput(
    const MolecularCluster &cluster_0,
    const std::vector<MassSpecInputFragmentationPathway> pathways,
    Gas gas,
    const Histogram &density_cluster,
    int cluster_charge_sign, VibIndexPtr vib_index = {}, RFBasisPtr rf_basis = {});

private:
  // Delegated to by the ClusterSpec constructors, so that the derived quantities
  // are computed once and every member is initialised rather than assigned.
  MassSpecSubstanceSingleInput(
    const ClusterDerived &derived,
    int cluster_charge_sign,
    const Histogram &density_cluster,
    std::vector<MassSpecInputFragmentationPathway> pathways,
    Gas gas, VibIndexPtr vib_index, RFBasisPtr rf_basis);
};

struct MSSubstanceTreeCluster
{
  double m_ion;
  double R_cluster;
  const Histogram density_cluster;
  const VibIndexPtr vib_index;
  const RFBasisPtr rf_basis;

  MSSubstanceTreeCluster(
    double m_ion,
    double R_cluster,
    Histogram density_cluster, VibIndexPtr vib_index = {}, RFBasisPtr rf_basis = {});

  MSSubstanceTreeCluster(
    const MolecularCluster &cluster_0,
    Histogram density_cluster, VibIndexPtr vib_index = {}, RFBasisPtr rf_basis = {});

private:
  MSSubstanceTreeCluster(
    const ClusterDerived &derived,
    Histogram density_cluster, VibIndexPtr vib_index = {}, RFBasisPtr rf_basis = {});
};

struct MSSubstanceTreeNode
{
  size_t payload_idx;
  std::vector<size_t> pathway_indices;
};

struct MSSubstanceTreePathway
{
  size_t payload_idx;
  std::optional<size_t> product_idx;
};

struct MassSpecSubstanceTreeInput
{
  int cluster_charge_sign;
  const Gas gas;
  const FragmentationStateMethod fragmentation_state_method;

  // Payloads are stored exactly once
  std::vector<MSSubstanceTreeCluster> cluster_payloads;
  std::vector<MassSpecInputFragmentationPathway> pathway_payloads;
  // The tree duplicates any DAG structures found
  std::vector<MSSubstanceTreeNode> tree_nodes;
  std::vector<MSSubstanceTreePathway> tree_pathways;

  MassSpecSubstanceTreeInput(
    int cluster_charge_sign,
    Gas gas,
    std::vector<MSSubstanceTreeCluster> cluster_payloads,
    std::vector<MassSpecInputFragmentationPathway> pathway_payloads,
    std::vector<MSSubstanceTreeNode> tree_nodes,
    std::vector<MSSubstanceTreePathway> tree_pathways,
    FragmentationStateMethod fragmentation_state_method = FragmentationStateMethod::zero);
};

struct Pressures
{
  InstrumentPressures P;
  Eigen::Array2d n;

  Pressures(const InstrumentPressures &pressures, double kT);
  Eigen::Array2d histogram_dts(double R_tot, double mobility_gas, double mobility_gas_inv, double multiplier, std::optional<Quadrupole> quadrupole) const;
};

struct CumulativeLengths
{
  double first_chamber_end;
  double sk_end;
  double quadrupole_start;
  double quadrupole_end;
  double second_chamber_end;
  double total_length;

  CumulativeLengths(const InstrumentDims &lengths);
  void info(std::ostream &out) const;
};

/**
 * @brief Struct to compute and store chamber quantities
 *
 * This struct computes and stores chamber quantities not related to
 * the cluster but including gas-derived quantities.
 *
 */
struct ChamberQuantities
{
  double kT;
  Pressures pressures;
  Eigen::Array2d gas_mean_free_paths;
  double mobility_gas;
  double mobility_gas_inv;
  CumulativeLengths clens;
  Eigen::Array4d E;

  ChamberQuantities(const MassSpectrometer &ms, const Gas &gas);
};

// Instantaneous total frequency, using the collision-time sampler's boundaries.
double collision_frequency(double z, const Eigen::Vector3d &velocity, const ChamberQuantities &chamber, double m_gas, double R, const SkimmerData &skimmer, double mesh_skimmer);

class CollisionProfile
{
  MassSpectrometer ms;
  Gas gas;
  ChamberQuantities chamber;
  double radius;

public:
  CollisionProfile(const MassSpectrometer &ms, const Gas &gas, double cluster_radius);
  Eigen::ArrayXd frequency(const Eigen::ArrayXd &z, const Eigen::Matrix<double, Eigen::Dynamic, 3> &velocity) const;
};

struct SubstanceQuantities
{
  double reduced_mass;
  double inertia;
  Eigen::Array4d acc;
  Eigen::Array2d dts;
  std::optional<Quadrupole> quadrupole;

  SubstanceQuantities(
    const MassSpectrometer &ms,
    const ChamberQuantities &chamber,
    const MassSpecSubstanceSingleInput &subs);

  SubstanceQuantities(
    const MassSpectrometer &ms,
    const ChamberQuantities &chamber,
    const Gas &gas,
    const int cluster_charge_sign,
    const MSSubstanceTreeCluster &cluster);

  SubstanceQuantities(
    const MassSpectrometer &ms,
    const ChamberQuantities &chamber,
    const Gas &gas,
    const int cluster_charge_sign,
    const double m_ion,
    const double R_cluster);
};

enum struct SampleMode
{
  dss_normalized,
  dss_unnormalized,
  rejection,
};

enum struct CollisionTimeMethod
{
  integrated_hazard,
  legacy,
};

struct CollisionTimeOptions
{
  const CollisionTimeMethod method;
  const double hazard_atol;
  const double hazard_rtol;

  CollisionTimeOptions(CollisionTimeMethod method = CollisionTimeMethod::integrated_hazard,
                       double hazard_atol = 1e-12, double hazard_rtol = 1e-9);
};

struct MassSpecLogConf
{
  int level = DEFAULT_LOGLEVEL;
  bool log_events = false;

  MassSpecLogConf(int level = DEFAULT_LOGLEVEL, bool log_events = false) : level(level), log_events(log_events)
  {
  }

  MassSpecLogConf(std::tuple<int, bool> tpl) : MassSpecLogConf(std::get<0>(tpl), std::get<1>(tpl))
  {
  }
};

const MassSpecLogConf DEFAULT_LOGCONF = MassSpecLogConf{};

using RuntimeDuration = std::chrono::high_resolution_clock::duration;
using SimulationResult = std::tuple<Eigen::ArrayXi, RuntimeDuration, RuntimeDuration>;

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceSingleInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict = true,
  MassSpecLogConf logconf = DEFAULT_LOGCONF,
  bool on_main_thread = false,
  const CollisionTimeOptions &collision_time = {});

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceSingleInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict,
  MassSpecLogConf logconf,
  OperationContext &operation,
  const CollisionTimeOptions &collision_time = {});

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict = true,
  MassSpecLogConf logconf = DEFAULT_LOGCONF,
  bool on_main_thread = false,
  const CollisionTimeOptions &collision_time = {});

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict,
  MassSpecLogConf logconf,
  OperationContext &operation,
  const CollisionTimeOptions &collision_time = {});

double particle_density(double pressure, double kT);
double evaluate_error(int n, int k);

Histogram scaled_density(const Histogram &density_cluster);
Histogram scaled_rate_const(const Histogram &rate_const);

#include "samplers.h"
