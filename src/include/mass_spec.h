#pragma once

#include "consts.h"
#include "warnlogcount.h"
#include "apitofsim.h"
#include "exceptions.h"

#include <Eigen/Dense>

class OperationContext;

typedef Eigen::Array<double, Eigen::Dynamic, 3> SkimmerData;
const int VEL_SKIMMER = 0;
const int TEMP_SKIMMER = 1;
const int PRESSURE_SKIMMER = 2;
typedef Eigen::Array<double, 5, 1> InstrumentDims;
const int SKIMMER_LENGTH = 4;
typedef Eigen::Array<double, 5, 1> InstrumentVoltages;
typedef Eigen::Array<double, 2, 1> InstrumentPressures;

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

  MassSpecInputFragmentationPathway(
    const ClusterData &cluster_0,
    const ClusterData &cluster_1,
    const ClusterData &cluster_2,
    const Histogram &rate_const,
    std::optional<double> fragmentation_energy = std::nullopt);

  MassSpecInputFragmentationPathway(
    const Histogram rate_const,
    double bonding_energy);
};

struct MassSpecSubstanceSingleInput
{
  int cluster_charge_sign;
  double m_ion;
  double R_cluster;
  const Histogram density_cluster;
  std::vector<MassSpecInputFragmentationPathway> pathways;
  const Gas gas;

  MassSpecSubstanceSingleInput(
    const ClusterData &cluster_0,
    const ClusterData &cluster_1,
    const ClusterData &cluster_2,
    Gas gas,
    const Histogram &density_cluster,
    const Histogram &rate_const,
    std::optional<double> fragmentation_energy = std::nullopt,
    int cluster_charge_sign = defaults::cluster_charge_sign);

  MassSpecSubstanceSingleInput(
    int cluster_charge_sign,
    double m_ion,
    double R_cluster,
    const Histogram density_cluster,
    std::vector<MassSpecInputFragmentationPathway> pathways,
    const Gas gas);

  MassSpecSubstanceSingleInput(
    const ClusterData &cluster_0,
    const std::vector<MassSpecInputFragmentationPathway> pathways,
    Gas gas,
    const Histogram &density_cluster,
    int cluster_charge_sign);
};

struct MSSubstanceTreeCluster
{
  double m_ion;
  double R_cluster;
  const Histogram density_cluster;

  MSSubstanceTreeCluster(
    double m_ion,
    double R_cluster,
    const Histogram density_cluster);

  MSSubstanceTreeCluster(
    const ClusterData &cluster_0,
    const Histogram density_cluster);
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
    std::vector<MSSubstanceTreePathway> tree_pathways);
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

struct SubstanceQuantities
{
  double reduced_mass;
  double inertia;
  Eigen::Array4d acc;
  Eigen::Array2d dts;
  std::optional<double> mathieu_factor = std::nullopt;

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

typedef std::chrono::high_resolution_clock::duration RuntimeDuration;
typedef std::tuple<Eigen::ArrayXi, RuntimeDuration, RuntimeDuration> SimulationResult;

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceSingleInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict = true,
  MassSpecLogConf logconf = DEFAULT_LOGCONF,
  bool on_main_thread = false);

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceSingleInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict,
  MassSpecLogConf logconf,
  OperationContext &operation);

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict = true,
  MassSpecLogConf logconf = DEFAULT_LOGCONF,
  bool on_main_thread = false);

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  const MassSpecSubstanceTreeInput &subs,
  int N,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict,
  MassSpecLogConf logconf,
  OperationContext &operation);

double particle_density(double pressure, double kT);
double evaluate_error(int n, int k);

Histogram scaled_density(const Histogram &density_cluster);
Histogram scaled_rate_const(const Histogram &rate_const);

#include "samplers.h"
