#pragma once

#include "warnlogcount.h"
#include "apitofsim.h"
#include "exceptions.h"

#include <Eigen/Dense>

typedef Eigen::Array<double, Eigen::Dynamic, 3> SkimmerData;
const int VEL_SKIMMER = 0;
const int TEMP_SKIMMER = 1;
const int PRESSURE_SKIMMER = 2;
typedef Eigen::Array<double, 5, 1> InstrumentDims;
const int SKIMMER_LENGTH = 4;
typedef Eigen::Array<double, 5, 1> InstrumentVoltages;

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

enum struct SampleMode
{
  dss_normalized,
  dss_unnormalized,
  rejection,
};

typedef std::chrono::high_resolution_clock::duration RuntimeDuration;
typedef std::tuple<Counters, RuntimeDuration, RuntimeDuration> SimulationResult;

SimulationResult apitof_pinhole(
  int cluster_charge_sign,
  double T,
  double pressure_first,
  double pressure_second,
  InstrumentDims lengths,
  InstrumentVoltages voltages,
  int N,
  double bonding_energy,
  Gas gas,
  std::optional<Quadrupole> quadrupole,
  double m_ion,
  double R_cluster,
  const Histogram &density_cluster,
  const Histogram &rate_const,
  const SkimmerData &skimmer,
  const double mesh_skimmer,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  int loglevel = DEFAULT_LOGLEVEL);

void rescale_density(Histogram &density);
void rescale_energies(Histogram &energies);
double particle_density(double pressure, double kT);
double evaluate_error(int n, int k);

#include "samplers.h"
