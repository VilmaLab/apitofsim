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

enum struct SampleMode
{
  dss_normalized,
  dss_unnormalized,
  rejection,
};

typedef std::chrono::high_resolution_clock::duration RuntimeDuration;
typedef std::tuple<Counters, RuntimeDuration, RuntimeDuration> SimulationResult;

SimulationResult apitof_mass_spec(
  const MassSpectrometer &mass_spec,
  int cluster_charge_sign,
  int N,
  double bonding_energy,
  Gas gas,
  double m_ion,
  double R_cluster,
  const Histogram &density_cluster,
  const Histogram &rate_const,
  unsigned long long root_seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  int loglevel = DEFAULT_LOGLEVEL,
  bool on_main_thread = false);

void rescale_density(Histogram &density);
void rescale_energies(Histogram &energies);
double particle_density(double pressure, double kT);
double evaluate_error(int n, int k);

#include "samplers.h"
