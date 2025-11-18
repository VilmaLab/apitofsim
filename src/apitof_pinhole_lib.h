#include <Eigen/Dense>
#include <iostream>
#include <omp.h>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <tuple>
#include <variant>
#include "utils.h"
#include <magic_enum/magic_enum.hpp>
#pragma clang attribute push(__attribute__((no_sanitize("unsigned-integer-overflow"))), apply_to = function)
#include <blockingconcurrentqueue.h>
#pragma clang attribute pop
#include "consts.h"

using namespace std;
using magic_enum::enum_count;
using moodycamel::BlockingConcurrentQueue;

typedef Eigen::Array<double, Eigen::Dynamic, 3> SkimmerData;
const int VEL_SKIMMER = 0;
const int TEMP_SKIMMER = 1;
const int PRESSURE_SKIMMER = 2;
typedef Eigen::Array<double, 5, 1> InstrumentDims;
const int SKIMMER_LENGTH = 4;
typedef Eigen::Array<double, 5, 1> InstrumentVoltages;

template <typename Callback>
std::string call_with_stringstream(Callback cb)
{
  stringstream ss;
  ss << std::scientific << std::setprecision(3);
  cb(ss);
  return ss.str();
}

class ApiTofError : public std::runtime_error
{
public:
  ApiTofError(const std::string &msg)
      : std::runtime_error(msg)
  {
  }

  ApiTofError(const char *msg)
      : std::runtime_error(msg)
  {
  }

  template <typename Callback>
  ApiTofError(Callback cb)
      : ApiTofError(call_with_stringstream(cb))
  {
  }
};

namespace Counter
{
enum Counter
{
  nwarnings,
  n_fragmented_total,
  n_escaped_total,
  ncoll_total,
  counter_collision_rejections
};
};
constexpr auto n_counters = enum_count<Counter::Counter>();
using Counters = Eigen::Array<int, n_counters, 1>;
#pragma omp declare reduction(+ : Counters : omp_out = omp_out + omp_in) \
  initializer(omp_priv = Counters::Zero())

struct PartialResult
{
  int thread_id;
  Counters counters;

  PartialResult(Counters counters)
      : thread_id(omp_get_thread_num()), counters(counters)
  {
  }
};

struct LogMessage
{
  enum LogType
  {
    collisions,
    warnings,
    fragments,
    probabilities,
    intenergy,
    tmp,
    tmp_evolution,
    file_energy_distribution,
    final_position,
    pinhole
  };

  LogType type;
  std::string message;

  LogMessage(LogType type, std::string message)
      : type(type), message(message)
  {
  }

  LogMessage(LogType type, const char *message)
      : type(type), message(message)
  {
  }

  template <typename Callback>
  LogMessage(LogType type, Callback cb)
      : type(type), message(call_with_stringstream(cb))
  {
  }
};

struct LogFileWriter
{
  ofstream out_streams[10];

  LogFileWriter(char *file_probabilities)
  {
    if (LOGLEVEL >= LOGLEVEL_MIN)
    {
      this->open(LogMessage::collisions, Filenames::COLLISIONS);
      this->open(LogMessage::warnings, Filenames::WARNINGS, false);
      this->open(LogMessage::fragments, Filenames::FRAGMENTS);
      this->open(LogMessage::probabilities, file_probabilities);
      this->open(LogMessage::intenergy, Filenames::INTENERGY);
      this->open(LogMessage::tmp, Filenames::TMP);
      this->open(LogMessage::tmp_evolution, Filenames::TMP_EVOLUTION);
      this->open(LogMessage::file_energy_distribution, Filenames::ENERGY_DISTRIBUTION);
      this->open(LogMessage::final_position, Filenames::FINAL_POSITION);
      this->open(LogMessage::pinhole, Filenames::PINHOLE);
    }
  }

  void close()
  {
    for (auto &stream : this->out_streams)
    {
      if (stream.is_open())
      {
        stream.close();
      }
    }
  }

private:
  void
  open(LogMessage::LogType type, const char *const filename, bool scientific = true)
  {
    this->out_streams[type].open(filename);
    if (scientific)
    {
      this->out_streams[type] << setprecision(12) << std::scientific;
    }
  }
};

using StreamingResultElement = std::variant<std::monostate, LogMessage, PartialResult, std::exception>;
using StreamingResultQueue = BlockingConcurrentQueue<StreamingResultElement>;

struct WarningHelper
{
  Counters &counters;
  StreamingResultQueue &result_queue;

  template <typename T>
  void operator()(T msg)
  {
    counters[Counter::nwarnings] += 1;
    result_queue.enqueue(LogMessage(LogMessage::warnings, msg));
  }
};

struct LogHelper
{
  StreamingResultQueue &result_queue;
  LogMessage::LogType type;

  template <typename T>
  void operator()(T msg)
  {
    result_queue.enqueue(LogMessage(type, msg));
  }
};

struct Histogram
{
  Eigen::ArrayXd x;
  Eigen::ArrayXd y;
  double bin_width;
  double x_max;

  Histogram(Eigen::ArrayXd x, Eigen::ArrayXd y)
      : x(x), y(y)
  {
    compute_derived();
  }

  void compute_derived()
  {
    bin_width = x[1] - x[0];
    x_max = bin_width * length();
  }

  int length() const
  {
    return x.rows();
  }
};

struct Quadrupole
{
  double dc_field;
  double ac_field;
  double radiofrequency;
  double r_quadrupole;

  double mathieu_factor;
  double angular_velocity;

  Quadrupole(
    double dc_field,
    double ac_field,
    double radiofrequency,
    double r_quadrupole)
      : dc_field(dc_field), ac_field(ac_field), radiofrequency(radiofrequency), r_quadrupole(r_quadrupole)
  {
    angular_velocity = 2.0 * consts::pi * radiofrequency;
  }

  void compute_mathieu_factor(double m_ion)
  {
    mathieu_factor = consts::eV / (m_ion * r_quadrupole * r_quadrupole);
  }
};

// LIST OF FUNCTIONS
// Here we are
double particle_density(double pressure, double kT);
double coll_freq(double n, double mobility_gas, double mobility_gas_inv, double R, double v);
template <typename GenT>
void init_vel(GenT &gen, normal_distribution<double> &gauss, double *v_cluster, double m, double kT);
template <typename GenT>
void init_ang_vel(GenT &gen, normal_distribution<double> &gauss, double *omega, double m, double kT, double R);
template <typename GenT>
void init_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, double kT, const Histogram &density_cluster);
double evaluate_rotational_energy(double *omega, double inertia);
double evaluate_internal_energy(double vib_energy, double rot_energy);
double evaluate_rate_const(const Histogram &rate_const, double energy, WarningHelper warn);
template <typename GenT>
void time_next_coll_quadrupole(GenT &gen, uniform_real_distribution<double> &unif, double rate_constant, double *v_cluster, double &v_cluster_norm, double n1, double n2, double mobility_gas, double mobility_gas_inv, double R, double dt1, double dt2, double &z, double &x, double &y, double &delta_t, double &t_fragmentation, double first_chamber_end, double sk_end, double quadrupole_start, double quadrupole_end, double second_chamber_end, double acc1, double acc2, double acc3, double acc4, double &t, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, std::optional<Quadrupole> quadrupole, LogHelper tmp_evolution);
void update_physical_quantities(double z, const SkimmerData skimmer, double mesh_skimmer, double &v_gas, double &temperature, double &pressure, double &density, double first_chamber_end, double sk_end, double P1, double P2, double n1, double n2, double T);
template <typename GenT>
double draw_theta_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double z, double n1, double n2, double m_gas, double mobility_gas, double mobility_gas_inv, double R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, WarningHelper warn, int mode = 0);
template <typename GenT>
double draw_u_norm_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double z, double du, double boundary_u, double theta, double n1, double n2, double m_gas, double mobility_gas, double mobility_gas_inv, double R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, WarningHelper warn, int mode = 0);
void evaluate_relative_velocity(double z, double *v_cluster, double &v_rel_norm, double v_gas, double *v_rel, double first_chamber_end, double sk_end);
template <typename GenT>
double draw_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double vib_energy_old, const Histogram &density_cluster, double reduced_mass, double u_norm, double v_cluster_norm, double theta, int mode = 0);
void update_velocities(double *v_cluster, double &v_cluster_norm, double *v_rel, double v_gas);
void update_rot_vel(double *omega, double rot_energy_old, double rot_energy);
int mod_func_int(int a, int b);
template <typename GenT>
void redistribute_internal_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, double &rot_energy, const Histogram &density_cluster);
void rescale_density(Eigen::ArrayXd &density, int m_max);
void rescale_energies(Eigen::ArrayXd &energies, int m_max, double &energy_max, double &bin_width);
void eval_velocities(double *v, double *omega, double *u, double vib_energy, double vib_energy_old, double M, double m, double R_cluster);
void change_coord(double *v_cluster, double theta, double phi, double alpha, double *x3, double *y3, double *z3);
template <typename GenT>
void eval_collision(GenT &gen, uniform_real_distribution<double> &unif, bool &collision_accepted, double gas_mean_free_path, double x, double y, double z, double L, double radius_pinhole, double quadrupole_end, double *v_cluster, double *omega, double u_norm, double theta, double R_cluster, double vib_energy, double vib_energy_old, double m_ion, double m_gas, double temperature, LogHelper pinhole);
double vec_norm(double *v);
template <typename GenT>
double onedimMaxwell(GenT &gen, normal_distribution<double> &gauss, double m, double kT);
double mean_free_path(double R, double kT, double pressure);
double evaluate_error(int n, int k);
double eval_solid_angle_stokes(double R, double L, double xx, double yy, double zz);
int zone(double z, double first_chamber_end, double sk_end, double quadrupole_start, double quadrupole_end, double second_chamber_end);

// MAIN PROGRAM
Counters apitof_pinhole(
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
  const int sample_mode = 0)
{
  using namespace consts;
  // TO BE DELETED ###############
  //  R_cluster=4.675e-10;
  // #############################
  double R_gas = gas.radius;
  double m_gas = gas.mass;
  double kT = boltzmann * T;
  double R_tot = R_cluster + R_gas;
  double reduced_mass = 1. / (1. / m_ion + 1. / m_gas);
  double inertia = 0.4 * m_ion * R_cluster * R_cluster;
  double mobility_gas = kT / m_gas; // thermal agitation
  // std_gas=sqrt(mobility_gas);
  double mobility_gas_inv = 1.0 / mobility_gas;
  double boundary_u = 5.0 * sqrt(mobility_gas);
  double du = 1.0e-4 * sqrt(mobility_gas);
  double E1 = -(voltages[1] - voltages[0]) / lengths[0];
  double E2 = -(voltages[2] - voltages[1]) / lengths[1];
  double E3 = -(voltages[3] - voltages[2]) / lengths[2];
  double E4 = -(voltages[4] - voltages[3]) / lengths[3];
  double first_chamber_end = lengths[0];
  double sk_end = first_chamber_end + lengths[SKIMMER_LENGTH];
  double quadrupole_start = sk_end + lengths[1];
  double quadrupole_end = quadrupole_start + lengths[2];
  double second_chamber_end = quadrupole_end + lengths[3];
  double total_length = second_chamber_end;

  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    std::cout << "Physical quantities:" << endl;
    std::cout << "L1: " << first_chamber_end << " m" << endl;
    std::cout << "L2: " << sk_end << " m" << endl;
    std::cout << "L3: " << quadrupole_start << " m" << endl;
    std::cout << "L4: " << quadrupole_end << " m" << endl;
    std::cout << "L5: " << second_chamber_end << " m" << endl;
  }

  auto start = std::chrono::high_resolution_clock::now();

  bonding_energy *= boltzmann; // convert in Joules
  if (quadrupole)
  {
    quadrupole->compute_mathieu_factor(m_ion);
  }
  double acc1 = E1 * consts::eV * cluster_charge_sign / m_ion;
  double acc2 = E2 * consts::eV * cluster_charge_sign / m_ion;
  double acc3 = E3 * consts::eV * cluster_charge_sign / m_ion;
  double acc4 = E4 * consts::eV * cluster_charge_sign / m_ion;
  double P1 = pressure_first;
  double P2 = pressure_second;
  double gas_mean_free_path = mean_free_path(R_gas, kT, P2);
  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    std::cout << "Cluster charge sign: " << cluster_charge_sign << endl;
    std::cout << "Pressure 1st chamber: " << P1 << " Pa" << endl;
    std::cout << "Pressure 2nd chamber: " << P2 << " Pa" << endl;
    std::cout << "E1: " << E1 << " V/m, Acceleration: " << acc1 << " m/s^2" << endl;
    std::cout << "E2: " << E2 << " V/m, Acceleration: " << acc2 << " m/s^2" << endl;
    std::cout << "E3: " << E3 << " V/m, Acceleration: " << acc3 << " m/s^2" << endl;
    std::cout << "E4: " << E4 << " V/m, Acceleration: " << acc4 << " m/s^2" << endl;
  }
  double n1 = particle_density(P1, kT);
  double n2 = particle_density(P2, kT);
  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    std::cout << "Fragmentation energy: " << bonding_energy / boltzmann << " K (" << bonding_energy * kcal << " kcal/mol)" << endl;
    std::cout << "Cluster mass: " << m_ion << " Kg" << endl;
    std::cout << "Inertia momentum: " << inertia << " kg*m^2" << endl;
    std::cout << "Cluster radius: " << R_cluster << " m" << endl;
    std::cout << "Particle density 1st chamber: " << n1 << " 1/m^3" << endl;
    std::cout << "Particle density 2nd chamber: " << n2 << " 1/m^3" << endl;
    std::cout << "Cluster mean free path 1st chamber: " << mean_free_path(R_tot, kT, P1) << " m" << endl;
    std::cout << "Cluster mean free path 2nd chamber: " << mean_free_path(R_tot, kT, P2) << " m" << endl;
    std::cout << "Gas mean free path 1st chamber: " << mean_free_path(R_gas, kT, P1) << " m" << endl;
    std::cout << "Gas mean free path 2nd chamber: " << mean_free_path(R_gas, kT, P2) << " m" << endl;
    std::cout << "Gas density 1st chamber: " << n1 << " 1/m^3" << endl;
    std::cout << "Gas density 2nd chamber: " << n2 << " 1/m^3" << endl;
    std::cout << "Collision frequency 1st chamber (at v=0): " << coll_freq(n1, mobility_gas, mobility_gas_inv, R_tot, 0.0) << " 1/s" << endl;
    std::cout << "Collision frequency 2nd chamber (at v=0): " << coll_freq(n2, mobility_gas, mobility_gas_inv, R_tot, 0.0) << " 1/s" << endl;
    std::cout << "Standard deviation velocity_x: " << sqrt(boltzmann * T / m_ion) << " m/s" << endl;
    std::cout << "R_tot: " << R_tot << " m" << endl;
  }

  // dt1=1.934e-16;
  double dt1 = 1.0e-3 / coll_freq(n1, mobility_gas, mobility_gas_inv, R_tot, 0.0);
  double dt2 = 1.0e-3 / coll_freq(n2, mobility_gas, mobility_gas_inv, R_tot, 0.0);
  if (quadrupole && dt2 > 1.0 / quadrupole->radiofrequency / 1000.0)
    dt2 = 1.0 / quadrupole->radiofrequency / 1000.0;

  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    std::cout << "Time step t1: " << dt1 << " s" << endl;
    std::cout << "Time step t2: " << dt2 << " s" << endl
              << endl;
  }

  Counters counters;

  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    result_queue.enqueue(LogMessage{LogMessage::probabilities, "#1_FragmentationEnergy 2_SurvivalProbability 3_Error\n"});
    result_queue.enqueue(LogMessage{LogMessage::fragments, "#1_Realization 2_Time 3_Position 4_FragmentationZone 5_PositionOfCollision 6_CollisionZone 7_VelocityAtCollision\n"});
  }

  // cout << bin_width_rate << endl;
  //  N realizations
  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    std::cout << "Simulating dynamics... (Fragments *, Intacts -)" << endl;
  }
  // All firstprivate variables *should* be constant within the loop
  // Truly private variables are declared in the loop
  auto loop_start = std::chrono::high_resolution_clock::now();
  OMPExceptionHelper exception_helper;
#pragma omp parallel for default(none) \
  firstprivate( \
      N, T, kT, m_ion, R_cluster, R_tot, density_cluster, rate_const, \
        inertia, second_chamber_end, n1, n2, dt1, dt2, \
        skimmer, mesh_skimmer, total_length, mobility_gas, \
        mobility_gas_inv, gas_mean_free_path, first_chamber_end, root_seed, \
        sk_end, quadrupole_start, quadrupole_end, acc1, acc2, acc3, acc4, \
        P1, P2, bonding_energy, m_gas, quadrupole, du, boundary_u, reduced_mass, pi, \
        sample_mode) \
  shared(exception_helper, result_queue) \
  reduction(+ : counters) \
  schedule(guided)
  for (int j = 0; j < N; j++)
  {
    // shared(nwarnings, collisions, intenergy, warnings, fragments, tmp, tmp_evolution, file_energy_distribution, final_position, pinhole, probabilities, std::cout, exception_helper)
    exception_helper.guard([&]
    {
      WarningHelper warn{counters, result_queue};
      LogHelper fragments{result_queue, LogMessage::fragments};
      LogHelper final_position{result_queue, LogMessage::final_position};
      mt19937 gen = mt19937(root_seed ^ j);
      // Define uniform distribution from 0 to 1
      static uniform_real_distribution<double> unif = uniform_real_distribution<>(0.0, 1.0);
      // Define normal (gaussian) distribution with 0 mean and 1 standard deviation
      static normal_distribution<double> gauss = normal_distribution<>(0.0, 1.0);

      int n_escaped = 0;
      int n_fragmented = 0;

      double t = 0.0;
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      int ncoll = 0;
      double coll_z = 0.0;

      double v_cluster[3];
      double v_rel[3];
      double v_rel_norm;
      double omega[3];
      double v_gas;
      double temperature;
      double density;
      double v_cluster_norm;

      double theta;
      double u_norm; // normal velocity of colliding gas molecule

      double vib_energy = 0.0;
      double rot_energy;

      double rate_constant;
      double delta_t;

      double t_fragmentation;

      // Draw initial random velocity from Maxwell-Boltzmann distribution
      init_vel(gen, gauss, v_cluster, m_ion, kT);
      init_ang_vel(gen, gauss, omega, m_ion, kT, R_cluster);
      init_vib_energy(gen, unif, vib_energy, kT, density_cluster);

      while (z < total_length) // single realization // TO BE CHANGED IN SECOND CHAMBER!!!!!!!!!!!
      {
        int a;
        double vib_energy_old = 0.0;
        double vib_energy_new;
        double rot_energy_old;
        double pressure = 10.0;
        double internal_energy;
        double delta_en;
        const double radius_pinhole = 1.0e-3;
        const int max_coll = 1e6;

        v_cluster_norm = vec_norm(v_cluster);

        // Checking the collision frequencies during the evolution
        // if(z<sk_end) tmp << coll_freq(n1, mobility_gas, mobility_gas_inv, R_tot, v_cluster_norm)<<endl;
        // else tmp << coll_freq(n2, mobility_gas, mobility_gas_inv, R_tot, v_cluster_norm)<<endl;

        rot_energy = evaluate_rotational_energy(omega, inertia);
        internal_energy = evaluate_internal_energy(vib_energy, rot_energy);
        delta_en = internal_energy - bonding_energy;

        // intenergy << j+1 << "\t" << ncoll << "\t" << internal_energy*kcal << endl;
        // intenergy << j+1 << "\t" << ncoll << "\t" << vib_energy/boltzmann << endl;

        a = 0; // variable that check if the cluster fragments when delta_en > energy_max_rate

        if (delta_en > 0.0)
        {
          auto energy_max_rate = rate_const.x_max;
          // tmp << delta_en << endl;
          if (delta_en > energy_max_rate)
          {
            auto overflow = (delta_en - energy_max_rate) / energy_max_rate;
            warn([&overflow](auto &warning)
            {
              warning << "Internal energy exceeds maximum rate energy by " << setprecision(3) << scientific << overflow << endl;
            });
            result_queue.enqueue(LogMessage{LogMessage::probabilities, [&overflow](auto &probabilities)
            {
              probabilities << "# Internal energy exceeds maximum rate energy: " << setprecision(3) << scientific << overflow << endl;
            }});
            delta_en = energy_max_rate;
            a = 1;
          }
          rate_constant = evaluate_rate_const(rate_const, delta_en, warn);
        }
        else
        {
          rate_constant = 0.0;
        }

        time_next_coll_quadrupole(gen, unif, rate_constant, v_cluster, v_cluster_norm, n1, n2, mobility_gas, mobility_gas_inv, R_tot, dt1, dt2, z, x, y, delta_t, t_fragmentation, first_chamber_end, sk_end, quadrupole_start, quadrupole_end, second_chamber_end, acc1, acc2, acc3, acc4, t, m_gas, skimmer, mesh_skimmer, quadrupole, LogHelper{result_queue, LogMessage::tmp_evolution});

        // tmp << kin_energy << "\t";
        // tmp_evolution << delta_t << " " << z << " " << v_cluster[0] << " " << v_cluster[1] << " " << v_cluster[2] << " " << kin_energy << endl;


        // In case we are still in the box
        if (z < total_length)
        {
          // Evaluate if the cluster fragments or not
          if (rate_constant > 0 && delta_t >= t_fragmentation)
          {
            n_fragmented++;
            // if(a==1) cout << "Fragmentation with max energy for rate exceeded. Realization: " << j+1 << endl;
            // if(coll_z>quadrupole_start && coll_z<quadrupole_end)
            if (LOGLEVEL >= LOGLEVEL_NORMAL)
            {
              fragments([&](auto &fragments)
              {
                fragments << j + 1 << "\t" << t << "\t" << z << "\t" << zone(z, first_chamber_end, sk_end, quadrupole_start, quadrupole_end, second_chamber_end) << "\t" << coll_z << "\t" << zone(coll_z, first_chamber_end, sk_end, quadrupole_start, quadrupole_end, second_chamber_end) << endl;
              });
            }
            break;
          }

          if (a == 1)
          {
            {
              throw ApiTofError([&](auto &msg)
              {
                msg << "FATAL ERROR: The internal energy exceeded the max energy related to rate constant (so the cluster should fragment), but the cluster did not fragment. Realization: " << j + 1 << endl
                    << "--> EVALUATE FRAGMENTATION RATE CONSTANT AT HIGHER ENERGIES" << endl
                    << "position= " << scientific << z << endl;
              });
            }
          }

          // Keep track on number of collisions per realization
          ncoll++;
          // cout << "Collision number: " << ncoll << endl;
          // cout << "Position z: " << z << endl;
          // if(z>quadrupole_start && z<quadrupole_end)
          // collisions << j+1 << "\t" << delta_t << "\t" << t << "\t" << x << '\t' << y << "\t" << z << "\t" << ncoll << "\t" << v_cluster_norm << endl;

          // XXX: For some reason these are written after they are read above

          // coll_z = z;

          if (ncoll > max_coll)
          {
            throw ApiTofError([&](auto &warning)
            {
              warning << "Got to the max collisions " << ncoll << " (max is " << max_coll << ")";
            });
          }

          update_physical_quantities(z, skimmer, mesh_skimmer, v_gas, temperature, pressure, density, first_chamber_end, sk_end, P1, P2, n1, n2, T);

          // Draw theta angle of collision
          theta = draw_theta_skimmer(gen, unif, z, n1, n2, m_gas, mobility_gas, mobility_gas_inv, R_tot, v_cluster, v_gas, pressure, temperature, first_chamber_end, sk_end, warn, sample_mode);

          // Draw normal velocity of carrier gas
          u_norm = draw_u_norm_skimmer(gen, unif, z, du, boundary_u, theta, n1, n2, m_gas, mobility_gas, mobility_gas_inv, R_tot, v_cluster, v_gas, pressure, temperature, first_chamber_end, sk_end, warn, sample_mode);

          vib_energy_old = vib_energy;

          evaluate_relative_velocity(z, v_cluster, v_rel_norm, v_gas, v_rel, first_chamber_end, sk_end);

          // Evaluate the dissipated energy in the collision (energy that goes to vibrational modes)

          vib_energy_new = draw_vib_energy(gen, unif, vib_energy_old, density_cluster, reduced_mass, u_norm, v_rel_norm, theta, sample_mode);

          bool collision_accepted = true;
          eval_collision(gen, unif, collision_accepted, gas_mean_free_path, x, y, z, total_length, radius_pinhole, quadrupole_end, v_rel, omega, u_norm, theta, R_cluster, vib_energy_new, vib_energy_old, m_ion, m_gas, temperature, LogHelper{result_queue, LogMessage::pinhole});

          if (collision_accepted)
          {
            vib_energy = vib_energy_new;
            update_velocities(v_cluster, v_cluster_norm, v_rel, v_gas);
            // tmp << kin_energy << endl;

            rot_energy_old = evaluate_rotational_energy(omega, inertia);
            rot_energy = rot_energy_old;
            redistribute_internal_energy(gen, unif, vib_energy, rot_energy, density_cluster);
            update_rot_vel(omega, rot_energy_old, rot_energy);
          }
          else
          {
            counters[Counter::counter_collision_rejections]++;
          }
        }

        else
        {
          if (a == 1)
          {
            throw ApiTofError("FATAL ERROR: The internal energy exceeded the max energy related to rate constant (so the cluster should fragment), but the cluster did not fragment");
          }
          n_escaped++; // Count how many clusters reached the end of the box intact
          if (LOGLEVEL >= LOGLEVEL_NORMAL)
          {
            final_position([&](auto &final_position)
            {
              final_position << x << "\t" << y << endl;
            });
          }
          // cout << "Distance from exit on x: " << x << "and y: " << y << endl; // Distance from the exit on x and y axes
        }
      }

      counters[Counter::n_fragmented_total] += n_fragmented;
      counters[Counter::n_escaped_total] += n_escaped;
      counters[Counter::ncoll_total] += ncoll;

      result_queue.enqueue(PartialResult(counters));

      // if(j%100==0 and j>0) cout << std::defaultfloat << 100.0*j/N << "%" << " Intacts: " << setw(5) << setfill(' ') << n_escaped << " | Fragments: " << setw(5) << setfill (' ') << n_fragmented << " | Survival probability: "  << std::setprecision(3) << 1.0*n_escaped/(n_escaped+n_fragmented)  << endl;
      // if(10*j%N==0)
      // {
      //   c= (int) 10.0*j/N;
      //   counter(c, n_escaped, n_fragmented);
      // }
    });
  }
  exception_helper.rethrow();
  // End of parallel loop

  // realizations = n_fragmented_total + n_escaped_total;

  auto end = std::chrono::high_resolution_clock::now();

  auto loop_time = std::chrono::duration_cast<std::chrono::microseconds>(end - loop_start);
  std::cout << endl
            << "<loop_time>" << loop_time.count() << "</loop_time>" << endl
            << endl;
  auto total_time = end - start;
  auto seconds_tot = std::chrono::duration_cast<std::chrono::seconds>(total_time).count();
  auto microseconds_tot = std::chrono::duration_cast<std::chrono::microseconds>(total_time).count();
  auto hours = (int)(seconds_tot / 3600);
  auto minutes = mod_func_int(seconds_tot / 60, 60);
  auto seconds = mod_func_int(seconds_tot, 60);
  std::cout << "Computational time: " << setw(3) << setfill(' ') << hours << "h" << setw(2) << setfill('0') << minutes << "m" << setw(2) << setfill('0') << seconds << "s" << microseconds_tot << "us" << endl;

  return counters;
}


double evaluate_error(int n, int k)
{
  return sqrt((6.0 * k * k - k * (6.0 + k) * n + (2.0 + k) * n * n) / (n * n * (3.0 + n) * (2.0 + n)));
}

// Geometrical mean of moment of inertia
void compute_inertia(double *rotations, double &inertia_moment)
{
  using namespace consts;
  inertia_moment = 0.5 * hbar * hbar / (boltzmann * pow(rotations[0] * rotations[1] * rotations[2], 1.0 / 3));
}

double vec_norm(double *v)
{
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}


double scalar(double *in1, double *in2)
{
  return in1[0] * in2[0] + in1[1] * in2[1] + in1[2] * in2[2];
}

// Compute cross product of vectors
void cross(double *in1, double *in2, double *out) // PRODOTTO VETTORIALE tra due vettori
{
  out[0] = in1[1] * in2[2] - in1[2] * in2[1];
  out[1] = in1[2] * in2[0] - in1[0] * in2[2];
  out[2] = in1[0] * in2[1] - in1[1] * in2[0];
}

// Compute normalized cross product of vectors
void cross_norm(double *in1, double *in2, double *out) // PRODOTTO VETTORIALE tra due vettori
{
  double norm;

  out[0] = in1[1] * in2[2] - in1[2] * in2[1];
  out[1] = in1[2] * in2[0] - in1[0] * in2[2];
  out[2] = in1[0] * in2[1] - in1[1] * in2[0];
  norm = sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
  // cout << norm << endl;
  if (norm > 0)
  {
    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;
  }
  else
  {
    throw ApiTofError("Zero result in evaluating the cross product");
  }
}


// Compute the mod() function for integers
int mod_func_int(int a, int b)
{
  int r, s;
  if (a < 0)
  {
    s = (int)fmod(a, b);
    if (s < 0)
    {
      r = s + b;
    }
    else
    {
      r = s;
    }
  }
  else
  {
    r = (int)fmod(a, b);
  }
  return r;
};

double particle_density(double pressure, double kT)
{
  using namespace consts;
  return pressure / kT;
}

// Total collision frequency
double coll_freq(double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  if (v > 0)
    return 2.0 * pi * n * R * R * (0.5 * (mobility_gas / v + v) * erf(sqrt(0.5 * mobility_gas_inv) * v) + sqrt(0.5 * mobility_gas / pi) * exp(-0.5 * mobility_gas_inv * v * v));
  else
    return 2.0 * sqrt(2.0 * pi * mobility_gas) * n * R * R;
}


// Collision frequency on angle theta
double coll_freq_theta(double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  double costheta = cos(theta);
  double sintheta = sin(theta);
  return pi * n * R * R * sintheta * (sqrt(mobility_gas * 2.0 / pi) * exp(-0.5 * mobility_gas_inv * v * v * costheta * costheta) + v * costheta * (erf(sqrt(0.5 * mobility_gas_inv) * v * costheta) + 1));
}


// Collision frequency on angle theta and gas velocity
double coll_freq_theta_u(double u, double theta, double n, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  double costheta = cos(theta);
  double sintheta = sin(theta);
  return 2.0 * pi * n * R * R * sqrt(0.5 * mobility_gas_inv / pi) * (u + v * costheta) * exp(-0.5 * mobility_gas_inv * u * u) * sintheta;
}


// Distribution of angle theta
double distr_theta(double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  using namespace consts;
  return coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v) / coll_freq(n, mobility_gas, mobility_gas_inv, R, v);
}


// Distribution of gas velocity
double distr_u(double u, double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v)
{
  return coll_freq_theta_u(u, theta, n, mobility_gas_inv, R, v) / coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v);
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
void init_vel(GenT &gen, normal_distribution<double> &gauss, double *v_cluster, double m, double kT)
{
  v_cluster[0] = onedimMaxwell(gen, gauss, m, kT);
  v_cluster[1] = onedimMaxwell(gen, gauss, m, kT);
  v_cluster[2] = onedimMaxwell(gen, gauss, m, kT);
}

// Inizialize the cluster angular velocity
template <typename GenT>
void init_ang_vel(GenT &gen, normal_distribution<double> &gauss, double *omega, double m, double kT, double R)
{
  omega[0] = onedimMaxwell_angular(gen, gauss, m, R, kT);
  omega[1] = onedimMaxwell_angular(gen, gauss, m, R, kT);
  omega[2] = onedimMaxwell_angular(gen, gauss, m, R, kT);
}


double evaluate_rate_const(const Histogram &rate_const, double energy, WarningHelper warn)
{
  int m;
  double coeff1;
  double coeff2;
  // m=int(energy/bin_width_rate);
  m = int((energy + 0.5 * rate_const.bin_width) / rate_const.bin_width);
  // coeff1=(energy-m*bin_width_rate)/bin_width_rate;
  coeff1 = (energy - (m - 0.5) * rate_const.bin_width) / rate_const.bin_width;
  coeff2 = 1.0 - coeff1;
  if (m >= rate_const.length())
  {
    warn([&energy](auto &warning)
    {
      warning << "delta_energy exceeded upper limit of rate_constant evaluation: delta_energy= " << energy << endl;
    });
    return rate_const.y[rate_const.length() - 1];
  }
  else if (m > 0)
  {
    return coeff2 * rate_const.y[m - 1] + coeff1 * rate_const.y[m];
  }
  else if (m == 0)
  {
    return rate_const.y[0];
  }
  else
  {
    warn([&energy](auto &warning)
    {
      warning << "Rate constant evaluation failed: delta_energy= " << energy << endl;
    });
    return 0;
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
    pressure = skimmer(m, TEMP_SKIMMER);
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

void update_physical_quantities(double z, const SkimmerData skimmer, double mesh_skimmer, double &v_gas, double &temperature, double &pressure, double &density, double first_chamber_end, double sk_end, double P1, double P2, double n1, double n2, double T)
{
  int m;
  double coeff1;
  double coeff2;
  double position;

  if (z < first_chamber_end)
  {
    density = n1;
    pressure = P1;
    temperature = T;
    v_gas = 0;
  }
  else if (z < sk_end)
  {
    position = z - first_chamber_end;
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
    density = n2;
    pressure = P2;
    temperature = T;
    v_gas = 0;
  }
}

// Draw initial vibrational energy
template <typename GenT>
void init_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, double kT, const Histogram &density_cluster)
{
  double sum1 = 0.0;
  double sum2 = 0.0;
  double r = unif(gen);
  int m;

  for (m = 0; m < density_cluster.length(); m++)
  {
    sum1 += density_cluster.y[m] * exp(-density_cluster.x[m] / kT);
  }

  m = 0;
  while (sum2 < r)
  {
    sum2 += density_cluster.y[m] * exp(-density_cluster.x[m] / kT) / sum1;
    m++;
  }
  vib_energy = density_cluster.x[m - 1];
}

// Evaluate time to next collision
template <typename GenT>
void time_next_coll_quadrupole(GenT &gen, uniform_real_distribution<double> &unif, double rate_constant, double *v_cluster, double &v_cluster_norm, double n1, double n2, double mobility_gas, double mobility_gas_inv, double R, double dt1, double dt2, double &z, double &x, double &y, double &delta_t, double &t_fragmentation, double first_chamber_end, double sk_end, double quadrupole_start, double quadrupole_end, double second_chamber_end, double acc1, double acc2, double acc3, double acc4, double &t, double m_gas, const SkimmerData &skimmer, double mesh_skimmer, std::optional<Quadrupole> quadrupole, LogHelper tmp_evolution)
{
  using namespace consts;
  double integral = 0.0;
  double P = 1.0;
  double c1;
  double c2;
  double v1;
  double dt;
  double v_cluster_norm_xy = v_cluster[0] * v_cluster[0] + v_cluster[1] * v_cluster[1];
  double r = unif(gen);
  double r2 = unif(gen);
  double mobility_gas_skimmer;
  double mobility_gas_inv_skimmer;
  double T_skimmer;
  double kT_skimmer;
  double P_skimmer;
  double n_skimmer = NAN;
  double v_gas;
  double v_rel_norm;
  double v1x;
  double v1y;
  double accx;
  double accy;

  delta_t = 0.0;
  v_cluster_norm = sqrt(v_cluster[0] * v_cluster[0] + v_cluster[1] * v_cluster[1] + v_cluster[2] * v_cluster[2]);

  if (z < first_chamber_end) // In first chamber
  {
    c1 = coll_freq(n1, mobility_gas, mobility_gas_inv, R, v_cluster_norm);
  }
  else if (z > sk_end) // In the second chamber
  {
    c1 = coll_freq(n2, mobility_gas, mobility_gas_inv, R, v_cluster_norm);
  }
  else // In the skimmer
  {
    update_skimmer_quantities(skimmer, z, first_chamber_end, mesh_skimmer, v_gas, T_skimmer, P_skimmer);
    kT_skimmer = boltzmann * T_skimmer;
    mobility_gas_skimmer = boltzmann * T_skimmer / m_gas;
    mobility_gas_inv_skimmer = 1.0 / mobility_gas_skimmer;
    n_skimmer = particle_density(P_skimmer, kT_skimmer);
    v_rel_norm = sqrt(v_cluster_norm_xy + pow(v_cluster[2] - v_gas, 2));
    c1 = coll_freq(n_skimmer, mobility_gas_skimmer, mobility_gas_inv_skimmer, R, v_rel_norm);
  }

  // tmp_evolution << z << " " << c1 << endl;
  // if(z<first_chamber_end) tmp_evolution << z << " " << c1 << endl;

  if (rate_constant > 0)
  {
    t_fragmentation = -log(r) / rate_constant;
  }
  else
  {
    t_fragmentation = 1.0e10; // Set huge fragmentation time for no fragmentation happening
  }
  while (r2 < P && z < second_chamber_end && delta_t < t_fragmentation)
  {
    v1 = v_cluster[2];
    v1x = v_cluster[0];
    v1y = v_cluster[1];

    if (z < first_chamber_end)
    {
      v_cluster[2] += acc1 * dt1;
    }

    else if (z >= sk_end and z < quadrupole_start)
    {
      v_cluster[2] += acc2 * dt2;
    }

    else if (z >= quadrupole_start and z < quadrupole_end)
    {
      if (quadrupole)
      {
        accx = quadrupole->mathieu_factor * (-quadrupole->dc_field + quadrupole->ac_field * cos(quadrupole->angular_velocity * t)) * (x + v_cluster[0] * dt2 / 2.0);
        accy = quadrupole->mathieu_factor * (quadrupole->dc_field - quadrupole->ac_field * cos(quadrupole->angular_velocity * t)) * (y + v_cluster[1] * dt2 / 2.0);
        v_cluster[0] += accx * dt2;
        v_cluster[1] += accy * dt2;
      }
      v_cluster[2] += acc3 * dt2;
    }

    else if (z >= quadrupole_end)
    {
      v_cluster[2] += acc4 * dt2;
    }

    v_cluster_norm = sqrt(v_cluster[0] * v_cluster[0] + v_cluster[1] * v_cluster[1] + v_cluster[2] * v_cluster[2]);

    if (z < first_chamber_end) // Dynamics in the 1st chamber
    {
      c2 = coll_freq(n1, mobility_gas, mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dt1 / 2.0;
      P = exp(-integral);
      delta_t += dt1;
      x += v1x * dt1;
      y += v1y * dt1;
      z += (v1 + v_cluster[2]) * dt1 / 2.0;
      t += dt1;
    }

    else if (z > sk_end and z < quadrupole_start) // Dynamics in the 2nd chamber
    {
      c2 = coll_freq(n2, mobility_gas, mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dt2 / 2.0;
      P = exp(-integral);
      delta_t += dt2;
      x += v1x * dt2;
      y += v1y * dt2;
      z += (v1 + v_cluster[2]) * dt2 / 2.0;
      t += dt2;
    }

    else if (z >= quadrupole_start and z < quadrupole_end) // Dynamics in the 2nd chamber
    {
      c2 = coll_freq(n2, mobility_gas, mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dt2 / 2.0;
      P = exp(-integral);
      delta_t += dt2;
      x += (v1x + v_cluster[0]) * dt2 / 2.0;
      y += (v1y + v_cluster[1]) * dt2 / 2.0;
      z += (v1 + v_cluster[2]) * dt2 / 2.0;
      t += dt2;
    }
    else if (z >= quadrupole_end) // Dynamics in the 2nd chamber
    {
      c2 = coll_freq(n2, mobility_gas, mobility_gas_inv, R, v_cluster_norm);
      integral += (c1 + c2) * dt2 / 2.0;
      P = exp(-integral);
      delta_t += dt2;
      x += v1x * dt2;
      y += v1y * dt2;
      z += (v1 + v_cluster[2]) * dt2 / 2.0;
      t += dt2;
    }

    else // Dynamics in the skimmer
    {
      update_skimmer_quantities(skimmer, z, first_chamber_end, mesh_skimmer, v_gas, T_skimmer, P_skimmer);
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
  if (LOGLEVEL >= LOGLEVEL_NORMAL)
  {
    if (z < first_chamber_end)
    {
      tmp_evolution([&](auto &tmp_evolution)
      {
        tmp_evolution << z << " " << delta_t << " " << v_gas << " " << v_cluster_norm << " " << n_skimmer << endl;
      });
    }
  }
}


// Draw theta angle of collision UPDATED
template <typename GenT>
double draw_theta_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double z, double n1, double n2, double m_gas, double mobility_gas, double mobility_gas_inv, double R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, WarningHelper warn, int mode)
{
  using namespace consts;
  double r = unif(gen);
  double n;
  double v_rel_norm;

  if (z < first_chamber_end)
  {
    n = n1;
    v_rel_norm = vec_norm(v_cluster);
  }
  else if (z < sk_end)
  {
    double v_rel[3];
    v_rel[0] = v_cluster[0];
    v_rel[1] = v_cluster[1];
    v_rel[2] = v_cluster[2] - v_gas;
    v_rel_norm = vec_norm(v_rel);
    double kT = boltzmann * temperature;
    mobility_gas = kT / m_gas;
    mobility_gas_inv = 1.0 / mobility_gas;
    n = particle_density(pressure, kT);
  }
  else
  {
    n = n2;
    v_rel_norm = vec_norm(v_cluster);
  }

  const double dtheta = 1.0e-3;
  double theta = 0.0;
  if (mode == 1)
  {
    double integral_unnorm = 0.0;
    double normalization = coll_freq(n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
    double r_unnorm = r * normalization;
    while (integral_unnorm < r_unnorm)
    {
      double c = coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
      integral_unnorm += c * dtheta;
      theta += dtheta;
    }
  }
  else
  {
    double integral = 0.0;
    while (r > integral && theta < pi)
    {
      double c = distr_theta(theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
      integral += c * dtheta;
      theta += dtheta;
    }
  }
  if (theta > pi)
  {
    theta = pi - 1.0e-3;
    warn([&r](auto &warning)
    {
      warning << "theta exceeded pi. random number r is: " << r << endl;
    });
  }
  return theta;
}
// Draw translational energy of cluster after the impact with carrier gas
// Here we are considering a constant density of states for vibrational mode, i.e. a single vibration (simplified model)
template <typename GenT>
double draw_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double vib_energy_old, const Histogram &density_cluster, double reduced_mass, double u_norm, double v_cluster_norm, double theta, int mode)
{
  using consts::boltzmann;

  double relative_speed = u_norm + v_cluster_norm * cos(theta);
  double E = vib_energy_old + reduced_mass * 0.5 * relative_speed * relative_speed;

  if (E > density_cluster.x_max)
  {
    throw ApiTofError([&](auto &warning)
    {
      warning << "Energy is exceeding the density of states file. E: " << E / boltzmann << endl;
    });
  }

  // 1st step: I evaluate the integral (normalization)
  double normalization = 0.0;
  int m = 0;
  while (density_cluster.x[m] < E)
  {
    normalization += sqrt(E - density_cluster.x[m]) * density_cluster.y[m];
    m++;
  }

  // 2nd step: I evaluate the random transferred energy to the cluster
  m = 0;
  double r = unif(gen);
  if (mode == 1)
  {
    double r_unnorm = r * normalization;
    double integral_unnorm = 0.0;
    while (integral_unnorm < r_unnorm)
    {
      integral_unnorm += sqrt(E - density_cluster.x[m]) * density_cluster.y[m];
      m++;
    }
  }
  else
  {
    double integral = 0.0;
    while (integral < r)
    {
      integral += sqrt(E - density_cluster.x[m]) * density_cluster.y[m] / normalization;
      m++;
    }
  }
  return density_cluster.x[m - 1];
}

// Redistribution of internal energy (between vibrational and rotational modes)
template <typename GenT>
void redistribute_internal_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, double &rot_energy, const Histogram &density_cluster)
{
  using consts::boltzmann;
  double r = unif(gen);
  double E = vib_energy + rot_energy;
  double integral = 0.0;
  double integral2 = 0.0;
  int m;


  if (E > density_cluster.x_max)
  {
    throw ApiTofError([&](auto &msg)
    {
      msg << "Energy is exceeding the density of states file. E: " << E / boltzmann << endl;
    });
  }

  // 1st step: I evaluate the integral (normalization)
  m = 0;
  while (density_cluster.x[m] < E)
  {
    if (E - density_cluster.x[m] < 0)
      std::cout << "ERROR!!" << endl
                << endl;
    integral += sqrt(E - density_cluster.x[m]) * density_cluster.y[m];
    m++;
  }

  // 2nd step: I evaluate the random transferred energy to the cluster
  m = 0;
  while (integral2 < r)
  {
    if (E - density_cluster.x[m] < 0)
      std::cout << "ERROR!!" << endl
                << endl;
    integral2 += sqrt(E - density_cluster.x[m]) * density_cluster.y[m] / integral;
    m++;
  }
  vib_energy = density_cluster.x[m - 1];
  rot_energy = E - vib_energy;
  // cout << vib_energy<< " " << rot_energy<<endl<<endl;
}


// Update angular velocity after redistribution of vibrational and rotational energy
void update_rot_vel(double *omega, double rot_energy_old, double rot_energy)
{
  omega[0] = omega[0] * sqrt(rot_energy / rot_energy_old);
  omega[1] = omega[1] * sqrt(rot_energy / rot_energy_old);
  omega[2] = omega[2] * sqrt(rot_energy / rot_energy_old);
}

// Draw normal velocity of carrier gas
template <typename GenT>
double draw_u_norm_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double z, double du, double boundary_u, double theta, double n1, double n2, double m_gas, double mobility_gas, double mobility_gas_inv, double R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, WarningHelper warn, int mode)
{
  using namespace consts;
  double n;
  double v_rel_norm;
  double kT;

  if (z < first_chamber_end)
  {
    n = n1;
    v_rel_norm = vec_norm(v_cluster);
  }
  else if (z < sk_end)
  {
    double v_rel[3];
    v_rel[0] = v_cluster[0];
    v_rel[1] = v_cluster[1];
    v_rel[2] = v_cluster[2] - v_gas;
    v_rel_norm = vec_norm(v_rel);
    kT = boltzmann * temperature;
    mobility_gas = kT / m_gas;
    mobility_gas_inv = 1.0 / mobility_gas;
    n = particle_density(pressure, kT);
  }
  else
  {
    n = n2;
    v_rel_norm = vec_norm(v_cluster);
  }

  double u_norm;
  if (mode == 1)
  {
    double costheta = cos(theta);
    double sintheta = sin(theta);
    if (v_rel_norm * costheta > boundary_u)
    {
      u_norm = -boundary_u;
    }
    else
    {
      u_norm = -v_rel_norm * costheta;
    }

    double r = unif(gen);
    double normalization = coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
    double common_factor = 2.0 * pi * n * R * R * sqrt(0.5 * mobility_gas_inv / pi) * sintheta;
    double r_unnorm = r * normalization / common_factor;
    double integral_unnorm = 0.0;
    while (integral_unnorm < r_unnorm)
    {
      double c = (u_norm + v_rel_norm * costheta) * exp(-0.5 * mobility_gas_inv * u_norm * u_norm);
      integral_unnorm += c * du;
      u_norm += du;
    }

    if (u_norm > boundary_u)
    {
      u_norm = boundary_u;
      warn([&](auto &warning)
      {
        warning << "u_norm exceeded boundary of the integration. random number r is: " << r << endl;
      });
    }
  }
  else
  {
    double costheta = cos(theta);
    if (v_rel_norm * costheta > boundary_u)
    {
      u_norm = -boundary_u;
    }
    else
    {
      u_norm = -v_rel_norm * costheta;
    }

    double r = unif(gen);
    double integral = 0.0;
    while (r > integral && u_norm < boundary_u)
    {
      double c = distr_u(u_norm, theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
      integral += c * du;
      u_norm += du;
    }

    if (u_norm > boundary_u)
    {
      warn([&](auto &warning)
      {
        warning << "u_norm exceeded boundary of the integration. random number r is: " << r << endl;
      });
    }
  }
  return u_norm;
}


void update_v_cluster_norm(double *v_cluster, double &v_cluster_norm)
{
  v_cluster_norm = sqrt(v_cluster[0] * v_cluster[0] + v_cluster[1] * v_cluster[1] + v_cluster[2] * v_cluster[2]);
}

// Evaluate (approximation of) kinetic energy of crashing gas molecule
double evaluate_energy_collision(double *v, double *omega, double inertia, double m_ion)
{
  double v_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  double omega_squared = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
  return 0.5 * (m_ion * v_squared + inertia * omega_squared);
}

// Evaluate internal energy (rotational+vibrational)
double evaluate_internal_energy(double vib_energy, double rot_energy)
{
  return rot_energy + vib_energy;
}

// Evaluate rotational energy
double evaluate_rotational_energy(double *omega, double inertia)
{
  double omega_squared = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
  return 0.5 * inertia * omega_squared;
}

// Mean free path
double mean_free_path(double R, double kT, double pressure)
{
  using consts::pi;
  return kT / (sqrt(2.0) * pi * 4.0 * R * R * pressure);
}


double energy_in_eV(double energy)
{
  return energy / 1.602e-19;
}

void evaluate_relative_velocity(double z, double *v_cluster, double &v_rel_norm, double v_gas, double *v_rel, double first_chamber_end, double sk_end)
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
  v_rel_norm = vec_norm(v_rel);
}

void update_velocities(double *v_cluster, double &v_cluster_norm, double *v_rel, double v_gas)
{
  v_cluster[0] = v_rel[0];
  v_cluster[1] = v_rel[1];
  v_cluster[2] = v_rel[2] + v_gas;
  v_cluster_norm = vec_norm(v_cluster);
}


// Evaluate the velocities after collision in the rotated reference system
void eval_velocities(double *v, double *omega, double *u, double vib_energy, double vib_energy_old, double M, double m, double R_cluster)
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
    throw ApiTofError([&](auto &msg)
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
void change_coord(double *v_cluster, double theta, double phi, double alpha, double *x3, double *y3, double *z3)
{
  using consts::pi;
  double v_cluster_norm = vec_norm(v_cluster);
  double x[3] = {1.0, 0.0, 0.0};
  double y[3] = {0.0, 1.0, 0.0};
  double x1[3];
  double y1[3];
  double z1[3];
  double x2[3];
  double y2[3];
  double z2[3];
  double foo[3];

  // check if v_cluster is null
  if (v_cluster_norm > 0)
  {
    for (int i = 0; i < 3; i++)
    {
      z1[i] = v_cluster[i] / v_cluster_norm;
    }
  }
  else
  {
    z1[0] = 0.0;
    z1[1] = 0.0;
    z1[2] = 1.0;
  }

  // build reference system with v_cluster aligned to z1 versor
  cross(z1, x, foo);
  if (vec_norm(foo) != 0.0)
  {
    cross_norm(z1, x, y1);
    cross_norm(y1, z1, x1);
  }
  else
  {
    cross_norm(y, z1, x1);
    cross_norm(z1, x1, y1);
  }

  // build reference of system centered in point of collision (x2,y2,z2)
  if (theta > 0 and theta < pi)
  {
    for (int i = 0; i < 3; i++)
    {
      z2[i] = sin(theta) * cos(phi) * x1[i] + sin(theta) * sin(phi) * y1[i] + cos(theta) * z1[i];
    }
    cross_norm(z2, z1, x2);
    cross_norm(z2, x2, y2);
  }
  else if (theta == 0.0)
  {
    for (int i = 0; i < 3; i++)
    {
      z2[i] = z1[i];
    }
    cross_norm(z2, x1, y2);
    cross_norm(y2, z2, x2);
  }
  else if (theta == pi)
  {
    for (int i = 0; i < 3; i++)
    {
      z2[i] = -z1[i];
    }
    cross_norm(z2, x1, y2);
    cross_norm(y2, z2, x2);
  }
  else
  {
    std::cout << endl
              << endl
              << "ERROR in defining reference system at theta: " << theta << endl
              << endl;
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
void eval_collision(GenT &gen, uniform_real_distribution<double> &unif, bool &collision_accepted, double gas_mean_free_path, double x, double y, double z, double L, double radius_pinhole, double quadrupole_end, double *v_cluster, double *omega, double u_norm, double theta, double R_cluster, double vib_energy, double vib_energy_old, double m_ion, double m_gas, double temperature, LogHelper pinhole)
{
  using namespace consts;
  double x3[3];
  double y3[3];
  double z3[3];
  double v2[3];
  double omega2[3];
  double phi = 2.0 * pi * unif(gen);
  double alpha = 2.0 * pi * unif(gen);
  double kT = boltzmann * temperature;
  double u[2];
  double velocity_gas[3];
  double target[2];
  bool inside_target = false;
  double prob_coll = 1.0;
  double distance;

  collision_accepted = true;
  change_coord(v_cluster, theta, phi, alpha, x3, y3, z3);


  v2[0] = scalar(v_cluster, x3);
  v2[1] = scalar(v_cluster, y3);
  v2[2] = scalar(v_cluster, z3);


  omega2[0] = scalar(omega, x3);
  omega2[1] = scalar(omega, y3);
  omega2[2] = scalar(omega, z3);


  // Normal component of air molecule velocity
  u[0] = -u_norm;
  // Tangential component of air molecule velocity
  u[1] = twodimMaxwell(gen, unif, m_gas, kT);
  // cout << kT << endl;
  if (u[0] > v2[2])
  {
    std::cout << endl
              << endl
              << "ERROR: relative velocities prevent collision!" << endl
              << endl;
  }


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
    pinhole([&](auto &pinhole)
    {
      pinhole << x << " " << y << " " << z << " " << velocity_gas[0] << " " << velocity_gas[1] << " " << velocity_gas[2] << " " << inside_target << endl;
    });
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

  if (collision_accepted) // Normal procedure
  {
    // Express new velocities in lab reference system
    for (int i = 0; i < 3; i++)
    {
      v_cluster[i] = v2[0] * x3[i] + v2[1] * y3[i] + v2[2] * z3[i];
      omega[i] = omega2[0] * x3[i] + omega2[1] * y3[i] + omega2[2] * z3[i];
    }

    eval_velocities(v2, omega2, u, vib_energy, vib_energy_old, m_ion, m_gas, R_cluster);
    // Express new velocities in lab reference system
    for (int i = 0; i < 3; i++)
    {
      v_cluster[i] = v2[0] * x3[i] + v2[1] * y3[i] + v2[2] * z3[i];
      omega[i] = omega2[0] * x3[i] + omega2[1] * y3[i] + omega2[2] * z3[i];
    }
  }
}

double modulus_squared(double *x)
{
  return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
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

int zone(double z, double first_chamber_end, double sk_end, double quadrupole_start, double quadrupole_end, double second_chamber_end)
{
  if (z < first_chamber_end)
    return 1;
  else if (z < sk_end)
    return 2;
  else if (z < quadrupole_start)
    return 3;
  else if (z < quadrupole_end)
    return 4;
  else if (z <= second_chamber_end)
    return 5;
  else
    return 9999999;
}
