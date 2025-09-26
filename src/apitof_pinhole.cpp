// CODE WITHOUT DSMC CORRECTIONS

// TO DO LIST:
// - pre-processing compilation (with flags)
// - HEADER FILE
// - Prandtl Meyer maximum turning angle (skimmer dynamics)
// - Free expansion gas Montecarlo at skimmer (DSMC - Direct Simulation Monte Carlo)
// - Electric field at electrode plates position
// - Expand the code with the possibility of multiple fragmentations
// - The multiple fragmentation brings some issues: once the first cluster is broken, you need to calculate the internal energy of the product cluster of your interest and the new momentum acquired by the "explosion".

// ratio_masses, kin_energy, std_gas, (err_down=)x --> useless variables

// QUESTIONS:
// - Does the cluster reach the distribution at equilibrium at temperature T? Seems that the theoretical distribution of internal energy does not match the one from simulation...
// Change the initial energy of clusters and see how the fragmentation probability changes

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <tuple>
#include "utils.h"

#define pi 3.14159265
#define eV 1.602176565e-19
#define boltzmann 1.38064852e-23
#define pmass 1.6726219e-27
#define hartK 3.157732e+5 // 1 hartree in Kelvin
#define kcal 1.439325e+20 // 1 Joule in kcal/mol

// Define Pi if it is not already defined
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

// FOR WINDOWS
// #define srand48(x) srand(x)
// #define drand48() ((double)rand()/RAND_MAX)

using namespace std;

typedef Eigen::Array<double, Eigen::Dynamic, 3> SkimmerData;
const int VEL_SKIMMER = 0;
const int TEMP_SKIMMER = 1;
const int PRESSURE_SKIMMER = 2;

class ApiTofError : public std::runtime_error
{
public:
  ApiTofError(const std::string &msg) : std::runtime_error(msg)
  {
  }

  ApiTofError(const char *msg) : std::runtime_error(msg)
  {
  }

  template <typename Callback>
  ApiTofError(Callback cb) : ApiTofError(call_with_stringstream(cb))
  {
  }

private:
  template <typename Callback>
  std::string call_with_stringstream(Callback cb)
  {
    std::string result;
    stringstream ss(result);
    ss << std::scientific << std::setprecision(3);
    cb(ss);
    return result;
  }
};

// LIST OF FUNCTIONS
// Here we are
void read_electronic_energy(char *filename, double &electronic_energy);
float particle_density(float pressure, float kT);
double coll_freq(float n, float mobility_gas, float mobility_gas_inv, float R, double v);
template <typename GenT>
void init_vel(GenT &gen, normal_distribution<double> &gauss, double *v_cluster, float m, float kT);
template <typename GenT>
void init_ang_vel(GenT &gen, normal_distribution<double> &gauss, double *omega, float m, float kT, float R);
template <typename GenT>
void init_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, float kT, const Eigen::ArrayXd &density_cluster, const Eigen::ArrayXd &energies_density, int m_max_density);
double evaluate_rotational_energy(double *omega, float inertia);
double evaluate_internal_energy(double vib_energy, double rot_energy);
double evaluate_rate_const(const Eigen::ArrayXd &rate_const, double energy, double bin_width_rate, int m_max_rate, ofstream &warnings, int &nwarnings);
template <typename GenT>
void time_next_coll_quadrupole(GenT &gen, uniform_real_distribution<double> &unif, double rate_constant, double *v_cluster, double &v_cluster_norm, float n1, float n2, float mobility_gas, float mobility_gas_inv, float R, double dt1, double dt2, double &z, double &x, double &y, double &delta_t, double &t_fragmentation, float first_chamber_end, float sk_end, float quadrupole_start, float quadrupole_end, float second_chamber_end, float acc1, float acc2, float acc3, float acc4, double &t, float m_gas, const SkimmerData &skimmer, double mesh_skimmer, double angular_velocity, double mathieu_factor, double dc_field, double ac_field, ofstream &tmp_evolution);
void update_physical_quantities(double z, const SkimmerData skimmer, double mesh_skimmer, double &v_gas, double &temperature, double &pressure, double &density, float first_chamber_end, float sk_end, float P1, float P2, float n1, float n2, float T);
template <typename GenT>
void draw_theta_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double &theta, double z, float n1, float n2, float m_gas, float mobility_gas, float mobility_gas_inv, float R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, ofstream &warnings, int &nwarnings);
void update_param(double &v_cluster_norm, double *v_cluster, double *v_cluster_versor, double theta, double phi, double &sintheta, double &costheta, double &sinphi, double &cosphi);
template <typename GenT>
void draw_u_norm_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double z, double du, double boundary_u, double &u_norm, double theta, float n1, float n2, float m_gas, float mobility_gas, float mobility_gas_inv, float R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, double costheta, ofstream &warnings, int &nwarnings);
void evaluate_relative_velocity(double z, double *v_cluster, double &v_rel_norm, double v_gas, double *v_rel, double first_chamber_end, double sk_end);
template <typename GenT>
double draw_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double vib_energy_old, const Eigen::ArrayXd &density_cluster, const Eigen::ArrayXd &energies_density, double energy_max_density, float reduced_mass, double u_norm, double v_cluster_norm, double theta, ofstream &warnings, int &nwarnings);
void update_velocities(double *v_cluster, double &v_cluster_norm, double *v_rel, double v_gas);
void update_rot_vel(double *omega, double rot_energy_old, double rot_energy);
int mod_func_int(int a, int b);
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, int, double, double> read_histogram(char *filename);
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd, int, double, double> read_two_density_histograms(char *file_density_cluster, char *file_density_combined_products);
template <typename GenT>
void redistribute_internal_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, double &rot_energy, const Eigen::ArrayXd &density_cluster, const Eigen::ArrayXd &energies_density, double energy_max_density, ofstream &warnings, int &nwarnings);
void rescale_density(Eigen::ArrayXd &density, int m_max);
void rescale_energies(Eigen::ArrayXd &energies, int m_max, double &energy_max, double &bin_width);
std::tuple<SkimmerData, double> read_skimmer(char *filename);
void eval_velocities(double *v, double *omega, double *u, double vib_energy, double vib_energy_old, float M, float m, double R_cluster, ofstream &warnings, int &nwarnings);
void change_coord(double *v_cluster, double theta, double phi, double alpha, double *x3, double *y3, double *z3);
template <typename GenT>
void eval_collision(GenT &gen, uniform_real_distribution<double> &unif, bool &collision_accepted, double gas_mean_free_path, double x, double y, double z, double L, double radius_pinhole, float quadrupole_end, double *v_cluster, double *omega, double u_norm, double theta, float R_cluster, double vib_energy, double vib_energy_old, float m_ion, float m_gas, float temperature, ofstream &pinhole, ofstream &warnings);
double vec_norm(double *v);
template <typename GenT>
double onedimMaxwell(GenT &gen, normal_distribution<double> &gauss, float m, float kT);
double mean_free_path(float R, float kT, float pressure);
double evaluate_error(int n, int k);
double polar_function(double phi, double theta1, double theta2);
double eval_solid_angle_stokes(double R, double L, double xx, double yy, double zz);
int zone(double z, float first_chamber_end, float sk_end, float quadrupole_start, float quadrupole_end, float second_chamber_end);

// MAIN PROGRAM

void apitof_pinhole()
{
  // Mersenne-Twister uniform random number generator
  mt19937 root_gen = mt19937(42ull);
  unsigned long long root_seed = root_gen();

  int hours;
  int seconds;
  int minutes;

  float L0;
  float Lsk;
  float L1;
  float L2;
  float L3;
  float first_chamber_end;
  float sk_end;
  float quadrupole_start;
  float quadrupole_end;
  float second_chamber_end;
  float V0;
  float V1;
  float V2;
  float V3;
  float V4;
  float E1;
  float E2;
  float E3;
  float E4;
  float acc1;
  float acc2;
  float acc3;
  float acc4;
  float kT;
  float T;
  float P1;
  float P2;
  float n1;
  float n2;
  double R_cluster;
  double R_gas;
  double R_tot;
  double m_ion;
  double m_gas;
  double reduced_mass;
  float q = 1.602e-19; // Coulombs
  float mobility_gas; // thermal agitation
  // float std_gas;
  float mobility_gas_inv;
  double dt1;
  double dt2;
  float survival_probability;
  float error_survival_probability;
  float pressure_first;
  float pressure_second;
  double bonding_energy;

  double du;
  double boundary_u;

  // double rate_const;
  double total_length;
  double mathieu_factor;
  double r_quadrupole;
  double radiofrequency;
  double angular_velocity;
  double dc_field;
  double ac_field;
  double gas_mean_free_path;
  int counter_collision_rejections = 0;

  int j;
  int ncoll_total = 0;
  int n_escaped_total;
  int n_fragmented_total;
  int N;
  int nwarnings = 0;
  int realizations;
  int amu;
  int cluster_charge_sign;

  char file_rate_const[150];
  char file_density_cluster[150];
  char file_density_combined_products[150];
  char file_skimmer[150];
  char file_rotations[150];
  char file_electronic_energy_0[150];
  char file_electronic_energy_1[150];
  char file_electronic_energy_2[150];
  char file_probabilities[150];

  // Set scientific notation
  std::cout << std::scientific << std::setprecision(3);

  // MANAGE FILES
  ofstream collisions;
  ofstream warnings;
  ofstream fragments;
  ofstream probabilities;
  ofstream intenergy;
  ofstream tmp;
  ofstream tmp_evolution;
  ofstream file_energy_distribution;
  ofstream final_position;
  ofstream pinhole;


  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    collisions.open(Filenames::COLLISIONS);
    collisions << setprecision(12) << std::scientific;
    intenergy.open(Filenames::INTENERGY);
    intenergy << setprecision(12) << std::scientific;
    warnings.open(Filenames::WARNINGS);
    fragments.open(Filenames::FRAGMENTS);
    fragments << setprecision(12) << std::scientific;
    tmp.open(Filenames::TMP);
    tmp << setprecision(12) << std::scientific;
    tmp_evolution.open(Filenames::TMP_EVOLUTION);
    tmp_evolution << setprecision(12) << std::scientific;
    file_energy_distribution.open(Filenames::ENERGY_DISTRIBUTION);
    file_energy_distribution << setprecision(12) << scientific;
    final_position.open(Filenames::FINAL_POSITION);
    final_position << setprecision(12) << std::scientific;
    pinhole.open(Filenames::PINHOLE);
    pinhole << setprecision(12) << std::scientific;
  }

  // SEED OF RANDOM NUMBERS GENERATOR
  // srand48(2);

  // READING THE INPUT FILE
  std::cout << endl
            << "Reading input..." << endl
            << endl;

  read_config(
    std::cin,
    nullptr, // title
    &cluster_charge_sign,
    &amu,
    (int *)nullptr, // amu_1
    (int *)nullptr, // amu_2
    &T,
    &pressure_first,
    &pressure_second,
    &L0,
    &Lsk,
    &L1,
    &L2,
    &L3,
    &V0,
    &V1,
    &V2,
    &V3,
    &V4,
    &N,
    nullptr, // dc
    nullptr, // alpha_factor
    &bonding_energy,
    nullptr,
    nullptr,
    nullptr,
    &R_gas,
    &m_gas,
    nullptr, // ga
    &dc_field,
    &ac_field,
    &radiofrequency,
    &r_quadrupole,
    file_skimmer,
    nullptr, // file_frequencies_0
    nullptr, // file_frequencies_1
    nullptr, // file_frequencies_2
    file_rotations,
    nullptr, // file_rotations_1
    nullptr, // file_rotations_2
    file_electronic_energy_0,
    file_electronic_energy_1,
    file_electronic_energy_2,
    file_density_cluster,
    nullptr,
    nullptr,
    file_density_combined_products,
    file_rate_const,
    file_probabilities,
    nullptr, // N_iter
    nullptr, // M_iter
    nullptr, // resolution
    nullptr // tolerance
  );

  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    probabilities.open(file_probabilities);
    probabilities << setprecision(6) << std::scientific;
  }

  // Read electronic energies
  auto electronic_energy_0 = read_electronic_energy(file_electronic_energy_0);
  auto electronic_energy_1 = read_electronic_energy(file_electronic_energy_1);
  auto electronic_energy_2 = read_electronic_energy(file_electronic_energy_2);

  // Compute fragmentation energy in Kelvin
  if (bonding_energy == 0)
  {
    bonding_energy = (electronic_energy_1 + electronic_energy_2 - electronic_energy_0) * hartK;
  }

  Eigen::ArrayXd energies_density;
  Eigen::ArrayXd density_cluster;
  Eigen::ArrayXd density_combined_products;
  int m_max_density;
  double energy_max_density;
  double bin_width_density;
  std::tie(energies_density, density_cluster, density_combined_products, m_max_density, energy_max_density, bin_width_density) = read_two_density_histograms(file_density_cluster, file_density_combined_products);

  Eigen::ArrayXd energies_rate;
  Eigen::ArrayXd rate_const;
  int m_max_rate;
  double energy_max_rate;
  double bin_width_rate;
  std::tie(energies_rate, rate_const, m_max_rate, energy_max_rate, bin_width_rate) = read_histogram(file_rate_const);

  SkimmerData skimmer;
  double mesh_skimmer;
  std::tie(skimmer, mesh_skimmer) = read_skimmer(file_skimmer);

  rescale_density(density_cluster, m_max_density);
  rescale_density(density_combined_products, m_max_density);

  rescale_energies(energies_density, m_max_density, energy_max_density, bin_width_density);
  rescale_energies(energies_rate, m_max_rate, energy_max_rate, bin_width_rate);

  // cout << "Density: " << energy_max_density << " " << m_max_density << " " << bin_width_density << endl;
  // cout << "Rate: " << energy_max_rate << " " << m_max_rate << " " << bin_width_rate << endl;

  auto rotations = read_rotations(file_rotations);
  auto inertia = compute_inertia(rotations);
  compute_mass_and_radius(inertia, amu, m_ion, R_cluster);

  // TO BE DELETED ###############
  //  R_cluster=4.675e-10;
  // #############################
  kT = boltzmann * T;
  R_tot = R_cluster + R_gas;
  reduced_mass = 1. / (1. / m_ion + 1. / m_gas);
  inertia = 0.4 * m_ion * R_cluster * R_cluster;
  mobility_gas = kT / m_gas;
  // std_gas=sqrt(mobility_gas);
  mobility_gas_inv = 1.0 / mobility_gas;
  boundary_u = 5.0 * sqrt(mobility_gas);
  du = 1.0e-4 * sqrt(mobility_gas);
  E1 = -(V1 - V0) / L0;
  E2 = -(V2 - V1) / L1;
  E3 = -(V3 - V2) / L2;
  E4 = -(V4 - V3) / L3;
  first_chamber_end = L0;
  sk_end = L0 + Lsk;
  quadrupole_start = L0 + Lsk + L1;
  quadrupole_end = L0 + Lsk + L1 + L2;
  second_chamber_end = L0 + Lsk + L1 + L2 + L3;
  total_length = second_chamber_end;

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
  mathieu_factor = q / (m_ion * r_quadrupole * r_quadrupole);
  angular_velocity = 2.0 * M_PI * radiofrequency;
  acc1 = E1 * q * cluster_charge_sign / m_ion;
  acc2 = E2 * q * cluster_charge_sign / m_ion;
  acc3 = E3 * q * cluster_charge_sign / m_ion;
  acc4 = E4 * q * cluster_charge_sign / m_ion;
  P1 = pressure_first;
  P2 = pressure_second;
  gas_mean_free_path = mean_free_path(R_gas, kT, P2);
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
  n1 = particle_density(P1, kT);
  n2 = particle_density(P2, kT);
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
  dt1 = 1.0e-3 / coll_freq(n1, mobility_gas, mobility_gas_inv, R_tot, 0.0);
  dt2 = 1.0e-3 / coll_freq(n2, mobility_gas, mobility_gas_inv, R_tot, 0.0);
  if (dt2 > 1.0 / radiofrequency / 1000.0)
    dt2 = 1.0 / radiofrequency / 1000.0;

  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    std::cout << "Time step t1: " << dt1 << " s" << endl;
    std::cout << "Time step t2: " << dt2 << " s" << endl
              << endl;
  }

  n_escaped_total = 0;
  n_fragmented_total = 0;

  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    probabilities << "#1_FragmentationEnergy 2_SurvivalProbability 3_Error" << endl;
    fragments << "#1_Realization 2_Time 3_Position 4_FragmentationZone 5_PositionOfCollision 6_CollisionZone 7_VelocityAtCollision" << endl;
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
      N, T, kT, m_ion, R_cluster, R_tot, m_max_density, energies_density, inertia, m_max_rate, second_chamber_end, mathieu_factor, rate_const, n1, n2, dt1, dt2, skimmer, mesh_skimmer, angular_velocity, total_length, bin_width_rate, mobility_gas, mobility_gas_inv, gas_mean_free_path, first_chamber_end, root_seed, density_cluster, energy_max_rate, energy_max_density, sk_end, quadrupole_start, quadrupole_end, acc1, acc2, acc3, acc4, P1, P2, bonding_energy, m_gas, dc_field, ac_field, du, boundary_u, reduced_mass) \
  shared(nwarnings, collisions, intenergy, warnings, fragments, tmp, tmp_evolution, file_energy_distribution, final_position, pinhole, probabilities, std::cout, exception_helper) \
  reduction(+ : n_fragmented_total, n_escaped_total, ncoll_total, counter_collision_rejections)
  for (j = 0; j < N; j++)
  {
    exception_helper.guard([&]
    {
      mt19937 gen = mt19937(root_seed ^ j);
      // Define uniform distribution from 0 to 1
      static uniform_real_distribution<double> unif = uniform_real_distribution<>(0.0, 1.0);
      // Define normal (gaussian) distribution with 0 mean and 1 standard deviation
      static normal_distribution<double> gauss = normal_distribution<>(0.0, 1.0);

      const int progress = 10; // Show progress of simulation every *progress* realizations

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
      double v_cluster_versor[3];
      double v_gas;
      double temperature;
      double density;
      double v_cluster_norm;
      double v_cluster_norm_old;

      double theta;
      double phi;
      double u_norm; // normal velocity of colliding gas molecule

      double sintheta;
      double costheta;
      double sinphi;
      double cosphi;

      double vib_energy = 0.0;
      double rot_energy;

      double rate_constant;
      double delta_t;

      double t_fragmentation;

      // Draw initial random velocity from Maxwell-Boltzmann distribution
      init_vel(gen, gauss, v_cluster, m_ion, kT);
      init_ang_vel(gen, gauss, omega, m_ion, kT, R_cluster);
      init_vib_energy(gen, unif, vib_energy, kT, density_cluster, energies_density, m_max_density);

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
          // tmp << delta_en << endl;
          if (delta_en > energy_max_rate)
          {
            warn_omp(nwarnings, [&warnings, &probabilities, &delta_en, &energy_max_rate]()
            {
            warnings << "Internal energy exceeds maximum rate energy by " << setprecision(3) << scientific << (delta_en-energy_max_rate)/energy_max_rate << endl;
            probabilities << "# Internal energy exceeds maximum rate energy: " << setprecision(3) << scientific << (delta_en-energy_max_rate)/energy_max_rate << endl; });
            delta_en = energy_max_rate;
            a = 1;
          }
          rate_constant = evaluate_rate_const(rate_const, delta_en, bin_width_rate, m_max_rate, warnings, nwarnings);
        }
        else
        {
          rate_constant = 0.0;
        }

        time_next_coll_quadrupole(gen, unif, rate_constant, v_cluster, v_cluster_norm, n1, n2, mobility_gas, mobility_gas_inv, R_tot, dt1, dt2, z, x, y, delta_t, t_fragmentation, first_chamber_end, sk_end, quadrupole_start, quadrupole_end, second_chamber_end, acc1, acc2, acc3, acc4, t, m_gas, skimmer, mesh_skimmer, angular_velocity, mathieu_factor, dc_field, ac_field, tmp_evolution);

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
#pragma omp critical
              {
                fragments << j + 1 << "\t" << t << "\t" << z << "\t" << zone(z, first_chamber_end, sk_end, quadrupole_start, quadrupole_end, second_chamber_end) << "\t" << coll_z << "\t" << zone(coll_z, first_chamber_end, sk_end, quadrupole_start, quadrupole_end, second_chamber_end) << "\t" << v_cluster_norm_old << endl;
              }
            }
            break;
          }

          if (a == 1)
          {
            {
              throw ApiTofError([&](auto &ss)
              {
                ss << "FATAL ERROR: The internal energy exceeded the max energy related to rate constant (so the cluster should fragment), but the cluster did not fragment. Realization: " << j + 1 << endl
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
          // v_cluster_norm_old = vec_norm(v_cluster);

          if (ncoll > max_coll)
          {
            throw ApiTofError([&](auto &ss)
            {
              ss << "Got to the max collisions " << ncoll << " (max is " << max_coll << ")";
            });
          }

          update_physical_quantities(z, skimmer, mesh_skimmer, v_gas, temperature, pressure, density, first_chamber_end, sk_end, P1, P2, n1, n2, T);

          // Draw theta angle of collision
          draw_theta_skimmer(gen, unif, theta, z, n1, n2, m_gas, mobility_gas, mobility_gas_inv, R_tot, v_cluster, v_gas, pressure, temperature, first_chamber_end, sk_end, warnings, nwarnings);

          phi = 2.0 * pi * unif(gen);

          // Update some parameters useful for calculations
          update_param(v_cluster_norm, v_cluster, v_cluster_versor, theta, phi, sintheta, costheta, sinphi, cosphi);

          // Draw normal velocity of carrier gas
          draw_u_norm_skimmer(gen, unif, z, du, boundary_u, u_norm, theta, n1, n2, m_gas, mobility_gas, mobility_gas_inv, R_tot, v_cluster, v_gas, pressure, temperature, first_chamber_end, sk_end, costheta, warnings, nwarnings);

          vib_energy_old = vib_energy;

          evaluate_relative_velocity(z, v_cluster, v_rel_norm, v_gas, v_rel, first_chamber_end, sk_end);

          // Evaluate the dissipated energy in the collision (energy that goes to vibrational modes)

          vib_energy_new = draw_vib_energy(gen, unif, vib_energy_old, density_cluster, energies_density, energy_max_density, reduced_mass, u_norm, v_rel_norm, theta, warnings, nwarnings);

          bool collision_accepted = true;
          eval_collision(gen, unif, collision_accepted, gas_mean_free_path, x, y, z, total_length, radius_pinhole, quadrupole_end, v_rel, omega, u_norm, theta, R_cluster, vib_energy_new, vib_energy_old, m_ion, m_gas, temperature, pinhole, warnings);

          if (collision_accepted)
          {
            vib_energy = vib_energy_new;
            update_velocities(v_cluster, v_cluster_norm, v_rel, v_gas);
            // tmp << kin_energy << endl;

            rot_energy_old = evaluate_rotational_energy(omega, inertia);
            rot_energy = rot_energy_old;
            redistribute_internal_energy(gen, unif, vib_energy, rot_energy, density_cluster, energies_density, energy_max_density, warnings, nwarnings);
            update_rot_vel(omega, rot_energy_old, rot_energy);
          }
          else
            counter_collision_rejections++;
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
#pragma omp critical
            {
              final_position << x << "\t" << y << endl;
            }
          }
          // cout << "Distance from exit on x: " << x << "and y: " << y << endl; // Distance from the exit on x and y axes
        }
      }

      if (LOGLEVEL >= LOGLEVEL_NORMAL)
      {
        if ((j + 1) % progress == 0 and j > 0)
        {
#pragma omp critical
          {
            std::cout << std::defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 * (j + 1) / N << "% " << string(n_fragmented, '*') << string(n_escaped, '-') << " (" << n_fragmented << "*, " << n_escaped << "-) P=" << setprecision(3) << (double)n_escaped_total / (j + 1) << endl;
          }
        }
      }
      n_fragmented_total += n_fragmented;
      n_escaped_total += n_escaped;
      ncoll_total += ncoll;

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

  realizations = n_fragmented_total + n_escaped_total;
  std::cout << setprecision(3);

  if (N != realizations)
  {
    warn_omp(nwarnings, [&warnings]()
    { warnings << "Number of total realizations does not correspond to input value!" << endl; });
  }
  else
  {
    if (LOGLEVEL >= LOGLEVEL_MIN)
    {
      // cout << std::defaultfloat << " 100%" << " Intacts: " << setw(5) << setfill(' ') << n_escaped << " | Fragments: " << setw(5) << setfill (' ') << n_fragmented << " | Survival probability: "  << std::setprecision(3) << 1.0*n_escaped/(n_escaped+n_fragmented)  << endl;
      std::cout << "Simulation completed" << endl
                << endl;
      // cout << std::defaultfloat << "Intacts: " << setw(5) << setfill(' ') << n_escaped << " | Fragments: " << setw(5) << setfill (' ') << n_fragmented << " | Survival probability: "  << std::setprecision(3) << 1.0*n_escaped/(n_escaped+n_fragmented)  << endl;
    }
  }
  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    // cout << "\033[F\033[J";
    std::cout << "Realizations: " << realizations << endl;
    std::cout << "Fragments: " << n_fragmented_total << endl;
    std::cout << "Intacts: " << n_escaped_total << endl;
    survival_probability = (float)n_escaped_total / realizations;
    // error_survival_probability=sqrt(survival_probability*(1.0-survival_probability)/realizations);
    error_survival_probability = evaluate_error(realizations, n_escaped_total);
    double avg_ncoll = (double)ncoll_total / N;
    std::cout << "Average number of collisions: " << avg_ncoll << endl;
    std::cout << "Number of collision rejections close to the pinhole: " << counter_collision_rejections << endl;
    std::cout << endl
              << "SURVIVAL PROBABILITY: " << std::setprecision(6) << survival_probability << " +/-" << std::setprecision(4) << error_survival_probability << endl
              << endl;
    if (nwarnings > 0)
      probabilities << "# WARNINGS GENERATED" << endl;
    // probabilities << bonding_energy/boltzmann << " " << survival_probability << " "  << median << " " << err_down << " "<< err_up << endl;
    probabilities << std::setprecision(6) << bonding_energy / boltzmann << " " << survival_probability << " " << error_survival_probability << endl;
    std::cout << "OUTPUT" << endl;
    std::cout << file_probabilities << endl
              << endl;

    collisions.close();
    intenergy.close();
    warnings.close();
    fragments.close();
    tmp.close();
    tmp_evolution.close();
    file_energy_distribution.close();
    final_position.close();
    pinhole.close();
    probabilities.close();
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto loop_time = std::chrono::duration_cast<std::chrono::microseconds>(end - loop_start);
  std::cout << endl
            << "<loop_time>" << loop_time.count() << "</loop_time>" << endl
            << endl;
  auto total_time = end - start;
  auto seconds_tot = std::chrono::duration_cast<std::chrono::seconds>(total_time).count();
  auto microseconds_tot = std::chrono::duration_cast<std::chrono::microseconds>(total_time).count();
  hours = (int)(seconds_tot / 3600);
  minutes = mod_func_int(seconds_tot / 60, 60);
  seconds = mod_func_int(seconds_tot, 60);
  std::cout << "Computational time: " << setw(3) << setfill(' ') << hours << "h" << setw(2) << setfill('0') << minutes << "m" << setw(2) << setfill('0') << seconds << "s" << microseconds_tot << "us" << endl;
  if (nwarnings > 0)
    std::cout << "$$$$$$$$$ WARNING $$$$$$$$$" << endl
              << nwarnings << " warnings have been generated: check the file " << Filenames::WARNINGS << endl
              << "$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;

  // cout << setprecision(8) << eval_solid_angle(radius_pinhole, total_length, 0.0, 0.0, total_length-1.0e-4) << endl;
  // cout << setprecision(8) << eval_solid_angle_stokes(1.0,10.0,0.0,0.0,9.0) << endl;
}

int main()
{
  try
  {
    apitof_pinhole();
  }
  catch (std::exception &ex)
  {
    std::cerr << ex.what() << std::endl;
    return -1;
  }
  return 0;
}

double evaluate_error(int n, int k)
{
  return sqrt((6.0 * k * k - k * (6.0 + k) * n + (2.0 + k) * n * n) / (n * n * (3.0 + n) * (2.0 + n)));
}

// Geometrical mean of moment of inertia
void compute_inertia(double *rotations, double &inertia_moment)
{
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
    std::cout << "Zero result in evaluating the cross product" << endl;
    exit(EXIT_FAILURE);
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

float particle_density(float pressure, float kT)
{
  return pressure / kT;
}

// Total collision frequency
double coll_freq(float n, float mobility_gas, float mobility_gas_inv, float R, double v)
{
  if (v > 0)
    return 2.0 * pi * n * R * R * (0.5 * (mobility_gas / v + v) * erf(sqrt(0.5 * mobility_gas_inv) * v) + sqrt(0.5 * mobility_gas / pi) * exp(-0.5 * mobility_gas_inv * v * v));
  else
    return 2.0 * sqrt(2.0 * pi * mobility_gas) * n * R * R;
}


// Collision frequency on angle theta
double coll_freq_theta(double theta, float n, float mobility_gas, float mobility_gas_inv, float R, double v)
{
  double costheta = cos(theta);
  double sintheta = sin(theta);
  return pi * n * R * R * sintheta * (sqrt(mobility_gas * 2.0 / pi) * exp(-0.5 * mobility_gas_inv * v * v * costheta * costheta) + v * costheta * (erf(sqrt(0.5 * mobility_gas_inv) * v * costheta) + 1));
}


// Collision frequency on angle theta and gas velocity
double coll_freq_theta_u(double u, double theta, float n, float mobility_gas_inv, float R, double v)
{
  double costheta = cos(theta);
  double sintheta = sin(theta);
  return 2.0 * pi * n * R * R * sqrt(0.5 * mobility_gas_inv / pi) * (u + v * costheta) * exp(-0.5 * mobility_gas_inv * u * u) * sintheta;
}


// Distribution of angle theta
double distr_theta(double theta, float n, float mobility_gas, float mobility_gas_inv, float R, double v)
{
  return coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v) / coll_freq(n, mobility_gas, mobility_gas_inv, R, v);
}


// Distribution of gas velocity
double distr_u(double u, double theta, float n, float mobility_gas, float mobility_gas_inv, float R, double v)
{
  return coll_freq_theta_u(u, theta, n, mobility_gas_inv, R, v) / coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v);
}


// Distribution of 1-dim Maxwell velocity
template <typename GenT>
double onedimMaxwell(GenT &gen, normal_distribution<double> &gauss, float m, float kT)
{
  return sqrt(kT / m) * gauss(gen);
}


// Distribution of 2-dim Maxwell velocity
template <typename GenT>
double twodimMaxwell(GenT &gen, uniform_real_distribution<double> &unif, float m, float kT)
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
double onedimMaxwell_angular(GenT &gen, normal_distribution<double> &gauss, float m, float R, float kT)
{
  return sqrt(2.5 * kT / (m * R * R)) * gauss(gen);
}


// Evaluate parameters
void update_param(double &v_cluster_norm, double *v_cluster, double *v_cluster_versor, double theta, double phi, double &sintheta, double &costheta, double &sinphi, double &cosphi)
{
  v_cluster_norm = sqrt(v_cluster[0] * v_cluster[0] + v_cluster[1] * v_cluster[1] + v_cluster[2] * v_cluster[2]);
  v_cluster_versor[0] = v_cluster[0] / v_cluster_norm;
  v_cluster_versor[1] = v_cluster[1] / v_cluster_norm;
  v_cluster_versor[2] = v_cluster[2] / v_cluster_norm;
  sintheta = sin(theta);
  costheta = cos(theta);
  sinphi = sin(phi);
  cosphi = cos(phi);
}


// Change of coordinates from the cluster reference system to the laboratory one -- Input vectors are normalized!
void coord_change(double *v_cluster_versor, double *u_versor, double sintheta, double costheta, double sinphi, double cosphi, ofstream &warnings, int &nwarnings)
{
  double c_x;
  double c_y;
  // double sintheta;
  // double sinphi;
  // double costheta;
  // double cosphi;
  double vel_squared;
  double versor_collision[3];

  if (v_cluster_versor[2] == 0)
  {
    warn_omp(nwarnings, [&warnings]()
    {
      std::cout << endl << "FAILURE: z-component of cluster velocity is zero." << endl;
      warnings << "z-component of cluster velocity is zero." << endl; });
    exit(EXIT_FAILURE);
  }
  // Coefficients
  c_x = 1.0 / sqrt(1.0 + pow(v_cluster_versor[0] / v_cluster_versor[2], 2));
  vel_squared = v_cluster_versor[0] * v_cluster_versor[0] + v_cluster_versor[2] * v_cluster_versor[2];
  c_y = 1.0 / sqrt((v_cluster_versor[1] * v_cluster_versor[1] * vel_squared) / (vel_squared * vel_squared) + 1.0);

  // // Trigonometric values
  // sintheta=sin(theta);
  // sinphi=sin(phi);
  // costheta=cos(theta);
  // cosphi=cos(phi);

  // Returning values
  versor_collision[0] = sintheta * cosphi * c_x - c_y * sintheta * sinphi * v_cluster_versor[0] * v_cluster_versor[1] / vel_squared + costheta * v_cluster_versor[0];
  versor_collision[1] = sintheta * sinphi * c_y + costheta * v_cluster_versor[1];
  versor_collision[2] = -(v_cluster_versor[0] / v_cluster_versor[2]) * sintheta * cosphi * c_x - sintheta * sinphi * c_y * v_cluster_versor[1] * v_cluster_versor[2] / vel_squared + costheta * v_cluster_versor[2];

  // cout << versor_collision[0]*versor_collision[0]+versor_collision[1]*versor_collision[1]+versor_collision[2]*versor_collision[2] << endl;

  u_versor[0] = -versor_collision[0];
  u_versor[1] = -versor_collision[1];
  u_versor[2] = -versor_collision[2];
}


// Evaluate cluster velocity after collision
void vel_after_coll(double *u_versor, double u_norm, double *v_cluster, double v_cluster_norm, double costheta, float m, float m_gas)
{
  double c1 = m / m_gas - 1.0;
  double c2 = m / m_gas + 1.0;
  double c3 = v_cluster_norm * costheta;

  double v_perp[3];
  double v_paral[3];

  // Evaluate parallel components
  v_paral[0] = -u_versor[0] * c3;
  v_paral[1] = -u_versor[1] * c3;
  v_paral[2] = -u_versor[2] * c3;

  // Evaluate perpendicular components
  v_perp[0] = v_cluster[0] - v_paral[0];
  v_perp[1] = v_cluster[1] - v_paral[1];
  v_perp[2] = v_cluster[2] - v_paral[2];

  // Final velocity
  v_cluster[0] = v_perp[0] + (c1 * v_paral[0] + 2.0 * u_versor[0] * u_norm) / c2;
  v_cluster[1] = v_perp[1] + (c1 * v_paral[1] + 2.0 * u_versor[1] * u_norm) / c2;
  v_cluster[2] = v_perp[2] + (c1 * v_paral[2] + 2.0 * u_versor[2] * u_norm) / c2;
}


// Inizialize the cluster velocity
template <typename GenT>
void init_vel(GenT &gen, normal_distribution<double> &gauss, double *v_cluster, float m, float kT)
{
  v_cluster[0] = onedimMaxwell(gen, gauss, m, kT);
  v_cluster[1] = onedimMaxwell(gen, gauss, m, kT);
  v_cluster[2] = onedimMaxwell(gen, gauss, m, kT);
}

// Inizialize the cluster angular velocity
template <typename GenT>
void init_ang_vel(GenT &gen, normal_distribution<double> &gauss, double *omega, float m, float kT, float R)
{
  omega[0] = onedimMaxwell_angular(gen, gauss, m, R, kT);
  omega[1] = onedimMaxwell_angular(gen, gauss, m, R, kT);
  omega[2] = onedimMaxwell_angular(gen, gauss, m, R, kT);
}


double avg_vib_energy(float kT, double *density_cluster, double *energies_density, int m_max_density)
{
  double sum = 0.0;
  double sum2 = 0.0;

  for (int m = 0; m < m_max_density; m++)
  {
    sum += density_cluster[m] * exp(-energies_density[m] / kT);
  }

  for (int m = 0; m < m_max_density; m++)
  {
    sum2 += energies_density[m] * density_cluster[m] * exp(-energies_density[m] / kT) / sum;
  }
  return sum2;
}

double evaluate_rate_const(const Eigen::ArrayXd &rate_const, double energy, double bin_width_rate, int m_max_rate, ofstream &warnings, int &nwarnings)
{
  int m;
  double coeff1;
  double coeff2;
  // m=int(energy/bin_width_rate);
  m = int((energy + 0.5 * bin_width_rate) / bin_width_rate);
  // coeff1=(energy-m*bin_width_rate)/bin_width_rate;
  coeff1 = (energy - (m - 0.5) * bin_width_rate) / bin_width_rate;
  coeff2 = 1.0 - coeff1;
  if (m > 0)
    return coeff2 * rate_const[m - 1] + coeff1 * rate_const[m];
  else if (m == 0)
    return rate_const[0];
  else if (m >= m_max_rate)
  {
    warn_omp(nwarnings, [&warnings, &energy]()
    { warnings << "delta_energy exceeded upper limit of rate_constant evaluation: delta_energy= " << energy << endl; });
    return rate_const[m_max_rate];
  }
  else
  {
    warn_omp(nwarnings, [&warnings, &energy]()
    { warnings << "Rate constant evaluation failed: delta_energy= " << energy << endl; });
    return 0;
  }
}


void update_skimmer_quantities(const SkimmerData &skimmer, double z, float first_chamber_end, double mesh_skimmer, double &v_gas, double &temp, double &pressure)
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

void update_physical_quantities(double z, const SkimmerData skimmer, double mesh_skimmer, double &v_gas, double &temperature, double &pressure, double &density, float first_chamber_end, float sk_end, float P1, float P2, float n1, float n2, float T)
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
    coeff1 = (position - m * mesh_skimmer) / mesh_skimmer;
    coeff2 = 1.0 - coeff1;
    v_gas = coeff2 * skimmer(m, VEL_SKIMMER) + coeff1 * skimmer(m + 1, VEL_SKIMMER);
    temperature = coeff2 * skimmer(m, TEMP_SKIMMER) + coeff1 * skimmer(m + 1, TEMP_SKIMMER);
    pressure = coeff2 * skimmer(m, PRESSURE_SKIMMER) + coeff1 * skimmer(m + 1, PRESSURE_SKIMMER);
  }
  else
  {
    density = n2;
    pressure = P2;
    temperature = T;
    v_gas = 0;
  }
}
// double evaluate_rate_const(double energy)
//{
//   //return pow(energy/boltzmann,6)/2.0e18; // good for 10 Pa and 100 Pa
//   //return pow(energy/boltzmann,7)/2.0e22; // good for 10 Pa
//   //return pow(energy/boltzmann,4)/3.0e9;
//   //return pow(energy/boltzmann,5)/1.0e14;
//   //return pow(energy/boltzmann,5)/2.0e14;
//   return pow(energy/boltzmann,8)/1.5e26; //good for 10 Pa and 30 Pa
//   //return pow(energy/boltzmann,10)/3.0e34; //good for 10 Pa and 30 Pa
// }

// Draw initial vibrational energy
template <typename GenT>
void init_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, float kT, const Eigen::ArrayXd &density_cluster, const Eigen::ArrayXd &energies_density, int m_max_density)
{
  double sum1 = 0.0;
  double sum2 = 0.0;
  double r = unif(gen);
  int m;

  for (m = 0; m < m_max_density; m++)
  {
    sum1 += density_cluster[m] * exp(-energies_density[m] / kT);
  }

  m = 0;
  while (sum2 < r)
  {
    sum2 += density_cluster[m] * exp(-energies_density[m] / kT) / sum1;
    m++;
  }
  vib_energy = energies_density[m - 1];
}

// Evaluate time to next collision
template <typename GenT>
void time_next_coll_quadrupole(GenT &gen, uniform_real_distribution<double> &unif, double rate_constant, double *v_cluster, double &v_cluster_norm, float n1, float n2, float mobility_gas, float mobility_gas_inv, float R, double dt1, double dt2, double &z, double &x, double &y, double &delta_t, double &t_fragmentation, float first_chamber_end, float sk_end, float quadrupole_start, float quadrupole_end, float second_chamber_end, float acc1, float acc2, float acc3, float acc4, double &t, float m_gas, const SkimmerData &skimmer, double mesh_skimmer, double angular_velocity, double mathieu_factor, double dc_field, double ac_field, ofstream &tmp_evolution)
{
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
  double n_skimmer;
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
      accx = mathieu_factor * (-dc_field + ac_field * cos(angular_velocity * t)) * (x + v_cluster[0] * dt2 / 2.0);
      accy = mathieu_factor * (dc_field - ac_field * cos(angular_velocity * t)) * (y + v_cluster[1] * dt2 / 2.0);
      v_cluster[0] += accx * dt2;
      v_cluster[1] += accy * dt2;
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
#pragma omp critical
      {
        tmp_evolution << z << " " << delta_t << " " << v_gas << " " << v_cluster_norm << " " << n_skimmer << endl;
      }
    }
  }
}


// Draw theta angle of collision UPDATED
template <typename GenT>
void draw_theta_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double &theta, double z, float n1, float n2, float m_gas, float mobility_gas, float mobility_gas_inv, float R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, ofstream &warnings, int &nwarnings)
{
  double r = unif(gen);
  double integral = 0.0;
  double c;
  double dtheta = 1.0e-3;
  double v_rel[3];
  double n;
  double v_rel_norm;
  double kT;

  theta = 0.0;


  if (z < first_chamber_end)
  {
    n = n1;
    v_rel_norm = vec_norm(v_cluster);
  }
  else if (z < sk_end)
  {
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

  while (r > integral && theta < pi)
  {
    c = distr_theta(theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
    integral += c * dtheta;
    theta += dtheta;
  }
  if (theta > pi)
  {
    theta = pi - 1.0e-3;
    warn_omp(nwarnings, [&warnings, &r]()
    { warnings << "theta exceeded pi. random number r is: " << r << endl; });
  }
}
// Draw translational energy of cluster after the impact with carrier gas
// Here we are considering a constant density of states for vibrational mode, i.e. a single vibration (simplified model)
template <typename GenT>
double draw_vib_energy(GenT &gen, uniform_real_distribution<double> &unif, double vib_energy_old, const Eigen::ArrayXd &density_cluster, const Eigen::ArrayXd &energies_density, double energy_max_density, float reduced_mass, double u_norm, double v_cluster_norm, double theta, ofstream &warnings, int &nwarnings)
{
  double r = unif(gen);
  double relative_speed = u_norm + v_cluster_norm * cos(theta);
  // cout << relative_speed << endl<<endl;
  double E = vib_energy_old + reduced_mass * 0.5 * relative_speed * relative_speed;
  // double E=reduced_mass*0.5*relative_speed*relative_speed;
  double integral = 0.0;
  double integral2 = 0.0;
  int m;


  // 1st step: I evaluate the integral (normalization)

  if (E > energy_max_density)
  {
    warn_omp(nwarnings, [&warnings, &E]()
    {
      std::cout << "\n\n\n WARNING!!! E: "<<E/boltzmann<<"\n\n\n";
      warnings << "Energy is exceeding the density of states file. E: "<<E/boltzmann<<endl; });
    exit(EXIT_FAILURE);
  }

  m = 0;
  while (energies_density[m] < E)
  {
    // integral+=1.0/sqrt(E-energies_density[m])*density_cluster[m];  // Not sure if the density of states for translational motion is correct (the treated motion is unidimensional, but the density of states is for 3-dim)
    integral += sqrt(E - energies_density[m]) * density_cluster[m];
    m++;
  }

  // 2nd step: I evaluate the random transferred energy to the cluster
  m = 0;
  while (integral2 < r)
  {
    // integral2+=1.0/sqrt(E-energies_density[m])*density_cluster[m]/integral;
    integral2 += sqrt(E - energies_density[m]) * density_cluster[m] / integral;
    m++;
  }
  // debug_file << r << "\t" << relative_speed << "\t" << E/boltzmann << "\t" << energies_density[m-1]/boltzmann << endl;
  return energies_density[m - 1];
}

// Redistribution of internal energy (between vibrational and rotational modes)
template <typename GenT>
void redistribute_internal_energy(GenT &gen, uniform_real_distribution<double> &unif, double &vib_energy, double &rot_energy, const Eigen::ArrayXd &density_cluster, const Eigen::ArrayXd &energies_density, double energy_max_density, ofstream &warnings, int &nwarnings)
{
  double r = unif(gen);
  double E = vib_energy + rot_energy;
  double integral = 0.0;
  double integral2 = 0.0;
  int m;


  if (E > energy_max_density)
  {
    warn_omp(nwarnings, [&warnings, &E]()
    {
      std::cout << "\n\n\n WARNING!!! E: "<<E/boltzmann<<"\n\n\n";
      warnings << "Energy is exceeding the density of states file. E: "<<E/boltzmann<<endl; });
    exit(EXIT_FAILURE);
  }

  // 1st step: I evaluate the integral (normalization)
  m = 0;
  while (energies_density[m] < E)
  {
    if (E - energies_density[m] < 0)
      std::cout << "ERROR!!" << endl
                << endl;
    integral += sqrt(E - energies_density[m]) * density_cluster[m];
    m++;
  }

  // 2nd step: I evaluate the random transferred energy to the cluster
  m = 0;
  while (integral2 < r)
  {
    if (E - energies_density[m] < 0)
      std::cout << "ERROR!!" << endl
                << endl;
    integral2 += sqrt(E - energies_density[m]) * density_cluster[m] / integral;
    m++;
  }
  vib_energy = energies_density[m - 1];
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
void draw_u_norm_skimmer(GenT &gen, uniform_real_distribution<double> &unif, double z, double du, double boundary_u, double &u_norm, double theta, float n1, float n2, float m_gas, float mobility_gas, float mobility_gas_inv, float R, double *v_cluster, double v_gas, double pressure, double temperature, double first_chamber_end, double sk_end, double costheta, ofstream &warnings, int &nwarnings)
{
  double r = unif(gen);
  double c;
  double integral = 0.0;
  double n;
  double v_rel[3];
  double v_rel_norm;
  double kT;

  if (z < first_chamber_end)
  {
    n = n1;
    v_rel_norm = vec_norm(v_cluster);
  }
  else if (z < sk_end)
  {
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

  if (v_rel_norm * costheta > boundary_u)
  {
    u_norm = -boundary_u;
  }
  else
    u_norm = -v_rel_norm * costheta;

  while (r > integral && u_norm < boundary_u)
  {
    c = distr_u(u_norm, theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
    integral += c * du;
    u_norm += du;
  }

  if (u_norm > boundary_u)
  {
    warn_omp(nwarnings, [&warnings, &r]()
    { warnings << "u_norm exceeded boundary of the integration. random number r is: " << r << endl; });
  }
}


void update_v_cluster_norm(double *v_cluster, double &v_cluster_norm)
{
  v_cluster_norm = sqrt(v_cluster[0] * v_cluster[0] + v_cluster[1] * v_cluster[1] + v_cluster[2] * v_cluster[2]);
}

//// Evaluate (approximation of) kinetic energy of crashing gas molecule
// double evaluate_energy_collision(double v_relative, float m_gas)
//{
//   return 0.5*m_gas*v_relative*v_relative;
// }

// Evaluate (approximation of) kinetic energy of crashing gas molecule
double evaluate_energy_collision(double *v, double *omega, float inertia, float m_ion)
{
  double v_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  double omega_squared = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
  return 0.5 * (m_ion * v_squared + inertia * omega_squared);
}

//// Evaluate internal energy (rotational+vibrational)
// double evaluate_internal_energy(double vib_energy, double * omega, float inertia, float m_ion)
//{
//   double omega_squared=omega[0]*omega[0]+omega[1]*omega[1]+omega[2]*omega[2];
//   return 0.5*inertia*omega_squared+vib_energy;
// }

// Evaluate internal energy (rotational+vibrational)
double evaluate_internal_energy(double vib_energy, double rot_energy)
{
  return rot_energy + vib_energy;
}

// Evaluate rotational energy
double evaluate_rotational_energy(double *omega, float inertia)
{
  double omega_squared = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
  return 0.5 * inertia * omega_squared;
}

// Mean free path
double mean_free_path(float R, float kT, float pressure)
{
  return kT / (sqrt(2.0) * pi * 4.0 * R * R * pressure);
}


double energy_in_eV(double energy)
{
  return energy / 1.602e-19;
}


//// Evaluate tangential component of carrier gas velocity in cluster reference system
// void eval_u_tan(double * u_versor, double * v_cluster, double * v_new, double * u_tan, float m_gas, float kT)
//{
//   double phi=2.0*pi*drand48();
//   double cosphi=cos(phi);
//   double sinphi=sin(phi);
//   double y_old[3];
//   double x_old[3];
//   double y_new[3];
//   double x_new[3];
//   double norm;
//   double scalarvx;
//   double scalarvy;
//   double xcomponent;
//   double ycomponent;
//
//   // Define reference system in the tangent plane
//   cross_norm(u_versor,v_cluster,y_old);
//   cross_norm(y_old,u_versor,x_old);
//
//   // Rotate the reference system in order to have x axis directed as u_tan
//   for(int i=0; i<3; i++)
//   {
//     x_new[i]=cosphi*x_old[i]+sinphi*y_old[i];
//     y_new[i]=-sinphi*x_old[i]+cosphi*y_old[i];
//   }
//
//   // Compute omega_old in the new reference system
//   for(int i=0; i<3; i++)
//   {
//     omega_new[i]=scalar(omega_old,x_new)*x_new[i]+scalar(omega_old,y_new)*y_new[i]+scalar(omega_old,z_new)*z_new[i];
//   }
//
//   // Compute new velocities after collision
//   vx=4.0*m_gas*(u_tanx+R_cluster*omega_oldz)/(7.0*m_gas+2.0*m_ion);
//   vz=-4.0*m_gas*R_cluster*omega_oldx/(7.0*m_gas+2.0*m_ion);
//   omega_newx=(2.0*m_ion-3.0*m_gas)*omega_oldx/(7.0*m_gas+2.0*m_ion);
//   omega_newz=-10.0*m_ion*(u_tanx+R_cluster*omega_oldz)/((7.0*m_gas+2.0*m_ion)*R_cluster)+omega_oldz;
//   // Draw norm of tangential velocity
//  // norm=twodimMaxwell(m_gas, kT);
//
//  // scalarvx=scalar(v_cluster,x);
//  // scalarvy=scalar(v_cluster,y);
//  // xcomponent=norm*cosphi;
//  // ycomponent=norm*sinphi;
//
//  // u_tan[0]=(xcomponent-scalarvx)*x[0]+(ycomponent-scalarvy)*y[0];
//  // u_tan[1]=(xcomponent-scalarvx)*x[1]+(ycomponent-scalarvy)*y[1];
//  // u_tan[2]=(xcomponent-scalarvx)*x[2]+(ycomponent-scalarvy)*y[2];
// }

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
void eval_velocities(double *v, double *omega, double *u, double vib_energy, double vib_energy_old, float M, float m, double R_cluster, ofstream &warnings)
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
    std::cout << radicand << endl;
    warnings << "sqrt of negative number in evaluation of velocities after collision!" << endl;
    exit(EXIT_FAILURE);
  }
  vz = m_reduced * u[0] + M_reduced * v[2] - sqrt(radicand);

  // if(m_reduced*m_reduced*pow(u[1]-v[1],2)-2.0*(vib_energy-vib_energy_old)*m_reduced/M<0)
  // {
  //   cout << std::scientific << "ERROR!!!  "<< u[1]-v[1]<< " " << u_norm+v_cluster_norm*costheta<< " "<<vib_energy-vib_energy_old<< " " << 0.5*m*M*(u[1]-v[1])*(u[1]-v[1])/(M+m)<< " " << m_reduced*m_reduced*pow(u[1]-v[1],2)-2.0*(vib_energy-vib_energy_old)*m_reduced/M <<endl<<endl<<endl;
  //   exit(EXIT_FAILURE);
  // }
  // cout << m_reduced*m_reduced*pow(u[1]-v[1],2)-2.0*vib_energy*m_reduced/M << endl<<endl;
  // cout<<"anelastic: "<<vy<<endl;
  // vy=(m*M*u[1]+M*M*v[1]+sqrt(m*m*M*M*pow(u[1]-v[1],2)-2.0*diss_energy*M*m*(M+m)))/(M*(m+M));

  // In case of elastic collision, vy becomes:
  // vy=(2.0*u[1]+(ratio_masses-1.0)*v[1])/(1.0+ratio_masses);
  // cout<<"elastic: "<<vy<<endl;

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
  if (theta > 0 and theta < M_PI)
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
  else if (theta == M_PI)
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

double polar_function(double phi, double theta1, double theta2)
{
  return theta1 * theta2 / sqrt(pow(theta1 * cos(phi), 2) + pow(theta2 * sin(phi), 2));
}

// Evaluate solid angle using Stokes theorem (1d integral) (REF: Eq 32, Conway, Nuclear Instruments and Methods in Physics Research A 614, 2010)
double eval_solid_angle_stokes(double R, double L, double xx, double yy, double z)
{
  int N = 1000;
  double dphi;
  double sum = 0.0;
  double integrand;
  double c;
  double phi;
  double xphi;
  double yphi;
  double zz = L - z;

  dphi = 2.0 * M_PI / N;

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

  phi = 2.0 * M_PI;
  xphi = R * xx * cos(phi);
  yphi = R * yy * sin(phi);
  c = R * R + xx * xx + yy * yy - 2.0 * xphi - 2.0 * yphi;
  integrand = (1.0 - zz / sqrt(c + zz * zz)) * (R * R - xphi - yphi) / c;
  sum += 0.5 * integrand;

  return sum * dphi;
}

//
template <typename GenT>
void eval_collision(GenT &gen, uniform_real_distribution<double> &unif, bool &collision_accepted, double gas_mean_free_path, double x, double y, double z, double L, double radius_pinhole, float quadrupole_end, double *v_cluster, double *omega, double u_norm, double theta, float R_cluster, double vib_energy, double vib_energy_old, float m_ion, float m_gas, float temperature, ofstream &pinhole, ofstream &warnings)
{
  double x3[3];
  double y3[3];
  double z3[3];
  double v2[3];
  double omega2[3];
  double phi = 2.0 * M_PI * unif(gen);
  double alpha = 2.0 * M_PI * unif(gen);
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
      inside_target = false;
    pinhole << x << " " << y << " " << z << " " << velocity_gas[0] << " " << velocity_gas[1] << " " << velocity_gas[2] << " " << inside_target << endl;
    if (inside_target)
    {
      double r = unif(gen);
      distance = sqrt(x * x + y * y + (L - z) * (L - z));
      // Probability to accept the collision prob_coll
      prob_coll = (1.0 - exp(-distance / gas_mean_free_path)) * (1.0 - eval_solid_angle_stokes(radius_pinhole, L, x, y, z) / (2.0 * M_PI));

      // prob_coll=1.0-eval_solid_angle(radius_pinhole, L, x, y, z)/(2.0*M_PI);
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

    eval_velocities(v2, omega2, u, vib_energy, vib_energy_old, m_ion, m_gas, R_cluster, warnings);
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

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, int, double, double> read_histogram(char *filename)
{
  ifstream file;
  char garb[150];
  file.open(filename);

  int m_max = 0;
  while (file >> garb >> garb)
  {
    m_max++;
  }
  file.close();
  if (m_max < 2)
  {
    std::cout << "Error in reading file " << filename << ". It should contain at least two rows." << endl;
    exit(EXIT_FAILURE);
  }
  Eigen::ArrayXd x(m_max);
  Eigen::ArrayXd y(m_max);

  file.open(filename);
  for (int m = 0; m < m_max; m++)
  {
    file >> x[m] >> y[m];
  }
  file.close();
  double bin_width = x[1] - x[0];
  double x_max = bin_width * m_max;

  return std::make_tuple(x, y, m_max, x_max, bin_width);
}

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd, int, double, double> read_two_density_histograms(char *file_density_cluster, char *file_density_combined_products)
{
  auto [energies_density, density_cluster, m_max_density, energy_max_density, bin_width_density] = read_histogram(file_density_cluster);
  auto [energies_density_other, density_combined_products, m_max_density_other, energy_max_density_other, bin_width_density_other] = read_histogram(file_density_combined_products);
  if (!energies_density.isApprox(energies_density_other) || bin_width_density != bin_width_density_other || m_max_density != m_max_density_other || energy_max_density != energy_max_density_other)
  {
    std::cout << "Error: The two density of states histograms are not compatible." << std::endl;
    exit(EXIT_FAILURE);
  }
  return std::make_tuple(energies_density, density_cluster, density_combined_products, m_max_density, energy_max_density, bin_width_density);
}

void rescale_density(Eigen::ArrayXd &density, int m_max)
{
  for (int m = 0; m < m_max; m++)
  {
    density[m] = density[m] / boltzmann;
  }
}

void rescale_energies(Eigen::ArrayXd &energies, int m_max, double &energy_max, double &bin_width)
{
  for (int m = 0; m < m_max; m++)
  {
    energies[m] = energies[m] * boltzmann;
  }
  energy_max = energy_max * boltzmann;
  bin_width *= boltzmann;
}

std::tuple<SkimmerData, double> read_skimmer(char *filename)
{
  int m;
  ifstream file;
  double pos0;
  double pos1;
  char garb[150];
  int m_max;
  file.open(filename);
  file >> garb;

  file >> pos0 >> garb >> garb >> garb >> garb >> garb;
  file >> pos1 >> garb >> garb >> garb >> garb >> garb;
  m = 2;

  double mesh_skimmer = pos1 - pos0;

  while (file >> garb >> garb >> garb >> garb >> garb >> garb)
    m++;
  file.close();

  m_max = m;
  SkimmerData data(m_max, 3);

  file.open(filename);
  file >> garb;
  for (m = 0; m < m_max; m++)
  {
    file >> garb >> data(m, VEL_SKIMMER) >> data(m, TEMP_SKIMMER) >> data(m, PRESSURE_SKIMMER) >> garb >> garb;
  }
  file.close();

  return std::make_tuple(data, mesh_skimmer);
}

int zone(double z, float first_chamber_end, float sk_end, float quadrupole_start, float quadrupole_end, float second_chamber_end)
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
