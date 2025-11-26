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

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include "apitof_pinhole_io.h"

void apitof_pinhole_config_in()
{
  using namespace consts;
  // Mersenne-Twister uniform random number generator
  mt19937 root_gen = mt19937(42ull);
  unsigned long long root_seed = root_gen();

  double L0;
  double Lsk;
  double L1;
  double L2;
  double L3;
  double V0;
  double V1;
  double V2;
  double V3;
  double V4;
  double T;
  double R_gas;
  double m_gas;
  double ga;
  // double std_gas;
  double pressure_first;
  double pressure_second;
  double bonding_energy;

  // double rate_const;
  double r_quadrupole;
  double radiofrequency;
  double dc_field;
  double ac_field;
  int counter_collision_rejections = 0;

  int ncoll_total = 0;
  int N;
  int nwarnings = 0;
  int amu;
  int cluster_charge_sign;

  char file_rate_const[150];
  char file_density_cluster[150];
  char file_skimmer[150];
  char file_rotations[150];
  char file_electronic_energy_0[150];
  char file_electronic_energy_1[150];
  char file_electronic_energy_2[150];
  char file_probabilities[150];

  // Set scientific notation
  std::cout << std::scientific << std::setprecision(3);

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
    &ga,
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
    nullptr,
    file_rate_const,
    file_probabilities,
    nullptr, // N_iter
    nullptr, // M_iter
    nullptr, // resolution
    nullptr // tolerance
  );

  std::optional<LogFileWriter> writer = std::nullopt;
  if (LOGLEVEL >= LOGLEVEL_MIN)
  {
    writer = LogFileWriter(file_probabilities);
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

  auto density_cluster = read_histogram(file_density_cluster);
  auto rate_const = read_histogram(file_rate_const);

  SkimmerData skimmer;
  double mesh_skimmer;
  std::tie(skimmer, mesh_skimmer) = read_skimmer(file_skimmer);

  auto rotations = read_rotations(file_rotations);
  auto inertia = compute_inertia(rotations);
  double m_ion;
  double R_cluster;
  compute_mass_and_radius(inertia, amu, m_ion, R_cluster);

  rescale_density(density_cluster);
  rescale_energies(density_cluster);
  rescale_energies(rate_const);

  StreamingResultQueue result_queue;
  Counters counters;
  OMPExceptionHelper exception_helper;
  std::thread execution_thread = std::thread([&]
  {
    // TODO: Probably want to switch to jthread when possible
    exception_helper.guard([&]
    {
      InstrumentDims lengths(5);
      lengths << L0, L1, L2, L3, Lsk;
      InstrumentVoltages voltages(5);
      voltages << V0, V1, V2, V3, V4;
      int sample_mode = 0;
      char *sample_mode_env = getenv("SAMPLE_MODE");
      if (sample_mode_env != nullptr && strcmp(sample_mode_env, "1") == 0)
      {
        std::cout << "Using SAMPLE_MODE=1\n";
        sample_mode = 1;
      }
      counters = apitof_pinhole(
        cluster_charge_sign,
        T,
        pressure_first,
        pressure_second,
        lengths,
        voltages,
        N,
        bonding_energy,
        Gas{
          R_gas,
          m_gas,
          ga},
        Quadrupole(
          dc_field,
          ac_field,
          radiofrequency,
          r_quadrupole),
        m_ion,
        R_cluster,
        density_cluster,
        rate_const,
        skimmer,
        mesh_skimmer,
        root_seed,
        result_queue,
        sample_mode);
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
      if (LOGLEVEL >= LOGLEVEL_NORMAL)
      {
        const PartialResult &partial_result = std::get<PartialResult>(result);
        partial_counters.row(partial_result.thread_id) = partial_result.counters.transpose();
        Counters cur_counters = partial_counters.colwise().sum();
        auto cur_iters = cur_counters[Counter::n_fragmented_total] + cur_counters[Counter::n_escaped_total];
        const int progress = 10; // Show progress of simulation every *progress* realizations
        if ((cur_iters + 1) % progress == 0 and cur_iters > 0)
        {
          int n_fragmented_total = cur_counters[Counter::n_fragmented_total];
          int n_escaped_total = cur_counters[Counter::n_escaped_total];
          double survival_ratio = (double)n_escaped_total / (cur_iters + 1);
          std::cout << std::defaultfloat << setw(5) << setfill(' ') << fixed << setprecision(1) << 100.0 * (cur_iters + 1) / N << "% " << string(n_fragmented_total, '*') << string(n_escaped_total, '-') << " (" << n_fragmented_total << "*, " << n_escaped_total << "-) P=" << setprecision(3) << survival_ratio << endl;
        }
      }
    }
    else if (std::holds_alternative<LogMessage>(result))
    {
      const LogMessage &msg = std::get<LogMessage>(result);
      if (writer.has_value())
      {
        writer.value().out_streams[msg.type] << msg.message;
      }
    }
  }
  execution_thread.join();
  exception_helper.rethrow();

  std::cout << setprecision(3);

  int realizations = counters[Counter::n_fragmented_total] + counters[Counter::n_escaped_total];

  if (N != realizations)
  {
    nwarnings++;
    if (writer.has_value())
    {
      writer.value().out_streams[LogMessage::warnings] << "Number of total realizations does not correspond to input value!" << endl;
    }
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
    std::cout << "Fragments: " << counters[Counter::n_fragmented_total] << endl;
    std::cout << "Intacts: " << counters[Counter::n_escaped_total] << endl;
    double survival_probability = (double)counters[Counter::n_escaped_total] / realizations;
    // error_survival_probability=sqrt(survival_probability*(1.0-survival_probability)/realizations);
    double error_survival_probability = evaluate_error(realizations, counters[Counter::n_escaped_total]);
    double avg_ncoll = (double)ncoll_total / N;
    std::cout << "Average number of collisions: " << avg_ncoll << endl;
    std::cout << "Number of collision rejections close to the pinhole: " << counter_collision_rejections << endl;
    std::cout << endl
              << "SURVIVAL PROBABILITY: " << std::setprecision(6) << survival_probability << " +/-" << std::setprecision(4) << error_survival_probability << endl
              << endl;
    if (writer.has_value())
    {
      if (nwarnings > 0)
      {
        writer.value().out_streams[LogMessage::probabilities] << "# WARNINGS GENERATED" << endl;
      }
      // probabilities << bonding_energy/boltzmann << " " << survival_probability << " "  << median << " " << err_down << " "<< err_up << endl;
      writer.value().out_streams[LogMessage::probabilities] << std::setprecision(6) << bonding_energy / boltzmann << " " << survival_probability << " " << error_survival_probability << endl;
    }
    std::cout << "OUTPUT" << endl;
    std::cout << file_probabilities << endl
              << endl;

    if (writer.has_value())
    {
      writer.value().close();
    }
  }
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
    apitof_pinhole_config_in();
  }
  catch (std::exception &ex)
  {
    std::cerr << ex.what() << std::endl;
    return -1;
  }
  return 0;
}
