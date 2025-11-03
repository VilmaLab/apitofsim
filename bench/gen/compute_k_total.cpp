#include "densityandrate_lib.h"

double energy_max = 2.0e4;
double energy_max_rate = 2.0e3;
double bin_width = 20.0;
int m_max = int(energy_max / bin_width);
int m_max_rate = int(energy_max_rate / bin_width);
Eigen::ArrayXd k_rate(m_max_rate);
Eigen::ArrayXd k0(m_max_rate);
Eigen::Vector3d rotations_1 = Eigen::Vector3d(0.23454, 0.23880, 0.24449);
Eigen::Vector3d rotations_2 = Eigen::Vector3d(0.22649, 0.23531, 0.23870);
Eigen::ArrayXd freq_0 = (Eigen::ArrayXd(33) << 79.7943,
                         168.125,
                         180.628,
                         212.597,
                         243.822,
                         325.479,
                         555.063,
                         567.105,
                         570.798,
                         607.604,
                         773.207,
                         773.623,
                         802.17,
                         809.182,
                         831.131,
                         831.296,
                         923.583,
                         1179.39,
                         1259.83,
                         1283.99,
                         1349.01,
                         1391.17,
                         1483.98,
                         1655.24,
                         1696.49,
                         1837.91,
                         1935.8,
                         1967.06,
                         2059.82,
                         2198.57,
                         3940.69,
                         4133.95,
                         5206.53)
                          .finished();
Eigen::ArrayXd freq_comb = (Eigen::ArrayXd(27) << 134.114,
                            548.287,
                            593.805,
                            769.959,
                            788.046,
                            800.748,
                            1051.24,
                            1506,
                            1651.14,
                            1795.83,
                            1871.14,
                            5603.24,
                            362.425,
                            456.536,
                            523.744,
                            613.491,
                            692.895,
                            763.395,
                            771.197,
                            1175.95,
                            1255.38,
                            1662.45,
                            1682.54,
                            1752.59,
                            2107.84,
                            5530.05,
                            5535.9)
                             .finished();

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    return -1;
  }
  std::ofstream out(argv[1]);
  Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
  out << "int m_max_rate = " << m_max_rate << ";\n";
  out << "double bin_width = " << bin_width << ";\n";
  out << "Eigen::Vector3d rotations_1 = Eigen::Vector3d(" << rotations_1.format(fmt) << ");\n";
  out << "Eigen::Vector3d rotations_2 = Eigen::Vector3d(" << rotations_2.format(fmt) << ");\n";
  out << "double inertia_moment_1 = " << compute_inertia(rotations_1) << ";\n";
  out << "double inertia_moment_2 = " << compute_inertia(rotations_2) << ";\n";
  Eigen::ArrayXd rho_0(m_max);
  compute_density_of_states(freq_0, rho_0, energy_max, bin_width);
  out << "Eigen::ArrayXd rho_0 = (Eigen::ArrayXd(" << m_max << ")";
  out << " << " << rho_0.format(fmt) << ").finished();\n";
  Eigen::ArrayXd rho_comb(m_max);
  compute_density_of_states(freq_comb, rho_comb, energy_max, bin_width);
  out << "Eigen::ArrayXd rho_comb = (Eigen::ArrayXd(" << m_max << ")";
  out << " << " << rho_comb.format(fmt) << ").finished();\n";
  out << "double fragmentation_energy = " << ((-698.927023650035) + (-699.423353287572) - (-1398.424546231439)) * consts::hartK << ";\n";
}
