#include <Eigen/Dense>
#include <iostream>
#include <random>
#include "consts.h"
#include "warnlogcount.h"

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

template <typename GenT>
double draw_theta_skimmer_dss_norm(GenT &gen, std::uniform_real_distribution<double> &unif, double dtheta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelper warn)
{
  using namespace consts;
  double r = unif(gen);

  double theta = 0.0;
  double integral = 0.0;
  while (r > integral && theta < pi)
  {
    double c = distr_theta(theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
    integral += c * dtheta;
    theta += dtheta;
  }
  if (theta > pi)
  {
    theta = pi - 1.0e-3;
    warn([&r](auto &warning)
    {
      warning << "theta exceeded pi. random number r is: " << r << std::endl;
    });
  }
  return theta;
}

template <typename GenT>
double draw_theta_skimmer_dss_unnorm(GenT &gen, std::uniform_real_distribution<double> &unif, double dtheta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelper warn)
{
  using namespace consts;
  double r = unif(gen);

  double theta = 0.0;
  double integral_unnorm = 0.0;
  double normalization = coll_freq(n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
  double r_unnorm = r * normalization;
  while (integral_unnorm < r_unnorm)
  {
    double c = coll_freq_theta(theta, n, mobility_gas, mobility_gas_inv, R, v_rel_norm);
    integral_unnorm += c * dtheta;
    theta += dtheta;
  }
  if (theta > pi)
  {
    theta = pi - 1.0e-3;
    warn([&r](auto &warning)
    {
      warning << "theta exceeded pi. random number r is: " << r << std::endl;
    });
  }
  return theta;
}

// Draw normal velocity of carrier gas
template <typename GenT>
double draw_u_norm_skimmer_dss_unnorm(GenT &gen, std::uniform_real_distribution<double> &unif, double du, double boundary_u, double theta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelper warn)
{
  using consts::pi;

  double u_norm;
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
      warning << "u_norm exceeded boundary of the integration. random number r is: " << r << std::endl;
    });
  }
  return u_norm;
}

template <typename GenT>
double draw_u_norm_skimmer_dss_norm(GenT &gen, std::uniform_real_distribution<double> &unif, double du, double boundary_u, double theta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelper warn)
{
  double u_norm;
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
      warning << "u_norm exceeded boundary of the integration. random number r is: " << r << std::endl;
    });
  }
  return u_norm;
}


struct GasCollCondHistDSSSamplerBase
{
  double dtheta;
  double du;
  double boundary_u;
  std::uniform_real_distribution<double> unif = std::uniform_real_distribution<>(0.0, 1.0);

  GasCollCondHistDSSSamplerBase(double dtheta, double du, double boundary_u) : dtheta(dtheta), du(du), boundary_u(boundary_u)
  {
  }
};

struct GasCollCondNormHistDSSSampler : GasCollCondHistDSSSamplerBase
{
  using GasCollCondHistDSSSamplerBase::GasCollCondHistDSSSamplerBase;

  template <typename GenT>
  std::tuple<double, double> sample(GenT &gen, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelper warn)
  {
    double theta = draw_theta_skimmer_dss_norm(gen, unif, dtheta, n, v_rel_norm, mobility_gas, mobility_gas_inv, R, warn);
    double u_norm = draw_u_norm_skimmer_dss_norm(gen, unif, du, boundary_u, theta, n, v_rel_norm, mobility_gas, mobility_gas_inv, R, warn);

    return std::make_tuple(theta, u_norm);
  }
};

struct GasCollCondUnnormHistDSSSampler : GasCollCondHistDSSSamplerBase
{
  using GasCollCondHistDSSSamplerBase::GasCollCondHistDSSSamplerBase;

  template <typename GenT>
  std::tuple<double, double> sample(GenT &gen, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelper warn)
  {
    double theta = draw_theta_skimmer_dss_unnorm(gen, unif, dtheta, n, v_rel_norm, mobility_gas, mobility_gas_inv, R, warn);
    double u_norm = draw_u_norm_skimmer_dss_unnorm(gen, unif, du, boundary_u, theta, n, v_rel_norm, mobility_gas, mobility_gas_inv, R, warn);

    return std::make_tuple(theta, u_norm);
  }
};

struct GasCollRejectionSampler
{
  std::uniform_real_distribution<double> theta_unif = std::uniform_real_distribution<>(0.0, consts::pi);
  double boundary_u;

  GasCollRejectionSampler(double boundary_u) : boundary_u(boundary_u)
  {
  }

  template <typename GenT>
  std::tuple<double, double> sample(GenT &gen, double /*n*/, double v_norm, double mobility_gas, double mobility_gas_inv, double /*R*/, WarningHelper /*warn*/)
  {

    // First work out a bound on the maximum probability density
    double u_for_boundary_func_max = (v_norm + sqrt(v_norm * v_norm + 4 * mobility_gas)) / 2;
    double u_v_diff = u_for_boundary_func_max - v_norm;
    double max_density = u_for_boundary_func_max * exp(-mobility_gas_inv * u_v_diff * u_v_diff / 2);
    std::uniform_real_distribution<double> accept_unif = std::uniform_real_distribution<>(0.0, max_density);

    std::uniform_real_distribution<double> u_unif = std::uniform_real_distribution<>(0.0, boundary_u + v_norm);
    while (true)
    {
      double theta = theta_unif(gen);
      double u = u_unif(gen);
      double u_norm = u - v_norm * cos(theta);
      double density = u * exp(-0.5 * mobility_gas_inv * u_norm * u_norm) * sin(theta);
      if (accept_unif(gen) < density)
      {
        // We sampled from a box rather than u_norm which has a range dependent on theta
        // Convert back now
        return std::make_tuple(theta, u_norm);
      }
    }
  }
};

struct VibEnergySamplerBase
{
  const Histogram &density_cluster;
  std::uniform_real_distribution<double> unif = std::uniform_real_distribution<>(0.0, 1.0);

  VibEnergySamplerBase(const Histogram &density_cluster) : density_cluster(density_cluster)
  {
  }

  // Draw translational energy of cluster after the impact with carrier gas
  // Here we are considering a constant density of states for vibrational mode, i.e. a single vibration (simplified model)
  double normalization(double E)
  {
    using consts::boltzmann;

    if (E > density_cluster.x_max)
    {
      throw ApiTofError([&](auto &warning)
      {
        warning << "Energy is exceeding the density of states file. E: " << E / boltzmann << std::endl;
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
    return normalization;
  }
};

struct VibEnergyNormSampler : VibEnergySamplerBase
{
  using VibEnergySamplerBase::VibEnergySamplerBase;

  template <typename GenT>
  double sample(GenT &gen, double E)
  {
    double nc = normalization(E);
    // 2nd step: I evaluate the random transferred energy to the cluster
    double r = unif(gen);
    double integral = 0.0;
    int m = 0;
    while (integral < r)
    {
      integral += sqrt(E - density_cluster.x[m]) * density_cluster.y[m] / nc;
      m++;
    }
    return density_cluster.x[m - 1];
  }
};

struct VibEnergyUnnormSampler : VibEnergySamplerBase
{
  using VibEnergySamplerBase::VibEnergySamplerBase;

  template <typename GenT>
  double sample(GenT &gen, double E)
  {
    double nc = normalization(E);
    // 2nd step: I evaluate the random transferred energy to the cluster
    double r_unnorm = unif(gen) * nc;
    double integral_unnorm = 0.0;
    int m = 0;
    while (integral_unnorm < r_unnorm)
    {
      integral_unnorm += sqrt(E - density_cluster.x[m]) * density_cluster.y[m];
      m++;
    }
    return density_cluster.x[m - 1];
  }
};
