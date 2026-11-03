#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include "consts.h"
#include "vib_index.h"
#include <algorithm>
#include <vector>
#include "exceptions.h"


template <typename GenT, typename WarningHelperT>
double draw_theta_skimmer_dss_norm(GenT &gen, std::uniform_real_distribution<double> &unif, double dtheta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelperT warn)
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

template <typename GenT, typename WarningHelperT>
double draw_theta_skimmer_dss_unnorm(GenT &gen, std::uniform_real_distribution<double> &unif, double dtheta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelperT warn)
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
template <typename GenT, typename WarningHelperT>
double draw_u_norm_skimmer_dss_unnorm(GenT &gen, std::uniform_real_distribution<double> &unif, double du, double boundary_u, double theta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelperT warn)
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
    double c = (u_norm + (v_rel_norm * costheta)) * exp(-0.5 * mobility_gas_inv * u_norm * u_norm);
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

template <typename GenT, typename WarningHelperT>
double draw_u_norm_skimmer_dss_norm(GenT &gen, std::uniform_real_distribution<double> &unif, double du, double boundary_u, double theta, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelperT warn)
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

  template <typename GenT, typename WarningHelperT>
  std::tuple<double, double> sample(GenT &gen, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelperT warn)
  {
    double theta = draw_theta_skimmer_dss_norm(gen, unif, dtheta, n, v_rel_norm, mobility_gas, mobility_gas_inv, R, warn);
    double u_norm = draw_u_norm_skimmer_dss_norm(gen, unif, du, boundary_u, theta, n, v_rel_norm, mobility_gas, mobility_gas_inv, R, warn);

    return std::make_tuple(theta, u_norm);
  }
};

struct GasCollCondUnnormHistDSSSampler : GasCollCondHistDSSSamplerBase
{
  using GasCollCondHistDSSSamplerBase::GasCollCondHistDSSSamplerBase;

  template <typename GenT, typename WarningHelperT>
  std::tuple<double, double> sample(GenT &gen, double n, double v_rel_norm, double mobility_gas, double mobility_gas_inv, double R, WarningHelperT warn)
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

  template <typename GenT, typename WarningHelperT>
  std::tuple<double, double> sample(GenT &gen, double /*n*/, double v_norm, double mobility_gas, double mobility_gas_inv, double /*R*/, WarningHelperT /*warn*/)
  {

    // First work out a bound on the maximum probability density
    double u_for_bound_func_max = (v_norm + sqrt((v_norm * v_norm) + (4 * mobility_gas))) / 2;
    double u_v_diff = u_for_bound_func_max - v_norm;
    double max_density = u_for_bound_func_max * exp(-mobility_gas_inv * u_v_diff * u_v_diff / 2);
    std::uniform_real_distribution<double> accept_unif = std::uniform_real_distribution<>(0.0, max_density);
    std::uniform_real_distribution<double> u_unif = std::uniform_real_distribution<>(0.0, boundary_u + v_norm);
    while (true)
    {
      double theta = theta_unif(gen);
      double u = u_unif(gen);
      double u_norm = u - (v_norm * cos(theta));
      double density = u * exp(-0.5 * mobility_gas_inv * u_norm * u_norm) * sin(theta);
      if (accept_unif(gen) < density)
      {
        return std::make_tuple(theta, u_norm);
      }
    }
  }
};

struct VibEnergySamplerBase
{
  std::reference_wrapper<const Histogram> density_cluster;
  std::uniform_real_distribution<double> unif = std::uniform_real_distribution<>(0.0, 1.0);

  VibEnergySamplerBase(const Histogram &density_cluster, const VibIndexPtr & /*unused*/ = {}) : density_cluster(density_cluster)
  {
  }

  const Histogram &density() const
  {
    return density_cluster.get();
  }

  void check_energy(double E) const
  {
    if (!std::isfinite(E) || E < 0)
    {
      throw ApiTofArgumentError("Available vibrational energy must be finite and nonnegative");
    }

    if (E > density().x_max())
    {
      throw ApiTofDosOverflow(density().x_max(), E / consts::boltzmann);
    }
  }

  double normalization(double E) const
  {
    check_energy(E);
    // 1st step: I evaluate the integral (normalization)
    double normalization = 0.0;
    int m = 0;
    while (m < density().length() && density().x()[m] < E)
    {
      normalization += sqrt(E - density().x()[m]) * density().y()[m];
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
    // No accessible vibrational states: leave the energy in the other modes.
    if (nc == 0)
    {
      return 0;
    }
    // 2nd step: I evaluate the random transferred energy to the cluster
    double r = unif(gen);
    double integral = 0.0;
    double last = 0;
    for (int m = 0; m < density().length() && density().x()[m] < E; ++m)
    {
      if (density().y()[m] == 0)
        continue;
      last = density().x()[m];
      integral += sqrt(E - last) * density().y()[m] / nc;
      if (integral >= r)
        break;
    }
    return last;
  }
};

struct VibEnergyUnnormSampler : VibEnergySamplerBase
{
  using VibEnergySamplerBase::VibEnergySamplerBase;

  template <typename GenT>
  double sample(GenT &gen, double E)
  {
    double nc = normalization(E);
    // No accessible vibrational states: leave the energy in the other modes.
    if (nc == 0)
    {
      return 0;
    }
    // 2nd step: I evaluate the random transferred energy to the cluster
    double r_unnorm = unif(gen) * nc;
    double integral_unnorm = 0.0;
    double last = 0;
    for (int m = 0; m < density().length() && density().x()[m] < E; ++m)
    {
      if (density().y()[m] == 0)
        continue;
      last = density().x()[m];
      integral_unnorm += sqrt(E - last) * density().y()[m];
      if (integral_unnorm >= r_unnorm)
        break;
    }
    return last;
  }
};

struct VibEnergyDosSampler : VibEnergySamplerBase
{
  const Eigen::ArrayXd *cdf;
  Eigen::Index first;

  VibEnergyDosSampler(const Histogram &density, const VibIndexPtr &index) : VibEnergySamplerBase(density), cdf(&index->data),
                                                                            first(std::upper_bound(cdf->data(), cdf->data() + cdf->size(), 0.) - cdf->data())
  {
  }

  template <typename GenT>
  double sample(GenT &gen, double E)
  {
    check_energy(E);
    const auto x = density().x();
    const auto count = std::lower_bound(x.data(), x.data() + x.size(), E) - x.data();
    if (first >= count)
      return 0;
    const double total = (*cdf)[count - 1];
    while (true)
    {
      double draw = std::min(unif(gen) * total, std::nextafter(total, 0.));
      auto bin = std::upper_bound(cdf->data(), cdf->data() + count, draw) - cdf->data();
      double u = unif(gen);
      if (u * u * (E - x[first]) < E - x[bin])
        return x[bin];
    }
  }
};

struct VibEnergyGroupSampler : VibEnergySamplerBase
{
  struct Group
  {
    Eigen::Index node;
    double residual;
    double weight;
  };
  const Eigen::ArrayXd *tree;
  std::vector<Group> groups;

  VibEnergyGroupSampler(const Histogram &density, const VibIndexPtr &index) : VibEnergySamplerBase(density), tree(&index->data)
  {
  }

  void cover(Eigen::Index node, Eigen::Index lo, Eigen::Index hi, double E)
  {
    const auto x = density().x();
    if (lo >= x.size() || E <= x[lo] || !std::isfinite((*tree)[node]))
      return;
    double residual = E - x[lo];
    if (hi <= x.size() && residual <= 2 * (E - x[hi - 1]))
    {
      groups.push_back({node, residual, (*tree)[node] + (0.5 * std::log(residual))});
      return;
    }
    const auto mid = lo + ((hi - lo) / 2);
    cover(2 * node, lo, mid, E);
    cover((2 * node) + 1, mid, hi, E);
  }

  template <typename GenT>
  double sample(GenT &gen, double E)
  {
    check_energy(E);
    groups.clear();
    auto leaves = tree->size() / 2;
    cover(1, 0, leaves, E);
    if (groups.empty())
      return 0;
    // Scale only the current cover before forming its small proposal CDF.
    const double peak = std::ranges::max_element(groups, {}, &Group::weight)->weight;
    double total = 0;
    for (auto &group : groups)
    {
      total += std::exp(group.weight - peak);
      group.weight = total;
    }
    while (true)
    {
      double draw = std::min(unif(gen) * total, std::nextafter(total, 0.));
      const auto &group = *std::ranges::upper_bound(groups, draw, {}, &Group::weight);
      auto node = group.node;
      while (node < leaves)
      {
        auto left = 2 * node;
        auto right = left + 1;
        // Test the lighter child directly: the heavier probability may round to one.
        bool left_lighter = (*tree)[left] <= (*tree)[right];
        auto lighter = left_lighter ? left : right;
        auto heavier = left_lighter ? right : left;
        double prob_lighter = std::exp((*tree)[lighter] - (*tree)[node]);
        node = unif(gen) < prob_lighter ? lighter : heavier;
      }
      double x = density().x()[node - leaves];
      double u = unif(gen);
      if (u * u * group.residual < E - x)
        return x;
    }
  }
};
