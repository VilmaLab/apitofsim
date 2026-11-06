#include "fragmentation.h"
#include "consts.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
void add_oscillators(Eigen::ArrayXd &logs, const Eigen::ArrayXd &frequencies, double bin_width)
{
  for (double frequency : frequencies)
  {
    const double bins = frequency * consts::boltzmann / bin_width;
    if (!std::isfinite(bins) || bins < 0.5)
      throw ApiTofArgumentError("Fragmentation bin_width must resolve every positive product vibrational frequency (at least half a bin)");
    if (bins >= static_cast<double>(logs.size()))
      continue; // This oscillator remains in its ground state on this grid.
    const int shift = static_cast<int>(std::floor(bins + 0.5));
    for (Eigen::Index i = shift; i < logs.size(); ++i)
    {
      const double previous = logs[i - shift];
      if (!std::isfinite(previous))
        continue;
      if (!std::isfinite(logs[i]))
        logs[i] = previous;
      else
        logs[i] = std::max(logs[i], previous) + std::log1p(std::exp(-std::abs(logs[i] - previous)));
    }
  }
}

// Log weights avoid products of very large state counts and powers of SI energy.
std::discrete_distribution<int> distribution(const Eigen::ArrayXd &logs)
{
  const double largest = logs.maxCoeff();
  if (!std::isfinite(largest))
    throw ApiTofArgumentError("No accessible fragmentation product states");
  Eigen::ArrayXd weights = (logs - largest).exp();
  return {weights.data(), weights.data() + weights.size()};
}

Eigen::Vector3d direction(double cosine, double azimuth, const Eigen::Vector3d &axis)
{
  const Eigen::Vector3d transverse = axis.unitOrthogonal();
  return (cosine * axis) + (std::sqrt(std::max(0., 1. - (cosine * cosine))) * ((std::cos(azimuth) * transverse) + (std::sin(azimuth) * axis.cross(transverse))));
}
} // namespace

FragmentationDos::FragmentationDos(const Eigen::ArrayXd &frequencies, double bin_width, double energy_max,
                                   std::optional<Eigen::ArrayXd> log_counts)
    : bin_width(bin_width), energy_max(energy_max)
{
  if (!std::isfinite(bin_width) || bin_width <= 0 || !std::isfinite(energy_max) || energy_max < bin_width || energy_max / bin_width >= std::numeric_limits<int>::max())
    throw ApiTofArgumentError("Fragmentation sampling requires a positive DOS grid");
  const int count = static_cast<int>(std::round(energy_max / bin_width));
  if (log_counts)
  {
    if (log_counts->size() != count || (*log_counts)[0] != 0 ||
        !(log_counts->isFinite() || (*log_counts == -std::numeric_limits<double>::infinity())).all())
      throw ApiTofArgumentError("Invalid prepared fragmentation log DOS");
    counts_ = std::move(*log_counts);
  }
  else
  {
    counts_ = Eigen::ArrayXd::Constant(count, -std::numeric_limits<double>::infinity());
    counts_[0] = 0;
    add_oscillators(counts_, frequencies, bin_width);
  }
}

FragmentationDosPtr FragmentationDos::extend(const Eigen::ArrayXd &frequencies) const
{
  auto result = std::make_shared<FragmentationDos>(*this);
  add_oscillators(result->counts_, frequencies, bin_width);
  return result;
}

namespace
{
void validate_products(const MolecularCluster &daughter_cluster, const ClusterSpec &neutral_cluster,
                       const ClusterDerived &daughter, const ClusterDerived &neutral)
{
  if (daughter.mass <= 0 || daughter.inertia_moment <= 0 || !std::isfinite(daughter.inertia_moment) || neutral.mass <= 0 || neutral.inertia_moment < 0 || !std::isfinite(neutral.inertia_moment) || daughter_cluster.charge == 0 || std::visit([](const auto &cluster)
  { return cluster.charge; }, neutral_cluster) != 0)
    throw ApiTofArgumentError("Fragmentation sampling requires a charged molecular daughter and a neutral product");
}
} // namespace

FragmentationProducts::FragmentationProducts(const MolecularCluster &daughter_cluster, const ClusterSpec &neutral_cluster,
                                             double bin_width, double energy_max, std::optional<Eigen::ArrayXXd> log_dos)
    : bin_width(bin_width), daughter(compute_derived(daughter_cluster)), neutral(compute_derived(neutral_cluster))
{
  validate_products(daughter_cluster, neutral_cluster, daughter, neutral);
  const auto *molecule = std::get_if<MolecularCluster>(&neutral_cluster);
  if (log_dos)
  {
    if (log_dos->cols() != 3 || (molecule == nullptr && !(log_dos->col(1) == -std::numeric_limits<double>::infinity()).all()))
      throw ApiTofArgumentError("Invalid prepared fragmentation log DOS");
    daughter_dos_ = std::make_shared<FragmentationDos>(daughter_cluster.frequencies, bin_width, energy_max, log_dos->col(0));
    if (molecule != nullptr)
      neutral_dos_ = std::make_shared<FragmentationDos>(molecule->frequencies, bin_width, energy_max, log_dos->col(1));
    combined_dos_ = std::make_shared<FragmentationDos>(Eigen::ArrayXd{}, bin_width, energy_max, log_dos->col(2));
  }
  else
  {
    daughter_dos_ = std::make_shared<FragmentationDos>(daughter_cluster.frequencies, bin_width, energy_max);
    if (molecule != nullptr)
      neutral_dos_ = std::make_shared<FragmentationDos>(molecule->frequencies, bin_width, energy_max);
    combined_dos_ = molecule != nullptr ? daughter_dos_->extend(molecule->frequencies) : daughter_dos_;
  }
}

FragmentationProducts::FragmentationProducts(const MolecularCluster &daughter_cluster, const ClusterSpec &neutral_cluster,
                                             FragmentationDosPtr daughter_dos_, FragmentationDosPtr neutral_dos_,
                                             FragmentationDosPtr combined_dos_)
    : bin_width(daughter_dos_ ? daughter_dos_->bin_width : 0.),
      daughter_dos_(std::move(daughter_dos_)), neutral_dos_(std::move(neutral_dos_)), combined_dos_(std::move(combined_dos_)),
      daughter(compute_derived(daughter_cluster)), neutral(compute_derived(neutral_cluster))
{
  validate_products(daughter_cluster, neutral_cluster, daughter, neutral);
  const bool molecular = std::holds_alternative<MolecularCluster>(neutral_cluster);
  const auto matches = [&](const FragmentationDosPtr &dos)
  { return dos && this->daughter_dos_ && dos->bin_width == bin_width && dos->energy_max == this->daughter_dos_->energy_max; };
  if (!matches(this->combined_dos_) || (molecular ? !matches(this->neutral_dos_) : this->neutral_dos_ != nullptr))
    throw ApiTofArgumentError("Fragmentation DOS must match the product kinds and grid");
}

Eigen::ArrayXXd FragmentationProducts::log_dos() const
{
  Eigen::ArrayXXd result(daughter_dos_->counts().size(), 3);
  result.col(0) = daughter_dos_->counts();
  if (neutral_dos_)
    result.col(1) = neutral_dos_->counts();
  else
    result.col(1).setConstant(-std::numeric_limits<double>::infinity());
  result.col(2) = combined_dos_->counts();
  return result;
}

std::optional<FragmentationState> FragmentationProducts::sample(
  std::mt19937 &gen, double available_energy, const Eigen::Vector3d &momentum,
  const Eigen::Vector3d &velocity, OperationContext &operation) const
{
  const auto &daughter_log_dos = daughter_dos_->counts();
  const auto &combined_log_dos = combined_dos_->counts();
  if (!std::isfinite(available_energy) || available_energy <= 0 || !momentum.allFinite() || !velocity.allFinite())
    throw ApiTofArgumentError("Fragmentation requires positive finite available energy and finite motion");
  const double max_energy = bin_width * static_cast<double>(combined_log_dos.size());
  if (available_energy > max_energy)
    throw ApiTofDosOverflow(max_energy / consts::boltzmann, available_energy / consts::boltzmann);
  if (!operation.checkpoint())
    return std::nullopt;

  const bool molecular = neutral.inertia_moment > 0;
  const double exponent = molecular ? 2.5 : 1.;
  const double inertia_sum = daughter.inertia_moment + neutral.inertia_moment;
  const double angular_energy = momentum.squaredNorm() / (2 * inertia_sum);
  if (!std::isfinite(angular_energy))
    throw ApiTofArgumentError("Fragmentation angular momentum exceeds the numerical range");
  const Eigen::Vector3d axis = momentum.norm() > 0 ? momentum.normalized().eval() : Eigen::Vector3d::UnitZ().eval();
  const double cosine_max = angular_energy > available_energy ? std::sqrt(available_energy / angular_energy) : 1.;
  const int count = std::min(static_cast<int>(combined_log_dos.size()), static_cast<int>(std::ceil(available_energy / bin_width)));
  Eigen::ArrayXd logs(count);
  for (int i = 0; i < count; ++i)
    logs[i] = combined_log_dos[i] + (exponent * std::log(available_energy - (i * bin_width)));
  auto vibration = distribution(logs);
  std::uniform_real_distribution<double> uniform(0., 1.);

  int total_bin;
  double cosine;
  double remaining;
  // Joint rejection from rho(V) (Q-V)^a and a uniform direction. Truncating
  // cos(theta) to its accessible range also handles large parent J efficiently.
  while (true)
  {
    if (!operation.checkpoint())
      return std::nullopt;
    total_bin = vibration(gen);
    cosine = ((2 * uniform(gen)) - 1) * cosine_max;
    const double budget = available_energy - (total_bin * bin_width);
    remaining = budget - (angular_energy * cosine * cosine);
    if (remaining > 0 && uniform(gen) < std::pow(remaining / budget, exponent))
      break;
  }
  const Eigen::Vector3d recoil = direction(cosine, 2 * consts::pi * uniform(gen), axis);

  int daughter_bin = total_bin;
  if (molecular)
  {
    logs.resize(total_bin + 1);
    for (int i = 0; i <= total_bin; ++i)
      logs[i] = daughter_log_dos[i] + neutral_dos_->counts()[total_bin - i];
    daughter_bin = distribution(logs)(gen);
  }

  // Dirichlet(1, 1, 3/2), or Dirichlet(1, 1) for a spinless neutral.
  std::gamma_distribution<double> gamma_one(1.);
  std::gamma_distribution<double> gamma_spin(1.5);
  double kinetic = gamma_one(gen);
  double transverse = gamma_one(gen);
  double spin = molecular ? gamma_spin(gen) : 0.;
  const double scale = remaining / (kinetic + transverse + spin);
  kinetic *= scale;
  transverse *= scale;
  spin *= scale;
  const Eigen::Vector3d orbital = momentum - (momentum.dot(recoil) * recoil) + (std::sqrt(2 * inertia_sum * transverse) * direction(0., 2 * consts::pi * uniform(gen), recoil));
  Eigen::Vector3d shifted_spin = Eigen::Vector3d::Zero();
  if (molecular)
    shifted_spin = std::sqrt(2 * (daughter.inertia_moment / inertia_sum) * neutral.inertia_moment * spin) * direction((2 * uniform(gen)) - 1, 2 * consts::pi * uniform(gen), Eigen::Vector3d::UnitZ());
  const Eigen::Vector3d daughter_spin = (daughter.inertia_moment / inertia_sum) * (momentum - orbital) + shifted_spin;
  const Eigen::Vector3d neutral_spin = (neutral.inertia_moment / inertia_sum) * (momentum - orbital) - shifted_spin;
  const double mass_sum = daughter.mass + neutral.mass;
  const double reduced_mass = (daughter.mass / mass_sum) * neutral.mass;
  const Eigen::Vector3d relative_velocity = std::sqrt(2 * kinetic / reduced_mass) * recoil;
  FragmentationState state;
  state.velocity = velocity + ((neutral.mass / mass_sum) * relative_velocity);
  state.neutral_velocity = velocity - ((daughter.mass / mass_sum) * relative_velocity);
  state.omega = daughter_spin / daughter.inertia_moment;
  state.neutral_omega = molecular ? (neutral_spin / neutral.inertia_moment).eval() : Eigen::Vector3d::Zero().eval();
  state.orbital_momentum = orbital;
  state.vibrational_energy = daughter_bin * bin_width;
  state.neutral_vibrational_energy = (total_bin - daughter_bin) * bin_width;
  return state;
}
