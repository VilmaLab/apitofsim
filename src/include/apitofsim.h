#pragma once

#include <cassert>
#include <tuple>
#include <utility>
#include <variant>
#include <Eigen/Dense>
#include "exceptions.h"

// Geometrical mean of moment of inertia
double compute_inertia(const Eigen::Vector3d &rotations);

// Compute mass and radius of cluster
std::tuple<double, double> compute_mass_and_radius(double inertia, double nucleon_count);

Eigen::ArrayXd prepare_energies(double bin_width, int m_max);

struct Gas
{
  double radius;
  double mass;
  double adiabatic_index;
};

// A histogrammed quantity: bin energies in x, bin values in y. The bin width and
// the upper edge are derived from x and cached alongside it.
class Histogram
{
public:
  enum OutOfBounds
  {
    underflow,
    overflow
  };

  Histogram(Eigen::ArrayXd x, Eigen::ArrayXd y)
      : x_(std::move(x)), y_(std::move(y)), derived_(compute_derived(x_))
  {
  }

  Histogram(double bin_width, int m_max, Eigen::ArrayXd y)
      : x_(prepare_energies(bin_width, m_max)), y_(std::move(y)), derived_(compute_derived(x_))
  {
  }

  Eigen::Map<const Eigen::ArrayXd> x() const
  {
    return {x_.data(), x_.size()};
  }

  Eigen::Map<const Eigen::ArrayXd> y() const
  {
    return {y_.data(), y_.size()};
  }

  double bin_width() const
  {
    return derived_.bin_width;
  }

  double x_max() const
  {
    return derived_.x_max;
  }

  // Scale the bin energies in place, keeping the cached derived quantities in
  // step with them.
  void rescale_x(double factor)
  {
    x_ *= factor;
    derived_.bin_width *= factor;
    derived_.x_max *= factor;
  }

  // Scale the bin values in place.
  void rescale_y(double factor)
  {
    y_ /= factor;
  }

  std::variant<double, OutOfBounds> get_lerp(double x) const
  {
    const double bin_width = derived_.bin_width;
    double bin_right = x + (0.5 * bin_width);
    int m = static_cast<int>(bin_right / bin_width);
    double coeff1 = (x - ((m - 0.5) * bin_width)) / bin_width;
    double coeff2 = 1.0 - coeff1;
    if (m >= length())
    {
      if (x > derived_.x_max)
      {
        return OutOfBounds::overflow;
      }
      else
      {
        return y_[length() - 1];
      }
    }
    else if (m > 0)
    {
      assert(coeff1 >= 0.0 && coeff1 <= 1.0);
      return (coeff2 * y_[m - 1]) + (coeff1 * y_[m]);
    }
    else if (m == 0 && x >= 0)
    {
      return y_[0];
    }
    else
    {
      return OutOfBounds::underflow;
    }
  }

  int length() const
  {
    return static_cast<int>(x_.rows());
  }

  double first_y() const
  {
    return y_[0];
  }

  double last_y() const
  {
    return y_[length() - 1];
  }

private:
  struct Derived
  {
    double bin_width{};
    double x_max{};
  };

  // Also rejects a histogram with fewer than two bins, for which x[1] is
  // undefined.
  static Derived compute_derived(const Eigen::ArrayXd &x)
  {
    if (x.size() < 2)
    {
      throw ApiTofArgumentError([&](auto &msg)
      {
        msg << "A histogram needs at least two bins, got " << x.size() << ".";
      });
    }
    Derived derived;
    derived.bin_width = x[1] - x[0];
    derived.x_max = derived.bin_width * static_cast<double>(x.size());
    return derived;
  }

  // Declared in this order: derived_ is computed from x_.
  Eigen::ArrayXd x_;
  Eigen::ArrayXd y_;
  Derived derived_;
};

// An atom-like product has no rotational or vibrational structure: it is
// treated as a point particle.
struct AtomLikeCluster
{
  int nucleon_count{};
  double electronic_energy{};
  int charge{};

  AtomLikeCluster() = default;
  AtomLikeCluster(int nucleon_count, double electronic_energy, int charge);
};

// A cluster with rotational and vibrational structure. The constructor rejects
// an empty set of vibrational temperatures, so every MolecularCluster has at
// least one vibrational mode.
struct MolecularCluster
{
  int nucleon_count{};
  double electronic_energy{};
  Eigen::Vector3d rotations = Eigen::Vector3d::Zero();
  Eigen::ArrayXd frequencies;
  int charge{};

  MolecularCluster(int nucleon_count, double electronic_energy, Eigen::Vector3d rotations, Eigen::ArrayXd frequencies, int charge);

  int num_oscillators() const;
};

using ClusterSpec = std::variant<AtomLikeCluster, MolecularCluster>;

// The products of a fragmentation. Only the second product may be atom-like.
struct MolecularProducts
{
  MolecularCluster first;
  MolecularCluster second;
};

struct AtomLikeProducts
{
  MolecularCluster first;
  AtomLikeCluster second;
};

using Products = std::variant<MolecularProducts, AtomLikeProducts>;

int num_oscillators(const ClusterSpec &spec);
double electronic_energy(const ClusterSpec &spec);
Products make_products(const MolecularCluster &first, const ClusterSpec &second);

// Quantities derived from a ClusterSpec. Computed by compute_derived and passed
// alongside the spec to everything that needs them.
struct ClusterDerived
{
  double inertia_moment{};
  double radius{};
  double mass{};
};

ClusterDerived compute_derived(const ClusterSpec &spec);

struct FragmentationPathway
{
  MolecularCluster parent;
  Products products;

  FragmentationPathway(const MolecularCluster &parent, const MolecularCluster &product1, const ClusterSpec &product2);

  double fragmentation_energy_kelvin() const;
};

void debug_info();

void debug_info_on_env();
