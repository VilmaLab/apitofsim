#pragma once

#include <cassert>
#include <tuple>
#include <variant>
#include <Eigen/Dense>

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

struct Histogram
{
  Eigen::ArrayXd x;
  Eigen::ArrayXd y;
  double bin_width;
  double x_max;

  enum OutOfBounds
  {
    underflow,
    overflow
  };

  Histogram(Eigen::ArrayXd x, Eigen::ArrayXd y)
      : x(x), y(y)
  {
    compute_derived();
  }

  Histogram(double bin_width, int m_max, Eigen::ArrayXd y)
      : x(prepare_energies(bin_width, m_max)), y(y)
  {
    compute_derived();
  }

  std::variant<double, OutOfBounds> get_lerp(double x) const
  {
    double bin_right = x + (0.5 * bin_width);
    int m = static_cast<int>(bin_right / bin_width);
    double coeff1 = (x - ((m - 0.5) * bin_width)) / bin_width;
    double coeff2 = 1.0 - coeff1;
    if (m >= length())
    {
      if (x > x_max)
      {
        return OutOfBounds::overflow;
      }
      else
      {
        return y[length() - 1];
      }
    }
    else if (m > 0)
    {
      assert(coeff1 >= 0.0 && coeff1 <= 1.0);
      return (coeff2 * y[m - 1]) + (coeff1 * y[m]);
    }
    else if (m == 0 && x >= 0)
    {
      return y[0];
    }
    else
    {
      return OutOfBounds::underflow;
    }
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

  double first_y() const
  {
    return y[0];
  }

  double last_y() const
  {
    return y[length() - 1];
  }
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
