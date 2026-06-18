#pragma once

#include <cassert>
#include <variant>
#include <Eigen/Dense>

// Geometrical mean of moment of inertia
double compute_inertia(const Eigen::Vector3d &rotations);

// Compute radius of cluster
void compute_mass_and_radius(double inertia, double amu, double &mass, double &radius);

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
    double bin_right = x + 0.5 * bin_width;
    int m = int(bin_right / bin_width);
    double coeff1 = (x - (m - 0.5) * bin_width) / bin_width;
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
      return coeff2 * y[m - 1] + coeff1 * y[m];
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

// TODO: Separate struct for atom-like products
struct ClusterData
{
  int atomic_mass;
  double electronic_energy;
  Eigen::Vector3d rotations;
  Eigen::ArrayXd frequencies;
  int charge;

  // Computed members
  double inertia_moment;
  double radius;
  double mass;

  ClusterData();
  ClusterData(int atomic_mass, double electronic_energy, Eigen::Vector3d rotations, Eigen::ArrayXd frequencies, int charge);
  void validate();
  int num_oscillators();
  bool is_atom_like_product();
  void compute_derived();
};

struct FragmentationPathway
{
  ClusterData parent;
  ClusterData product1;
  ClusterData product2;

  double fragmentation_energy_kelvin();
};

void debug_info();

void debug_info_on_env();
