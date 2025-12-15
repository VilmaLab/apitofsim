#pragma once

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

// TODO: Separate struct for atom-like products
struct ClusterData
{
  int atomic_mass;
  double electronic_energy;
  Eigen::Vector3d rotations;
  Eigen::ArrayXd frequencies;

  // Computed members
  double inertia_moment;
  double radius;
  double mass;

  ClusterData();
  ClusterData(int atomic_mass, double electronic_energy, Eigen::Vector3d rotations, Eigen::ArrayXd frequencies);
  void validate();
  int num_oscillators();
  bool is_atom_like_product();
  void compute_derived();
};

struct FragmentationPathway
{
  ClusterData &parent;
  ClusterData &product1;
  ClusterData &product2;

  double fragmentation_energy_kelvin();
};
