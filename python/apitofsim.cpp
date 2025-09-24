#include <iostream>
#include <stdlib.h>

#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include "skimmer_lib.h"
#include "densityandrate_lib.h"

namespace nb = nanobind;

typedef Eigen::Array<double, Eigen::Dynamic, 6> SkimmerResult;

SkimmerResult skimmer(
  double T0,
  double P0,
  double rmax,
  double dc,
  double alpha_factor,
  double m,
  double ga,
  int N,
  int M,
  int resolution,
  double tolerance)
{
  int nwarnings = 0;
  std::ofstream warnings;
  warnings.open("warnings_skimmer.dat");
  warnings << std::scientific;
  Skimmer s = {
    T0,
    P0,
    rmax,
    dc,
    alpha_factor,
    m,
    ga,
    N,
    M,
    resolution,
    tolerance,
    nwarnings,
    warnings,
  };

  SkimmerResult result(resolution, 6);

  int i = 0;
  while (true)
  {
    s.next();
    auto r = s.get();
    if (r.has_value())
    {
      result(i, 0) = r->r;
      result(i, 1) = r->vel;
      result(i, 2) = r->T;
      result(i, 3) = r->P;
      result(i, 4) = r->rho;
      result(i, 5) = r->speed_of_sound;
      i++;
    }
    else
    {
      break;
    }
  }
  return result;
}

nb::tuple densityandrate(
  ClusterData &cluster_0,
  ClusterData &cluster_1,
  ClusterData &cluster_2,
  double energy_max,
  double energy_max_rate,
  double bin_width,
  double fragmentation_energy)
{
  // TODO:: The individual product densities are not used -- should be possible to not compute them
  DensityResult rhos = compute_density_of_states_all(cluster_0, cluster_1, cluster_2, energy_max, bin_width);
  const Eigen::ArrayXd k_rate = compute_k_total_full(
    cluster_0,
    cluster_1,
    cluster_2,
    rhos,
    fragmentation_energy,
    energy_max_rate,
    bin_width);
  return nb::make_tuple(rhos, k_rate);
}

NB_MODULE(_apitofsim, m)
{
  m.doc() = "APi-TOF-MS simulation module";
  m.def("skimmer", &skimmer);

  nb::class_<ClusterData>(m, "ClusterData")
    .def(nb::init<int, double, Eigen::Vector3d, Eigen::ArrayXd>(),
         nb::arg("atomic_mass"),
         nb::arg("electronic_energy"),
         nb::arg("rotations"),
         nb::arg("frequencies"))
    .def_ro("atomic_mass", &ClusterData::atomic_mass)
    .def_ro("electronic_energy", &ClusterData::electronic_energy)
    .def_ro("rotations", &ClusterData::rotations)
    .def_ro("frequencies", &ClusterData::frequencies)
    .def_ro("inertia_moment", &ClusterData::inertia_moment)
    .def_ro("radius", &ClusterData::radius)
    .def_ro("mass", &ClusterData::mass)
    .def("num_oscillators", &ClusterData::num_oscillators)
    .def("is_atom_like_product", &ClusterData::is_atom_like_product)
    .def("compute_derived", &ClusterData::compute_derived);

  nb::class_<Gas>(m, "Gas")
    .def(nb::init<double, double>(),
         nb::arg("radius"),
         nb::arg("mass"))
    .def_ro("radius", &Gas::radius)
    .def_ro("mass", &Gas::mass);

  m.def("densityandrate", &densityandrate);
}
