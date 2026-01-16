#pragma once

#include "apitofsim.h"

#include <Eigen/Dense>
#include <optional>

typedef Eigen::Array<double, Eigen::Dynamic, 4> DensityResult;
const int C0_ROW = 0;
const int C1_ROW = 1;
const int C2_ROW = 2;
const int COMB_ROW = 3;

enum struct MeshMode
{
  no_mesh,
  compute_mesh_single_threaded,
  compute_mesh_diagonal_single_threaded,
  compute_mesh_multithreaded,
  compute_mesh_diagonal_multithreaded
};

struct KTotalInput
{
  ClusterData cluster_1;
  ClusterData cluster_2;
  double fragmentation_energy;
  Eigen::Ref<Eigen::ArrayXd> rho_parent;
  Eigen::Ref<Eigen::ArrayXd> rho_comb;
};


void validate_max_energies(double fragmentation_energy, double energy_max, double energy_max_rate, double bin_width);
void validate_max_energies(int n_fragmentation, int m_max, int m_max_rate, double bin_width);
void validate_max_energies(double fragmentation_energy, int m_max, int m_max_rate, double bin_width);
void compute_density_of_states(Eigen::ArrayXd &frequencies, Eigen::Ref<Eigen::ArrayXd> rho, double energy_max, double bin_width);
void compute_combined_density_of_states(Eigen::Ref<Eigen::ArrayXd> rho_comb, Eigen::ArrayXd &frequencies_1, Eigen::ArrayXd &frequencies_2, double energy_max, double bin_width);
Eigen::ArrayXd combine_frequencies(Eigen::ArrayXd &frequencies_1, Eigen::ArrayXd &frequencies_2);
Eigen::ArrayXd prepare_energies(double bin_width, int m_max);
void compute_k_total(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, double inertia_moment_2, Eigen::Vector3d &rotations_1, Eigen::Vector3d &rotations_2, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy);
void compute_k_total_atom(Eigen::ArrayXd &k0, Eigen::Ref<Eigen::ArrayXd> k_rate, double inertia_moment_1, const Eigen::Ref<const Eigen::ArrayXd> rho_comb, const Eigen::Ref<const Eigen::ArrayXd> rho_0, double bin_width, int m_max_rate, double fragmentation_energy);
Eigen::ArrayXd compute_k_total_full(ClusterData &cluster_0, ClusterData &cluster_1, ClusterData &cluster_2, DensityResult &rhos, double fragmentation_energy, double energy_max_rate, double bin_width);
DensityResult compute_density_of_states_all(ClusterData &cluster_0, ClusterData &cluster_1, ClusterData &cluster_2, double energy_max, double bin_width);
Eigen::ArrayXXd compute_density_of_states_batch(std::vector<Eigen::ArrayXd> batch_frequencies, double energy_max, double bin_width, bool use_old_impl = false);
Eigen::ArrayXd precompute_mesh(double energy_max_rate, double bin_width, MeshMode mesh_mode = MeshMode::compute_mesh_single_threaded);
Eigen::ArrayXXd compute_k_total_batch(std::vector<KTotalInput> batch_input, double energy_max_rate, double bin_width, std::optional<Eigen::ArrayXd> mesh, std::optional<std::function<void(size_t)>> progress_callback = std::nullopt);
Eigen::ArrayXXd compute_k_total_batch(std::vector<KTotalInput> batch_input, double energy_max_rate, double bin_width, MeshMode mesh_mode = MeshMode::compute_mesh_diagonal_multithreaded, std::optional<std::function<void(size_t)>> progress_callback = std::nullopt);
