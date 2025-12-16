#pragma once

// Total collision frequency
double coll_freq(double n, double mobility_gas, double mobility_gas_inv, double R, double v);

// Collision frequency on angle theta
double coll_freq_theta(double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v);

// Collision frequency on angle theta and gas velocity
double coll_freq_theta_u(double u, double theta, double n, double mobility_gas_inv, double R, double v);

// Distribution of angle theta
double distr_theta(double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v);

// Distribution of gas velocity
double distr_u(double u, double theta, double n, double mobility_gas, double mobility_gas_inv, double R, double v);

#include "samplers.tpp"
