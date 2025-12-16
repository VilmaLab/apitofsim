#pragma once

#include <iostream>
#include <optional>

struct SkimmerRow
{
  double r;
  double vel;
  double T;
  double P;
  double rho;
  double speed_of_sound;
};

struct Skimmer
{
  double T0;
  double P0;
  double rmax;
  double dc;
  double alpha_factor;
  double m;
  double ga;
  int N;
  int M;
  int resolution;
  double tolerance;
  int &nwarnings;
  std::ostream &warnings;

  Skimmer(double T0_, double P0_, double rmax_, double dc_,
          double alpha_factor_, double m_, double ga_, int N_, int M_,
          int resolution_, double tolerance_, int &nwarnings_,
          std::ostream &warnings_);

  void next();

  std::optional<SkimmerRow> get();

private:
  double rho0{};
  double k = 1.380648e-23;
  double r{};
  double vc{};
  double v_alert{};
  double c[5]{};
  double alpha{};
  double mesh{};
  double vel{};
  double T{};
  double P{};
  double rho{};
  double speed_of_sound{};

  int i{};
  std::optional<SkimmerRow> cur_row = std::nullopt;
};
