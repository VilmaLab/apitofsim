#include "utils.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <optional>
#include <stdlib.h>

using namespace std;

double secant_step(double x0, double x1, double f0, double f1)
{
  return (x0 * f1 - x1 * f0) / (f1 - f0);
}

double f(double x, double *c)
{
  return pow(abs((c[0] / (c[1] - c[2] * x * x))), c[3]) - c[4] * x;
}

// Find lower solution --> a=0 (subsonic flow), find upper solution --> a=1
// (supersonic flow)
double solve_eqn(double c[5], double v0, double v1, double tolerance, int N,
                 int M, int a, int &nwarnings, ofstream &warnings)
{
  double mesh;
  double v;
  double v2 = 0.0;
  double f0;
  double f1;

  mesh = (v1 - v0) / N;

  // find solution for supersonic flow
  if (a == 1)
  {
    for (int i = 0; i < N; i++)
    {
      v = v0 + mesh * i;
      f0 = f(v, c);
      if (f0 > 0)
      {
        v0 = v;
        v1 = v + mesh;
        break;
      }
    }
  }

  // find solution for subsonic flow
  else
  {
    for (int i = 0; i < N; i++)
    {
      v = v0 - mesh * i;
      f0 = f(v, c);
      if (f0 > 0)
      {
        v0 = v - mesh;
        v1 = v;
        break;
      }
    }
  }
  for (int i = 0; i < M; i++)
  {
    f0 = f(v0, c);
    f1 = f(v1, c);
    if (abs((v1 - v0) / v0) < tolerance)
    {
      v2 = v1;
      return v2;
      break;
    }
    v2 = secant_step(v0, v1, f0, f1);
    v0 = v1;
    v1 = v2;
  }
  nwarnings++;
  warnings << "tolerance not reached at " << c[4] << endl;
  return v2;
}

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
  ofstream &warnings;

  Skimmer(double T0_, double P0_, double rmax_, double dc_,
          double alpha_factor_, double m_, double ga_, int N_, int M_,
          int resolution_, double tolerance_, int &nwarnings_,
          ofstream &warnings_)
      : T0(T0_), P0(P0_), rmax(rmax_), dc(dc_), alpha_factor(alpha_factor_),
        m(m_), ga(ga_), N(N_), M(M_), resolution(resolution_),
        tolerance(tolerance_), nwarnings(nwarnings_), warnings(warnings_)
  {
  }

  void next()
  {
    if (i >= resolution)
    {
      cur_row = std::nullopt;
      return;
    }

    if (i == 0)
    {
      rho0 = m * P0 / k / T0;
      alpha = alpha_factor * M_PI;

      vc = sqrt(2.0 * ga * k * T0 / (m * (ga + 1)));
      v_alert = sqrt(2.0 * k * ga * T0 / (m * (ga - 1)));

      c[1] = ga * k * T0 / m;
      c[0] = c[1] - 0.5 * (ga - 1.0) * vc * vc;
      c[2] = 0.5 * (ga - 1.0);
      c[3] = 1.0 / (ga - 1.0);

      r = 1.0e-3;
      c[4] = pow(dc + r * tan(alpha), 2.0) / (vc * dc * dc);

      mesh = rmax / resolution;
    }

    r = mesh * i;
    c[4] = pow(dc + r * tan(alpha), 2.0) / (vc * dc * dc);
    vel = solve_eqn(c, vc, v_alert, tolerance, N, M, 1, nwarnings, warnings);
    T = T0 - 0.5 * vel * vel * m / k * (ga - 1.0) / ga;
    P = P0 * pow(T / T0, ga / (ga - 1.0));
    rho = rho0 * pow(T / T0, 1 / (ga - 1));
    speed_of_sound = sqrt(ga * k * T / m);

    cur_row = {r, vel, T, P, rho, speed_of_sound};
    i++;
  }

  std::optional<SkimmerRow> get()
  {
    return cur_row;
  }

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

  int i = 0;
  std::optional<SkimmerRow> cur_row = std::nullopt;
};
