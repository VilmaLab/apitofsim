#include "skimmer_lib.h"

int main()
{
  double tolerance;
  double m;
  double T0;
  double P0;
  double ga;
  double dc;
  double alpha_factor;
  double rmax;
  int N; // number of iterations in finding location of solution in solve_eqn
  int M; // number of iterations in solve_eqn
  int resolution; // number of solved points
  int nwarnings = 0;
  char file_output[150];

  ofstream warnings;
  ofstream output;

  warnings.open("warnings_skimmer.dat");
  warnings << std::scientific;

  // Reading from input
  read_config(
    std::cin,
    nullptr,
    nullptr,
    (int *)nullptr,
    (int *)nullptr,
    (int *)nullptr,
    &T0,
    &P0,
    (double *)nullptr,
    nullptr,
    &rmax,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    &dc,
    &alpha_factor,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    &m,
    &ga,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    file_output,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    &N,
    &M,
    &resolution,
    &tolerance);

  output.open(file_output);
  output << std::scientific << "#Distance_Velocity_Temperature_Pressure_GasMassDensity_SpeedOfSound" << endl;

  Skimmer skimmer = {
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

  while (true)
  {
    skimmer.next();
    auto r = skimmer.get();
    if (r.has_value())
    {
      output << r->r << " " << r->vel << " " << r->T << " " << r->P << " " << r->rho << " " << r->speed_of_sound << endl;
    }
    else
    {
      break;
    }
  }

  if (nwarnings > 0)
    cout << nwarnings << " warnings have been generated: check the file warnings_skimmer.dat" << endl;

  cout << "END OF COMPUTATION" << endl
       << endl;
  cout << "OUTPUT" << endl
       << file_output << endl;
  cout << "###" << endl;

  warnings.close();
  output.close();
  return 0;
}
