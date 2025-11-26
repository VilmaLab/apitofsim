#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include "apitof_pinhole_lib.h"

Histogram read_histogram(char const *filename)
{
  ifstream file;
  char garb[150];
  file.open(filename);

  int m_max = 0;
  while (file >> garb >> garb)
  {
    m_max++;
  }
  file.close();
  if (m_max < 2)
  {
    throw ApiTofError([&](auto &msg)
    {
      msg << "Error in reading file " << filename << ". It should contain at least two rows." << endl;
    });
  }
  Eigen::ArrayXd x(m_max);
  Eigen::ArrayXd y(m_max);

  file.open(filename);
  for (int m = 0; m < m_max; m++)
  {
    file >> x[m] >> y[m];
  }
  file.close();

  return Histogram(x, y);
}

std::tuple<SkimmerData, double> read_skimmer(char const *filename)
{
  int m;
  ifstream file;
  double pos0;
  double pos1;
  char garb[150];
  int m_max;
  file.open(filename);
  file >> garb;

  file >> pos0 >> garb >> garb >> garb >> garb >> garb;
  file >> pos1 >> garb >> garb >> garb >> garb >> garb;
  m = 2;

  double mesh_skimmer = pos1 - pos0;

  while (file >> garb >> garb >> garb >> garb >> garb >> garb)
    m++;
  file.close();

  m_max = m;
  SkimmerData data(m_max, 3);

  file.open(filename);
  file >> garb;
  for (m = 0; m < m_max; m++)
  {
    file >> garb >> data(m, VEL_SKIMMER) >> data(m, TEMP_SKIMMER) >> data(m, PRESSURE_SKIMMER) >> garb >> garb;
  }
  file.close();

  return std::make_tuple(data, mesh_skimmer);
}
