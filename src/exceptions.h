#pragma once

#include <stdexcept>
#include "messages.h"

class ApiTofError : public std::runtime_error
{
public:
  template <typename Arg>
  ApiTofError(Arg arg)
      : std::runtime_error(prepare_message(arg))
  {
  }
};

class ApiTofArgumentError : public ApiTofError
{
  using ApiTofError::ApiTofError;
};

class ApiTofOverflowError : public ApiTofError
{
  using ApiTofError::ApiTofError;
};

template <typename ScaleT>
class ApiTofOverflowErrorTmpl : public ApiTofOverflowError
{
public:
  template <typename Arg>
  ApiTofOverflowErrorTmpl(Arg arg, ScaleT max, ScaleT current)
      : ApiTofOverflowError(arg), max(max), current(current)
  {
  }
  ScaleT max;
  ScaleT current;
};

template <typename ScaleT>
auto mk_msg(const char *main_msg, ScaleT max, ScaleT current)
{
  return ([main_msg, max, current](auto &msg)
  {
    msg << std::setprecision(3) << std::scientific;
    msg << main_msg << " by " << (current - max) << "\n";
    msg << "Current: " << current << " Max: " << max << "\n";
  });
}

class ApiTofDosOverflow : public ApiTofOverflowErrorTmpl<double>
{
public:
  ApiTofDosOverflow(double max, double current) : ApiTofOverflowErrorTmpl(mk_msg("Internal energy exceeds maximum rate energy", max, current), max, current)
  {
  }
};

class ApiTofRateConstantOverflow : public ApiTofOverflowErrorTmpl<double>
{
public:
  ApiTofRateConstantOverflow(double max, double current) : ApiTofOverflowErrorTmpl(mk_msg("Energy exceeds density of states", max, current), max, current)
  {
  }
};

class ApiTofMaxCollisions : public ApiTofOverflowErrorTmpl<int>
{
public:
  ApiTofMaxCollisions(int max, int current) : ApiTofOverflowErrorTmpl(mk_msg("Collisions exceeds maximum", max, current), max, current)
  {
  }
};

class ApiTofUnexpectedNumericalError : public ApiTofError
{
  using ApiTofError::ApiTofError;
};
