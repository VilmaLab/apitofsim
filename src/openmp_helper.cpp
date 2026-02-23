#include "openmp_helper.h"

#include <stdexcept>
#include <iostream>

std::atomic<int> saved_signal = -1;

extern "C"
{
  static inline void set_flag_handler(int signal)
  {
    saved_signal.store(signal);
  }
}

class SignalError : public std::runtime_error
{
public:
  int signum;

  SignalError(int signum) : std::runtime_error("Signal-as-exception"), signum(signum)
  {
  }
};

OMPExceptionHelper::OMPExceptionHelper()
{
  for (int i = 0; i < NUM_SIGNALS; i++)
  {
    saved_handlers[i] = std::signal(signals[i], set_flag_handler);
  }
}

OMPExceptionHelper::~OMPExceptionHelper()
{
  if (!rethrow_called)
  {
    if (this->exception)
    {
      std::cerr << "\nException lost! OMPExceptionHelper holding exception destroyed without rethrowing\n"
                << std::flush;
      std::terminate();
    }
    // Do *not* check saved_signal here because if we have nested OMPExceptionHelpers,
    // it will be set after calling rethrow due to the outer exception handler
    // Although ideally we would use signals_as_exceptions in the inner nested handler
  }
}

void OMPExceptionHelper::rethrow(bool signals_as_exceptions)
{
  rethrow_called = true;
  for (int i = 0; i < NUM_SIGNALS; i++)
  {
    std::signal(signals[i], saved_handlers[i]);
  }
  int signal = saved_signal.load();
  if (signal != -1)
  {
    // Prefer to raise the signal over the exception
    // Reason: Application exception is more likely to be caught
    saved_signal.store(-1);
    if (signals_as_exceptions)
    {
      throw SignalError(signal);
    }
    else
    {
      raise(signal);
    }
  }
  if (this->exception)
  {
    try
    {
      std::rethrow_exception(this->exception);
    }
    catch (SignalError &err)
    {
      if (signals_as_exceptions)
      {
        throw;
      }
      else
      {
        raise(err.signum);
      }
    }
  }
}

bool OMPExceptionHelper::should_continue() const
{
  return !this->exception && saved_signal.load() == -1;
}

void OMPExceptionHelper::capture()
{
#pragma omp critical
  if (!this->exception)
  {
    this->exception = std::current_exception();
  }
}
