#pragma once

// This file is responsible for including OpenMP -- don't include OpenMP directly, include this file instead

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

// The rest of the file is a helper for exception/signal handling

#include <atomic>
#include <csignal>
#include <iostream>

static std::atomic<int> saved_signal = -1;

extern "C" void set_flag_handler(int signal)
{
  saved_signal.store(signal);
}

typedef void (*SignalHandler)(int);

/* Exceptions can't pass between threads.
 * The solution is to capture and rethrow.
 * Additionally once the shared exception is set, no other guarded code can run, preventing further processing. */
class OMPExceptionHelper
{
  std::exception_ptr exception = nullptr;
  bool rethrow_called = false;
  static const int NUM_SIGNALS = 3;
  static constexpr int signals[NUM_SIGNALS] = {SIGTERM, SIGINT, SIGABRT};
  SignalHandler saved_handlers[NUM_SIGNALS];

public:
  OMPExceptionHelper()
  {
    for (int i = 0; i < NUM_SIGNALS; i++)
    {
      saved_handlers[i] = std::signal(signals[i], set_flag_handler);
    }
  }

  ~OMPExceptionHelper()
  {
    if (!rethrow_called)
    {
      bool should_die = false;
      if (this->exception)
      {
        std::cerr << "\nException lost! OMPExceptionHelper holding exception destroyed without rethrowing\n"
                  << std::flush;
        should_die = true;
      }
      if (saved_signal.load() != -1)
      {
        std::cerr << "\nSIGTERM flag set, but OMPExceptionHelper was destroyed without rethrowing\n"
                  << std::flush;
        should_die = true;
      }
      if (should_die)
      {
        std::terminate();
      }
    }
  }

  void rethrow()
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
      raise(signal);
    }
    if (this->exception)
    {
      std::rethrow_exception(this->exception);
    }
  }

  void capture()
  {
#pragma omp critical
    if (!this->exception)
    {
      this->exception = std::current_exception();
    }
  }

  template <typename Function, typename... Parameters>
  void guard(Function f, Parameters... params)
  {
    if (!this->exception && saved_signal.load() == -1)
    {
      try
      {
        f(params...);
      }
      catch (...)
      {
        capture();
      }
    }
  }
};
