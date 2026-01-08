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
#include <exception>

extern "C"
{
  static inline void set_flag_handler(int signal);
}

extern std::atomic<int> saved_signal;

typedef void (*SignalHandler)(int);

/* Exceptions can't pass between threads.
 * The solution is to capture and rethrow.
 * Additionally once the shared exception is set, no other guarded code can run, preventing further processing.
 * This feature is useful for OpenMP loops, which can't otherwise be cancelled. */
class OMPExceptionHelper
{
  std::exception_ptr exception = nullptr;
  bool rethrow_called = false;
  static const int NUM_SIGNALS = 3;
  static constexpr int signals[NUM_SIGNALS] = {SIGTERM, SIGINT, SIGABRT};
  SignalHandler saved_handlers[NUM_SIGNALS];

public:
  OMPExceptionHelper();
  ~OMPExceptionHelper();
  void rethrow(bool signals_as_exceptions = false);
  void capture();

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
