#pragma once

#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_group.h>

class SignalError : public std::runtime_error
{
public:
  int signum;

  explicit SignalError(int signum)
      : std::runtime_error("Signal-as-exception"), signum(signum)
  {
  }
};

class OperationContext
{
  std::unique_ptr<oneapi::tbb::global_control> concurrency_limit;
  oneapi::tbb::task_group_context context;

public:
  OperationContext();
  ~OperationContext();

  OperationContext(const OperationContext &) = delete;
  OperationContext &operator=(const OperationContext &) = delete;

  bool checkpoint();
  bool should_continue();
  oneapi::tbb::task_group_context &tbb_context();
  void rethrow_pending_signal(bool signals_as_exceptions = false);

  template <typename Function>
  decltype(auto) run(Function &&function, bool signals_as_exceptions = false)
  {
    try
    {
      if constexpr (std::is_void_v<std::invoke_result_t<Function>>)
      {
        std::forward<Function>(function)();
        rethrow_pending_signal(signals_as_exceptions);
        return;
      }
      else
      {
        auto result = std::forward<Function>(function)();
        rethrow_pending_signal(signals_as_exceptions);
        return result;
      }
    }
    catch (...)
    {
      rethrow_pending_signal(signals_as_exceptions);
      throw;
    }
  }
};

class ExceptionTransport
{
  mutable std::mutex mutex;
  std::exception_ptr exception;

public:
  void capture();
  bool should_continue() const;
  void rethrow() const;

  template <typename Function, typename... Parameters>
  void guard(Function &&function, Parameters &&...parameters)
  {
    if (!should_continue())
    {
      return;
    }
    try
    {
      std::forward<Function>(function)(std::forward<Parameters>(parameters)...);
    }
    catch (...)
    {
      capture();
    }
  }
};
