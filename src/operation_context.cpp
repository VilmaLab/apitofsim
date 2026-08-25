#include "operation_context.h"

#include <array>
#include <csignal>

namespace
{
using SignalHandler = void (*)(int);

constexpr std::array<int, 2> cooperative_signals{SIGTERM, SIGINT};
volatile std::sig_atomic_t pending_signal = 0;
std::array<SignalHandler, cooperative_signals.size()> saved_handlers{};
std::mutex handler_mutex;
size_t active_contexts = 0;
bool handlers_installed = false;

extern "C" void record_signal(int signum)
{
  if (pending_signal == 0)
  {
    pending_signal = signum;
  }
}

void install_handlers()
{
  for (size_t i = 0; i < cooperative_signals.size(); ++i)
  {
    saved_handlers[i] = std::signal(cooperative_signals[i], record_signal);
  }
  handlers_installed = true;
}

void restore_handlers()
{
  for (size_t i = 0; i < cooperative_signals.size(); ++i)
  {
    std::signal(cooperative_signals[i], saved_handlers[i]);
  }
  handlers_installed = false;
}
} // namespace

OperationContext::OperationContext()
    : concurrency_limit(
#ifdef APITOFSIM_MAX_PARALLELISM
        std::make_unique<oneapi::tbb::global_control>(
          oneapi::tbb::global_control::max_allowed_parallelism,
          APITOFSIM_MAX_PARALLELISM)),
#else
        nullptr),
#endif
      context(oneapi::tbb::task_group_context::isolated)
{
  const std::lock_guard<std::mutex> lock(handler_mutex);
  if (active_contexts++ == 0)
  {
    pending_signal = 0;
    install_handlers();
  }
}

OperationContext::~OperationContext()
{
  const std::lock_guard<std::mutex> lock(handler_mutex);
  if (--active_contexts == 0)
  {
    if (handlers_installed)
    {
      restore_handlers();
    }
    pending_signal = 0;
  }
}

bool OperationContext::checkpoint()
{
  if (pending_signal != 0)
  {
    context.cancel_group_execution();
  }
  return !context.is_group_execution_cancelled();
}

bool OperationContext::should_continue()
{
  return checkpoint();
}

oneapi::tbb::task_group_context &OperationContext::tbb_context()
{
  return context;
}

void OperationContext::rethrow_pending_signal(bool signals_as_exceptions)
{
  checkpoint();
  const int signum = pending_signal;
  if (signum == 0)
  {
    return;
  }
  if (signals_as_exceptions)
  {
    throw SignalError(signum);
  }

  {
    const std::lock_guard<std::mutex> lock(handler_mutex);
    if (handlers_installed)
    {
      restore_handlers();
    }
    pending_signal = 0;
  }
  std::raise(signum);
  {
    const std::lock_guard<std::mutex> lock(handler_mutex);
    if (active_contexts > 0 && !handlers_installed)
    {
      install_handlers();
    }
  }
}

void ExceptionTransport::capture()
{
  const std::lock_guard<std::mutex> lock(mutex);
  if (!exception)
  {
    exception = std::current_exception();
  }
}

bool ExceptionTransport::should_continue() const
{
  const std::lock_guard<std::mutex> lock(mutex);
  return !exception;
}

void ExceptionTransport::rethrow() const
{
  std::exception_ptr captured;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    captured = exception;
  }
  if (captured)
  {
    std::rethrow_exception(captured);
  }
}
