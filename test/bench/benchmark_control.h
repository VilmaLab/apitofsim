#pragma once

#include <cstdlib>
#include <string>
#include <utility>

#include <oneapi/tbb/task_arena.h>

template <typename Function>
int run_benchmark(Function &&function)
{
  const char *value = std::getenv("APITOFSIM_BENCH_CONCURRENCY");
  if (value == nullptr)
  {
    return std::forward<Function>(function)();
  }
  oneapi::tbb::task_arena arena(std::stoul(value));
  return arena.execute(std::forward<Function>(function));
}
