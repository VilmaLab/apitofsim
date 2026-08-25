#include "operation_context.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

int main()
{
  OperationContext operation;
  std::cout << "ready" << std::endl;
  oneapi::tbb::parallel_for(
    oneapi::tbb::blocked_range<int>(0, 100000),
    [&](const oneapi::tbb::blocked_range<int> &range)
  {
    for (int i = range.begin(); i != range.end() && operation.checkpoint(); ++i)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  },
    operation.tbb_context());
  operation.rethrow_pending_signal();
}
