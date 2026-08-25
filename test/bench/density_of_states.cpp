#include "densityandrate_smoke.h"
#include "benchmark_control.h"

int main()
{
  return run_benchmark([]
  {
    volatile auto result = dos_smoke();
    return 0;
  });
}
