incdirs = [
   include_dir,
   include_directories('..')
]

density_of_states = executable(
   'density_of_states',
   ['density_of_states.cpp', gen_compute_k_total],
   dependencies : [eigen3, tbb],
   link_with: libapitofsim,
   include_directories: incdirs,
)

benchmark('density_of_states', density_of_states)
benchmark(
   'density_of_states-one-thread',
   density_of_states,
   env: 'APITOFSIM_BENCH_CONCURRENCY=1',
)

compute_k_total = executable(
   'compute_k_total',
   ['compute_k_total.cpp', gen_compute_k_total],
   dependencies : [eigen3, tbb],
   link_with: libapitofsim,
   include_directories: incdirs,
)

benchmark('compute_k_total', compute_k_total)
benchmark(
   'compute_k_total-one-thread',
   compute_k_total,
   env: 'APITOFSIM_BENCH_CONCURRENCY=1',
)

mass_spec_benchmark_env = {
   'DATA_DIR': fs.parent(fs.parent(ready_test_data_files[0].full_path())),
}
benchmark(
   'mass_spec',
   tests,
   args: ['--test-case=apitof pinhole smoke tests'],
   env: mass_spec_benchmark_env,
)
mass_spec_benchmark_env += {'APITOFSIM_TEST_CONCURRENCY': '1'}
benchmark(
   'mass_spec-one-thread',
   tests,
   args: ['--test-case=apitof pinhole smoke tests'],
   env: mass_spec_benchmark_env,
)
