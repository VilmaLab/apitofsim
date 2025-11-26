incdir = [
   include_directories('../../src'),
   include_directories('..')
]

density_of_states = executable(
   'density_of_states',
   'density_of_states.cpp',
   dependencies : [eigen3],
   include_directories: incdir
)

benchmark('density_of_states', density_of_states)

compute_k_total = executable(
   'compute_k_total',
   ['compute_k_total.cpp', gen_compute_k_total],
   dependencies : [eigen3],
   include_directories: incdir
)

benchmark('compute_k_total', compute_k_total)
