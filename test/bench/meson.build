incdirs = [
   inc_src,
   include_dir,
   include_directories('..')
]

density_of_states = executable(
   'density_of_states',
   ['density_of_states.cpp', gen_compute_k_total],
   dependencies : [eigen3],
   link_with: libapitofsim,
   include_directories: incdirs,
)

benchmark('density_of_states', density_of_states)

compute_k_total = executable(
   'compute_k_total',
   ['compute_k_total.cpp', gen_compute_k_total],
   dependencies : [eigen3],
   link_with: libapitofsim,
   include_directories: incdirs,
)

benchmark('compute_k_total', compute_k_total)
