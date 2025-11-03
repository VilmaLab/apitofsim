incdir = include_directories('../src')

density_of_states = executable(
   'density_of_states',
   'density_of_states.cpp',
   dependencies : [eigen3, ittnotify],
   include_directories: incdir
)

benchmark('density_of_states', density_of_states)

gen_compute_k_total_bin = executable(
   'gen_compute_k_total_bin',
   'gen/compute_k_total.cpp',
   dependencies : [eigen3, ittnotify],
   include_directories: incdir
)

gen_compute_k_total = custom_target('gen_compute_k_total',
                        input : [],
                        output : ['compute_k_total_data.h'],
                        command : [gen_compute_k_total_bin, '@OUTPUT0@'])

compute_k_total = executable(
   'compute_k_total',
   ['compute_k_total.cpp', gen_compute_k_total],
   dependencies : [eigen3, ittnotify],
   include_directories: incdir
)

benchmark('compute_k_total', compute_k_total)
