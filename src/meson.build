inc_src = include_directories('.')

subdir('include')

libapitofsim_files = ['apitofsim.cpp', 'skimmer.cpp', 'densityandrate.cpp', 'mass_spec.cpp', 'samplers.cpp', 'openmp_helper.cpp']

libapitofsim = static_library(
  'apitofsim',
  libapitofsim_files,
  include_directories: [include_dir],
  dependencies : [eigen3, openmp, concurrentqueue, magic_enum],
)

if openmp_option == 'disabled'
  libapitofsim_no_omp = libapitofsim
else
  libapitofsim_no_omp = static_library(
    'apitofsim_no_omp',
    libapitofsim_files,
    include_directories: [include_dir],
    dependencies : [eigen3, concurrentqueue, magic_enum],
    build_by_default : false
  )
endif

subdir('cli')
