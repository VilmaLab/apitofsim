inc_src = include_directories('.')

subdir('include')

libapitofsim_files = ['apitofsim.cpp', 'skimmer.cpp', 'densityandrate.cpp', 'mass_spec.cpp', 'samplers.cpp', 'openmp_helper.cpp']

libapitofsim = static_library(
  'apitofsim',
  libapitofsim_files,
  include_directories: [include_dir],
  dependencies : [eigen3, openmp, concurrentqueue, magic_enum],
)

subdir('cli')
