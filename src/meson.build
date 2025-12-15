inc_src = include_directories('.')

subdir('include')

libapitofsim = static_library(
  'apitofsim',
  ['apitofsim.cpp', 'skimmer.cpp', 'densityandrate.cpp'],
  include_directories: [include_dir],
  dependencies : [eigen3, openmp],
)

subdir('cli')
