inc_src = include_directories('.')

subdir('include')

libapitofsim = static_library(
  'apitofsim',
  'skimmer.cpp',
  include_directories: [include_dir],
  dependencies : [eigen3, openmp],
)

subdir('cli')
