executable(
  'apitofsim-skimmer',
  'skimmer.cpp',
  dependencies : [eigen3, tbb],
  link_with: [libapitofsim],
  include_directories: [include_dir],
  install : true
)
executable(
  'apitofsim-densityandrate',
  'densityandrate.cpp',
  dependencies: [eigen3, tbb],
  link_with: [libapitofsim],
  include_directories: [include_dir],
  install : true
)
executable(
  'apitofsim-mass-spec',
  'mass_spec.cpp',
  dependencies: [eigen3, tbb, concurrentqueue, magic_enum],
  link_with: [libapitofsim],
  include_directories: [include_dir],
  install : true
)
