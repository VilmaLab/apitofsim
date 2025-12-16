executable(
  'apitofsim-skimmer',
  'skimmer_win.cpp',
  dependencies : [eigen3, openmp],
  link_with: [libapitofsim],
  include_directories: [include_dir],
  install : true
)
executable(
  'apitofsim-densityandrate',
  'densityandrate_win.cpp',
  dependencies: [eigen3, openmp],
  link_with: [libapitofsim],
  include_directories: [include_dir],
  install : true
)
executable(
  'apitofsim-main',
  'apitof_pinhole.cpp',
  dependencies: [eigen3, openmp, concurrentqueue, magic_enum],
  link_with: [libapitofsim],
  include_directories: [include_dir],
  install : true
)
