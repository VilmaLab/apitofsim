executable(
  'apitofsim-skimmer',
  'skimmer_win.cpp',
  dependencies : [eigen3, openmp],
  link_with: [libapitofsim],
  include_directories: [include_dir, inc_src],
  install : true
)
executable(
  'apitofsim-densityandrate',
  'densityandrate_win.cpp',
  dependencies: [eigen3, openmp],
  link_with: [libapitofsim],
  include_directories: [include_dir, inc_src],
  install : true
)
executable(
  'apitofsim-main',
  'apitof_pinhole.cpp',
  dependencies: [eigen3, openmp, concurrentqueue, magic_enum],
  include_directories: [include_dir, inc_src],
  install : true
)
