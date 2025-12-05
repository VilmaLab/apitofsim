inc_src = include_directories('.')

executable(
   'apitofsim-skimmer',
   'skimmer_win.cpp',
   dependencies : [eigen3, openmp],
   install : true
)
executable(
  'apitofsim-densityandrate',
  'densityandrate_win.cpp',
  dependencies: [eigen3, openmp],
  install : true
)
executable(
  'apitofsim-main',
  'apitof_pinhole.cpp',
  dependencies: [eigen3, openmp, concurrentqueue, magic_enum],
  install : true
)
