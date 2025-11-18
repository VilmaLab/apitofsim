inc_src = include_directories('.')

executable(
   'skimmer_win',
   'skimmer_win.cpp',
   dependencies : [eigen3, openmp],
   install : true
)
executable(
  'densityandrate_win',
  'densityandrate_win.cpp',
  dependencies: [eigen3, openmp],
  install : true
)
executable(
  'apitof_pinhole',
  'apitof_pinhole.cpp',
  dependencies: [eigen3, openmp, concurrentqueue, magic_enum],
  install : true
)
