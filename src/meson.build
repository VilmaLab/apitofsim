inc_src = include_directories('.')

executable(
  'apitof_pinhole',
  'apitof_pinhole.cpp',
  dependencies: [eigen3, openmp],
  install : true
)
executable(
  'densityandrate_win',
  'densityandrate_win.cpp',
  dependencies: [eigen3, openmp],
  install : true
)
executable('skimmer_win', 'skimmer_win.cpp', dependencies : openmp, install : true)
