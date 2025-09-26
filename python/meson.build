python = import('python').find_installation()
apitofsim = python.extension_module(
  '_apitofsim',
  sources: ['apitofsim.cpp'],
  dependencies: [nanobind, eigen3, openmp],
  include_directories: [inc_src],
  install: true,
)
python.install_sources(['apitofsim.py'], pure : true)

stubgen = nanobind.get_variable('stubgen')
custom_target(
  output: '_apitofsim.pyi',
  depends: apitofsim,
  command: [python, stubgen, '-m', '_apitofsim', '-M', 'py.typed'],
  #build_by_default: true
)
