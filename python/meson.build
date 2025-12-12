fs = import('fs')

python = import('python').find_installation()
apitofsim_python = python.extension_module(
  'apitofsimraw',
  sources: ['apitofsim.cpp'],
  dependencies: [nanobind, eigen3, openmp, concurrentqueue, magic_enum],
  include_directories: [inc_src],
  subdir: 'apitofsim',
  install: true,
  limited_api: '3.12',
)

subdir('apitofsim')

stubgen = nanobind.get_variable('stubgen')
custom_target(
  output: 'apitofsimraw.pyi',
  depends: apitofsim_python,
  env: {'PYTHONPATH': fs.parent(apitofsim_python.full_path())},
  command: [python, stubgen, '-m', 'apitofsimraw'],
  install: true,
  install_dir: python.get_install_dir(pure : false) / 'apitofsim',
  build_by_default: true
)
