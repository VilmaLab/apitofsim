fs = import('fs')

python = import('python').find_installation()
apitofsim = python.extension_module(
  '_apitofsim',
  sources: ['apitofsim.cpp'],
  dependencies: [nanobind, eigen3, openmp, concurrentqueue, magic_enum],
  include_directories: [inc_src],
  install: true,
  limited_api: '3.12',
)
python.install_sources([
'apitofsim/api.py',
'apitofsim/cli.py',
'apitofsim/config.py',
'apitofsim/__init__.py',
'apitofsim/__main__.py',
], subdir: 'apitofsim', pure : true)

stubgen = nanobind.get_variable('stubgen')
custom_target(
  output: '_apitofsim.pyi',
  depends: apitofsim,
  env: {'PYTHONPATH': fs.parent(apitofsim.full_path())},
  command: [python, stubgen, '-m', '_apitofsim'],
  install_dir: python.get_install_dir(),
  build_by_default: true
)
