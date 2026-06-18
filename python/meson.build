fs = import('fs')

python = import('python').find_installation()
apitofsim_python = python.extension_module(
  'apitofsimraw',
  sources: ['apitofsim.cpp'],
  dependencies: [nanobind, eigen3, openmp, concurrentqueue, magic_enum],
  link_with: libapitofsim,
  include_directories: [include_dir],
  subdir: 'apitofsim',
  install: true,
  limited_api: '3.12',
)

subdir('apitofsim')

if stubgen_feat.enabled()
  stubgen = nanobind.get_variable('stubgen')
  custom_target(
    output: ['__init__.pyi', 'defaults.pyi', 'consts.pyi'],
    build_subdir: 'apitofsimraw',
    depends: apitofsim_python,
    env: {'PYTHONPATH': fs.parent(apitofsim_python.full_path())},
    command: [python, stubgen, '-r', '-m', 'apitofsimraw'],
    install: true,
    install_dir: python.get_install_dir(pure : false) / 'apitofsim' / 'apitofsimraw',
    build_by_default: true
  )
endif
