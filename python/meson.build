fs = import('fs')

python = import('python').find_installation()
python_cpp = meson.get_compiler('cpp')
python_link_args = []
if python_cpp.get_id() == 'clang-cl'
  # Meson only suppresses CPython's version-specific auto-link pragma for MSVC.
  python_link_args += '/NODEFAULTLIB:python312.lib'
endif
apitofsim_python = python.extension_module(
  'apitofsimraw',
  sources: ['apitofsim.cpp'],
  dependencies: [nanobind, eigen3, tbb, concurrentqueue, magic_enum],
  cpp_args: python_cpp.get_supported_arguments('-Wno-maybe-uninitialized'),
  link_args: python_link_args,
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
