project(
  'apitofsim',
  'cpp',
  license : 'MIT',
  default_options: {
    'cpp_std': 'c++17',
    'b_ndebug': 'if-release',
  },
  meson_version : '>=1.3.0'
)
if meson.get_compiler('cpp').get_argument_syntax() == 'msvc'
  # Equivalent to no-unknown-pragmas
  add_project_arguments('/wd4068', language : 'cpp')
else
  add_project_arguments('-Wno-unknown-pragmas', language : 'cpp')
  add_project_arguments('-fno-math-errno', language : 'cpp')
endif

python_feat = get_option('python')
ittnotify_feat = get_option('ittnotify')
use_intel_openmp = get_option('use_intel_openmp')

if ittnotify_feat.enabled()
  ittnotify = dependency('ittnotify', include_type : 'system')
  add_project_dependencies(ittnotify, language: 'cpp')
endif
if use_intel_openmp
  openmp = declare_dependency(
    compile_args : ['-fiopenmp'],
    link_args : ['-fiopenmp'],
  )
else
  openmp = dependency('openmp', include_type : 'system')
endif
eigen3 = dependency('eigen3', include_type : 'system', version : '>= 3.4')
nanobind = dependency('nanobind', required : python_feat, include_type : 'system')
concurrentqueue = dependency('concurrentqueue', include_type : 'system')
magic_enum = dependency(
  'magic_enum',
  include_type : 'system',
  default_options : {
    'test': false
  }
)

subdir('src')
subdir('bench')

if python_feat.enabled()
  subdir('python')
endif
