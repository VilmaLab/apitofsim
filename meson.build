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
  add_project_arguments('/wd4068', language : 'cpp')
else
  add_project_arguments('-Wno-unknown-pragmas', language : 'cpp')
endif

python_feat = get_option('python')

openmp = dependency('openmp', include_type : 'system')
eigen3 = dependency('eigen3', include_type : 'system')
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

if python_feat.enabled()
  subdir('python')
endif
