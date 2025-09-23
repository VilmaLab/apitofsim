project(
  'apitofsim',
  'cpp',
  license : 'MIT',
  default_options: {
    'cpp_std': 'c++17',
    'b_ndebug': 'if-release',
  },
)

python_feat = get_option('python')

openmp = dependency('openmp', include_type : 'system')
eigen3 = dependency('eigen3', include_type : 'system')
nanobind = dependency('nanobind', required : python_feat, include_type : 'system')

subdir('src')

if python_feat.enabled()
  subdir('python')
endif
