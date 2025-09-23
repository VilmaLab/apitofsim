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

openmp = dependency('openmp')
eigen3 = dependency('eigen3')
nanobind = dependency('nanobind', required : python_feat)

subdir('src')

if python_feat.enabled()
  subdir('python')
endif
