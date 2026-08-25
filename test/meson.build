fs = import('fs')
subdir('data')

gen_compute_k_total_bin = executable(
   'gen_compute_k_total_bin',
   'gen/compute_k_total.cpp',
   dependencies : [eigen3],
   link_with: libapitofsim,
   include_directories: [include_dir],
)

gen_compute_k_total = custom_target('gen_compute_k_total',
                        input : [],
                        output : ['compute_k_total_data.h'],
                        command : [gen_compute_k_total_bin, '@OUTPUT0@'],
                        build_by_default : true)
doctest_dep = dependency('doctest')

signal_helper = executable(
   'signal-helper',
   'signal_helper.cpp',
   link_with: libapitofsim,
   include_directories: [include_dir],
)

tests = executable('apitofsim-tests',
                   ['main.cpp', gen_compute_k_total],
                   dependencies : [doctest_dep, eigen3, concurrentqueue, magic_enum],
                   link_with: libapitofsim,
                   include_directories: [inc_src, include_dir],
                   install : false)

# Longer timeout needed mainly for Valgrind runs
timeout = 900

test(
   'apitofsim-tests',
   tests,
   env : {
      'DATA_DIR': fs.parent(fs.parent(ready_test_data_files[0].full_path())),
      'APITOFSIM_TEST_CONCURRENCY': '1',
   },
   depends: ready_test_data_files,
   timeout : timeout
)

test(
   'apitofsim-tests-parallel',
   tests,
   env : 'DATA_DIR=' + fs.parent(fs.parent(ready_test_data_files[0].full_path())),
   depends: ready_test_data_files,
   timeout : timeout
)

if python_feat.enabled()
   env = environment()
   env.set('DATA_DIR', fs.parent(fs.parent(raw_test_data_files[0].full_path())))
   env.set('SIGNAL_HELPER', signal_helper.full_path())
   env.append(
      'PYTHONPATH',
      meson.project_source_root() / 'python',
      fs.parent(apitofsim_python.full_path()),
      fs.parent(apitofsim_python_py_builddir[0].full_path()),
   )
   r = run_command(python, '-c', 'import pytest, pint', check: false)
   if r.returncode() != 0
      message('Cannot run tests without pytest + other Python requirements, skipping Python tests.')
      message('Usual fix: Run Python tests with UV like so:')
      message('uv run meson setup --clearcache --reconfigure mybuilddir && uv run meson test -C mybuilddir')
   else
      test(
         'apitofsim-python-tests',
         python,
         args: ['-m', 'pytest', files('test_python.py')],
         env: env,
         depends: [raw_test_data_files, apitofsim_python, apitofsim_python_py_builddir, signal_helper],
         timeout: timeout
      )
   endif
endif

subdir('bench')
