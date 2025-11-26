fs = import('fs')
subdir('data')

incdir = include_directories('../src')

gen_compute_k_total_bin = executable(
   'gen_compute_k_total_bin',
   'gen/compute_k_total.cpp',
   dependencies : [eigen3],
   include_directories: incdir
)

gen_compute_k_total = custom_target('gen_compute_k_total',
                        input : [],
                        output : ['compute_k_total_data.h'],
                        command : [gen_compute_k_total_bin, '@OUTPUT0@'],
                        build_by_default : true)
doctest_dep = dependency('doctest', required : true)

tests = executable('apitofsim-tests',
                   ['main.cpp', gen_compute_k_total],
                   dependencies : [doctest_dep, nanobind, eigen3, concurrentqueue, magic_enum],
                   include_directories: incdir,
                   install : false)

test(
   'apitofsim-tests',
   tests,
   env : 'DATA_DIR=' + fs.parent(test_data_files[0].full_path()),
   depends: test_data_files,
   timeout : 300
)

subdir('bench')
