project('nanoarrow', 'c', version: '0.9.0', default_options: ['c_std=c11', 'warning_level=0', 'werror=false'])
subdir('src/nanoarrow')
nanoarrow_lib = static_library('nanoarrow',
  'src/nanoarrow/common/array.c', 'src/nanoarrow/common/schema.c',
  'src/nanoarrow/common/array_stream.c', 'src/nanoarrow/common/utils.c',
  'src/nanoarrow/ipc/codecs.c', 'src/nanoarrow/ipc/decoder.c',
  'src/nanoarrow/ipc/encoder.c', 'src/nanoarrow/ipc/reader.c', 'src/nanoarrow/ipc/writer.c',
  'thirdparty/flatcc/src/runtime/builder.c', 'thirdparty/flatcc/src/runtime/emitter.c',
  'thirdparty/flatcc/src/runtime/verifier.c', 'thirdparty/flatcc/src/runtime/refmap.c',
  include_directories: include_directories('src', 'thirdparty/flatcc/include'),
  c_args: ['-DNANOARROW_STATIC'],
  install: false,
)
nanoarrow_dep = declare_dependency(link_with: nanoarrow_lib,
  include_directories: include_directories('src', 'thirdparty/flatcc/include', is_system: true))
