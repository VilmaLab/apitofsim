inc_src = include_directories('.')

subdir('include')

libapitofsim_files = ['apitofsim.cpp', 'skimmer.cpp', 'densityandrate.cpp', 'mass_spec.cpp', 'fragmentation.cpp', 'trajectory.cpp', 'samplers.cpp', 'operation_context.cpp', 'event_recording.cpp']
if native_parquet
  libapitofsim_files += ['parquet_event_writer.cpp']
endif

libapitofsim = static_library(
  'apitofsim',
  libapitofsim_files,
  include_directories: [include_dir],
  dependencies : [eigen3, tbb, concurrentqueue, magic_enum, nanoarrow],
)

subdir('cli')
