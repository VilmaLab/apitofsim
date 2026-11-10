# dyld can coalesce weak C++ symbols from Arrow and PyArrow even when
# the two libraries have different install names.
set(apitofsim_private_arrow_definitions
  arrow=apitofsim_bundled_arrow
  parquet=apitofsim_bundled_parquet
)
