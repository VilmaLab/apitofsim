include("${VCPKG_ROOT_DIR}/triplets/x64-linux.cmake")

# oneTBB supports shared linkage. Keep the other Unix ports static.
if(PORT STREQUAL "tbb")
  set(VCPKG_LIBRARY_LINKAGE dynamic)
  # libtbb needs no private library search path. Avoid rewriting its ELF image.
  set(VCPKG_FIXUP_ELF_RPATH OFF)
endif()

# Some compilers default to AVX. The Snappy port disables the feature checks
# that would otherwise include the intrinsics header for that code path.
if(PORT STREQUAL "snappy")
  string(APPEND VCPKG_CXX_FLAGS " -mno-avx")
endif()
