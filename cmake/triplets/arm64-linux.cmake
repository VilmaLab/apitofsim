include("${VCPKG_ROOT_DIR}/triplets/arm64-linux.cmake")

# oneTBB supports shared linkage. Keep the other Unix ports static.
if(PORT STREQUAL "tbb")
  set(VCPKG_LIBRARY_LINKAGE dynamic)
  # libtbb needs no private library search path. Avoid rewriting its ELF image.
  set(VCPKG_FIXUP_ELF_RPATH OFF)
endif()
