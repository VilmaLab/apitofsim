include("${VCPKG_ROOT_DIR}/triplets/arm64-osx.cmake")

set(VCPKG_OSX_DEPLOYMENT_TARGET 11.0)
# oneTBB supports shared linkage. Keep the other Unix ports static.
if(PORT STREQUAL "tbb")
  set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
