include("${CMAKE_CURRENT_LIST_DIR}/x64-osx.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/osx-wheel-common.cmake")

# Included linkage policy must participate in the binary-cache ABI key.
list(APPEND VCPKG_HASH_ADDITIONAL_FILES
  "${CMAKE_CURRENT_LIST_DIR}/x64-osx.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/osx-wheel-common.cmake"
)
