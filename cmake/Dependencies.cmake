include(FetchContent)

find_package(Eigen3 CONFIG REQUIRED)
if(Eigen3_VERSION VERSION_LESS 3.4)
  message(FATAL_ERROR "Eigen 3.4 or later is required (found ${Eigen3_VERSION}).")
endif()
find_package(TBB 2023.0 CONFIG REQUIRED)

if(APITOFSIM_USE_VCPKG)
  find_package(concurrentqueue CONFIG REQUIRED)
  find_package(magic_enum CONFIG REQUIRED)
else()
  find_package(concurrentqueue CONFIG QUIET)
  if(NOT TARGET concurrentqueue::concurrentqueue)
    FetchContent_Declare(concurrentqueue
      URL https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.4.tar.gz
      URL_HASH SHA256=87fbc9884d60d0d4bf3462c18f4c0ee0a9311d0519341cac7cbd361c885e5281
      SYSTEM EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(concurrentqueue)
    add_library(concurrentqueue::concurrentqueue ALIAS concurrentqueue)
  endif()
  find_package(magic_enum CONFIG QUIET)
  if(NOT TARGET magic_enum::magic_enum)
    set(MAGIC_ENUM_OPT_BUILD_TESTS OFF)
    set(MAGIC_ENUM_OPT_BUILD_EXAMPLES OFF)
    FetchContent_Declare(magic_enum
      URL https://github.com/Neargye/magic_enum/archive/refs/tags/v0.9.7.tar.gz
      URL_HASH SHA256=b403d3dad4ef542fdc3024fa37d3a6cedb4ad33c72e31b6d9bab89dcaf69edf7
      SYSTEM EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(magic_enum)
  endif()
endif()

# Build nanoarrow privately with IPC and the existing symbol prefix. The writer
# also includes flatcc_generated.h by its path in the nanoarrow source tree.
set(NANOARROW_NAMESPACE apitofsim_ CACHE INTERNAL "Private nanoarrow symbol prefix")
set(NANOARROW_IPC ON CACHE INTERNAL "Build nanoarrow IPC")
set(NANOARROW_INSTALL_SHARED OFF CACHE INTERNAL "Use static nanoarrow")
FetchContent_Declare(nanoarrow
  URL https://github.com/apache/arrow-nanoarrow/archive/refs/tags/apache-arrow-nanoarrow-0.9.0.tar.gz
  URL_HASH SHA256=22c3edeaa5cb384c6da091f32a4e1c937779f2d62f84d646028de658b03d4385
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nanoarrow)
