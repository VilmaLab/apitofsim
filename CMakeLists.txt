cmake_minimum_required(VERSION 3.28)

option(APITOFSIM_USE_VCPKG "Acquire native dependencies using VCPKG_ROOT" ON)
option(APITOFSIM_BUILD_PYTHON "Build the Python extension" OFF)
option(APITOFSIM_GENERATE_STUBS "Generate Python type stubs" ON)
option(APITOFSIM_ITTNOTIFY "Link Intel's ITT profiling API" OFF)
option(APITOFSIM_STRICT_WARNINGS "Enable extra warnings and treat them as errors" OFF)
option(BUILD_TESTING "Build the test suite and benchmarks" ON)
if(APITOFSIM_USE_VCPKG)
  if(NOT CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
      set(CMAKE_TOOLCHAIN_FILE "$ENV{CMAKE_TOOLCHAIN_FILE}")
    elseif(DEFINED ENV{VCPKG_ROOT})
      set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    endif()
  endif()
  if(NOT CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "Set VCPKG_ROOT to a bootstrapped vcpkg checkout or use -DAPITOFSIM_USE_VCPKG=OFF with installed dependencies.")
  endif()
  list(PREPEND VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triplets")
  if(BUILD_TESTING)
    list(APPEND VCPKG_MANIFEST_FEATURES tests)
  endif()
endif()

project(apitofsim LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
include(CheckCXXCompilerFlag)
include(cmake/Dependencies.cmake)

add_library(apitofsim_core OBJECT
  src/apitofsim.cpp
  src/skimmer.cpp
  src/densityandrate.cpp
  src/mass_spec.cpp
  src/fragmentation.cpp
  src/trajectory.cpp
  src/samplers.cpp
  src/operation_context.cpp
)
target_include_directories(apitofsim_core PUBLIC src/include)
target_link_libraries(apitofsim_core PUBLIC
  Eigen3::Eigen TBB::tbb concurrentqueue::concurrentqueue magic_enum::magic_enum
)
if(APITOFSIM_ITTNOTIFY)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ITTNOTIFY REQUIRED IMPORTED_TARGET ittnotify)
  target_link_libraries(apitofsim_core PUBLIC PkgConfig::ITTNOTIFY)
endif()

if(MSVC)
  target_compile_options(apitofsim_core PUBLIC /wd4068)
else()
  target_compile_options(apitofsim_core PUBLIC -fno-math-errno -Wno-unknown-pragmas)
  foreach(flag IN ITEMS -fopenmp-simd -Wno-pass-failed)
    # Check the positive warning form, since GCC accepts unknown -Wno-* flags.
    string(REPLACE "-Wno-" "-W" check_flag "${flag}")
    check_cxx_compiler_flag("${check_flag}" "APITOFSIM_SUPPORTS_${check_flag}")
    if(APITOFSIM_SUPPORTS_${check_flag})
      target_compile_options(apitofsim_core PUBLIC "${flag}")
    endif()
  endforeach()
endif()

if(APITOFSIM_STRICT_WARNINGS)
  if(MSVC)
    target_compile_options(apitofsim_core PUBLIC /W4 /WX)
  else()
    target_compile_options(apitofsim_core PUBLIC -Wall -Wextra -Wpedantic -Werror)
  endif()
endif()

if(APITOFSIM_BUILD_PYTHON)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
endif()
# carquet supplies the Parquet encoder; nanoarrow supplies IPC and C Data types.
set(CARQUET_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(CARQUET_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(CARQUET_ENABLE_BROTLI OFF CACHE BOOL "" FORCE)
find_package(OpenMP REQUIRED COMPONENTS C CXX)
FetchContent_Declare(carquet
  GIT_REPOSITORY https://github.com/Vitruves/carquet.git
  GIT_TAG 10d50f721a1cf695917281ba388ed826084bee64
  SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(carquet)
# Upstream enables LTO in Release independently of the selected linker. Leave
# that choice to this project's toolchain instead of requiring LLVM's linker.
if(NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU"
    AND NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(carquet PRIVATE -fno-lto)
endif()

add_library(apitofsim STATIC src/event_recording.cpp src/parquet_event_writer.cpp)
set_target_properties(apitofsim PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(carquet nanoarrow_static nanoarrow_ipc_static flatccrt PROPERTIES C_VISIBILITY_PRESET hidden)
target_link_libraries(apitofsim PUBLIC apitofsim_core nanoarrow_ipc_static flatccrt
  PRIVATE carquet OpenMP::OpenMP_CXX)
# A separately installed Arrow package can contain unprefixed nanoarrow headers.
target_include_directories(apitofsim SYSTEM BEFORE PUBLIC
  "${nanoarrow_SOURCE_DIR}/src" "${nanoarrow_BINARY_DIR}/src")

include(GNUInstallDirs)
set(apitofsim_bin_dir "${CMAKE_INSTALL_BINDIR}")
if(SKBUILD)
  set(apitofsim_bin_dir "${SKBUILD_SCRIPTS_DIR}")
endif()
if(SKBUILD AND APITOFSIM_USE_VCPKG)
  # Keep the vcpkg runtime available after an isolated wheel build is removed.
  if(WIN32 OR APPLE)
    # vcpkg's Mach-O install name uses the full dylib filename.
    set(tbb_runtime_name "$<TARGET_FILE_NAME:TBB::tbb>")
  else()
    set(tbb_runtime_name "$<TARGET_SONAME_FILE_NAME:TBB::tbb>")
  endif()
  if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
  endif()
  foreach(destination IN ITEMS "${apitofsim_bin_dir}" apitofsim)
    install(FILES "$<TARGET_FILE:TBB::tbb>" DESTINATION "${destination}" RENAME "${tbb_runtime_name}")
  endforeach()
  install(FILES "${TBB_DIR}/copyright" DESTINATION apitofsim/licenses RENAME tbb.LICENSE.txt)
else()
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()
foreach(cli IN ITEMS skimmer densityandrate mass-spec)
  string(REPLACE "-" "_" source "${cli}")
  add_executable(apitofsim-${cli} "src/cli/${source}.cpp")
  target_link_libraries(apitofsim-${cli} PRIVATE apitofsim_core)
  install(TARGETS apitofsim-${cli} RUNTIME DESTINATION "${apitofsim_bin_dir}")
endforeach()

if(APITOFSIM_BUILD_PYTHON)
  add_subdirectory(python)
endif()
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
