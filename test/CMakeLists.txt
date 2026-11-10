find_package(doctest CONFIG REQUIRED)
add_executable(gen_compute_k_total_bin gen/compute_k_total.cpp)
target_link_libraries(gen_compute_k_total_bin PRIVATE apitofsim)
add_custom_command(OUTPUT compute_k_total_data.h
  COMMAND gen_compute_k_total_bin "${CMAKE_CURRENT_BINARY_DIR}/compute_k_total_data.h"
  DEPENDS gen_compute_k_total_bin
  VERBATIM
)

add_executable(apitofsim-tests
  main.cpp fragmentation.cpp test_rf_thinning.cpp event_recording.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/compute_k_total_data.h"
)
target_include_directories(apitofsim-tests PRIVATE "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(apitofsim-tests PRIVATE apitofsim doctest::doctest)

add_executable(signal-helper signal_helper.cpp)
target_link_libraries(signal-helper PRIVATE apitofsim)
add_test(NAME apitofsim-tests COMMAND apitofsim-tests)
add_test(NAME apitofsim-tests-parallel COMMAND apitofsim-tests)
set_tests_properties(apitofsim-tests apitofsim-tests-parallel PROPERTIES
  ENVIRONMENT "DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data"
  TIMEOUT 900
)
set_property(TEST apitofsim-tests APPEND PROPERTY ENVIRONMENT APITOFSIM_TEST_CONCURRENCY=1)

if(APITOFSIM_BUILD_PYTHON)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pytest, pint"
    RESULT_VARIABLE pytest_status OUTPUT_QUIET ERROR_QUIET)
  if(pytest_status EQUAL 0)
    add_test(NAME apitofsim-python-tests
      COMMAND "${Python_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(apitofsim-python-tests PROPERTIES
      ENVIRONMENT "DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data;SIGNAL_HELPER=$<TARGET_FILE:signal-helper>;PYTHONPATH=$<TARGET_FILE_DIR:apitofsimraw>/.."
      TIMEOUT 900
    )
  else()
    message(STATUS "Python tests require pytest and the apitofsim Python dependencies")
  endif()
endif()

foreach(bench IN ITEMS density_of_states compute_k_total compute_k_total_batch)
  add_executable(${bench} bench/${bench}.cpp "${CMAKE_CURRENT_BINARY_DIR}/compute_k_total_data.h")
  target_include_directories(${bench} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${bench} PRIVATE apitofsim)
endforeach()

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE format_sources CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/test/*.cpp" "${PROJECT_SOURCE_DIR}/test/*.h"
    "${PROJECT_SOURCE_DIR}/python/*.cpp")
  add_custom_target(clang-format-check COMMAND "${CLANG_FORMAT}" --dry-run --Werror ${format_sources}
    VERBATIM)
endif()

find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
  get_target_property(tidy_sources apitofsim SOURCES)
  list(TRANSFORM tidy_sources PREPEND "${PROJECT_SOURCE_DIR}/")
  foreach(cli IN ITEMS skimmer densityandrate mass_spec)
    list(APPEND tidy_sources "${PROJECT_SOURCE_DIR}/src/cli/${cli}.cpp")
  endforeach()
  if(APITOFSIM_BUILD_PYTHON)
    list(APPEND tidy_sources "${PROJECT_SOURCE_DIR}/python/apitofsim.cpp")
  endif()
  add_custom_target(clang-tidy COMMAND "${CLANG_TIDY}" -p "${PROJECT_BINARY_DIR}" ${tidy_sources} VERBATIM)
endif()
