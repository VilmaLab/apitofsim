if(NOT DEFINED nanobind_DIR)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(apitofsimraw STABLE_ABI NOMINSIZE NB_SUPPRESS_WARNINGS apitofsim.cpp)
target_link_libraries(apitofsimraw PRIVATE apitofsim)
if(SKBUILD AND APITOFSIM_USE_VCPKG AND TARGET Arrow::arrow_shared)
  # CMake omits link directories inside the build tree from the install RPATH.
  # Give the repair tools the vcpkg path so they can bundle and relink the libraries.
  if(UNIX)
    set_property(TARGET apitofsimraw APPEND PROPERTY INSTALL_RPATH "$<TARGET_FILE_DIR:Arrow::arrow_shared>")
  endif()
  file(GLOB vcpkg_notices "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/*/copyright")
  foreach(notice IN LISTS vcpkg_notices)
    get_filename_component(port "${notice}" DIRECTORY)
    get_filename_component(port "${port}" NAME)
    install(FILES "${notice}" DESTINATION apitofsim/licenses/vcpkg RENAME "${port}.txt")
  endforeach()
endif()
if(TARGET Arrow::arrow_static AND NOT TARGET Arrow::arrow_shared AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Keep statically linked Arrow symbols private when PyArrow is also loaded.
  target_link_options(apitofsimraw PRIVATE "LINKER:--exclude-libs,ALL")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(apitofsimraw PRIVATE -Wno-maybe-uninitialized)
elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # Suppress the version-specific import library when using the stable ABI.
  target_link_options(apitofsimraw PRIVATE "/NODEFAULTLIB:python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.lib")
endif()
if(SKBUILD)
  set(apitofsim_python_dir apitofsim)
else()
  set(apitofsim_python_dir "${Python_SITEARCH}/apitofsim")
  install(DIRECTORY apitofsim/ DESTINATION "${apitofsim_python_dir}"
    FILES_MATCHING PATTERN "*.py" PATTERN "*.sql" PATTERN "*.txt" PATTERN "py.typed"
    PATTERN "__pycache__" EXCLUDE)
endif()
install(TARGETS apitofsimraw LIBRARY DESTINATION "${apitofsim_python_dir}")

if(APITOFSIM_GENERATE_STUBS)
  nanobind_add_stub(apitofsim_stubs
    MODULE apitofsimraw RECURSIVE
    OUTPUT apitofsimraw/__init__.pyi apitofsimraw/defaults.pyi apitofsimraw/consts.pyi
    PYTHON_PATH "$<TARGET_FILE_DIR:apitofsimraw>"
    DEPENDS apitofsimraw
  )
  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/apitofsimraw"
    DESTINATION "${apitofsim_python_dir}" FILES_MATCHING PATTERN "*.pyi")
endif()

if(BUILD_TESTING)
  # Mirror the package next to the extension for tests without an install.
  file(GLOB_RECURSE package_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS
    apitofsim/*.py apitofsim/*.sql apitofsim/*.txt apitofsim/py.typed)
  if(CMAKE_CONFIGURATION_TYPES)
    set(package_configs ${CMAKE_CONFIGURATION_TYPES})
  elseif(CMAKE_BUILD_TYPE)
    set(package_configs ${CMAKE_BUILD_TYPE})
  else()
    set(package_configs .)
  endif()
  foreach(config IN LISTS package_configs)
    foreach(file IN LISTS package_files)
      configure_file("${file}" "${config}/${file}" COPYONLY)
    endforeach()
  endforeach()
  set_target_properties(apitofsimraw PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/apitofsim")

endif()
