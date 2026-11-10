execute_process(
  COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/python/pyarrow_build.py"
  OUTPUT_VARIABLE pyarrow_build OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(JSON pyarrow_version GET "${pyarrow_build}" version)
string(JSON pyarrow_libdir GET "${pyarrow_build}" libdir)
string(JSON pyarrow_include GET "${pyarrow_build}" include)
string(JSON pyarrow_so_version GET "${pyarrow_build}" so_version)

foreach(lib IN ITEMS Arrow Parquet)
  string(TOLOWER "${lib}" name)
  add_library(PyArrow::${lib} SHARED IMPORTED)
  if(WIN32)
    set(location "${pyarrow_libdir}/${name}.dll")
    set(import_lib "${pyarrow_libdir}/${name}.lib")
    if(NOT EXISTS "${import_lib}")
      message(FATAL_ERROR "PyArrow import library is missing: ${import_lib}")
    endif()
    set_property(TARGET PyArrow::${lib} PROPERTY IMPORTED_IMPLIB "${import_lib}")
  elseif(APPLE)
    set(location "${pyarrow_libdir}/lib${name}.${pyarrow_so_version}.dylib")
  else()
    set(location "${pyarrow_libdir}/lib${name}.so.${pyarrow_so_version}")
  endif()
  if(NOT EXISTS "${location}")
    message(FATAL_ERROR "PyArrow wheel library is missing: ${location}")
  endif()
  set_target_properties(PyArrow::${lib} PROPERTIES
    IMPORTED_LOCATION "${location}"
    INTERFACE_INCLUDE_DIRECTORIES "${pyarrow_include}"
  )
endforeach()
