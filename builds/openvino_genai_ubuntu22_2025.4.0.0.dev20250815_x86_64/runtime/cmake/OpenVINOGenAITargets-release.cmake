#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvino::genai" for configuration "Release"
set_property(TARGET openvino::genai APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::genai PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_genai.so.2025.4.0.0"
  IMPORTED_SONAME_RELEASE "libopenvino_genai.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::genai )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::genai "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_genai.so.2025.4.0.0" )

# Import target "openvino::genai::c" for configuration "Release"
set_property(TARGET openvino::genai::c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::genai::c PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "openvino::genai"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_genai_c.so.2025.4.0.0"
  IMPORTED_SONAME_RELEASE "libopenvino_genai_c.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::genai::c )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::genai::c "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_genai_c.so.2025.4.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
