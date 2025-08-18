#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvino::frontend::onnx" for configuration "Release"
set_property(TARGET openvino::frontend::onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::onnx PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_onnx_frontend.so.2025.4.0"
  IMPORTED_SONAME_RELEASE "libopenvino_onnx_frontend.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::onnx )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::onnx "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_onnx_frontend.so.2025.4.0" )

# Import target "openvino::frontend::pytorch" for configuration "Release"
set_property(TARGET openvino::frontend::pytorch APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::pytorch PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_pytorch_frontend.so.2025.4.0"
  IMPORTED_SONAME_RELEASE "libopenvino_pytorch_frontend.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::pytorch )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::pytorch "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_pytorch_frontend.so.2025.4.0" )

# Import target "openvino::frontend::tensorflow" for configuration "Release"
set_property(TARGET openvino::frontend::tensorflow APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::tensorflow PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_tensorflow_frontend.so.2025.4.0"
  IMPORTED_SONAME_RELEASE "libopenvino_tensorflow_frontend.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::tensorflow )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::tensorflow "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_tensorflow_frontend.so.2025.4.0" )

# Import target "openvino::frontend::tensorflow_lite" for configuration "Release"
set_property(TARGET openvino::frontend::tensorflow_lite APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::tensorflow_lite PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_tensorflow_lite_frontend.so.2025.4.0"
  IMPORTED_SONAME_RELEASE "libopenvino_tensorflow_lite_frontend.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::tensorflow_lite )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::tensorflow_lite "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_tensorflow_lite_frontend.so.2025.4.0" )

# Import target "openvino::frontend::paddle" for configuration "Release"
set_property(TARGET openvino::frontend::paddle APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::paddle PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_paddle_frontend.so.2025.4.0"
  IMPORTED_SONAME_RELEASE "libopenvino_paddle_frontend.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::paddle )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::paddle "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_paddle_frontend.so.2025.4.0" )

# Import target "openvino::runtime" for configuration "Release"
set_property(TARGET openvino::runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::runtime PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "TBB::tbb"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino.so.2025.4.0"
  IMPORTED_SONAME_RELEASE "libopenvino.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino.so.2025.4.0" )

# Import target "openvino::runtime::c" for configuration "Release"
set_property(TARGET openvino::runtime::c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::runtime::c PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "openvino::runtime"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_c.so.2025.4.0"
  IMPORTED_SONAME_RELEASE "libopenvino_c.so.2540"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime::c )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime::c "${_IMPORT_PREFIX}/runtime/lib/intel64/libopenvino_c.so.2025.4.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
