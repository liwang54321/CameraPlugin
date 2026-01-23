#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "uvgrtp::uvgrtp" for configuration "Release"
set_property(TARGET uvgrtp::uvgrtp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(uvgrtp::uvgrtp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libuvgrtp.a"
  )

list(APPEND _cmake_import_check_targets uvgrtp::uvgrtp )
list(APPEND _cmake_import_check_files_for_uvgrtp::uvgrtp "${_IMPORT_PREFIX}/lib/libuvgrtp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
