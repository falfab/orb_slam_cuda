# - Try to find NVXIO lib
#
# Once done this will define
#
#  NVXIO_FOUND - system has eigen lib with correct version
#  NVXIO_INCLUDE_DIR - the eigen include directory
#  NVXIO_LIBRARIES - The libraries needed to use NVXIO
#  NVXIO_DEFINITIONS - Compiler switches required for using NVXIO

find_package(PkgConfig)
pkg_check_modules(PC_NVXIO QUIET nvxio)
set(NVXIO_DEFINITIONS ${PC_NVXIO_CFLAGS_OTHER})

find_path(NVXIO_INCLUDE_DIR NVX/Utility.hpp
          HINTS ${PC_NVXIO_INCLUDEDIR} ${PC_VISIOWNORKS_INCLUDE_DIRS}
	  ${PROJECT_SOURCE_DIR}/nvxio/include/
          PATH_SUFFIXES NVX )

find_library(NVXIO_LIBRARY_NVX NAMES nvx
             HINTS ${PC_NVXIO_LIBDIR} ${PC_NVXIO_LIBRARY_DIRS} ${PROJECT_SOURCE_DIR}/nvxio/lib)

find_library(NVXIO_LIBRARY_OVX NAMES ovx
             HINTS ${PC_NVXIO_LIBDIR} ${PC_NVXIO_LIBRARY_DIRS} ${PROJECT_SOURCE_DIR}/nvxio/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NVXIO_FOUND to TRUE
# if all listed variables are TRUE


set(NVXIO_LIBRARIES ${NVXIO_LIBRARY_OVX} ${NVXIO_LIBRARY_NVX} )
set(NVXIO_INCLUDE_DIRS ${NVXIO_INCLUDE_DIR} )

find_package_handle_standard_args(nvxio  DEFAULT_MSG
                                  NVXIO_LIBRARIES NVXIO_INCLUDE_DIR)

mark_as_advanced(NVXIO_INCLUDE_DIR NVXIO_LIBRARY_NVX NVXIO_LIBRARY_OVX )

