# - Try to find Visionworks lib
#
# Once done this will define
#
#  VISIONWORKS_FOUND - system has eigen lib with correct version
#  VISIONWORKS_INCLUDE_DIR - the eigen include directory
#  VISIONWORKS_LIBRARIES - The libraries needed to use Visionworks
#  VISIONWORKS_DEFINITIONS - Compiler switches required for using Visionworks

find_package(PkgConfig)
pkg_check_modules(PC_VISIONWORKS QUIET visionworks)
set(VISIONWORKS_DEFINITIONS ${PC_VISIONWORKS_CFLAGS_OTHER})

find_path(VISIONWORKS_INCLUDE_DIR NVX/nvx.h
          HINTS ${PC_VISIONWORKS_INCLUDEDIR} ${PC_VISIONWORKS_INCLUDE_DIRS}
          PATH_SUFFIXES visionworks )


find_library(VISIONWORKS_LIBRARY NAMES visionworks
             HINTS ${PC_VISIONWORKS_LIBDIR} ${PC_VISIONWORKS_LIBRARY_DIRS} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set VISIONWORKS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(visionworks  DEFAULT_MSG
                                  VISIONWORKS_LIBRARY VISIONWORKS_INCLUDE_DIR)

mark_as_advanced(VISIONWORKS_INCLUDE_DIR VISIONWORKS_LIBRARY )

set(VISIONWORKS_LIBRARIES ${VISIONWORKS_LIBRARY} )
set(VISIONWORKS_INCLUDE_DIRS ${VISIONWORKS_INCLUDE_DIR} )


