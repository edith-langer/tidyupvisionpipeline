# FindCMPH.cmake
#
# Finds the cmph library
#
# This will define the following variables
#
#    CMPH_FOUND
#    CMPH_INCLUDE_DIRS
#
# and the following imported targets
#
#     CMPH::CMPH
#
# Author: Jeremy Itamah - jeremy.itamah@tuwien.ac.at

find_package(PkgConfig QUIET)
pkg_check_modules(PC_CMPH QUIET cmph)

find_path(CMPH_INCLUDE_DIR
    NAMES cmph.h
    PATHS ${PC_CMPH_INCLUDE_DIRS}
    PATH_SUFFIXES cmph
)

find_library(CMPH_LIBRARY
    NAMES cmph
    PATHS ${PC_CMPH_LIBRARY_DIRS}
)

set(CMPH_VERSION ${PC_CMPH_VERSION})
# set(CMPH_FOUND ${PC_CMPH_FOUND})

mark_as_advanced(CMPH_FOUND CMPH_INCLUDE_DIR CMPH_LIBRARY CMPH_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CMPH
    REQUIRED_VARS CMPH_INCLUDE_DIR CMPH_LIBRARY
    VERSION_VAR CMPH_VERSION
)

if(CMPH_FOUND)
    set(CMPH_INCLUDE_DIRS ${CMPH_INCLUDE_DIR})
endif()

if(CMPH_FOUND AND NOT TARGET CMPH::CMPH)
    add_library(CMPH::CMPH SHARED IMPORTED)
    set_target_properties(CMPH::CMPH PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMPH_INCLUDE_DIR}"
        IMPORTED_LOCATION "${CMPH_LIBRARY}"
    )
endif()