# defines the target PCL::PCL
# call after find_package(PCL ...)
macro(define_pcl_imported_target)

  # find full paths for pcl libraries so they can be used with target_link_libraries
  foreach(lib ${PCL_LIBRARIES})
      if(    (IS_ABSOLUTE ${lib})
      # OR (${lib} STREQUAL debug)      # pass through debug/optimized keywords
      # OR (${lib} STREQUAL optimized)
      )
          list(APPEND PCL_LIBS ${lib})
      else()
          find_library(PCL_LIB
              ${lib}
              HINTS ${PCL_LIBRARY_DIRS})

          if(${PCL_LIB})
              list(APPEND PCL_LIBS ${PCL_LIB})
              unset(PCL_LIB CACHE)
          endif()
      endif()
  endforeach()

  # sanitize PCL_DEFINITIONS
  # newer versions of cmake allow calling target_** commands on
  # imported targets which would take care of all these little details
  set(pcl_definitions ${PCL_DEFINITIONS})
  set(pcl_options ${PCL_DEFINITIONS})

list(FILTER pcl_definitions INCLUDE REGEX "^[ ]*-D")
list(FILTER pcl_options EXCLUDE REGEX "^[ ]*-D")

  # strip extra -D from pcl_definitions
  foreach(def ${pcl_definitions})
    string(REGEX REPLACE "^-D" "" def_out ${def})
    list(APPEND PCL_DEFS ${def_out})
  endforeach()

  # remove empty items, for some reason PCL_DEFINITIONS contains items that are " "
  list(REMOVE_ITEM PCL_DEFS " ")

  # split strings with options into individual list items
  string(REGEX REPLACE "[ ]+" ";" PCL_OPTS "${pcl_options}")

  #define imported target for PCL
  add_library(PCL::PCL INTERFACE IMPORTED)
  set_target_properties(PCL::PCL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PCL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${PCL_LIBS}"
    INTERFACE_COMPILE_DEFINITIONS "${PCL_DEFS}"
    INTERFACE_COMPILE_OPTIONS "${PCL_OPTS}"
  )

endmacro()
