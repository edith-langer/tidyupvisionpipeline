cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(pcl_1_8)

# This project contains files from PCL which are either not available in or have a different ABI
# than the currently released PCL version.

find_package(PCL REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS})

set(SOURCES
  uniform_sampling.cpp
        )

set(HEADERS
  pcl_1_8/keypoints/impl/uniform_sampling.hpp
  pcl_1_8/keypoints/uniform_sampling.h
)

add_library(pcl_1_8 STATIC ${SOURCES})
target_include_directories(pcl_1_8 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pcl_1_8 ${PCL_LIBRARIES})

set_target_properties(pcl_1_8 PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

install(DIRECTORY pcl_1_8 DESTINATION "include")
install(TARGETS pcl_1_8 EXPORT ppf-targets ARCHIVE DESTINATION lib)
