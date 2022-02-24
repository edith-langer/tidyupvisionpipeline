cmake_minimum_required(VERSION 3.10)
project(change_detection)

set(CMAKE_BUILD_TYPE Release)

set (CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

SET(CMAKE_PREFIX_PATH "/home/edith/Software/pcl-1.10.0/build/install")
SET(PCL_DIR "/home/edith/Software/pcl-1.10.0/build")
SET(PCL_INCLUDE_DIRS "/home/edith/Software/pcl-1.10.0/build/install/include/pcl-1.10")
list(APPEND CMAKE_MODULE_PATH "/home/edith/Software/pcl-1.10.0/cmake/Modules")
find_package(PCL REQUIRED)

find_package(OpenCV 3 REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#add_executable(${PROJECT_NAME} src/test_change_detection.cpp src/change_detection.cpp src/scene_differencing_points.cpp
#    src/plane_object_extraction.cpp src/local_object_verification.cpp src/object_visualization.cpp src/color_histogram.cpp
#    src/object_matching.cpp)
#TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ppf-recognizer Boost::program_options)

add_executable(all_scenes_comparison src/all_scenes_comparison.cpp src/change_detection.cpp src/scene_differencing_points.cpp
    src/plane_object_extraction.cpp src/local_object_verification.cpp src/object_visualization.cpp src/color_histogram.cpp
    src/object_matching.cpp)
TARGET_LINK_LIBRARIES(all_scenes_comparison ${PCL_LIBRARIES} ${OpenCV_LIBS} ppf-recognizer Boost::program_options)

#add_executable(all_scenes_comparison_matching_only src/all_scenes_comparison_matching_only.cpp src/change_detection.cpp src/scene_differencing_points.cpp
#    src/plane_object_extraction.cpp src/local_object_verification.cpp src/object_visualization.cpp src/color_histogram.cpp
#    src/object_matching.cpp)
#TARGET_LINK_LIBRARIES(all_scenes_comparison_matching_only ${PCL_LIBRARIES} ${OpenCV_LIBS} ppf-recognizer Boost::program_options)
