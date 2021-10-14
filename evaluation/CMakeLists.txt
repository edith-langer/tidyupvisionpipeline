cmake_minimum_required(VERSION 3.10)
project(evaluation)

set(CMAKE_BUILD_TYPE Debug)

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
include_directories("${PROJECT_SOURCE_DIR}/../change_detection/include") #for settings.h
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} src/evaluation.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ppf-recognizer Boost::program_options)

add_executable(fromDStoOrig src/fromDStoOrig.cpp)
TARGET_LINK_LIBRARIES(fromDStoOrig ${PCL_LIBRARIES} ${OpenCV_LIBS} ppf-recognizer Boost::program_options)


