cmake_minimum_required(VERSION 3.10)
project(ppf
    VERSION 0.2.2
    LANGUAGES CXX
)

## compile settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_VERBOSE_MAKEFILE ON)

## add self defined find scripts to cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

## find dependencies
find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

find_package(Eigen3 REQUIRED)

SET(CMAKE_PREFIX_PATH "/home/edith/Software/pcl-1.10.0/build/install")
SET(PCL_DIR "/home/edith/Software/pcl-1.10.0/build")
SET(PCL_INCLUDE_DIRS "/home/edith/Software/pcl-1.10.0/build/install/include/pcl-1.10")
list(APPEND CMAKE_MODULE_PATH "/home/edith/Software/pcl-1.10.0/cmake/Modules")

#set(OpenCV_DIR /home/edith/Software/opencv/build/install/lib/cmake/opencv4)
#include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED COMPONENTS
    common
    keypoints
    search
    filters
    features
    io
    recognition
    registration
    segmentation
    surface
    visualization
)


include(DefinePCLImportedTarget)
define_pcl_imported_target()

## the following dependencies should be required but are already satisfied ?
# find_package(yaml-cpp ...)
# find_package(eigen3 ...)

find_package(CMPH REQUIRED)

# Glog  -- only works if glog was built with cmake ?
# libgoogle-glog-dev package doesn't include the cmake config files
find_package(glog 0.4.0 REQUIRED)

find_package(OpenMP REQUIRED)

find_package(OpenCV 3 REQUIRED)

## add subdirectories
add_subdirectory(3rdparty/pcl_1_8) # v4r depends on 3rdparty/pcl_1_8 so this probably needs to be done before defining the targets


## collect include and source files
# yes i know file(GLOB ... is not the recommended way to do this
file(GLOB_RECURSE V4R_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/v4r/*.cpp
)

file(GLOB_RECURSE PPF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppf/*.cpp
)

file(GLOB RECOGNIZER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

## add targets
add_library(v4r-extracts STATIC ${V4R_SOURCES})
add_library(ppf STATIC ${PPF_SOURCES})
add_library(ppf-recognizer STATIC ${RECOGNIZER_SOURCES})

add_executable(test_ppf_recognizer ${CMAKE_CURRENT_SOURCE_DIR}/src/test_ppf_recognizer.cpp)

## specify dependencies
target_include_directories(v4r-extracts
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(v4r-extracts
    Boost::boost
    glog::glog
    PCL::PCL
    ${OpenCV_LIBS}  # @todo imported target for OpenCV ?
    pcl_1_8
)

target_include_directories(ppf
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ppf
    Boost::boost
    glog::glog
    CMPH::CMPH
    v4r-extracts
    PCL::PCL
)

target_include_directories(ppf-recognizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
target_link_libraries(ppf-recognizer
    Boost::boost
    v4r-extracts
    ppf
    PCL::PCL
    OpenMP::OpenMP_CXX
)

target_include_directories(test_ppf_recognizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_link_libraries(test_ppf_recognizer
    ppf-recognizer
    Boost::program_options
    PCL::PCL
)



## install targets / define export set for exporting from build tree
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/PPFRecognizer)

install(TARGETS ppf-recognizer v4r-extracts ppf
    EXPORT ppf-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# install header files
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# copy find scripts to build/cmake
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCMPH.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FindCMPH.cmake
    COPYONLY
)

## create Config files
include(CMakePackageConfigHelpers)

# Create a ConfigVersion.cmake file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PPFRecognizerConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# create a Config.cmake file
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/PPFRecognizerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PPFRecognizerConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

## export targets from build tree
export(EXPORT ppf-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/PPFRecognizerTargets.cmake NAMESPACE PPFRecognizer::)
