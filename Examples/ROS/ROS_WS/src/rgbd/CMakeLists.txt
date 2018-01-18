cmake_minimum_required(VERSION 2.4.6)
project(rgbd)

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# ------------------------------------------------------------------------------
# SET THIS PARAMETERS TODO
SET(DONT_USE_OPENVX FALSE)      #"Disable OpenVX build"
SET(USE_G2O_OPENMP FALSE)       #"Use openmp version of Thirdparty library g2o"
SET(USE_TEGRA_CV FALSE)         #"Use OpenCV4Tegra 2.4 instead of standard OpenCV"
# ------------------------------------------------------------------------------


MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../../../cmake_modules)

IF(USE_TEGRA_CV)
    find_package(OpenCV 2.4.3 QUIET)
ELSE()
    find_package(OpenCV 3.0 QUIET)
    if(NOT OpenCV_FOUND)
        find_package(OpenCV 2.4.3 QUIET)
    endif()
ENDIF()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
endif()

IF(USE_G2O_OPENMP)
    SET(G2O_SUFFIX _openmp)
ELSE()
    SET(G2O_SUFFIX )
ENDIF()

IF(USE_TEGRA_CV)
    SET(DBOW2_SUFFIX _tegracv)
ELSE()
    SET(DBOW2_SUFFIX )
ENDIF()

IF(DONT_USE_OPENVX)
    SET(OPENVX_SUFFIX )
ELSE()
    SET(OPENVX_SUFFIX _openvx)
ENDIF()

SET(LIB_ORB_SLAM ${PROJECT_SOURCE_DIR}/../../../../../lib/${CMAKE_BUILD_TYPE}${OPENVX_SUFFIX}${G2O_SUFFIX}${DBOW2_SUFFIX}/libORB_SLAM24Matlab.so)
message("Using orb-slam library ${PROJECT_SOURCE_DIR}/../../../../../lib/${CMAKE_BUILD_TYPE}${OPENVX_SUFFIX}${G2O_SUFFIX}${DBOW2_SUFFIX}/libORB_SLAM24Matlab.so") 

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(Boost COMPONENTS system REQUIRED)

find_package(catkin REQUIRED 
    COMPONENTS roscpp tf sensor_msgs image_transport cv_bridge
    )

  catkin_package(
    CATKIN_DEPENDS roscpp tf sensor_msgs image_transport cv_bridge    
    )

include_directories(
${catkin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../../../
${PROJECT_SOURCE_DIR}/../../../../../include
${Pangolin_INCLUDE_DIRS}
)

set(LIBS 
${catkin_LIBRARIES}
${Boost_LIBRARIES}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${LIB_ORB_SLAM}
)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# Node for rgbdcular camera
add_executable(rgbd
src/ros_rgbd.cc
)

target_link_libraries(rgbd
${LIBS}
)

install(TARGETS rgbd
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

