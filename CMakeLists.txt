cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM24Matlab)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

SET(DONT_USE_OPENVX FALSE CACHE BOOL "Disable OpenVX build")
SET(USE_G2O_OPENMP FALSE CACHE BOOL "Use openmp version of Thirdparty library g2o")
SET(USE_TEGRA_CV TRUE CACHE BOOL "Use OpenCV4Tegra 2.4 instead of standard OpenCV")


IF(DONT_USE_OPENVX)
    remove_definitions(-DOPENVX)
    MESSAGE("Using ORB-SLAM2 as provided")
    SET(OPENVX_SUFFIX )
ELSE()
    add_definitions(-DOPENVX)
    MESSAGE("Using ORB-SLAM2 with OPENVX")
    SET(OPENVX_SUFFIX _openvx)
ENDIF()

IF(USE_G2O_OPENMP)
    set(G2O_LIBRARIES ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_openmp.so)
    SET(G2O_SUFFIX _openmp)
ELSE()
    set(G2O_LIBRARIES ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so)
    SET(G2O_SUFFIX )
ENDIF()

IF(USE_TEGRA_CV)
    set(DBOW2_LIBRARIES ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2_tegra_cv.so)
    SET(DBOW2_SUFFIX _tegracv)
    add_definitions(-DTEGRACV)
ELSE()
    set(DBOW2_LIBRARIES ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)
    SET(DBOW2_SUFFIX )
    remove_definitions(-DTEGRACV)
ENDIF()

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_62,code=sm_62)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-register")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

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

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

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

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Visionworks REQUIRED)
find_package(NVXIO REQUIRED)
find_package(Pangolin REQUIRED)
find_package(CUDA QUIET REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${VISIONWORKS_INCLUDE_DIRS}
${NVXIO_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE}${OPENVX_SUFFIX}${G2O_SUFFIX}${DBOW2_SUFFIX})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}${OPENVX_SUFFIX}${G2O_SUFFIX}${DBOW2_SUFFIX})


# Find source files
file(GLOB INCLUDES include/*.h*)

cuda_add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
#src/BinaryMat.cc
src/Util.cc
src/powermon.cpp
src/CustomNodes.cc
src/Timer.cpp
${INCLUDES}
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${VISIONWORKS_LIBRARIES}
${NVXIO_LIBRARIES}
${DBOW2_LIBRARIES}
${G2O_LIBRARIES}
)

#add_executable(rgbd_tum
#Examples/RGB-D/rgbd_tum.cc)
#target_link_libraries(rgbd_tum ${PROJECT_NAME})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

#add_executable(stereo_kitti
#Examples/Stereo/stereo_kitti.cc)
#target_link_libraries(stereo_kitti ${PROJECT_NAME})

#add_executable(stereo_euroc
#Examples/Stereo/stereo_euroc.cc)
#target_link_libraries(stereo_euroc ${PROJECT_NAME})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

#add_executable(mono_tum
#Examples/Monocular/mono_tum.cc)
#target_link_libraries(mono_tum ${PROJECT_NAME})

cuda_add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

#add_executable(mono_euroc
#Examples/Monocular/mono_euroc.cc)
#target_link_libraries(mono_euroc ${PROJECT_NAME})

#add_executable(mono_intcatch
#Examples/Monocular/mono_intcatch.cc)
#target_link_libraries(mono_intcatch ${PROJECT_NAME})

# add_executable(driver Examples/PbaDriver/driver.cpp)
# target_link_libraries(driver ${PROJECT_NAME})


#add_custom_target(update
#                  COMMAND cmake ..
#                  COMMAND make -j
#                  COMMENT "UPDATE RELEASE")
#add_custom_target(update_debug
#                  COMMAND cmake -DCMAKE_BUILD_TYPE=Debug ..
#                  COMMAND make -j
#                  COMMENT "UPDATE DEBUG")
