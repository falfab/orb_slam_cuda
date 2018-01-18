cmake_minimum_required(VERSION 2.8.3)
project(tx2_camera)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  gscam
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES jetson_csi_cam
#  CATKIN_DEPENDS gscam
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

