#!/bin/bash

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir -p build4ros
cd build4ros
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TEGRA_CV=0
make -j4

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir -p build4ros
cd build4ros
cmake .. -DCMAKE_BUILD_TYPE=Release -DG2O_USE_OPENMP=0
make -j4

cd ../../../

echo "Building ORB-SLAM2"
mkdir -p build4ros
cd build4ros
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TEGRA_CV=0 -USE_G2O_OPENMP=0
make -j4

cd ..

echo "Building ROS nodes"

cd Examples/ROS/ROS_WS
catkin_make

echo 'Built! Now go to Examples/ROS/ROS_WS and execute run.sh'

