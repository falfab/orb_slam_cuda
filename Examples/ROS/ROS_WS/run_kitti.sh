#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_kitti.sh sequence_path"
    exit 1
fi

if [ ! -d "./devel" ]; then
    echo "Compiling ros nodes!"
    cd ../../../
    ./build_ros.sh
    cd Examples/ROS/ROS_WS
fi

NUM=$(basename "$1")
FOL=$1/../

source devel/setup.bash
roslaunch mono mono_with_kitti.launch sequences_folder:="$FOL" sequence_number:="$NUM"
echo "--------------------------------"
echo "Map and .csv files are in folder devel/lib/mono!"
