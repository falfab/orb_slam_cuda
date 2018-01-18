#!/bin/bash

#
# How to change resolution or fps.
# Go to src/tx2_camera/launch.
# Edit tx2_camera.launch arguments.
# Go to src/tx2_camera.
# Edit TX2CAMERA.yaml to match tx2_camera.launch values.
# 
# Different values may bring mono to crash!

source devel/setup.bash
roslaunch mono mono_with_tx2_camera.launch

