#!/bin/bash

# echo load keys
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 0xB01FA116

# install ROS
sudo apt update
sudo apt install ros-lunar-ros-base -y
sudo apt install python-rosdep -y
sudo apt install python-rosinstall -y
sudo apt install python-catkin-tools -y
sudo c_rehash /etc/ssl/certs

# prepare ros
sudo rosdep init
rosdep update

# prepare .bashrc -> only ONE ip address is required
source ~/.bashrc
if [ -z $ROS_MASTER_URI ]; then # don't add this two times
    echo -e "\nsource /opt/ros/lunar/setup.bash" >> ~/.bashrc
    echo -e $'MY_IP=$(ip addr | grep inet | grep -v 127 | grep -v inet6 | awk \'{ print $2 }\' | cut -d/ -f1)\nexport ROS_MASTER_URI=http://$MY_IP:11311\nexport ROS_HOSTNAME=$MY_IP\n' >> ~/.bashrc
fi

# depencencies for orb-slam2
sudo apt install -y ros-lunar-tf ros-lunar-roscpp ros-lunar-sensor-msgs ros-lunar-image-transport ros-lunar-cv-bridge ros-lunar-rqt-image-view ros-lunar-camera-calibration-parsers ros-lunar-camera-info-manager gstreamer1.0-tools

echo "Installed! Please close this bash and open a new one to get ros working."
