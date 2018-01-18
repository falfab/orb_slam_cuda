/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

// Parameters

// send_topic: the topic to publish images (default /camera/image_raw)
// sequence_path: the path of kitti sequence


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<ros/ros.h>
#include <std_msgs/String.h>

using namespace std;


void topic_callback(const std_msgs::String::ConstPtr& msg) {
    std::string response = msg->data;

    std::cout << "[CONTROLLER] Orb-slam response" << response << std::endl;
    std::cerr << "[CONTROLLER] Orb-slam response" << response << std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Orb_Controller");
    ros::start();

    std::string query_topic;
    std::string response_topic;

    ros::NodeHandle nodeHandler("~");

    if (!nodeHandler.getParam("response_topic", response_topic))
        ROS_ERROR("Missing response_topic! Aborting...");
    if (!nodeHandler.getParam("query_topic", query_topic))
        ROS_ERROR("Missing query_topic! Aborting...");

    ros::Subscriber query_sub = nodeHandler.subscribe(response_topic.c_str(), 10, topic_callback);
    ros::Publisher query_pub = nodeHandler.advertise<std_msgs::String>(query_topic.c_str(), 10);


    ros::Rate r(0.5);
    while(ros::ok()) {
        query_pub.publish(string("query"));

        ros::spinOnce();
        r.sleep();
    }

    ros::shutdown();

    return 0;
}


