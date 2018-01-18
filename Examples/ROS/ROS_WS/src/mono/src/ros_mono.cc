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

// receive_topic: the topic to listen for images (default /camera/image_raw)
// vocabulary_file: the path of vocabulary (default ../../../Vocabulary/ORBvoc.txt)
// settings_file: the .yaml settings file (default ../../Monocular/KITTI00-02.yaml)
// query_topic: the topic used to get query messages (default /orbslam2/query)
// response_topic: the topic used to answer to query messages (default: /orbslam2/response)

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include "std_msgs/String.h"
#include <cv_bridge/cv_bridge.h>

#include<opencv2/core/core.hpp>

#include"../../../../../include/System.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
    void AnswerQuery(const std_msgs::String::ConstPtr& msg);
    
    void setPublisher(ros::Publisher pub);

    ORB_SLAM2::System* mpSLAM;
    ros::Publisher ros_pub;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::start();
    
    ros::NodeHandle nodeHandler("~");

    std::string camera_topic;
    std::string query_topic;
    std::string response_topic;
    std::string vocabulary_file;
    std::string settings_file;

    // TODO rinominare receive_topic in camera_topic prima o poi! (anche nei .launch)
    nodeHandler.param<std::string>("receive_topic", camera_topic, "/camera/image_raw");
    nodeHandler.param<std::string>("query_topic", query_topic, "/orbslam2/query");
    nodeHandler.param<std::string>("response_topic", response_topic, "/orbslam2/response");
    nodeHandler.param<std::string>("vocabulary_file", vocabulary_file, "../../../Vocabulary/ORBvoc.txt");
    nodeHandler.param<std::string>("settings_file", settings_file, "../../Monocular/KITTI00-02.yaml");
    
    Powermon p;
    p.prepare();

    //if(argc != 3)
    //{
    //    cerr << endl << "Usage: rosrun ORB_SLAM2 Mono path_to_vocabulary path_to_settings" << endl;        
    //    ros::shutdown();
    //    return 1;
    //}    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    cout << "voc: " << vocabulary_file << endl;
    ORB_SLAM2::System SLAM(vocabulary_file.c_str(),settings_file.c_str(),ORB_SLAM2::System::MONOCULAR,true);
    

    ImageGrabber igb(&SLAM);

    ros::Subscriber image_subscriber = nodeHandler.subscribe(camera_topic.c_str(), 1, &ImageGrabber::GrabImage,&igb);
    ros::Subscriber query_sub = nodeHandler.subscribe(query_topic.c_str(), 1000, &ImageGrabber::AnswerQuery, &igb);
    ros::Publisher query_pub = nodeHandler.advertise<std_msgs::String>(response_topic.c_str(), 1000);

    igb.setPublisher(query_pub);


    //ros::Rate r(20); // go at 20Hz
    //while(ros::ok()) {
    //    ros::spinOnce();
    //    r.sleep();
    //}
    
    p.resetDataCollected();
    p.startAsync();

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    //SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    p.stopAsync();

    p.readSync();
    p.printStats();

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    std::cout << "GrabImage!\n";
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());
}

void ImageGrabber::AnswerQuery(const std_msgs::String::ConstPtr& msg) {
    std::string query = msg->data;

    std::cout << "Answering query id:"  << query << std::endl;

    // idea, basing of query value... do something
    // maybe using a structured field, with a requestid

    // for now... sending if there is a huge map change
    bool isMapChanged = this->mpSLAM->MapChanged();

    std::string res = string("Mapchanged =") + (isMapChanged?"true":"false");

    this->ros_pub.publish(res);
}

void ImageGrabber::setPublisher(ros::Publisher pub) {
    this->ros_pub = pub;
}


