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
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include<opencv2/core/core.hpp>

#include"../../../../../include/System.h"

using namespace std;


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps) {
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}


int main(int argc, char **argv)
{
    //if(argc != 2)
    //{
    //    cerr << endl << "Usage: ./ros_mono_kitti_source path_to_sequence" << endl;
    //    return 1;
    //}

    ros::init(argc, argv, "Mono_kitti_source");
    ros::start();
    
    ros::NodeHandle nodeHandler("~");

    std::string topic;
    std::string sequence_path;
    nodeHandler.param<std::string>("send_topic", topic, "/camera/image_raw");
    if (!nodeHandler.getParam("sequence_path", sequence_path)) {
        ROS_ERROR("Missing sequence_path! Aborting...");
    }

    image_transport::ImageTransport it(nodeHandler);
    image_transport::Publisher pub = it.advertise(topic.c_str(), 5);

    //Powermon p;
    //p.prepare();

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    std::cout << "Loading images!" << std::endl << std::flush;
    LoadImages(sequence_path, vstrImageFilenames, vTimestamps);
    std::cout << "End images load!" << std::endl << std::flush;

    int nImages = vstrImageFilenames.size();
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;

    //p.resetDataCollected();
    //p.startAsync();

    for(int nLoop = 0; nLoop < 1; nLoop++)
        for(int ni=0; ni<nImages; ni++)
        {
            // Read image from file
            im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
            double tframe = vTimestamps[ni];

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
                return 1;
            }

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // Pass the image to the SLAM system via ROS
            // TODO tframe is not used...
            std::cout << "Publishing image " << ni << std::endl;
            sensor_msgs::ImageConstPtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", im).toImageMsg();
            pub.publish(msg);
            ros::spinOnce();

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();


            // Wait to load the next frame
            double T=0;
            if(ni<nImages-1)
                T = vTimestamps[ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestamps[ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6);
        }

    //p.stopAsync();

    //p.readSync();
    //p.printStats();

    ros::shutdown();

    return 0;
}


