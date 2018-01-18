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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>
#include <NVX/nvx.h>

#ifdef TEGRACV
    #include <NVX/nvx_opencv_interop.hpp> // GpuMat only on Opencv2.4 (4tegra)
#endif

#include <chrono>
#include "powermon.hpp"

namespace ORB_SLAM2
{

class ORBextractor;

struct times_t
{
    int frame;
    std::string   name;
    int           level;
    long long time;
};

class GetTime{
public:
    GetTime(ORBextractor *o, std::string name, int level);
    GetTime(std::vector<times_t> &times, int nFrame, std::string name, int level);
    ~GetTime();
private:
    GetTime();
    ORBextractor *o;
    times_t t;
    std::chrono::steady_clock::time_point start;
    std::vector<times_t> &times;
};

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
    friend class GetTime;
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST, int width, int height);

    ~ORBextractor();

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    long long totalTime;
    unsigned int nFrame;

    std::vector<times_t> times;
//#ifdef OPENVX

    void buildGraph(int width, int height);

    vx_context ctx;
    vx_graph globalGraph;

    // "constants"
    vx_float32 strength_threshold1;
    vx_float32 strength_threshold2;
    vx_scalar s_strength_threshold2;
    vx_scalar s_strength_threshold1;
    vx_scalar s_EDGE_THRESHOLD;
    vx_scalar s_WINDOW;
    vx_scalar *s_NUM_FEATURES;
    vx_scalar *s_multiplier;

    vx_array umax_pattern;
    vx_array patterns;

    vx_convolution gaussianConvVertical;
    vx_convolution gaussianConvHorizontal;
    vx_pyramid gaussianPyramid;

    // data structures
    vx_image imageStart;
    vx_image *imagesResized;
    vx_array *keypointsFAST;
    vx_array *keypointsFAST2;
    vx_array *keypointsDistributed;
    vx_array *keypointsOriented;

    vx_image *imagesBlurred;
    vx_array *descriptors;
    vx_array *keypointsScaled;

    vx_scalar *num_corners;
    vx_scalar *coordinatesForGrid;

    //nodes
    vx_node pyramid_node;
    vx_node *resize_node;
    vx_node *fast_nodemin;
    vx_node *grid_node;
    vx_node *quadtree_node;
    vx_node *orientation_node;
    vx_node *blur_node;
    vx_node *orb_node;
    vx_node *scale_node;
//#endif
};

} //namespace ORB_SLAM

#endif

