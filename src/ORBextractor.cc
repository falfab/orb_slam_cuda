/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <NVX/Utility.hpp> // TODO
#include <OVX/UtilityOVX.hpp>
#include <chrono>
#include <fstream>

#include "ORBextractor.h"
#include "Util_vx.h"
#include "CustomNodes.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

bool b_pyramid_VX = false;
bool b_gridfast_VX = false;
bool b_quadtree_VX = false;
bool b_orient_VX = false;
bool b_blur_VX = false;
bool b_orb_VX = false;
bool b_scale_VX = false;

#ifdef OPENVX
bool b_complete_VX = true;
bool b_use_virtual = true;
#else
bool b_complete_VX = false;
bool b_use_virtual = false;
#endif



const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;
const int WINDOW_SIZE = 30;

template <typename TP>
void arrayToVX(vx_array vx_v, std::vector<TP> v)
{
    vxTruncateArray(vx_v, 0);
    vxAddArrayItems(vx_v, v.size(), v.data(), sizeof(TP));
}
template <typename TP>
std::vector<TP> vxToArray(vx_array vx_v)
{
    std::vector<TP> v;
    vx_size n;
    vxQueryArray(vx_v, VX_ARRAY_NUMITEMS, &n, sizeof(n));
    v.resize(n);
    vxCopyArrayRange(vx_v, 0, n, sizeof(TP), v, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    return v;
}
template <>
void arrayToVX<cv::KeyPoint>(vx_array vx_v, std::vector<cv::KeyPoint> v)
{
    vxTruncateArray(vx_v, 0);

    for(cv::KeyPoint p : v)
    {
        vx_keypoint_t kp;
        kp.x = p.pt.x;
        kp.y = p.pt.y;
        kp.error = 0;
        kp.orientation = p.angle;
        kp.strength = p.response;
        kp.scale = p.octave;

        vxAddArrayItems(vx_v, 1, &kp, sizeof(vx_keypoint_t));
    }

}
template <>
std::vector<cv::KeyPoint> vxToArray(vx_array vx_v)
{
    std::vector<cv::KeyPoint> v;

    vx_size i, stride = sizeof(vx_size), sz;
    void *base = NULL;
    vx_map_id map_id;
    vxQueryArray(vx_v, VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size));
    vxMapArrayRange(vx_v, 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    for (i = 0; i < sz; i++)
    {
        vx_keypoint_t kp = vxArrayItem(vx_keypoint_t, base, i, stride);

        cv::KeyPoint p;
        p.pt = cv::Point2f(kp.x, kp.y);
        p.response = kp.strength;
        p.octave = kp.scale;
        p.angle = kp.orientation;

        v.push_back(p);
    }
    vxUnmapArrayRange(vx_v, map_id);

    return v;
}


static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}


const float factorPI = (float)(CV_PI/180.f);
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}


static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,VX_FAILURE
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         int _iniThFAST, int _minThFAST, int width, int height):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST), totalTime(0), nFrame(0)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    const int npoints = 512;
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    times.reserve(200);

    buildGraph(width, height);
}

void ORBextractor::buildGraph(int width, int height)
{
    ctx = vxCreateContext();
    NVXIO_CHECK_REFERENCE( ctx );
    vxDirective((vx_reference)ctx, NVX_DIRECTIVE_ENABLE_PERFORMANCE);

    registerMakeGrid(ctx);
    registerMakeQuadtree(ctx);
    registerComputeAngle(ctx);
    registerORB(ctx);
    registerScaleArray(ctx);

    globalGraph = vxCreateGraph(ctx);
    NVXIO_CHECK_REFERENCE( globalGraph );

    vx_size sz_array = 0;
    strength_threshold1 = iniThFAST;
    strength_threshold2 = minThFAST;
    s_strength_threshold1 = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &strength_threshold1);
    NVXIO_CHECK_REFERENCE( s_strength_threshold1 );
    s_strength_threshold2 = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &strength_threshold2);
    NVXIO_CHECK_REFERENCE( s_strength_threshold2 );

    s_EDGE_THRESHOLD = vxCreateScalar(ctx, VX_TYPE_UINT32, &EDGE_THRESHOLD);
    NVXIO_CHECK_REFERENCE( s_EDGE_THRESHOLD);
    s_WINDOW         = vxCreateScalar(ctx, VX_TYPE_UINT32, &WINDOW_SIZE);
    NVXIO_CHECK_REFERENCE( s_WINDOW);
    s_NUM_FEATURES   = new vx_scalar[nlevels];
    s_multiplier     = new vx_scalar[nlevels];
    for(int i = 0; i < nlevels; i++)
    {
        vx_size tmpSz = mnFeaturesPerLevel[i];
        s_NUM_FEATURES[i] = vxCreateScalar(ctx, VX_TYPE_SIZE, &tmpSz);
        //s_NUM_FEATURES[i] = vxCreateScalar(ctx, VX_TYPE_SIZE, &(mnFeaturesPerLevel[i]));
        NVXIO_CHECK_REFERENCE( s_NUM_FEATURES[i] );
    }

    umax_pattern = vxCreateArray(ctx, VX_TYPE_INT32, umax.size());
    NVXIO_CHECK_REFERENCE( umax_pattern );
    NVXIO_SAFE_CALL( vxAddArrayItems(umax_pattern, umax.size(), umax.data(), sizeof(int)) );

    patterns = vxCreateArray(ctx, VX_TYPE_INT32, 256*4);
    NVXIO_CHECK_REFERENCE( patterns );
    NVXIO_SAFE_CALL( vxAddArrayItems(patterns, 256*4, bit_pattern_31_, sizeof(int)) );

    num_corners       = new vx_scalar[nlevels];
    for(int i = 0; i < nlevels; ++i)
    {
        num_corners[i]    = vxCreateScalar(ctx, VX_TYPE_SIZE, &sz_array);
        NVXIO_CHECK_REFERENCE( num_corners[i] );
    }

    gaussianConvHorizontal = createGaussianConvolution(ctx, 7, 2, 85, true);
    NVXIO_CHECK_REFERENCE( gaussianConvHorizontal );
    gaussianConvVertical   = createGaussianConvolution(ctx, 7, 2, 85, false);
    NVXIO_CHECK_REFERENCE( gaussianConvVertical );

    imagesResized        = new vx_image[nlevels];
    keypointsFAST        = new vx_array[nlevels];
    keypointsFAST2       = new vx_array[nlevels];
    keypointsDistributed = new vx_array[nlevels];
    keypointsOriented    = new vx_array[nlevels];

    imagesBlurred        = new vx_image[2*nlevels];
    descriptors          = new vx_array[nlevels];
    keypointsScaled      = new vx_array[nlevels];

    coordinatesForGrid   = new vx_scalar[nlevels*4];

    resize_node      = new vx_node[nlevels-1];
    fast_nodemin     = new vx_node[nlevels];
    grid_node        = new vx_node[nlevels];
    quadtree_node    = new vx_node[nlevels];
    orientation_node = new vx_node[nlevels];
    blur_node        = new vx_node[2*nlevels];
    orb_node         = new vx_node[nlevels];
    scale_node       = new vx_node[nlevels-1];

#define USE_PYRAMID

#ifdef USE_PYRAMID
    if(b_use_virtual)
        gaussianPyramid = vxCreateVirtualPyramid(globalGraph, nlevels, VX_SCALE_PYRAMID_ORB, width, height, VX_DF_IMAGE_U8);
    else
        gaussianPyramid = vxCreatePyramid(ctx, nlevels, VX_SCALE_PYRAMID_ORB, width, height, VX_DF_IMAGE_U8);
std::cout << width << "," << height << std::endl;
    imageStart      = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(imageStart);
    vx_image image2 = vxCreateVirtualImage(globalGraph, width, height, VX_DF_IMAGE_U8);
    nvxCopyImageNode(globalGraph, imageStart, image2);

    //imagesResized[0] = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
    //pyramid_node = vxGaussianPyramidNode(globalGraph, imageStart, gaussianPyramid);
    pyramid_node = vxGaussianPyramidNode(globalGraph, image2, gaussianPyramid);
#endif
    for (int level = 0; level < nlevels; ++level)
    {
        vx_uint32 v_width, v_height;

#ifdef USE_PYRAMID
        imagesResized[level] = vxGetPyramidLevel(gaussianPyramid, level);
#else
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)width*scale), cvRound((float)height*scale));

        //imagesResize[level] = vxCreateVirtualImage(globalGraph, sz.width, sz.height, VX_DF_IMAGE_U8);
        imagesResized[level] = vxCreateImage(ctx, sz.width, sz.height, VX_DF_IMAGE_U8);
        if(level == 0) imageStart = imagesResized[level];
#endif
        NVXIO_CHECK_REFERENCE( imagesResized[level] );
        vxQueryImage(imagesResized[level], VX_IMAGE_WIDTH, &v_width, sizeof(vx_uint32));
        vxQueryImage(imagesResized[level], VX_IMAGE_HEIGHT, &v_height, sizeof(vx_uint32));
#ifdef USE_PYRAMID
        //if(b_complete_VX || b_pyramid_VX)
        {
            mvScaleFactor[level] = ((float) width) / v_width;
            mvInvScaleFactor[level] = ((float) v_width) / width;
        }
#endif
        s_multiplier[level]   = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &mvScaleFactor[level]);
        NVXIO_CHECK_REFERENCE( s_multiplier[level] );

        if(b_use_virtual)
            keypointsFAST[level]        = vxCreateVirtualArray(globalGraph, VX_TYPE_KEYPOINT, 5*nfeatures);
        else
            keypointsFAST[level]        = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 5*nfeatures);
        NVXIO_CHECK_REFERENCE( keypointsFAST[level] );
        if(b_use_virtual)
            keypointsFAST2[level]       = vxCreateVirtualArray(globalGraph, VX_TYPE_KEYPOINT, 5*nfeatures);
        else
            keypointsFAST2[level]       = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 5*nfeatures);
        NVXIO_CHECK_REFERENCE( keypointsFAST2[level] );
        if(b_use_virtual)
            keypointsDistributed[level] = vxCreateVirtualArray(globalGraph, VX_TYPE_KEYPOINT, mnFeaturesPerLevel[level]);
        else
            keypointsDistributed[level] = vxCreateArray(ctx, VX_TYPE_KEYPOINT, mnFeaturesPerLevel[level]);
        NVXIO_CHECK_REFERENCE( keypointsDistributed[level] );
        if(b_use_virtual)
            keypointsOriented[level] = vxCreateVirtualArray(globalGraph, VX_TYPE_KEYPOINT, mnFeaturesPerLevel[level]);
        else
            keypointsOriented[level] = vxCreateArray(ctx, VX_TYPE_KEYPOINT, mnFeaturesPerLevel[level]);
        NVXIO_CHECK_REFERENCE( keypointsOriented[level] );

        if(b_use_virtual)
            imagesBlurred[level] = vxCreateVirtualImage(globalGraph, v_width, v_height, VX_DF_IMAGE_U8);
        else
            imagesBlurred[level] = vxCreateImage(ctx, v_width, v_height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE( imagesBlurred[level] );
        if(b_use_virtual)
            imagesBlurred[level+nlevels] = vxCreateImage(ctx, v_width, v_height, VX_DF_IMAGE_U8);
        else
            imagesBlurred[level+nlevels] = vxCreateImage(ctx, v_width, v_height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE( imagesBlurred[level+nlevels] );
        if(b_use_virtual)
            descriptors[level] = vxCreateVirtualArray(globalGraph, zvx_descriptor, mnFeaturesPerLevel[level]);
        else
            descriptors[level] = vxCreateArray(ctx, zvx_descriptor, mnFeaturesPerLevel[level]);
        NVXIO_CHECK_REFERENCE( descriptors[level] );

        if(level == 0)
            keypointsScaled[level] = keypointsOriented[level];
        else
            if(b_use_virtual)
                keypointsScaled[level] = vxCreateVirtualArray(globalGraph, VX_TYPE_KEYPOINT, mnFeaturesPerLevel[level]);
            else
                keypointsScaled[level] = vxCreateArray(ctx, VX_TYPE_KEYPOINT, mnFeaturesPerLevel[level]);
        NVXIO_CHECK_REFERENCE( keypointsScaled[level] );

        int base = level*4;
        vx_uint32 val;
        //minX
        val = EDGE_THRESHOLD -3;
        coordinatesForGrid[base+0] = vxCreateScalar(ctx, VX_TYPE_UINT32, &val);
        NVXIO_CHECK_REFERENCE( coordinatesForGrid[base+0] );
        //minY
        coordinatesForGrid[base+1] = vxCreateScalar(ctx, VX_TYPE_UINT32, &val);
        NVXIO_CHECK_REFERENCE( coordinatesForGrid[base+1] );
        //maxX
        val = v_width - EDGE_THRESHOLD+3;
        coordinatesForGrid[base+2] = vxCreateScalar(ctx, VX_TYPE_UINT32, &val);
        NVXIO_CHECK_REFERENCE( coordinatesForGrid[base+2] );
        //maxY
        val = v_height - EDGE_THRESHOLD+3;
        coordinatesForGrid[base+3] = vxCreateScalar(ctx, VX_TYPE_UINT32, &val);

        NVXIO_CHECK_REFERENCE( coordinatesForGrid[base+3] );

        vx_scalar min_x = coordinatesForGrid[base+0],
                  min_y = coordinatesForGrid[base+1],
                  max_x = coordinatesForGrid[base+2],
                  max_y = coordinatesForGrid[base+3];
        if(level > 0)
        {
            //resize_node[level-1] = vxScaleImageNode(globalGraph, imagesResize[level-1], imagesResize[level], VX_INTERPOLATION_BILINEAR);
            #ifndef USE_PYRAMID
            resize_node[level-1] = vxScaleImageNode(globalGraph, imagesResized[0], imagesResized[level], VX_INTERPOLATION_BILINEAR);
            NVXIO_CHECK_REFERENCE( resize_node[level-1] );
            #endif
        }
        vxSetReferenceName((vx_reference)coordinatesForGrid[base], "TEST_LEVEL");

        fast_nodemin[level] = vxFastCornersNode(globalGraph, imagesResized[level], s_strength_threshold2, vx_true_e, keypointsFAST[level], num_corners[level]);
        NVXIO_CHECK_REFERENCE( fast_nodemin[level] );

        grid_node[level] = zvxMakeGridNode(globalGraph, keypointsFAST[level],
                  min_x, min_y, max_x, max_y,
                s_EDGE_THRESHOLD, s_strength_threshold1, s_WINDOW, keypointsFAST2[level]);
        NVXIO_CHECK_REFERENCE( grid_node[level] );

        quadtree_node[level] = zvxMakeQuadtreeNode(globalGraph, keypointsFAST2[level], min_x, min_y, max_x, max_y, s_NUM_FEATURES[level], keypointsDistributed[level] );
        NVXIO_CHECK_REFERENCE( quadtree_node[level] );

        vxSetReferenceName((vx_reference) quadtree_node[level], "TEST_LEVEL");

        orientation_node[level] = zvxComputeAngleNode(globalGraph, imagesResized[level], keypointsDistributed[level], umax_pattern, keypointsOriented[level]);
        NVXIO_CHECK_REFERENCE( orientation_node[level] );

        blur_node[level+nlevels] = vxConvolveNode(globalGraph, imagesResized[level], gaussianConvHorizontal, imagesBlurred[level+nlevels]);
        //blur_node[level] = vxGaussian3x3Node(globalGraph, imagesResized[level], imagesBlurred[level]);
        NVXIO_CHECK_REFERENCE( blur_node[level+nlevels] );

        blur_node[level] = vxConvolveNode(globalGraph, imagesBlurred[level+nlevels], gaussianConvVertical, imagesBlurred[level]);
        //blur_node[level] = vxGaussian3x3Node(globalGraph, imagesResized[level], imagesBlurred[level]);
        NVXIO_CHECK_REFERENCE( blur_node[level] );

        orb_node[level]  = zvxORBNode(globalGraph, imagesBlurred[level], keypointsOriented[level], patterns, descriptors[level]);
        NVXIO_CHECK_REFERENCE( orb_node[level] );

        if(level > 0)
        {
            scale_node[level - 1] = zvxScaleArrayNode(globalGraph, keypointsOriented[level], s_multiplier[level], keypointsScaled[level]);
            NVXIO_CHECK_REFERENCE( scale_node[level-1] );
        }
    }

    NVXIO_SAFE_CALL( vxVerifyGraph(globalGraph) );
}

ORBextractor::~ORBextractor()
{
    std::cout << "Avg computed frame ORB: " << ((double) totalTime / nFrame) / 1000000.0 << "ms" << std::endl;

    if(!times.empty())
    {
        std::ofstream timesFile;
        timesFile.open ("times.csv", std::ios_base::app);
        timesFile << "#Frame;Name Processing function;Level;Time spent (ns);Time spent (ms)" << std::endl;
        for(times_t t : times)
        {
            timesFile << t.frame << ";";
            timesFile << t.name  << ";";
            timesFile << t.level << ";";
            timesFile << t.time  << ";";
            timesFile << t.time / 1000000.0 << ";";
            timesFile << std::endl;
        }
        timesFile.close();
    }
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}

vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    GetTime tmp(this, "Make quadtree", level);
    // Compute how many initial nodes   
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if((int)lNodes.size()>=N)
            {
                bFinish = true;
                break;
            }

            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);                    
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }       

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(nlevels);

    const float W = 30;

    for (int level = 0; level < nlevels; ++level)
    {
        {
            GetTime tmp(this, "FAST+Grid", level);

            const int minBorderX = EDGE_THRESHOLD-3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
            const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures*10);

            const float width = (maxBorderX-minBorderX);
            const float height = (maxBorderY-minBorderY);

            const int nCols = width/W;
            const int nRows = height/W;
            const int wCell = ceil(width/nCols);
            const int hCell = ceil(height/nRows);


            if(b_gridfast_VX)//switch openvx/original implementation (true: use OpenVX)
            {
                if(true)// switch between FAST or GRID point calculation (true: only FAST with OpenVX)
                {
                    vx_size sz = 0;
                    vxCopyScalar(num_corners[level], &sz, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                    vxQueryArray(keypointsFAST[level], VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size));

                    std::map<std::pair<int, int>, bool> grid;

                    {
                        vx_size i, stride = sizeof(vx_size);
                        void *base = NULL;
                        vx_map_id map_id;
                        vxMapArrayRange(keypointsFAST[level], 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

                        // first step: read and add certain threshold
                        for (i = 0; i < sz; i++)
                        {
                            vx_keypoint_t *kp1 = &vxArrayItem(vx_keypoint_t, base, i, stride);
                            vx_keypoint_t kp = *kp1;
                            if(kp.strength < strength_threshold1) continue; //not corner in first metric
                            cv::KeyPoint p;
                            p.pt = cv::Point2f(kp.x - minBorderX, kp.y - minBorderY);
                            p.response = kp.strength;

                            //precalcola la cella
                            int nColumn = floor((p.pt.x)/wCell);
                            int nRow    = floor((p.pt.y)/hCell);

                            bool b1 = nColumn < nCols;
                            bool b2 = nRow < nRows;
                            bool b3 = nRow >= 0;
                            bool b4 = nColumn >= 0;
                            bool b5 = p.pt.x >= 0;
                            bool b6 = p.pt.x < maxBorderX-minBorderX;
                            bool b7 = p.pt.y >= 0;
                            bool b8 = p.pt.y < maxBorderY-minBorderY;

                            if(b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8)
                            {
                                vToDistributeKeys.push_back(p);
                                grid[std::pair<int, int>(nRow, nColumn)] = true;
                            }
                        }

                        // second step: add threshold if grid empty
                        for (i = 0; i < sz; i++)
                        {
                            vx_keypoint_t kp = vxArrayItem(vx_keypoint_t, base, i, stride);
                            if(kp.strength >= strength_threshold1) continue; //corner already processed in first metric
                            cv::KeyPoint p;
                            p.pt = cv::Point2f(kp.x - minBorderX, kp.y - minBorderY);
                            p.response = kp.strength;

                            //precalcola la cella
                            int nColumn = floor((p.pt.x)/wCell);
                            int nRow    = floor((p.pt.y)/hCell);

                            //if there is points in grid, move on
                            if(grid[std::pair<int, int>(nRow, nColumn)]) continue;

                            bool b1 = nColumn < nCols;
                            bool b2 = nRow < nRows;
                            bool b3 = nRow >= 0;
                            bool b4 = nColumn >= 0;
                            bool b5 = p.pt.x >= 0;
                            bool b6 = p.pt.x < maxBorderX-minBorderX;
                            bool b7 = p.pt.y >= 0;
                            bool b8 = p.pt.y < maxBorderY-minBorderY;

                            if(b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8)
                            {
                                vToDistributeKeys.push_back(p);
                            }
                        }

                        vxUnmapArrayRange(keypointsFAST[level], map_id);
                    }
                }
                else
                {
                    vx_size sz = 0;
                    vxQueryArray(keypointsFAST2[level], VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size));

                    std::map<std::pair<int, int>, bool> grid;

                    {
                        vx_size i, stride = sizeof(vx_size);
                        void *base = NULL;
                        vx_map_id map_id;
                        vxMapArrayRange(keypointsFAST2[level], 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
                        for (i = 0; i < sz; i++)
                        {
                            vx_keypoint_t kp = vxArrayItem(vx_keypoint_t, base, i, stride);

                            cv::KeyPoint p;
                            p.pt = cv::Point2f(kp.x, kp.y);
                            p.response = kp.strength;

                            vToDistributeKeys.push_back(p);
                        }
                        vxUnmapArrayRange(keypointsFAST2[level], map_id);
                    }
                }
            }
            else
            {
                for(int i=0; i<nRows; i++)
                {
                    const float iniY =minBorderY+i*hCell;
                    float maxY = iniY+hCell+6;

                    if(iniY>=maxBorderY-3)
                        continue;
                    if(maxY>maxBorderY)
                        maxY = maxBorderY;

                    for(int j=0; j<nCols; j++)
                    {
                        const float iniX =minBorderX+j*wCell;
                        float maxX = iniX+wCell+6;
                        if(iniX>=maxBorderX-6)
                            continue;
                        if(maxX>maxBorderX)
                            maxX = maxBorderX;

                        vector<cv::KeyPoint> vKeysCell;
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,iniThFAST,true);

                        if(vKeysCell.empty())
                        {
                            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                                 vKeysCell,minThFAST,true);
                        }

                        if(!vKeysCell.empty())
                        {
                            for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                            {
                                (*vit).pt.x+=j*wCell;
                                (*vit).pt.y+=i*hCell;
                                vToDistributeKeys.push_back(*vit);
                            }
                        }

                    }
                }
            }

            vector<KeyPoint> & keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            if(b_quadtree_VX)
            {
                //...
            }
            else
            {
                keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                          minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);
            }

            const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

            // Add border to coordinates and scale information
            const int nkps = keypoints.size();
            for(int i=0; i<nkps ; i++)
            {
                keypoints[i].pt.x+=minBorderX;
                keypoints[i].pt.y+=minBorderY;
                keypoints[i].octave=level;
                keypoints[i].size = scaledPatchSize;
            }
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level)
    {
        GetTime tmp(this, "Compute angle", level);
        if(b_orient_VX)
        {
            //...
        }
        else
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }

}

void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        const int nCells = levelRows*levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;


        float hY = cellH + 6;

        for(int i=0; i<levelRows; i++)
        {
            const float iniY = minBorderY + i*cellH - 3;
            iniYRow[i] = iniY;

            if(i == levelRows-1)
            {
                hY = maxBorderY+3-iniY;
                if(hY<=0)
                    continue;
            }

            float hX = cellW + 6;

            for(int j=0; j<levelCols; j++)
            {
                float iniX;

                if(i==0)
                {
                    iniX = minBorderX + j*cellW - 3;
                    iniXCol[j] = iniX;
                }
                else
                {
                    iniX = iniXCol[j];
                }


                if(j == levelCols-1)
                {
                    hX = maxBorderX+3-iniX;
                    if(hX<=0)
                        continue;
                }


                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                FAST(cellImage,cellKeyPoints[i][j],iniThFAST,true);

                if(cellKeyPoints[i][j].size()<=3)
                {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                }


                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if(nKeys>nfeaturesCell)
                {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                }
                else
                {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell-nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }

            }
        }


        // Retain by score

        while(nToDistribute>0 && nNoMore<nCells)
        {
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
            nToDistribute = 0;

            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    if(!bNoMore[i][j])
                    {
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures*2);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates
        for(int i=0; i<levelRows; i++)
        {
            for(int j=0; j<levelCols; j++)
            {
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                if((int)keysCell.size()>nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);


                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    keysCell[k].octave=level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        if((int)keypoints.size()>nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // and compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

vx_uint64 getNodeTime(vx_node node)
{
    vx_perf_t perf;
    vxQueryNode(node, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf));
    return perf.tmp;
    return 0;
}

void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors)
{ 
    auto start = std::chrono::steady_clock::now();
    if(_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    GetTime tmp(this, "Total Time ORB extraction", -1);

    if(b_complete_VX)//true: full OpenVX
    {
        vx_image vx_img = nvx_cv::createVXImageFromCVMat(ctx, image);
        nvxuCopyImage(ctx, vx_img, imageStart);
        vxReleaseImage(&vx_img);

        NVXIO_SAFE_CALL( vxProcessGraph(globalGraph) );
        //NVXIO_SAFE_CALL( vxScheduleGraph(globalGraph) );
        //NVXIO_SAFE_CALL( vxWaitGraph(globalGraph) );

        {
            times_t t;
            t.frame = nFrame;


#ifdef USE_PYRAMID
            t.name = "Pyramid/Resize";
            t.level = -1;
            t.time = getNodeTime(pyramid_node);
            times.push_back(t);
#else
            t.name = "Pyramid/Resize";
            for(int i = 1; i < nlevels; ++i)
            {
                t.level = i;
                t.time = getNodeTime(resize_node[i-1]);
                times.push_back(t);
            }
#endif
            t.name = "FAST";
            for(int i = 0; i < nlevels; ++i)
            {
                t.level = i;
                t.time = getNodeTime(fast_nodemin[i]);
                times.push_back(t);
            }
            t.name = "Grid";
            for(int i = 0; i < nlevels; ++i)
            {
                t.level = i;
                t.time = getNodeTime(grid_node[i]);
                times.push_back(t);
            }
            t.name = "Make quadtree";
            for(int i = 0; i < nlevels; ++i)
            {
                t.level = i;
                t.time = getNodeTime(quadtree_node[i]);
                times.push_back(t);
            }

            t.name = "Compute angle";
            for(int i = 0; i < nlevels; ++i)
            {
                t.level = i;
                t.time = getNodeTime(orientation_node[i]);
                times.push_back(t);
            }

            t.name = "Gaussian Blur";
            for(int i = 0; i < nlevels; ++i)
            {
                t.level = i;
                t.time = getNodeTime(blur_node[i]);
                times.push_back(t);
            }
            t.name = "ORB descriptor";
            for(int i = 0; i < nlevels; ++i)
            {
                t.level = i;
                t.time = getNodeTime(orb_node[i]);
                times.push_back(t);
            }
            t.name = "Compute scale";
            for(int i = 0; i < nlevels-1; ++i)
            {
                t.level = i+1;
                t.time = getNodeTime(scale_node[i]);
                times.push_back(t);
            }
        }

        vx_size n_kp = 0;
        for (int level = 0; level < nlevels; ++level)
        {
            vx_size sz_keypoints = 0;
            vxQueryArray(keypointsOriented[level], VX_ARRAY_NUMITEMS, &sz_keypoints, sizeof(vx_size));
//            std::cout << "Level " << level << ": " << sz_keypoints << std::endl;
            n_kp += sz_keypoints;
        }
        if(n_kp == 0)
            _descriptors.release();
        else
            _descriptors.create(n_kp, 32, CV_8U);

        _keypoints.reserve(n_kp);

        int offset = 0;

        cv::Mat descriptorsMat = _descriptors.getMat();

        for (int level = 0; level < nlevels; ++level)
        {
            vx_size sz_keypoints = 0;
            NVXIO_SAFE_CALL(vxQueryArray(keypointsScaled[level], VX_ARRAY_NUMITEMS, &sz_keypoints, sizeof(vx_size)));
            //std::cout << "Level " << level << " kp: " << sz_keypoints << std::endl;
            cv::Mat descriptorsMatOffset = descriptorsMat.rowRange(offset, offset + sz_keypoints);
            descriptorsMatOffset = Mat::zeros(sz_keypoints, 32, CV_8UC1);

            vx_size stride = sizeof(vx_size);
            void *base = NULL;
            vx_map_id map_id;
            //vx_size sz_descriptors = 0;
            //NVXIO_SAFE_CALL(vxQueryArray(descriptors[level], VX_ARRAY_NUMITEMS, &sz_descriptors, sizeof(vx_size)));
            //std::cout << "Level " << level << " descr: " << sz_descriptors << std::endl;
            NVXIO_SAFE_CALL( vxMapArrayRange(keypointsScaled[level], 0, sz_keypoints, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) );

            void* base_d = NULL;
            vx_map_id map_id_d;
            vx_size stride_d = sizeof(vx_size);
            NVXIO_SAFE_CALL( vxMapArrayRange(descriptors[level], 0, sz_keypoints, &map_id_d, &stride_d, &base_d, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) );

            for(vx_size i = 0; i < sz_keypoints; ++i)
            {
                vx_keypoint_t kpt = vxArrayItem(vx_keypoint_t, base, i, stride);
                cv::KeyPoint kp;
                kp.pt.x = kpt.x;
                kp.pt.y = kpt.y;
                kp.angle = kpt.orientation;
                kp.response = kpt.strength;
                kp.octave   = level;

                _keypoints.push_back(kp);

                descriptor_t d = vxArrayItem(descriptor_t, base_d, i, stride_d);

                uint8_t* u  = descriptorsMatOffset.ptr((int)i);
                for(int i2 = 0; i2 < 32; i2++)
                {
                    u[i2] = d.descr[i2];
                }
            }

            //uint8_t *d = _descriptors.getMat().ptr<uint8_t>(offset);
            //std::copy((uint8_t*)base_d, (uint8_t*)base_d + sz_keypoints*sizeof(descriptor_t), d);
            offset += sz_keypoints;

            vxUnmapArrayRange(descriptors[level], map_id_d);
            vxUnmapArrayRange(keypointsScaled[level], map_id);
        }
    }
    else
    {
        // Pre-compute the scale pyramid
        ComputePyramid(image);

        vector < vector<KeyPoint> > allKeypoints;
        ComputeKeyPointsOctTree(allKeypoints);
        //ComputeKeyPointsOld(allKeypoints);

        Mat descriptors;

        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += (int)allKeypoints[level].size();
        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        _keypoints.clear();
        _keypoints.reserve(nkeypoints);

        int offset = 0;
        for (int level = 0; level < nlevels; ++level)
        {
            vector<KeyPoint>& keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();

            if(nkeypointsLevel==0)
                continue;

            Mat workingMat;
            {
                GetTime tmp(this, "Gaussian Blur", level);
                // preprocess the resized image
                if(b_blur_VX)
                {
                    //nvx_cv::VXImageToCVMatMapper map(imagesBlurred[level]);
                    //blur
                }
                else
                {
                    workingMat = mvImagePyramid[level].clone();
                    GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
                }
            }
            // Compute the descriptors
            Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
            {
                GetTime tmp(this, "ORB descriptor", level);
                if(b_orb_VX)
                {

                }
                else
                {
                    computeDescriptors(workingMat, keypoints, desc, pattern);
                }
            }


            {
                GetTime tmp(this, "Compute scale", level);
                if(b_scale_VX)
                {
                    vx_size sz = 0;
                    vxQueryArray(keypointsScaled[level], VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size));
                    {
                        vx_size i, stride = sizeof(vx_size);
                        void *base = NULL;
                        vx_map_id map_id;
                        vxMapArrayRange(keypointsScaled[level], 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
                        for (i = 0; i < sz; i++)
                        {
                            vx_keypoint_t kp = vxArrayItem(vx_keypoint_t, base, i, stride);

                            cv::KeyPoint p;
                            p.pt = cv::Point2f(kp.x, kp.y);
                            p.response = kp.strength;
                            p.octave = level;
                            p.angle = kp.orientation;

                            _keypoints.push_back(p);
                        }
                        vxUnmapArrayRange(keypointsScaled[level], map_id);
                    }
                }
                else
                {
                    // Scale keypoint coordinates
                    if (level != 0)
                    {
                        GetTime tmp(this, "Compute scale", level);
                        float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
                        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                             keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                            keypoint->pt *= scale;
                    }
                    // And add the keypoints to the output
                    _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
                }
            }

            offset += nkeypointsLevel;
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    //std::cout << "Total time: " << std::chrono::duration <long long, nano> (diff).count() << " ns" << std::endl;
    totalTime += std::chrono::duration <long long, nano> (diff).count();
    nFrame ++;
}

void ORBextractor::ComputePyramid(cv::Mat image)
{
//    auto start = std::chrono::steady_clock::now();

    if(b_pyramid_VX)
    {
        vx_image vx_img = nvx_cv::createVXImageFromCVMat(ctx, image);
        //nvxuCopyImage(ctx, vx_img, imageStart);

        //NVXIO_SAFE_CALL( vxProcessGraph(globalGraph) );
        for (int level = 0; level < nlevels; ++level)
        {
            NVXIO_SAFE_CALL( vxuScaleImage(ctx, vx_img, imagesResized[level], VX_INTERPOLATION_BILINEAR));
            //std::cout << "Level #" << level << std::endl;
            nvx_cv::VXImageToCVMatMapper map(imagesResized[level]);
            mvImagePyramid[level] = map.getMat().clone();
        }//*/

        vxReleaseImage(&vx_img);
    }
    else
    {
        for (int level = 0; level < nlevels; ++level)
        {
            GetTime tmp(this, "Pyramid/Resize", level);

            float scale = mvInvScaleFactor[level];
            Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
            Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101+BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }
    }
/*
    for (int level = 0; level < nlevels; ++level)
    {
        cv::imshow(std::to_string(level), mvImagePyramid[level]);
    }
    cv::waitKey();*/
//    auto end = std::chrono::steady_clock::now();
//    auto diff = end - start;
    //std::cout << "Resize: " << std::chrono::duration <long long, nano> (diff).count() << " ns" << std::endl;
}


GetTime::GetTime(std::vector<times_t> &times, int nFrame, string name, int level)
    : o(NULL), times(times)
{
    t.frame = nFrame;
    t.name = name;
    t.level = level;

    start = std::chrono::steady_clock::now();
    //std::cout << t.name << " constructed" << std::endl;
}
GetTime::GetTime(ORBextractor *o, string name, int level)
    : o(o), times(o->times)
{
    t.frame = o->nFrame;
    t.name = name;
    t.level = level;

    start = std::chrono::steady_clock::now();
    //std::cout << t.name << " constructed" << std::endl;
}
GetTime::~GetTime()
{
    //std::cout << t.name << " destructed" << std::endl;
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    t.time = std::chrono::duration <long long, nano> (diff).count();
    times.push_back(t);
}

} //namespace ORB_SLAM
