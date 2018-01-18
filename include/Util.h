#pragma once

#include <opencv2//opencv.hpp>
#include <string>
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

template<typename m1, typename m2>
std::map<m1, m2> loadMap(cv::FileNode n);



cv::Mat loadMat(cv::FileNode n);

template<typename c1>
std::set<c1> loadSet(cv::FileNode n);

DBoW2::BowVector loadBowVec(cv::FileNode n);
DBoW2::FeatureVector loadFeatVec(cv::FileNode n);

template<typename c1>
std::vector<c1> loadVector1(cv::FileNode n);
template<typename c1>
std::vector<std::vector<c1>> loadVector2(cv::FileNode n);
template<typename c1>
std::vector<std::vector<std::vector<c1>>> loadVector3(cv::FileNode n);

//#include "Util.t.h"

DBoW2::BowVector loadBowVec(cv::FileNode n)
{
    DBoW2::BowVector vec;
    cv::FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it)
    {
        int i = *it;
        double v = *(++it);
        vec.addWeight(i,v);
    }
    return vec;
}

DBoW2::FeatureVector loadFeatVec(cv::FileNode n)
{
    DBoW2::FeatureVector vec;
    cv::FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it)
    {
        int i = *it;
        ++it;
        cv::FileNodeIterator it2 = (*it).begin(), it2_end = (*it).end();
        for( ; it2 != it2_end; ++it2)
        {
            vec.addFeature(i, (int) *(it2));
        }
    }
    return vec;
}

std::map<int, std::vector<int>> loadFeatVecId(cv::FileNode n)
{
    std::map<int, std::vector<int>> conversion;

    cv::FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it)
    {
        std::vector<int> idx;

        int i = *it;
        ++it;
        cv::read(*it, idx);
        conversion.insert(std::pair<int, std::vector<int>>(i, idx));
    }
    return conversion;
}

template<typename m1, typename m2>
std::map<m1, m2> loadMap(cv::FileNode n)
{
    std::map<m1, m2> m;
    cv::FileNode obj = n;
    cv::FileNodeIterator it = obj.begin(), it_end = obj.end();

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        m1 i;
        m2 v;
        cv::read(*it, (m1&) i, -1);
        cv::read(*(++it), (m2&) v, -1);
        m[i] = v;
    }

    return m;
}

template<>
std::map<int, std::vector<int>> loadMap(cv::FileNode n)
{
    std::map<int, std::vector<int>> m;
    cv::FileNode obj = n;
    cv::FileNodeIterator it = obj.begin(), it_end = obj.end();

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        int i;
        std::vector<int> v;
        cv::read(*it, (int&) i, -1);
        cv::read(*(++it), v);
        m[i] = v;
    }

    return m;
}

template<typename c1>
std::set<c1> loadSet(cv::FileNode n)
{
    std::set<c1> s;
    cv::FileNode obj = n;
    cv::FileNodeIterator it = obj.begin(), it_end = obj.end();

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        c1 v;
        cv::read(*it, v, 0);
        s.insert(v);
    }

    return s;
}

template<typename c1>
std::vector<c1> loadVector1(cv::FileNode n)
{
    std::vector<c1> v;
    cv::read(n, v);

    return v;
}

template<typename c1>
std::vector<std::vector<c1>> loadVector2(cv::FileNode n)
{
    std::vector<std::vector<c1>> v2;
    cv::FileNode obj = n;
    cv::FileNodeIterator it = obj.begin(), it_end = obj.end();

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        std::vector<c1> v1;
        cv::read(*it, v1);
        v2.push_back(v1);
    }

    return v2;
}

template<typename c1>
std::vector<std::vector<std::vector<c1>>> loadVector3(cv::FileNode n)
{
    std::vector<std::vector<std::vector<c1>>> v3;
    cv::FileNode obj = n;
    cv::FileNodeIterator it = obj.begin(), it_end = obj.end();

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        std::vector<std::vector<c1>> v2;
        cv::FileNodeIterator it2 = (*it).begin(), it2_end = (*it).end();
        for( ; it2 != it2_end; ++it2)
        {
            std::vector<c1> v1;
            cv::read(*it2, v1);
            v2.push_back(v1);
        }
        v3.push_back(v2);
    }

    return v3;
}

cv::Mat loadMat(cv::FileNode n)
{
    cv::Mat m;
    n >> m;
    return m;
}


