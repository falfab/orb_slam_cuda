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

#include "Map.h"

#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

void Map::saveMap(cv::FileStorage& mapFile)
{
    mapFile << "points" << "[";
    //get all the map points
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
    {
        MapPoint *p = *sit;
        if(p->isBad()) continue;
        mapFile << "{";
        p->write(mapFile);
        mapFile << "}";
    }
    mapFile << "]";

    //get all the keyframe
    mapFile << "keyframes" << "[";
    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
    {
        KeyFrame *k = *sit;
        if(k->isBad()) continue;
        mapFile << "{";
        k->write(mapFile);
        mapFile << "}";
    }
    mapFile << "]";

    mapFile << "mvpKeyFrameOrigins" << "[";
    for(vector<KeyFrame*>::iterator sit=mvpKeyFrameOrigins.begin(), send=mvpKeyFrameOrigins.end(); sit!=send; sit++)
    {
        if((*sit)->isBad()) continue;
        mapFile << (int) (*sit)->mnId;
    }
    mapFile << "]";

    //get all the keyframe
    mapFile << "mvpReferenceMapPoints" << "[";
    for(vector<MapPoint*>::iterator sit=mvpReferenceMapPoints.begin(), send=mvpReferenceMapPoints.end(); sit!=send; sit++)
    {
        if((*sit)->isBad()) continue;
        mapFile << (int) (*sit)->mnId;
    }
    mapFile << "]";

    mapFile << "mnMaxKFid" << (int) mnMaxKFid;
    mapFile << "mnBigChangeIdx" << mnBigChangeIdx;
    //reference map points
    //keyframe origins

}

void Map::loadMap(cv::FileNode mapFile, ORBVocabulary* voc, KeyFrameDatabase* db)
{
    cv::FileNode points = mapFile["points"];
    cv::FileNodeIterator it = points.begin(), it_end = points.end();

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        cv::FileNode f = *it;
        mspMapPoints.insert(new MapPoint(f, this));
    }

    cv::FileNode keyframe = mapFile["keyframes"];
    it = keyframe.begin(), it_end = keyframe.end();

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        cv::FileNode f = *it;
        int id = f["mnId"];
        mspKeyFrames.insert(new KeyFrame(f, voc, db, this));
    }

    std::vector<MapPoint*> pointsVec;
    std::vector<KeyFrame*> keyframesVec;
    MapPoint *badPoint = new MapPoint();
    KeyFrame *badFrame = new KeyFrame();
    unsigned long maxId = 0;
    for(auto const &p : mspMapPoints)
    {
        if(p->mnId > maxId) maxId = p->mnId;
    }
    pointsVec.reserve(maxId+1);
    maxId = 0;
    for(auto const &k : mspKeyFrames)
    {
        if(k->mnId > maxId) maxId = k->mnId;
    }
    keyframesVec.reserve(maxId+1);
    for(unsigned int i = 0; i < pointsVec.capacity(); ++i)    pointsVec.push_back(badPoint);
    for(unsigned int i = 0; i < keyframesVec.capacity(); ++i) keyframesVec.push_back(badFrame);
    for(auto const &p : mspMapPoints)
    {
        pointsVec[p->mnId] = p;
    }
    maxId = 0;
    for(auto const &k : mspKeyFrames)
    {
        keyframesVec[k->mnId] = k;
    }


    cv::FileNode obj = mapFile["mvpKeyFrameOrigins"];
    it = obj.begin(), it_end = obj.end();
    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        if(-1 == (int) (*it)) continue;
        mvpKeyFrameOrigins.push_back(keyframesVec[(int) *it]);
    }

    obj = mapFile["mvpReferenceMapPoints"];
    it = obj.begin(), it_end = obj.end();
    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it)
    {
        if(-1 == (int) *it) continue;
        mvpReferenceMapPoints.push_back(pointsVec[(int) *it]);
    }

    for(auto const &p : mspMapPoints)
    {
        p->updateLinks(keyframesVec, pointsVec);
    }
    for(auto const &k : mspKeyFrames)
    {
        k->updateLinks(keyframesVec, pointsVec);
    }

    cv::read(mapFile["mnMaxKFid"],mnMaxKFid, 0);
    mapFile["mnBigChangeIdx"] >> mnBigChangeIdx;

    /*for(auto const &f : keyframesVec)
    {
        db->add(f);
    }*/
    db->load(mapFile["kfDB"], keyframesVec);
}

/*
void Map::loadMap()
{
    ifstream mapFile ("map.bin", ios::in | ios::binary);
    if(!mapFile.is_open()) return;
    size_t points;
    mapFile.read((char*)&points, sizeof(size_t));
    std::cout << "#points: " << points << std::endl;
    mapFile.read((char*)&points, sizeof(size_t));
    std::cout << "#keyframe: " << points << std::endl;

    mapFile.close();
}*/

} //namespace ORB_SLAM
