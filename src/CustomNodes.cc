#include "CustomNodes.h"
#include <iostream>
#include <cmath>
#include <map>
#include <OVX/UtilityOVX.hpp>
#include <opencv2/opencv.hpp>
#include <list>

#define CHECK_ERROR(call)                                                                 \
    {                                                                                     \
        cudaError_t err = call;                                                           \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err);                                                                    \
        }                                                                                 \
    }

#define PATTERN_SIZE 512 // 512 points that is 1024 elements

using namespace std;

vx_enum zvx_descriptor;

vx_char makeGrid_name[] = "user.kernel.makeGrid";
vx_char makeQuadtree_name[] = "user.kernel.makeQuadtree";
vx_char computeAngle_name[] = "user.kernel.computeAngle";
//vx_char ORB_name[] = "gpu:user.kernel.ORB"; // TODO per andare su GPU inizia con questo nome!
vx_char ORB_name[] = "user.kernel.ORB";
vx_char scaleArray_name[] = "user.kernel.scaleArray";

class ExtractorNode
{
  public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
        const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

        //Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x + halfX, UL.y);
        n1.BL = cv::Point2i(UL.x, UL.y + halfY);
        n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x, UL.y + halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x, BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        //Associate points to childs
        for (size_t i = 0; i < vKeys.size(); i++)
        {
            const vx_keypoint_t &kp = vKeys[i];
            if (kp.x < n1.UR.x)
            {
                if (kp.y < n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if (kp.y < n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if (n1.vKeys.size() == 1)
            n1.bNoMore = true;
        if (n2.vKeys.size() == 1)
            n2.bNoMore = true;
        if (n3.vKeys.size() == 1)
            n3.bNoMore = true;
        if (n4.vKeys.size() == 1)
            n4.bNoMore = true;
    }

    std::vector<vx_keypoint_t> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

//-------------------------------------------------------------------------------------------------
//-------------------------------- Kernel core functions ------------------------------------------
//-------------------------------------------------------------------------------------------------

#ifdef false
vx_status computeAngle_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 2)
        return VX_FAILURE;
    vx_image input = (vx_image)parameters[0];
    vx_image output = (vx_image)parameters[1];
    // Gets the CUDA stream used for current node.
    //cudaStream_t stream = NULL;
    //vxQueryNode(node, NVX_NODE_CUDA_STREAM, &stream, sizeof(stream));
    // Maps the objects.
    vx_imagepatch_addressing_t in_addr, out_addr;
    vx_uint8 *in_ptr = NULL, *out_ptr = NULL;
    vx_map_id in_map_id, out_map_id;
    vxMapImagePatch(input, NULL, 0, &in_map_id, &in_addr, (void **)&in_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    vxMapImagePatch(output, NULL, 0, &out_map_id, &out_addr, (void **)&out_ptr, VX_WRITE_ONLY, NVX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    /*    std::cout << "dim_x" << in_addr.dim_x << std::endl;
    std::cout << "dim_y" << in_addr.dim_y << std::endl;
    std::cout << "scale_x" << in_addr.scale_x << std::endl;
    std::cout << "scale_y" << in_addr.scale_y << std::endl;
    std::cout << "step_x" << in_addr.step_x << std::endl;
    std::cout << "step_y" << in_addr.step_y << std::endl;
    std::cout << "stride_x" << in_addr.stride_x << std::endl;
    std::cout << "stride_y" << in_addr.stride_y << std::endl;*/

    //    vx_uint8 value = 0;
    // Unmaps the objects.
    vxUnmapImagePatch(input, in_map_id);
    vxUnmapImagePatch(output, out_map_id);
    return VX_SUCCESS;
}
#endif
vx_status makeGrid_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    //return VX_SUCCESS;
    if (num != 9)
        return VX_ERROR_INVALID_PARAMETERS;

    vx_array kp_in = (vx_array)parameters[0];
    vx_uint32
        min_x,
        min_y,
        max_x,
        max_y;
    vxCopyScalar((vx_scalar)parameters[1], &min_x, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)parameters[2], &min_y, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)parameters[3], &max_x, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)parameters[4], &max_y, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_uint32 edge_threshold;
    vxCopyScalar((vx_scalar)parameters[5], &edge_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vx_float32 newfast_value;
    vxCopyScalar((vx_scalar)parameters[6], &newfast_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vx_uint32 window;
    vxCopyScalar((vx_scalar)parameters[7], &window, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_array kp_out = (vx_array)parameters[8];

    const vx_int32 width = max_x - min_x;
    const vx_int32 height = max_y - min_y;

    const vx_int32 nCols = width / window;
    const vx_int32 nRows = height / window;

    const vx_int32 wCell = ceil(width / nCols);
    const vx_int32 hCell = ceil(height / nRows);

    //temporary scratchpad memory
    const int MAX_SIZE = 256;
    vx_keypoint_t tmpKp[MAX_SIZE];
    int currentIdx = 0;

    {
        vxTruncateArray(kp_out, 0);

        vx_size sz = 0;
        vxQueryArray(kp_in, VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size));

        std::map<std::pair<int, int>, bool> grid;

        vx_size i, stride = sizeof(vx_size);
        void *base = NULL;
        vx_map_id map_id;
        vxMapArrayRange(kp_in, 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

        // first step: read and add certain threshold
        for (i = 0; i < sz; i++)
        {
            tmpKp[currentIdx] = vxArrayItem(vx_keypoint_t, base, i, stride);

            if (tmpKp[currentIdx].strength < newfast_value)
                continue; //not corner in first metric

            //precalcola la cella
            int nColumn = floor((tmpKp[currentIdx].x - min_x) / wCell);
            int nRow = floor((tmpKp[currentIdx].y - min_y) / hCell);

            bool b1 = nColumn < nCols;
            bool b2 = nRow < nRows;
            bool b3 = nRow >= 0;
            bool b4 = nColumn >= 0;
            bool b5 = tmpKp[currentIdx].x >= min_x;
            bool b6 = tmpKp[currentIdx].x < max_x;
            bool b7 = tmpKp[currentIdx].y >= min_x;
            bool b8 = tmpKp[currentIdx].y < max_x;

            //tmpKp[currentIdx].x -= min_x;
            //tmpKp[currentIdx].y -= min_y;

            if (b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8)
            {
                //vToDistributeKeys.push_back(p);
                if (++currentIdx == MAX_SIZE)
                {
                    vxAddArrayItems(kp_out, MAX_SIZE, tmpKp, sizeof(vx_keypoint_t));
                    currentIdx = 0;
                }
                grid[std::pair<int, int>(nRow, nColumn)] = true;
            }
        }

        // second step: add threshold if grid empty
        for (i = 0; i < sz; i++)
        {
            tmpKp[currentIdx] = vxArrayItem(vx_keypoint_t, base, i, stride);
            if (tmpKp[currentIdx].strength >= newfast_value)
                continue; //corner already processed in first metric

            //precalcola la cella
            int nColumn = floor((tmpKp[currentIdx].x - min_x) / wCell);
            int nRow = floor((tmpKp[currentIdx].y - min_y) / hCell);

            //if there is points in grid, move on
            if (grid[std::pair<int, int>(nRow, nColumn)])
                continue;

            bool b1 = nColumn < nCols;
            bool b2 = nRow < nRows;
            bool b3 = nRow >= 0;
            bool b4 = nColumn >= 0;
            bool b5 = tmpKp[currentIdx].x >= min_x;
            bool b6 = tmpKp[currentIdx].x < max_x;
            bool b7 = tmpKp[currentIdx].y >= min_x;
            bool b8 = tmpKp[currentIdx].y < max_x;

            //tmpKp[currentIdx].x -= min_x;
            //tmpKp[currentIdx].y -= min_y;

            if (b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8)
            {
                if (++currentIdx == MAX_SIZE)
                {
                    vxAddArrayItems(kp_out, currentIdx, tmpKp, sizeof(vx_keypoint_t));
                    currentIdx = 0;
                }
            }
        }

        if (currentIdx > 0)
        {
            vxAddArrayItems(kp_out, currentIdx, tmpKp, sizeof(vx_keypoint_t));
            currentIdx = 0;
        }

        vxUnmapArrayRange(kp_in, map_id);
    }

    return VX_SUCCESS;
}
vx_status makeQuadtree_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    //return VX_SUCCESS;
    if (num != 7)
        return VX_ERROR_INVALID_PARAMETERS;

    vx_array kp_in = (vx_array)parameters[0];
    vx_uint32
        min_x,
        min_y,
        max_x,
        max_y;
    vxCopyScalar((vx_scalar)parameters[1], &min_x, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)parameters[2], &min_y, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)parameters[3], &max_x, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)parameters[4], &max_y, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vx_size num_features = 0;
    vxCopyScalar((vx_scalar)parameters[5], &num_features, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vx_array kp_out = (vx_array)parameters[6];

    vxTruncateArray(kp_out, 0);
    vx_size sz = 0;
    vxQueryArray(kp_in, VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size));

    std::map<std::pair<int, int>, bool> grid;

    {
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(max_x - min_x) / (max_y - min_y));

        const vx_size N = num_features;

        const float hX = static_cast<float>(max_x - min_x) / nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode *> vpIniNodes;
        vpIniNodes.resize(nIni);

        for (int i = 0; i < nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
            ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
            ni.BL = cv::Point2i(ni.UL.x, max_y - min_y);
            ni.BR = cv::Point2i(ni.UR.x, max_y - min_y);
            ni.vKeys.reserve(sz); //vToDistributeKeys.size();

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        {
            vx_size i, stride = sizeof(vx_size);
            void *base = NULL;
            vx_map_id map_id;
            vxMapArrayRange(kp_in, 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

            //Associate points to childs
            for (i = 0; i < sz; i++)
            {
                vx_keypoint_t kp = vxArrayItem(vx_keypoint_t, base, i, stride); // vToDistributeKeys[i];
                kp.x -= min_x;
                kp.y -= min_y;
                vpIniNodes[kp.x / hX]->vKeys.push_back(kp);
            }

            vxUnmapArrayRange(kp_in, map_id);
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while (lit != lNodes.end())
        {
            if (lit->vKeys.size() == 1)
            {
                lit->bNoMore = true;
                lit++;
            }
            else if (lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        bool bFinish = false;

        int iteration = 0;

        vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size() * 4);
        while (!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while (lit != lNodes.end())
            {
                if ((int)lNodes.size() >= N)
                {
                    bFinish = true;
                    break;
                }

                if (lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1, n2, n3, n4;
                    lit->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit = lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
            {
                bFinish = true;
            }
            else if (((int)lNodes.size() + nToExpand * 3) > N)
            {

                while (!bFinish)
                {

                    prevSize = lNodes.size();

                    vector<pair<int, ExtractorNode *>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                    for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                    {
                        ExtractorNode n1, n2, n3, n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                        // Add childs if they contain points
                        if (n1.vKeys.size() > 0)
                        {
                            lNodes.push_front(n1);
                            if (n1.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n2.vKeys.size() > 0)
                        {
                            lNodes.push_front(n2);
                            if (n2.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n3.vKeys.size() > 0)
                        {
                            lNodes.push_front(n3);
                            if (n3.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n4.vKeys.size() > 0)
                        {
                            lNodes.push_front(n4);
                            if (n4.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if ((int)lNodes.size() >= N)
                            break;
                    }

                    if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                        bFinish = true;
                }
            }
        }

        // Retain the best point in each node
        {
            const int MAX_SIZE = 256;
            int localIdx = 0;
            vx_keypoint_t localBuffer[MAX_SIZE];

            vxTruncateArray(kp_out, 0);
            int countFeatures = 0;
            vx_size stride = sizeof(vx_size);
            void *base = NULL;
            vx_map_id map_id;
            vxMapArrayRange(kp_in, 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
            for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
            {

                vector<vx_keypoint_t> &vNodeKeys = lit->vKeys;
                vx_keypoint_t *pKP = &vNodeKeys[0];
                float maxResponse = pKP->strength;

                for (size_t k = 1; k < vNodeKeys.size(); k++)
                {
                    if (vNodeKeys[k].strength > maxResponse)
                    {
                        pKP = &vNodeKeys[k];
                        maxResponse = vNodeKeys[k].strength;
                    }
                }

                if (++countFeatures > N)
                {
                    //limit to current size
                    break;
                    //std::cerr << "WARNING: more features than thought!" << std::endl << std::flush;
                }
                localBuffer[localIdx] = *pKP;
                localBuffer[localIdx].x += min_x;
                localBuffer[localIdx].y += min_y;
                if (++localIdx == MAX_SIZE)
                {
                    vxAddArrayItems(kp_out, MAX_SIZE, localBuffer, sizeof(vx_keypoint_t));
                    localIdx = 0;
                }
                //vResultKeys.push_back(*pKP);
            }

            if (localIdx > 0)
            {
                vxAddArrayItems(kp_out, localIdx, localBuffer, sizeof(vx_keypoint_t));
                localIdx = 0;
            }

            vxUnmapArrayRange(kp_in, map_id);
        }
    }

    return VX_SUCCESS;
}

const int HALF_PATCH_SIZE = 15;
vx_status computeAngle_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 4)
        return VX_ERROR_INVALID_PARAMETERS;

    vx_image img = (vx_image)parameters[0];
    vx_array kp_in = (vx_array)parameters[1];
    vx_array patterns = (vx_array)parameters[2];
    vx_array kp_out = (vx_array)parameters[3];

    vx_size sz_kp = 0;
    vxQueryArray(kp_in, VX_ARRAY_NUMITEMS, &sz_kp, sizeof(vx_size));
    vx_size sz_patterns = 0;
    vxQueryArray(patterns, VX_ARRAY_NUMITEMS, &sz_patterns, sizeof(vx_size));

    vx_size i,
        stride_kp_in = sizeof(vx_size),
        stride_patterns = sizeof(vx_size);
    void
        *base_kp_in = NULL,
        *base_patterns = NULL;
    vx_map_id
        map_id_kp_in,
        map_id_patterns;
    vxMapArrayRange(kp_in, 0, sz_kp, &map_id_kp_in, &stride_kp_in, &base_kp_in, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    vxMapArrayRange(patterns, 0, sz_patterns, &map_id_patterns, &stride_patterns, &base_patterns, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

    vx_uint32 width, height;
    vxQueryImage(img, VX_IMAGE_WIDTH, &width, sizeof(width));
    vxQueryImage(img, VX_IMAGE_HEIGHT, &height, sizeof(height));
    vx_rectangle_t rect = {
        0u, 0u,
        width, height};
    void *ptr_img;
    vx_map_id map_id_img;
    vx_imagepatch_addressing_t addr_img;
    vxMapImagePatch(img, &rect, 0, &map_id_img, &addr_img, &ptr_img, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

    const int MAX_SIZE = 256;
    int localIdx = 0;
    vx_keypoint_t localBuffer[MAX_SIZE];

    vxTruncateArray(kp_out, 0);
    for (i = 0; i < sz_kp; ++i)
    {
        localBuffer[localIdx] = vxArrayItem(vx_keypoint_t, base_kp_in, i, stride_kp_in);
        int m_01 = 0, m_10 = 0;

        const vx_uint8 *center = ((vx_uint8 *)ptr_img) + localBuffer[localIdx].y * addr_img.stride_y * addr_img.step_y + localBuffer[localIdx].x * addr_img.stride_x * addr_img.step_x; //  &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = vxArrayItem(vx_int32, base_patterns, v, stride_patterns);
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v * addr_img.stride_y], val_minus = center[u - v * addr_img.stride_y];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        localBuffer[localIdx].orientation = cv::fastAtan2((float)m_01, (float)m_10);

        if (++localIdx == MAX_SIZE)
        {
            vxAddArrayItems(kp_out, MAX_SIZE, localBuffer, sizeof(vx_keypoint_t));
            localIdx = 0;
        }
    }
    if (localIdx > 0)
    {
        vxAddArrayItems(kp_out, localIdx, localBuffer, sizeof(vx_keypoint_t));
        localIdx = 0;
    }

    vxUnmapImagePatch(img, map_id_img);
    vxUnmapArrayRange(kp_in, map_id_kp_in);
    vxUnmapArrayRange(patterns, map_id_patterns);

    return VX_SUCCESS;
}

// TODO
// Cuda kernel Simone
/*void computeOrbDescriptorCUDA_kernel(cv::KeyPoint *kpt, uchar *img, int imgW, uchar* desc, int kpt_size, const int step, vector<cv::Point> patt){
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;

    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;

    if(i >= kpt_size){
        return;
    }

    float angle = (float)kpt[i].angle*factorPI;
    float a = (float)cos(angle);
    float b = (float)sin(angle);

    const uchar *center = &img[(int)((floor(kpt[i].pt.y)) * imgW + floor(kpt[i].pt.x))];
#define GET_VALUE(idx) \
    center[(int)(floor(pat[idx].x*b + pat[idx].y*a)*step + \
    (floor(pat[idx].x*a - pat[idx].y*b)))]

    d_Point* pat = patt + 16*j;

    int t0, t1, val;
    t0 = GET_VALUE(0);
    t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2);
    t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4);
    t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6);
    t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8);
    t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10);
    t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12);
    t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14);
    t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    desc[i*32+j] = (uchar)val;
#undef GET_VALUE

}*/

vx_status ORB_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    //    std::cout << "HELLO, ORB KERNEL!" << std::endl << std::flush;
    if (num != 4)
        return VX_ERROR_INVALID_PARAMETERS;

    vx_image img = (vx_image)parameters[0];
    vx_array kp_in = (vx_array)parameters[1];
    vx_array patterns = (vx_array)parameters[2];
    vx_array descr_out = (vx_array)parameters[3];

    NVXIO_CHECK_REFERENCE(descr_out);

    vx_size num_keypoints = 0;
    vxQueryArray(kp_in, VX_ARRAY_NUMITEMS, &num_keypoints, sizeof(vx_size));
    //    std::cout << "Num keypoints: " << num_keypoints << std::endl << std::flush;
    vx_size i, stride = sizeof(vx_size);
    void *base = NULL;
    vx_map_id map_id;
    vxMapArrayRange(kp_in, 0, num_keypoints, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

    int sz_patterns = 0;
    vxQueryArray(patterns, VX_ARRAY_NUMITEMS, &sz_patterns, sizeof(vx_size));
    vx_size stride_p = sizeof(vx_size);
    void *base_p = NULL;
    vx_map_id map_id_p;
    vxMapArrayRange(patterns, 0, sz_patterns, &map_id_p, &stride_p, &base_p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

    vx_uint32 width, height;
    vxQueryImage(img, VX_IMAGE_WIDTH, &width, sizeof(width));
    vxQueryImage(img, VX_IMAGE_HEIGHT, &height, sizeof(height));
    vx_rectangle_t rect = {
        0u, 0u,
        width, height};
    void *ptr_img;
    vx_map_id map_id_img;
    vx_imagepatch_addressing_t addr_img;
    vxMapImagePatch(img, &rect, 0, &map_id_img, &addr_img, &ptr_img, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

    vxTruncateArray(descr_out, 0);

    // mappatch truncate array
    void *ptr_descr;
    vx_map_id map_id_descr;
    vx_size stride_descr = sizeof(vx_size);
    vxMapArrayRange(descr_out, 0, num_keypoints * 32, &map_id_descr, &stride_descr, &ptr_descr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

    //computeDescriptorsCUDA(const Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors, const vector<Point>& pattern)
    //computeDescriptorsCUDA((Mat&)ptr_img, (vector<KeyPoint>&)base, (Mat&)descr..., (vector<Point>&) base_p);

    //computeOrbDescriptorCUDA_kernel((cv::KeyPoint*) base, (uchar*) ptr_img, width, (uchar*) ptr_descr, ((vector<cv::KeyPoint>) base).size(), ((cv::Mat)ptr_img).step, (vector<cv::Point>)base_p);

    const float factorPI = (float)(CV_PI / 180.f);

    //const int step = addr_img.step_y * addr_img.stride_y;//p(int)img.step;
    //std::cout << "width: " << width << " step: " << step << std::endl;
    const int MAX_SIZE = 256;
    descriptor_t localBuffer[MAX_SIZE];
    int localIdx = 0;

    int totalAdd = 0;

    for (i = 0; i < num_keypoints; i++)
    {
        vx_keypoint_t kp = vxArrayItem(vx_keypoint_t, base, i, stride);

        float angle = (float)kp.orientation * factorPI;
        float a = (float)cos(angle), b = (float)sin(angle);

        const vx_uint8 *center = ((vx_uint8 *)ptr_img) + kp.y * addr_img.stride_y * addr_img.step_y + kp.x * addr_img.stride_x * addr_img.step_x; //&img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
        const int step = addr_img.step_y * addr_img.stride_y;                                                                                     //p(int)img.step;

        vx_int32 *pattern = (vx_int32 *)base_p;

#define GET_VALUE(idx)                                                           \
    center[cvRound(pattern[idx << 1] * b + pattern[(idx << 1) + 1] * a) * step + \
           cvRound(pattern[idx << 1] * a - pattern[(idx << 1) + 1] * b)]
        for (int lv = 0; lv < 32; ++lv, pattern += 32)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0);
            t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2);
            t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4);
            t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6);
            t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8);
            t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10);
            t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12);
            t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14);
            t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            localBuffer[localIdx].descr[lv] = (vx_uint8)val;
        }
#undef GET_VALUE

        if (++localIdx == MAX_SIZE)
        {
            totalAdd += MAX_SIZE;
            vxAddArrayItems(descr_out, MAX_SIZE, localBuffer, sizeof(descriptor_t));
            localIdx = 0;
        }
    }
    if (localIdx > 0)
    {
        totalAdd += localIdx;
        vxAddArrayItems(descr_out, localIdx, localBuffer, sizeof(descriptor_t));
        localIdx = 0;
    }

    //std::cout << "ORB extractor: got " << num_keypoints << " kp, computed " << totalAdd << " descriptors" << std::endl;

    //    std::cout << "Unmap pattenrs!" << std::endl << std::flush;
    vxUnmapArrayRange(patterns, map_id_p);
    //    std::cout << "Unmap kp_in!" << std::endl << std::flush;
    vxUnmapArrayRange(kp_in, map_id);
    vxUnmapImagePatch(img, map_id_img);
    vxUnmapArrayRange(descr_out, map_id_descr);
    //    std::cout << "HELLO, END ORB KERNEL!" << std::endl << std::flush;
    return VX_SUCCESS;
}
vx_status scaleArray_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    //std::cout << "HELLO!" << std::endl << std::flush;
    if (num != 3)
        return VX_ERROR_INVALID_PARAMETERS;

    vx_array kp_in = (vx_array)parameters[0];
    vx_float32 mult;
    NVXIO_SAFE_CALL(vxCopyScalar((vx_scalar)parameters[1], &mult, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    vx_array kp_out = (vx_array)parameters[2];
    NVXIO_CHECK_REFERENCE(parameters[2]);

    vxTruncateArray(kp_out, 0);
    int sz = 0;
    //std::cout << "Querying array size!" << std::endl;
    NVXIO_SAFE_CALL(vxQueryArray(kp_in, VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size)));
    vx_size i, stride = sizeof(vx_size);
    void *base = NULL;
    vx_map_id map_id;
    NVXIO_SAFE_CALL(vxMapArrayRange(kp_in, 0, sz, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    //std::cout << "Level: scale: " << mult<< std::endl;
    //std::cout << "Multiplier: " << mult << " size " << sz << std::endl;

    //vx_size cp;
    //NVXIO_SAFE_CALL( vxQueryArray(kp_out, VX_ARRAY_CAPACITY, &cp, sizeof(vx_size)) );
    //std::cout << "Capacity size: " << cp << std::endl;

    const int MAX_SIZE = 128;
    vx_keypoint_t localBuffer[MAX_SIZE];
    int localIdx = 0;

    for (i = 0; i < sz; ++i)
    {
        localBuffer[localIdx] = vxArrayItem(vx_keypoint_t, base, i, stride);
        localBuffer[localIdx].x *= mult;
        localBuffer[localIdx].y *= mult;
        if (++localIdx == MAX_SIZE)
        {
            //std::cout << "Adding " << MAX_SIZE << " elements to kp_out" << std::endl;
            NVXIO_SAFE_CALL(vxAddArrayItems(kp_out, MAX_SIZE, localBuffer, sizeof(vx_keypoint_t)));
            localIdx = 0;
        }
    }

    if (localIdx > 0)
    {
        NVXIO_SAFE_CALL(vxAddArrayItems(kp_out, localIdx, localBuffer, sizeof(vx_keypoint_t)));
        localIdx = 0;
    }

    vxUnmapArrayRange(kp_in, map_id);
    NVXIO_SAFE_CALL(vxQueryArray(kp_out, VX_ARRAY_NUMITEMS, &sz, sizeof(vx_size)));
    return VX_SUCCESS;
}

//-------------------------------------------------------------------------------------------------
//-------------------------------- Validation functions ---------------------------------------------
//-------------------------------------------------------------------------------------------------
vx_status makeGrid_validate(vx_node node, const vx_reference parameters[],
                            vx_uint32 num, vx_meta_format metas[])
{
    if (num != 9)
        return VX_ERROR_INVALID_PARAMETERS;
    vx_status status = VX_SUCCESS;

    {
        vx_array src = (vx_array)parameters[0];
        vx_size capacity;
        vx_enum itemType;

        vxQueryArray(src, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        vxQueryArray(src, VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));

        vxSetMetaFormatAttribute(metas[8], VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        vxSetMetaFormatAttribute(metas[8], VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));
    }

    return status;
}

vx_status makeQuadtree_validate(vx_node node, const vx_reference parameters[],
                                vx_uint32 num, vx_meta_format metas[])
{
    if (num != 7)
        return VX_ERROR_INVALID_PARAMETERS;
    vx_status status = VX_SUCCESS;

    {
        vx_array src = (vx_array)parameters[0];
        vx_size capacity;
        vx_enum itemType;

        vx_scalar s_numFeatures = (vx_scalar)parameters[5];
        vx_uint32 numFeatures;
        vxCopyScalar(s_numFeatures, &numFeatures, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        capacity = static_cast<vx_size>(numFeatures);
        //vxQueryArray(src, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        vxQueryArray(src, VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));

        vxSetMetaFormatAttribute(metas[6], VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        vxSetMetaFormatAttribute(metas[6], VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));
    }

    return status;
}

vx_status computeAngle_validate(vx_node node, const vx_reference parameters[],
                                vx_uint32 num, vx_meta_format metas[])
{
    if (num != 4)
        return VX_ERROR_INVALID_PARAMETERS;
    vx_status status = VX_SUCCESS;

    vx_array src = (vx_array)parameters[1];
    vx_size capacity;
    vx_enum itemType;

    vxQueryArray(src, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
    vxQueryArray(src, VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));

    vxSetMetaFormatAttribute(metas[3], VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
    vxSetMetaFormatAttribute(metas[3], VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));

    return status;
}

vx_status ORB_validate(vx_node node, const vx_reference parameters[],
                       vx_uint32 num, vx_meta_format metas[])
{
    if (num != 4)
        return VX_ERROR_INVALID_PARAMETERS;
    vx_status status = VX_SUCCESS;

    {
        vx_array src = (vx_array)parameters[1];
        vx_size capacity;
        vx_enum itemType;

        vxQueryArray(src, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        itemType = zvx_descriptor;

        vxSetMetaFormatAttribute(metas[3], VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        vxSetMetaFormatAttribute(metas[3], VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));
    }

    return status;
}

vx_status scaleArray_validate(vx_node node, const vx_reference parameters[],
                              vx_uint32 num, vx_meta_format metas[])
{
    if (num != 3)
        return VX_ERROR_INVALID_PARAMETERS;
    vx_status status = VX_SUCCESS;

    {
        vx_array src = (vx_array)parameters[0];
        vx_size capacity;
        vx_enum itemType;

        vxQueryArray(src, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        vxQueryArray(src, VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));

        vxSetMetaFormatAttribute(metas[2], VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
        vxSetMetaFormatAttribute(metas[2], VX_ARRAY_ITEMTYPE, &itemType, sizeof(vx_enum));
    }

    return status;
}

//-------------------------------------------------------------------------------------------------
//-------------------------------- Register functions ---------------------------------------------
//-------------------------------------------------------------------------------------------------

vx_status registerMakeGrid(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_uint32 num_params = 9;
    vx_kernel kernel = vxAddUserKernel(context, makeGrid_name, USER_KERNEL_MAKE_GRID_NODE,
                                       makeGrid_kernel,
                                       num_params,
                                       makeGrid_validate,
                                       NULL, // init
                                       NULL  // deinit
    );
    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create makeGrid Kernel");
        return status;
    }
    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // src
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // minX
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // minY
    status |= vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // maxX
    status |= vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // maxY
    status |= vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // edge threshold
    status |= vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // FAST second threshold
    status |= vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // window size
    status |= vxAddParameterToKernel(kernel, 8, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // dst
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize makeGrid Kernel parameters");
        return VX_FAILURE;
    }
    status = vxFinalizeKernel(kernel);
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize makeGrid Kernel");
        return VX_FAILURE;
    }
    return status;
}

vx_status registerMakeQuadtree(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_uint32 num_params = 7;
    vx_kernel kernel = vxAddUserKernel(context, makeQuadtree_name, USER_KERNEL_MAKE_QUADTREE_NODE,
                                       makeQuadtree_kernel,
                                       num_params,
                                       makeQuadtree_validate,
                                       NULL, // init
                                       NULL  // deinit
    );
    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create makeQuadtree Kernel");
        return status;
    }
    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // src
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // minX
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // minY
    status |= vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // maxX
    status |= vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // maxY
    status |= vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // num features
    status |= vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // dst
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize makeQuadtree Kernel parameters");
        return VX_FAILURE;
    }
    status = vxFinalizeKernel(kernel);
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize makeQuadtree Kernel");
        return VX_FAILURE;
    }
    return status;
}

vx_status registerComputeAngle(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_uint32 num_params = 4;
    vx_kernel kernel = vxAddUserKernel(context, computeAngle_name, USER_KERNEL_COMPUTE_ANGLE_NODE,
                                       computeAngle_kernel,
                                       num_params,
                                       computeAngle_validate,
                                       NULL, // init
                                       NULL  // deinit
    );
    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create computeAngle Kernel");
        return status;
    }
    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);  // src
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // images
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // patterns
    status |= vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // keypoints rotated
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize computeAngle Kernel parameters");
        return VX_FAILURE;
    }
    status = vxFinalizeKernel(kernel);
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize computeAngle Kernel");
        return VX_FAILURE;
    }
    return status;
}

vx_status registerORB(vx_context context)
{
    vx_status status = VX_SUCCESS;

    zvx_descriptor = vxRegisterUserStruct(context, sizeof(descriptor_t));

    vx_uint32 num_params = 4;
    vx_kernel kernel = vxAddUserKernel(context, ORB_name, USER_KERNEL_ORB_NODE,
                                       ORB_kernel,
                                       num_params,
                                       ORB_validate,
                                       NULL, // init
                                       NULL  // deinit
    );
    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create ORB Kernel");
        return status;
    }
    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);  // source image
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // keypoints
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // patterns
    status |= vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // descriptor
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize ORB Kernel parameters");
        return VX_FAILURE;
    }
    status = vxFinalizeKernel(kernel);
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize ORB Kernel");
        return VX_FAILURE;
    }
    return status;
}

vx_status registerScaleArray(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_uint32 num_params = 3;
    vx_kernel kernel = vxAddUserKernel(context, scaleArray_name, USER_KERNEL_SCALE_ARRAY_NODE,
                                       scaleArray_kernel,
                                       num_params,
                                       scaleArray_validate,
                                       NULL, // init
                                       NULL  // deinit
    );
    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create scaleArray Kernel");
        return status;
    }
    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // keypoints
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // scale factor
    status |= vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // new keypoints
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize scaleArray Kernel parameters");
        return VX_FAILURE;
    }
    status = vxFinalizeKernel(kernel);
    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize scaleArray Kernel");
        return VX_FAILURE;
    }
    return status;
}

//-------------------------------------------------------------------------------------------------
//-------------------------------- Node creation functions ----------------------------------------
//-------------------------------------------------------------------------------------------------

vx_node zvxMakeGridNode(vx_graph graph, vx_array kp_in, vx_scalar min_x, vx_scalar min_y, vx_scalar max_x, vx_scalar max_y, vx_scalar edge_threshold, vx_scalar newFast_value, vx_scalar window, vx_array kp_out)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_MAKE_GRID_NODE);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);
        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)kp_in);
            vxSetParameterByIndex(node, 1, (vx_reference)min_x);
            vxSetParameterByIndex(node, 2, (vx_reference)min_y);
            vxSetParameterByIndex(node, 3, (vx_reference)max_x);
            vxSetParameterByIndex(node, 4, (vx_reference)max_y);
            vxSetParameterByIndex(node, 5, (vx_reference)edge_threshold);
            vxSetParameterByIndex(node, 6, (vx_reference)newFast_value);
            vxSetParameterByIndex(node, 7, (vx_reference)window);
            vxSetParameterByIndex(node, 8, (vx_reference)kp_out);
        }
    }
    return node;
}
vx_node zvxMakeQuadtreeNode(vx_graph graph, vx_array kp_in, vx_scalar min_x, vx_scalar min_y, vx_scalar max_x, vx_scalar max_y, vx_scalar num_features, vx_array kp_out)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_MAKE_QUADTREE_NODE);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);
        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)kp_in);
            vxSetParameterByIndex(node, 1, (vx_reference)min_x);
            vxSetParameterByIndex(node, 2, (vx_reference)min_y);
            vxSetParameterByIndex(node, 3, (vx_reference)max_x);
            vxSetParameterByIndex(node, 4, (vx_reference)max_y);
            vxSetParameterByIndex(node, 5, (vx_reference)num_features);
            vxSetParameterByIndex(node, 6, (vx_reference)kp_out);
        }
    }
    return node;
}

vx_node zvxComputeAngleNode(vx_graph graph, vx_image img, vx_array kp_in, vx_array patterns, vx_array kp_out)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_COMPUTE_ANGLE_NODE);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);
        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)img);
            vxSetParameterByIndex(node, 1, (vx_reference)kp_in);
            vxSetParameterByIndex(node, 2, (vx_reference)patterns);
            vxSetParameterByIndex(node, 3, (vx_reference)kp_out);
        }
    }
    return node;
}

vx_node zvxORBNode(vx_graph graph, vx_image img, vx_array kp_in, vx_array patterns, vx_array descr_out)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_ORB_NODE);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);
        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)img);
            vxSetParameterByIndex(node, 1, (vx_reference)kp_in);
            vxSetParameterByIndex(node, 2, (vx_reference)patterns);
            vxSetParameterByIndex(node, 3, (vx_reference)descr_out);
        }
    }
    return node;
}

vx_node zvxScaleArrayNode(vx_graph graph, vx_array kp_in, vx_scalar multiplier, vx_array kp_out)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_SCALE_ARRAY_NODE);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);
        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)kp_in);
            vxSetParameterByIndex(node, 1, (vx_reference)multiplier);
            vxSetParameterByIndex(node, 2, (vx_reference)kp_out);
        }
    }
    return node;
}

vx_node zvxComputeAngleNode(vx_graph graph, vx_image src, vx_image dst)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_COMPUTE_ANGLE_NODE);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);
        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)src);
            vxSetParameterByIndex(node, 1, (vx_reference)dst);
        }
    }
    return node;
}

//-------------------------------------------------------------------------------------------------
//-------------------------------- Custom functions -----------------------------------------------
//-------------------------------------------------------------------------------------------------

// BLUR TODO ?
vx_convolution createGaussianConvolution(vx_context c, vx_size size, vx_float32 sigma, vx_uint32 scale, bool horizontal)
{
    vx_convolution conv;
    if (horizontal)
        conv = vxCreateConvolution(c, size, 3);
    else
        conv = vxCreateConvolution(c, 3, size);
    NVXIO_CHECK_REFERENCE(conv);

    //vx_size sz;
    //vxQueryConvolution(conv, VX_CONVOLUTION_SIZE, &sz, sizeof(c));
    //std::cout << "Total size: " << sz << std::endl;

    vx_int16 *tmp = new vx_int16[size * 3];

    //std::cout << "Creating convolution! " << std::endl;

    vx_float32 sigma2 = 2 * sigma;
    int hsize = size / 2;
    // size must be ODD!
    int idx = 0;
    vx_uint32 totscale = 0;
    //for(int y = -hsize; y <= hsize; y++)
    int y = 0;
    {
        if (horizontal)
            for (; idx < size; ++idx)
                tmp[idx] = 0;
        for (int x = -hsize; x <= hsize; x++)
        {
            int16_t val = static_cast<int16_t>(scale * exp(-x * x / sigma2 - y * y / sigma2));
            //std::cout << val << "\t";// << " (" << static_cast<int>(scale*exp(-x*x/sigma2 - y*y/sigma2)) << ")\t";
            totscale += val;
            if (!horizontal)
                tmp[idx++] = 0;
            tmp[idx++] = val;
            if (!horizontal)
                tmp[idx++] = 0;
        }
        if (horizontal)
            for (; idx < 3 * size; ++idx)
                tmp[idx] = 0;
        //std::cout << std::endl;
    }

    NVXIO_SAFE_CALL(vxCopyConvolutionCoefficients(conv, (vx_int16 *)tmp, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    //std::cout << "Scale: " << totscale << std::endl;
    totscale = (vx_uint32)pow(2, ceil(log2(totscale)));

    //vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(vx_uint32));
    NVXIO_SAFE_CALL(vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, &totscale, sizeof(vx_uint32)));
    return conv;
}

vx_status zvxuMakeGrid(vx_context context, vx_array kp_in, vx_uint32 min_x, vx_uint32 min_y, vx_uint32 max_x, vx_uint32 max_y, vx_uint32 edge_threshold, vx_float32 newFast_value, vx_uint32 window, vx_array kp_out)
{
    vx_status s = VX_SUCCESS;
    vx_graph g = vxCreateGraph(context);
    s |= vxGetStatus((vx_reference)g);
    if (s != VX_SUCCESS)
        return s;

    vx_scalar
        s_min_x = vxCreateScalar(context, VX_TYPE_UINT32, &min_x),
        s_min_y = vxCreateScalar(context, VX_TYPE_UINT32, &min_y),
        s_max_x = vxCreateScalar(context, VX_TYPE_UINT32, &max_x),
        s_max_y = vxCreateScalar(context, VX_TYPE_UINT32, &max_y),
        s_edge = vxCreateScalar(context, VX_TYPE_UINT32, &edge_threshold),
        s_newF = vxCreateScalar(context, VX_TYPE_FLOAT32, &newFast_value),
        s_wind = vxCreateScalar(context, VX_TYPE_UINT32, &window);

    vx_node n = zvxMakeGridNode(g, kp_in, s_min_x, s_min_y, s_max_x, s_max_y, s_edge, s_newF, s_wind, kp_out);
    s |= vxGetStatus((vx_reference)n);
    s |= vxProcessGraph(g);

    vxReleaseScalar(&s_min_x);
    vxReleaseScalar(&s_min_y);
    vxReleaseScalar(&s_max_x);
    vxReleaseScalar(&s_max_y);
    vxReleaseScalar(&s_edge);
    vxReleaseScalar(&s_newF);
    vxReleaseScalar(&s_wind);

    vxReleaseNode(&n);

    vxReleaseGraph(&g);

    return s;
}

/**
 * @brief makeQuadtreeNode
 * @param graph
 * @param kp_in type VX_TYPE_KEYPOINT
 * @param min_x type VX_UINT32
 * @param min_y type VX_UINT32
 * @param max_x type VX_UINT32
 * @param max_y type VX_UINT32
 * @param num_features type VX_TYPE_SIZE
 * @param kp_out type VX_TYPE_KEYPOINT
 * @return
 */
vx_status zvxuMakeQuadtree(vx_context context, vx_array kp_in, vx_uint32 min_x, vx_uint32 min_y, vx_uint32 max_x, vx_uint32 max_y, vx_size num_features, vx_array kp_out)
{
    vx_status s = VX_SUCCESS;
    vx_graph g = vxCreateGraph(context);
    s |= vxGetStatus((vx_reference)g);
    if (s != VX_SUCCESS)
        return s;

    vx_scalar
        s_min_x = vxCreateScalar(context, VX_TYPE_UINT32, &min_x),
        s_min_y = vxCreateScalar(context, VX_TYPE_UINT32, &min_y),
        s_max_x = vxCreateScalar(context, VX_TYPE_UINT32, &max_x),
        s_max_y = vxCreateScalar(context, VX_TYPE_UINT32, &max_y),
        s_numF = vxCreateScalar(context, VX_TYPE_UINT32, &num_features);

    vx_node n = zvxMakeQuadtreeNode(g, kp_in, s_min_x, s_min_y, s_max_x, s_max_y, s_numF, kp_out);
    s |= vxGetStatus((vx_reference)n);
    s |= vxProcessGraph(g);

    vxReleaseScalar(&s_min_x);
    vxReleaseScalar(&s_min_y);
    vxReleaseScalar(&s_max_x);
    vxReleaseScalar(&s_max_y);
    vxReleaseScalar(&s_numF);

    vxReleaseNode(&n);

    vxReleaseGraph(&g);

    return s;
}

/**
 * @brief computeAngleNode
 * @param graph
 * @param img type VX_DF_IMAGE_U8
 * @param kp_in type VX_TYPE_KEYPOINT
 * @param patterns type VX_INT32
 * @param kp_out type VX_TYPE_KEYPOINT
 * @return
 */
vx_status zvxuComputeAngle(vx_context context, vx_image img, vx_array kp_in, vx_array patterns, vx_array kp_out)
{
    vx_status s = VX_SUCCESS;
    vx_graph g = vxCreateGraph(context);
    s |= vxGetStatus((vx_reference)g);
    if (s != VX_SUCCESS)
        return s;

    vx_node n = zvxComputeAngleNode(g, img, kp_in, patterns, kp_out);
    s |= vxGetStatus((vx_reference)n);
    s |= vxProcessGraph(g);

    vxReleaseNode(&n);

    vxReleaseGraph(&g);

    return s;
}

/**
 * @brief ORBNode
 * @param graph
 * @param img type VX_DF_IMAGE_U8
 * @param kp_in type VX_YTPE_KEYPOINT
 * @param patterns type VX_INT32
 * @param descr_out ZVX_DESCRIPTOR
 * @return
 */
vx_status zvxuORB(vx_context context, vx_image img, vx_array kp_in, vx_array patterns, vx_array descr_out)
{
    vx_status s = VX_SUCCESS;
    vx_graph g = vxCreateGraph(context);
    s |= vxGetStatus((vx_reference)g);
    if (s != VX_SUCCESS)
        return s;

    vx_node n = zvxORBNode(g, img, kp_in, patterns, descr_out);
    s |= vxGetStatus((vx_reference)n);
    s |= vxProcessGraph(g);

    vxReleaseNode(&n);

    vxReleaseGraph(&g);

    return s;
}

/**
 * @brief scaleArrayNode
 * @param graph
 * @param kp_in VX_TYPE_KEYPOINT
 * @param multiplier VX_TYPE_FLOAT32
 * @param kp_out VX_TYPE_KEYPOINT
 * @return
 */
vx_status zvxuScaleArray(vx_context context, vx_array kp_in, vx_float32 multiplier, vx_array kp_out)
{
    vx_status s = VX_SUCCESS;
    vx_graph g = vxCreateGraph(context);
    s |= vxGetStatus((vx_reference)g);
    if (s != VX_SUCCESS)
        return s;

    vx_scalar
        s_mult = vxCreateScalar(context, VX_TYPE_FLOAT32, &multiplier);

    vx_node n = zvxScaleArrayNode(g, kp_in, s_mult, kp_out);
    s |= vxGetStatus((vx_reference)n);
    s |= vxProcessGraph(g);

    vxReleaseNode(&n);

    vxReleaseGraph(&g);

    return s;
}

/*
// questa chiama il kernel
void computeDescriptorsCUDA(const Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors, const vector<Point>& pattern) {
    // initializzation of descriptor
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);
    uchar *h_desc = descriptors.ptr<uchar>();
    //d_Point patternPoints[PATTERN_SIZE];

    for(int i = 0; i < PATTERN_SIZE; i++){
        patternPoints[i].x = pattern[i].x;
        patternPoints[i].y = pattern[i].y;
    }

    uchar *d_N;
    KeyPoint *d_Kpt;
    uchar *d_Desc;

    //int imgH = img.size().height;
    //int imgW = img.size().width;

    //int size_img = imgW * imgH * sizeof(uchar);
    //int size_desc = descriptors.size().width * descriptors.size().height * sizeof(uchar);
    //int size_kpt = keypoints.size() * sizeof(cv::KeyPoint);

    // 1.Allocate device memory
    //CHECK_ERROR(cudaMalloc((void**)&d_N, size_img));
    //CHECK_ERROR(cudaMalloc((void**)&d_Desc, size_desc));
    //CHECK_ERROR(cudaMalloc((void**)&d_Kpt, size_kpt));

    // copy pattern and step to device constant memory
    //CHECK_ERROR(cudaMemcpyToSymbol(patt, patternPoints, sizeof(d_Point) * PATTERN_SIZE ));

    // copy N and descriptors and keypoints to device global memory
    //CHECK_ERROR(cudaMemcpy(d_N, img.ptr<uchar>(), size_img, cudaMemcpyHostToDevice));
    //CHECK_ERROR(cudaMemcpy(d_Kpt, &keypoints[0], size_kpt, cudaMemcpyHostToDevice));

    // 2.Kernel launch code
    //dim3 dimGrid(1, ceil((float)keypoints.size()/32.0), 1);
    //dim3 dimBlock(32, 32, 1);

    //computeOrbDescriptorCUDA_kernel <<<dimGrid, dimBlock>>> (d_Kpt, d_N, imgW, d_Desc, keypoints.size(), (int)img.step);
    //cudaDeviceSynchronize();

    // 3.Copy descriptors from the device memory
    //CHECK_ERROR(cudaMemcpy(h_desc, d_Desc, size_desc, cudaMemcpyDeviceToHost));

    // free device memory
    //CHECK_ERROR(cudaFree(d_N));
    //CHECK_ERROR(cudaFree(d_Desc));
    //CHECK_ERROR(cudaFree(d_Kpt));
}
*/