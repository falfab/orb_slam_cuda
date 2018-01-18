#pragma once
#include <NVX/nvx.h>

#include "cuda_runtime.h"
#include <cuda.h>
#include <iostream>
#include <opencv2/opencv.hpp>


//enum and node name
enum {
    // Library ID
    USER_LIBRARY = 0x1,
    // Kernel ID
    USER_KERNEL_MAKE_GRID_NODE     = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x0,
    USER_KERNEL_MAKE_QUADTREE_NODE = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x1,
    USER_KERNEL_COMPUTE_ANGLE_NODE = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x2,
    USER_KERNEL_ORB_NODE           = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x3,
    USER_KERNEL_SCALE_ARRAY_NODE   = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x4,
};

/*vx_char distributeGrid_name[] = "user.kernel.distributeGrid";
vx_char makeQuadtree_name[] = "user.kernel.makeQuadtree";
vx_char computeAngle_name[] = "user.kernel.computeAngle";
vx_char ORB_name[] = "user.kernel.ORB";
vx_char scaleArray_name[] = "user.kernel.scaleArray";*/

typedef struct _descriptor_t {
    vx_uint8 descr[32];
} descriptor_t;
extern vx_enum zvx_descriptor;



//function kernel and validation
vx_status makeGrid_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status makeGrid_validate(vx_node node, const vx_reference parameters[],
                            vx_uint32 num, vx_meta_format metas[]);

vx_status makeQuadtree_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status makeQuadtree_validate(vx_node node, const vx_reference parameters[],
                                vx_uint32 num, vx_meta_format metas[]);

vx_status computeAngle_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status computeAngle_validate(vx_node node, const vx_reference parameters[],
                                vx_uint32 num, vx_meta_format metas[]);

vx_status ORB_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status ORB_validate(vx_node node, const vx_reference parameters[],
                       vx_uint32 num, vx_meta_format metas[]);

vx_status scaleArray_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status scaleArray_validate(vx_node node, const vx_reference parameters[],
                              vx_uint32 num, vx_meta_format metas[]);


//registering function
vx_status registerMakeGrid(vx_context context);
vx_status registerMakeQuadtree(vx_context context);
vx_status registerComputeAngle(vx_context context);
vx_status registerORB(vx_context context);
vx_status registerScaleArray(vx_context context);

vx_status registerCustomKernels(vx_context context);

//node creation function

/**
 * @brief makeGridNode
 * @param graph
 * @param kp_in type VX_TYPE_KEYPOINT
 * @param min_x type VX_UINT32
 * @param min_y type VX_UINT32
 * @param max_x type VX_UINT32
 * @param max_y type VX_UINT32
 * @param edge_threshold type VX_UINT32
 * @param newFast_value type VX_FLOAT32
 * @param window type VX_UINT32
 * @param kp_out type VX_TYPE_KEYPOINT
 */
vx_node zvxMakeGridNode(vx_graph graph, vx_array kp_in, vx_scalar min_x, vx_scalar min_y, vx_scalar max_x, vx_scalar max_y, vx_scalar edge_threshold, vx_scalar newFast_value, vx_scalar window, vx_array kp_out);
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
vx_node zvxMakeQuadtreeNode(vx_graph graph, vx_array kp_in, vx_scalar min_x, vx_scalar min_y, vx_scalar max_x, vx_scalar max_y, vx_scalar num_features, vx_array kp_out);
/**
 * @brief computeAngleNode
 * @param graph
 * @param img type VX_DF_IMAGE_U8
 * @param kp_in type VX_TYPE_KEYPOINT
 * @param patterns type VX_INT32
 * @param kp_out type VX_TYPE_KEYPOINT
 * @return
 */
vx_node zvxComputeAngleNode(vx_graph graph, vx_image img, vx_array kp_in, vx_array patterns, vx_array kp_out);
/**
 * @brief ORBNode
 * @param graph
 * @param img type VX_DF_IMAGE_U8
 * @param kp_in type VX_YTPE_KEYPOINT
 * @param patterns type VX_INT32
 * @param descr_out ZVX_DESCRIPTOR
 * @return
 */
vx_node zvxORBNode(vx_graph graph, vx_image img, vx_array kp_in, vx_array patterns, vx_array descr_out);
/**
 * @brief scaleArrayNode
 * @param graph
 * @param kp_in VX_TYPE_KEYPOINT
 * @param multiplier VX_TYPE_FLOAT32
 * @param kp_out VX_TYPE_KEYPOINT
 * @return
 */
vx_node zvxScaleArrayNode(vx_graph graph, vx_array kp_in, vx_scalar multiplier, vx_array kp_out);



/**
 * @brief makeGridNode
 * @param graph
 * @param kp_in type VX_TYPE_KEYPOINT
 * @param min_x type VX_UINT32
 * @param min_y type VX_UINT32
 * @param max_x type VX_UINT32
 * @param max_y type VX_UINT32
 * @param edge_threshold type VX_UINT32
 * @param newFast_value type VX_FLOAT32
 * @param window type VX_UINT32
 * @param kp_out type VX_TYPE_KEYPOINT
 */
vx_status zvxuMakeGrid(vx_context context, vx_array kp_in, vx_uint32 min_x, vx_uint32 min_y, vx_uint32 max_x, vx_uint32 max_y, vx_uint32 edge_threshold, vx_float32 newFast_value, vx_uint32 window, vx_array kp_out);
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
vx_status zvxuMakeQuadtree(vx_context context, vx_array kp_in, vx_uint32 min_x, vx_uint32 min_y, vx_uint32 max_x, vx_uint32 max_y, vx_size num_features, vx_array kp_out);
/**
 * @brief computeAngleNode
 * @param graph
 * @param img type VX_DF_IMAGE_U8
 * @param kp_in type VX_TYPE_KEYPOINT
 * @param patterns type VX_INT32
 * @param kp_out type VX_TYPE_KEYPOINT
 * @return
 */
vx_status zvxuComputeAngle(vx_context context, vx_image img, vx_array kp_in, vx_array patterns, vx_array kp_out);
/**
 * @brief ORBNode
 * @param graph
 * @param img type VX_DF_IMAGE_U8
 * @param kp_in type VX_YTPE_KEYPOINT
 * @param patterns type VX_INT32
 * @param descr_out ZVX_DESCRIPTOR
 * @return
 */
vx_status zvxuORB(vx_context context, vx_image img, vx_array kp_in, vx_array patterns, vx_array descr_out);
/**
 * @brief scaleArrayNode
 * @param graph
 * @param kp_in VX_TYPE_KEYPOINT
 * @param multiplier VX_TYPE_FLOAT32
 * @param kp_out VX_TYPE_KEYPOINT
 * @return
 */
vx_status zvxuScaleArray(vx_context context, vx_array kp_in, vx_float32 multiplier, vx_array kp_out);

vx_convolution createGaussianConvolution(vx_context c, vx_size size, vx_float32 sigma, vx_uint32 scale, bool horizontal);

