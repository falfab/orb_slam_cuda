#include <NVX/nvx.h>
//#include <NVX/nvx_opencv_interop.hpp>
#include <opencv2/opencv.hpp>

void printReferenceInfo(vx_reference r, std::string prefix = "")
{
    vx_uint32 count;
    vx_char name[VX_MAX_REFERENCE_NAME];
    vxQueryReference(r, VX_REF_ATTRIBUTE_COUNT, &count, sizeof(vx_uint32));
    vxQueryReference(r, VX_REF_ATTRIBUTE_NAME, &name, VX_MAX_REFERENCE_NAME*sizeof(vx_char));

    std::cout << prefix << "Ref count: " << count << std::endl;
    //std::cout << prefix << "Name: " << name  << std::endl;
}

void printParameterInfo(vx_parameter p, std::string prefix = "")
{
    vx_direction_e d;
    vx_type_e t;
    vx_parameter_state_e s;


    vxQueryParameter(p, VX_PARAMETER_DIRECTION, &d, sizeof(vx_direction_e));
    std::cout << prefix << "Direction: ";
    switch(d)
    {
        case VX_INPUT: std::cout << "Input"; break;
        case VX_OUTPUT: std::cout << "Output"; break;
        case VX_BIDIRECTIONAL: std::cout << "Bidirectional"; break;
    default: std::cout << d;
    }
    std::cout << ", ";


    vxQueryParameter(p, VX_PARAMETER_STATE, &s, sizeof(vx_parameter_state_e));
    switch(s)
    {
        case VX_PARAMETER_STATE_REQUIRED: std::cout << "required"; break;
        case VX_PARAMETER_STATE_OPTIONAL: std::cout << "optional"; break;
    default: std::cout << s;
    }
    std::cout << std::endl;

    vxQueryParameter(p, VX_PARAMETER_TYPE, &t, sizeof(vx_type_e));
    std::cout << prefix << "Type: ";
    switch(t)
    {
        case VX_TYPE_IMAGE: std::cout << "VX_TYPE_IMAGE"; break;
        case VX_TYPE_SCALAR: std::cout << "VX_TYPE_SCALAR"; break;
    default: std::cout << t;
    }
    std::cout << std::endl;

    vx_reference r;
    vxQueryParameter(p, VX_PARAMETER_REF, &r, sizeof(vx_reference));
    printReferenceInfo(r, prefix);
    vxReleaseReference(&r);
}

void printNodeInfo(vx_node n, std::string prefix = "")
{
    vx_uint32 num_param = 0;
    vxQueryNode(n, VX_NODE_PARAMETERS, &num_param, sizeof(vx_uint32));
    std::cout << prefix;
    if(!prefix.empty()) std::cout << std::endl;
    for(vx_uint32 i = 0; i < num_param; i++)
    {
        std::cout << ((prefix.empty()) ? "" : "\t") << "Parameter #" << i << ":" << std::endl;
        vx_parameter p = vxGetParameterByIndex(n, i);

        printParameterInfo(p, (prefix.empty()) ? "\t" : "\t\t");

        vxReleaseParameter(&p);
    }
}

#ifdef TEGRACV

#include <NVX/nvx_opencv_interop.hpp> // using GpuMat

#else // discarding GpuMat (opencv 3)


/*
 * Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief OpenCV Interoperability APIs.
 */

#ifndef NVX_OPENCV_INTEROP_HPP
#define NVX_OPENCV_INTEROP_HPP

#ifndef __cplusplus
#   error nvx_opencv_interop.hpp header must be compiled as C++
#endif

#include <limits>

#include <NVX/nvx.h>
#include <opencv2/core/core.hpp>

/**
 * \brief NVIDIA VisionWorks OpenCV Interoperability APIs.
 */
namespace nvx_cv {

/**
 * \ingroup nvx_opencv_interop
 * \brief Converts from OpenVX Matrix' type to OpenCV Mat's type.
 *
 * \param [in] matrix_type  OpenVX Matrix' type.
 *
 * \return OpenCV Mat's type.
 */
inline int convertVXMatrixTypeToCVMatType(vx_enum matrix_type)
{
    switch(matrix_type)
    {
    case VX_TYPE_UINT8:
        return CV_8UC1;
    case VX_TYPE_INT8:
        return CV_8SC1;
    case VX_TYPE_UINT16:
        return CV_16UC1;
    case VX_TYPE_INT16:
        return CV_16SC1;
    case VX_TYPE_UINT32:
    case VX_TYPE_INT32:
        return CV_32SC1;
    case VX_TYPE_FLOAT32:
        return CV_32FC1;
    case VX_TYPE_FLOAT64:
        return CV_64FC1;
    case NVX_TYPE_POINT2F:
        return CV_32FC2;
    case NVX_TYPE_POINT3F:
        return CV_32FC3;
    }
    CV_Error(CV_StsUnsupportedFormat, "Unsupported format");
    return 0;
}

/**
 * \ingroup nvx_opencv_interop
 * \brief Converts from OpenCV Mat's type to OpenVX Matrix' type.
 *
 * \param [in] mat_type     OpenCV Mat's type.
 *
 * \return OpenVX Matrix' type.
 */
inline vx_enum convertCVMatTypeToVXMatrixType(int mat_type)
{
    switch(mat_type)
    {
    case CV_8UC1:
        return VX_TYPE_UINT8;
    case CV_8SC1:
        return VX_TYPE_INT8;
    case CV_16UC1:
        return VX_TYPE_UINT16;
    case CV_16SC1:
        return VX_TYPE_INT16;
    case CV_32SC1:
        return VX_TYPE_INT32;
    case CV_32FC1:
        return VX_TYPE_FLOAT32;
    case CV_64FC1:
        return VX_TYPE_FLOAT64;
    case CV_32FC2:
        return NVX_TYPE_POINT2F;
    case CV_32FC3:
        return NVX_TYPE_POINT3F;
    }
    CV_Error(CV_StsUnsupportedFormat, "Unsupported format");
    return 0;
}

/**
 * \ingroup nvx_opencv_interop
 * \brief Converts from OpenVX Image's format to OpenCV Mat's type.
 *
 * \param [in] format       OpenVX Image's format.
 * \param [in] plane_index  Optional OpenVX Image's plane index.
 *
 * \return OpenCV Mat's type.
 */
inline int convertVXImageFormatToCVMatType(vx_df_image format, vx_uint32 plane_index = 0)
{
    switch (format)
    {
    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_YUV4:
    case VX_DF_IMAGE_IYUV:
        return CV_8UC1;
    case VX_DF_IMAGE_U16:
        return CV_16UC1;
    case VX_DF_IMAGE_S16:
        return CV_16SC1;
    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
        return CV_32SC1;
    case NVX_DF_IMAGE_F32:
        return CV_32FC1;
    case NVX_DF_IMAGE_2F32:
        return CV_32FC2;
    case NVX_DF_IMAGE_2S16:
        return CV_16SC2;
    case VX_DF_IMAGE_UYVY:
    case VX_DF_IMAGE_YUYV:
        return CV_8UC2;
    case VX_DF_IMAGE_RGB:
        return CV_8UC3;
    case NVX_DF_IMAGE_RGB16:
        return CV_16SC3;
    case VX_DF_IMAGE_RGBX:
        return CV_8UC4;
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
        return plane_index == 0 ? CV_8UC1 : CV_8UC2;
    }
    CV_Error(CV_StsUnsupportedFormat, "Unsupported format");
    return 0;
}

/**
 * \ingroup nvx_opencv_interop
 * \brief Converts from OpenCV Mat's type to OpenVX Image's format.
 *
 * \param [in] mat_type     OpenCV Mat's type.
 *
 * \return OpenVX Image's format.
 */
inline vx_df_image convertCVMatTypeToVXImageFormat(int mat_type)
{
    switch (mat_type)
    {
    case CV_8UC1:
        return VX_DF_IMAGE_U8;
    case CV_16UC1:
        return VX_DF_IMAGE_U16;
    case CV_16SC1:
        return VX_DF_IMAGE_S16;
    case CV_32SC1:
        return VX_DF_IMAGE_S32;
    case CV_32FC1:
        return NVX_DF_IMAGE_F32;
    case CV_32FC2:
        return NVX_DF_IMAGE_2F32;
    case CV_16SC2:
        return NVX_DF_IMAGE_2S16;
    case CV_8UC3:
        return VX_DF_IMAGE_RGB;
    case CV_8UC4:
        return VX_DF_IMAGE_RGBX;
    case CV_16SC3:
        return NVX_DF_IMAGE_RGB16;
    }
    CV_Error(CV_StsUnsupportedFormat, "Unsupported format");
    return 0;
}

/**
 * \ingroup nvx_opencv_interop
 * \brief Copies data from OpenVX Matrix to OpenCV Mat.
 *
 * \param [in]  src_mat     Input OpenVX Matrix.
 * \param [out] dst_mat     Output OpenCV Mat.
 */
inline void copyVXMatrixToCVMat(vx_matrix src_mat, cv::Mat& dst_mat)
{
    vx_status status = VX_SUCCESS;
    vx_size rows_num = 0, cols_num = 0;
    vx_enum elem_type = 0;

    status = vxQueryMatrix(src_mat, VX_MATRIX_ATTRIBUTE_TYPE, &elem_type, sizeof(elem_type) );
    CV_Assert(status == VX_SUCCESS);

    status = vxQueryMatrix(src_mat, VX_MATRIX_ATTRIBUTE_ROWS, &rows_num, sizeof(rows_num) );
    CV_Assert(status == VX_SUCCESS);
    CV_Assert(rows_num <= static_cast<vx_size>(std::numeric_limits<int>::max()));

    status = vxQueryMatrix(src_mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols_num, sizeof(cols_num) );
    CV_Assert(status == VX_SUCCESS);
    CV_Assert(cols_num <= static_cast<vx_size>(std::numeric_limits<int>::max()));

    dst_mat.create(static_cast<int>(rows_num), static_cast<int>(cols_num), convertVXMatrixTypeToCVMatType(elem_type));
    CV_Assert(dst_mat.isContinuous());

    status = vxReadMatrix(src_mat, dst_mat.data);
    CV_Assert(status == VX_SUCCESS);
}

/**
 * \ingroup nvx_opencv_interop
 * \brief Copies data from OpenCV Mat to OpenVX Matrix.
 *
 * \param [in]  src_mat     Input OpenCV Mat. It must be a continuous matrix.
 * \param [out] dst_mat     Output OpenVX Matrix.
 */
inline void copyCVMatToVXMatrix(const cv::Mat& src_mat, vx_matrix dst_mat)
{
    vx_status status = VX_SUCCESS;
    vx_size dst_rows_num = 0, dst_cols_num = 0;
    vx_enum dst_elem_type = 0;

    status = vxQueryMatrix(dst_mat, VX_MATRIX_ATTRIBUTE_TYPE, &dst_elem_type, sizeof(dst_elem_type) );
    CV_Assert(status == VX_SUCCESS);

    status = vxQueryMatrix(dst_mat, VX_MATRIX_ATTRIBUTE_ROWS, &dst_rows_num, sizeof(dst_rows_num) );
    CV_Assert(status == VX_SUCCESS);

    status = vxQueryMatrix(dst_mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &dst_cols_num, sizeof(dst_cols_num) );
    CV_Assert(status == VX_SUCCESS);

    CV_Assert(src_mat.isContinuous());
    CV_Assert(static_cast<vx_size>(src_mat.cols) == dst_cols_num && static_cast<vx_size>(src_mat.rows) == dst_rows_num);
    CV_Assert(src_mat.type() == convertVXMatrixTypeToCVMatType(dst_elem_type));

    status = vxWriteMatrix(dst_mat, src_mat.data);
    CV_Assert(status == VX_SUCCESS);
}

/**
 * \ingroup nvx_opencv_interop
 * \brief Creates OpenVX Matrix object and copies data from OpenCV Mat.
 *
 * \param [in] context      OpenVX context.
 * \param [in] src_mat      Input OpenCV Mat.
 *
 * \return OpenVX Matrix.
 */
inline vx_matrix cloneCVMatToVXMatrix(vx_context context, const cv::Mat& src_mat)
{
    vx_matrix res_matrix = vxCreateMatrix(context, convertCVMatTypeToVXMatrixType(src_mat.type()),
                                          src_mat.cols, src_mat.rows);
    CV_Assert(vxGetStatus((vx_reference)res_matrix) == VX_SUCCESS);

    copyCVMatToVXMatrix(src_mat, res_matrix);

    return res_matrix;
}

/**
 * \ingroup nvx_opencv_interop
 * \brief Creates OpenVX Image from OpenCV Mat without data copy.
 *
 * The OpenVX Image points to the same pixels as the OpenCV Mat.
 *
 * \note The OpenCV Mat object must not be released until the OpenVX Image object is destroyed.
 *
 * \param [in] context      OpenVX context.
 * \param [in] mat          Input OpenCV Mat to import.
 *
 * \return OpenVX Image object imported from input OpenCV Mat.
 */
inline vx_image createVXImageFromCVMat(vx_context context, const cv::Mat& mat)
{
    vx_df_image format = convertCVMatTypeToVXImageFormat(mat.type());

    vx_imagepatch_addressing_t addrs[1];
    addrs[0].dim_x = mat.cols;
    addrs[0].dim_y = mat.rows;
    addrs[0].stride_x = static_cast<vx_int32>(mat.elemSize());
    addrs[0].stride_y = static_cast<vx_int32>(mat.step);

    void *ptrs[1] = { const_cast<uchar *>(mat.ptr()) };

    vx_image img = vxCreateImageFromHandle(context, format, addrs, ptrs, VX_MEMORY_TYPE_HOST);
    CV_Assert(vxGetStatus((vx_reference)img) == VX_SUCCESS);

    return img;
}



/**
 * \ingroup nvx_opencv_interop
 * \brief Utility class to map OpenVX Image into OpenCV Mat or GpuMat.
 *
 * The class maps OpenVX Image into OpenCV Mat or GpuMat in constructor and
 * unmaps OpenVX Image in destructor.
 *
 * \note The input OpenVX Image must not be destroyed until the class' destructor is called.
 */
class VXImageToCVMatMapper
{
public:
    /**
     * \brief Maps input OpenVX Image into OpenCV Mat or GpuMat without memory copy.
     *
     * \param [in] image            Input OpenVX Image.
     * \param [in] plane_index      Plane index to map.
     * \param [in] rect             Region to map. Use `NULL` to map the whole Image region.
     * \param [in] usage            Intended usage of mapped memory
     *                                  (`VX_READ_ONLY`, `VX_WRITE_ONLY` or `VX_READ_AND_WRITE`).
     * \param [in] memory_type      Intended memory type of the mapped memory
     *                                  (`VX_MEMORY_TYPE_HOST` for Mat or `NVX_MEMORY_TYPE_CUDA` for GpuMat).
     */
    explicit VXImageToCVMatMapper(vx_image image,
                                  vx_uint32 plane_index = 0,
                                  const vx_rectangle_t *rect = NULL,
                                  vx_enum usage = VX_READ_AND_WRITE,
                                  vx_enum memory_type = VX_MEMORY_TYPE_HOST) : image_(image)
    {
        vx_status status = VX_SUCCESS;

        vx_rectangle_t rect_;
        if (rect)
        {
            rect_.start_x = rect->start_x;
            rect_.start_y = rect->start_y;
            rect_.end_x = rect->end_x;
            rect_.end_y = rect->end_y;
        }
        else
        {
            rect_.start_x = rect_.start_y = 0;

            status = vxQueryImage(image_, VX_IMAGE_ATTRIBUTE_WIDTH, &rect_.end_x, sizeof(rect_.end_x));
            CV_Assert(status == VX_SUCCESS);

            status = vxQueryImage(image_, VX_IMAGE_ATTRIBUTE_HEIGHT, &rect_.end_y, sizeof(rect_.end_y));
            CV_Assert(status == VX_SUCCESS);
        }

        vx_df_image format = VX_DF_IMAGE_VIRT;
        status = vxQueryImage(image_, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format));
        CV_Assert(status == VX_SUCCESS);

        int type = convertVXImageFormatToCVMatType(format, plane_index);

        vx_imagepatch_addressing_t addr;
        void* ptr = NULL;
        status = vxMapImagePatch(image_, &rect_, plane_index, &map_id_, &addr, &ptr, usage, memory_type, VX_NOGAP_X);
        CV_Assert(status == VX_SUCCESS);

        int cols = addr.dim_x * addr.scale_x / VX_SCALE_UNITY;
        int rows = addr.dim_y * addr.scale_y / VX_SCALE_UNITY;
        int step = addr.stride_y;

        if (memory_type == VX_MEMORY_TYPE_HOST)
        {
            cv_mat_ = cv::Mat(rows, cols, type, ptr, step);
        }
    }

    /**
     * \brief Unmaps a mapped region of an OpenVX Image.
     */
    ~VXImageToCVMatMapper()
    {
        vxUnmapImagePatch(image_, map_id_);
    }

    /**
     * \brief Gets mapped OpenCV Mat.
     *
     * \return Mapped OpenCV Mat.
     */
    cv::Mat getMat()
    {
        CV_Assert(!cv_mat_.empty());
        return cv_mat_;
    }


private:
    vx_image image_;
    vx_map_id map_id_;

    cv::Mat cv_mat_;
};

}


#endif // TEGRACV

#endif
