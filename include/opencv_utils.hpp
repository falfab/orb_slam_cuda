
#ifndef OPENCV_UTILS_H
#define OPENCV_UTILS_H

#include <opencv2/opencv.hpp>

namespace cv {

static inline void read(const FileNode& node, size_t& value, size_t default_value)
{
    int temp; 
    read(node, temp, (int)default_value);
    value = temp;
}

}

#endif
