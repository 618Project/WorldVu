#include <gflags/gflags.h>
#include <glog/logging.h>
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/cudaarithm.hpp"
cv::Mat sharpenGPU (cv::Mat);
//cv::Mat IIRFilter (cv::Mat Input,
//                   float amount,
//                   cv::Mat& lpImage,
//                   const H& hBoundary,
//                   const V& vBoundary,
//                   const float maxVal = 255.0f
//                  );
