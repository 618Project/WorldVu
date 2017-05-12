#include <gflags/gflags.h>
#include <glog/logging.h>
#include "opencv2/cudaoptflow.hpp"
#include "MathUtil.h"
#include "opencv2/highgui/highgui.hpp"
#include "SystemUtil.h"
//#include "opencv2/cudaarithm.hpp"
using namespace cv;
using namespace cv::cuda;



cv::Mat sharpenGPU (cv::Mat);
//template <typename H, typename V, typename P>
void IIRFilter(cv::Mat Input,
           float amount,
           cv::Mat& lpImage,
           const float maxVal = 255.0f
          );
