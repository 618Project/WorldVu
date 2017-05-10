
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"

using namespace cv;
using namespace cv::cuda;

void temp_fn();

void gpu_patch(
  const Mat& I0,
  const Mat& I1,
  const Mat& alpha0,
  const Mat& alpha1,
  const int kKernelSize,
  const Size kGradientBlurSize,
  const float kGradientBlurSigma,
  const float kUpdateAlphaThreshold,
  const Mat& flow,
  const Mat& I0x,
  const Mat& I0y,
  const Mat& I1x,
  const Mat& I1y,
  const Mat& blurredFlow,
  bool correctness_mode);

void gpu_denseflow (
  const Mat& I0,
  const Mat& I1,
  Mat& gpuFlow,
  Mat& flow);
