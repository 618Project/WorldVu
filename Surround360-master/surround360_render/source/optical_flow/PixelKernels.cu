
#include "PixelKernels.h"
#include "CvUtil.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

__global__ void temporary () {
  int a;
  a = 3*4;
  printf("Inside temporary\n");
//  cv::cuda::GpuMat b;
}

void temp_fn() {
  printf ("Inside temp_fn()\n");
  temporary<<<10,1>>>();
  printf ("Inside temp_fn()\n");
}

__global__ void lucasKanade () {

}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void gpu_denseflow (
  const Mat& I0,
  const Mat& I1,
  Mat& gpuFlow,
  Mat& flow) {

  cv::cuda::GpuMat gpuI0, gpuI1, gpugpuFlow;
  cv::cuda::GpuMat gpuI0c, gpuI1c, gpuResFlow;
  cv::Mat resFlow;
  gpuI0.upload(I0);
  gpuI1.upload(I1);
  gpugpuFlow.upload(gpuFlow);

  Mat2f diff;
  Ptr<cuda::DensePyrLKOpticalFlow> lk = cv::cuda::DensePyrLKOpticalFlow::create(Size(7,7));
  gpuI0.convertTo(gpuI0c, CV_8UC1);
  gpuI1.convertTo(gpuI1c, CV_8UC1);
  gpugpuFlow.convertTo(gpuResFlow, CV_32FC2);
  lk->calc (gpuI0c, gpuI1c, gpuResFlow);

  gpuResFlow.download(resFlow);
  //resFlow.convertTo(resFlow0, CV_32FC1);

//  cvtColor(resFlow, resFlow, CV_BGR2GRAY);
//  cvtColor(resFlow, resFlow, CV_BGR2GRAY);
  // cv::bitwise_xor(resFlow, flow, diff);
  // Mat1f diff1 = diff.reshape(1);
  // int type = diff1.type();
  // LOG(INFO) << "diff type: " << type2str(type);

  // bool eq1 = (cv::countNonZero(diff1) == 0);

  // if (eq1)
  //   LOG(INFO) << "DF correctness passed" ;
  // else
  //   LOG(INFO) << "DF diff: " << cv::countNonZero(diff);
}

/*
// __device__ inline float gpu_errorFunction(
float gpu_errorFunction(
    const Mat& I0,
    const Mat& I1,
    const Mat& alpha0,
    const Mat& alpha1,
    const Mat& I0x,
    const Mat& I0y,
    const Mat& I1x,
    const Mat& I1y,
    const int x,
    const int y,
    const Mat& flow,
    const Mat& blurredFlow,
    const Point2f& flowDir) {

  const float matchX      = x + flowDir.x;
  const float matchY      = y + flowDir.y;
  const float i0x         = I0x.at<float>(y, x);
  const float i0y         = I0y.at<float>(y, x);
  const float i1x         = getPixBilinear32FExtend(I1x, matchX, matchY);
  const float i1y         = getPixBilinear32FExtend(I1y, matchX, matchY);
  const Point2f flowDiff  = blurredFlow.at<Point2f>(y, x) - flowDir;
  const float smoothness  = sqrtf(flowDiff.dot(flowDiff));

  float err = sqrtf((i0x - i1x) * (i0x - i1x) + (i0y - i1y) * (i0y - i1y))
    + smoothness * smoothnessCoef
    + verticalRegularizationCoef * fabsf(flowDir.y) / float(I0.cols)
    + horizontalRegularizationCoef * fabsf(flowDir.x) / float(I0.cols);

  if (UseDirectionalRegularization) {
    Point2f bf = blurredFlow.at<Point2f>(y, x);
    const float blurredFlowMag = sqrtf(bf.dot(bf));
    const static float kEpsilon = 0.001f;
    bf /= blurredFlowMag + kEpsilon;
    const float flowMag = sqrtf(flowDir.dot(flowDir));
    Point2f normalizedFlowDir = flowDir / (flowMag + kEpsilon);
    const float dot = bf.dot(normalizedFlowDir);
    err -= directionalRegularizationCoef * dot;
  }

  return err;
}
*/

void gpu_patch(
  const cv::Mat& I0,
  const cv::Mat& I1,
  const cv::Mat& alpha0,
  const cv::Mat& alpha1,
  const int kKernelSize,
  const cv::Size kGradientBlurSize,
  const float kGradientBlurSigma,
  const float kUpdateAlphaThreshold,
  const Mat& flow,
  const cv::Mat& I0x,
  const cv::Mat& I0y,
  const cv::Mat& I1x,
  const cv::Mat& I1y,
  const Mat& blurredFlow,
  bool correctness_mode
  ) {


  cv::cuda::GpuMat gpuI0, gpuI1, gpuAlpha0, gpuAlpha1;
  cv::cuda::GpuMat gpuI0x_s, gpuI0y_s, gpuI1x_s, gpuI1y_s;
  cv::cuda::GpuMat gpuI0x_g, gpuI0y_g, gpuI1x_g, gpuI1y_g;
  cv::cuda::GpuMat gpuI0x_t, gpuI0y_t, gpuI1x_t, gpuI1y_t;
  cv::Mat gpuI0x, gpuI0y, gpuI1x, gpuI1y;

  // Uploading data to GPU
  gpuI0.upload(I0);
  gpuI1.upload(I1);
  gpuAlpha0.upload(alpha0);
  gpuAlpha1.upload(alpha1);

  // Sobel phase
  cv::Ptr<cv::cuda::Filter> sobelXFilter = cv::cuda::createSobelFilter(I0.type(), I0.type(), 1, 0, kKernelSize, 1, cv::BORDER_REPLICATE, cv::BORDER_REPLICATE);
  cv::Ptr<cv::cuda::Filter> sobelYFilter = cv::cuda::createSobelFilter(I0.type(), I0.type(), 0, 1, kKernelSize, 1, cv::BORDER_REPLICATE, cv::BORDER_REPLICATE);

  // TODO: Consider replacing with non-blocking calls
  sobelXFilter->apply(gpuI0, gpuI0x_s);
  sobelXFilter->apply(gpuI1, gpuI1x_s);
  sobelYFilter->apply(gpuI0, gpuI0y_s);
  sobelYFilter->apply(gpuI1, gpuI1y_s);

  // Gaussian - Not passing correctness
  cv::Ptr<cv::cuda::Filter> gaussianXFilter = cv::cuda::createGaussianFilter(
    gpuI0x_s.type(), gpuI0x_s.type(), kGradientBlurSize, kGradientBlurSigma,
    kGradientBlurSigma);
  cv::Ptr<cv::cuda::Filter> gaussianYFilter = cv::cuda::createGaussianFilter(
    gpuI0y_s.type(), gpuI0y_s.type(), kGradientBlurSize, kGradientBlurSigma,
    kGradientBlurSigma);

  gaussianXFilter->apply(gpuI0x_s, gpuI0x_g);
  gaussianXFilter->apply(gpuI1x_s, gpuI1x_g);
  gaussianYFilter->apply(gpuI0y_s, gpuI0y_g);
  gaussianYFilter->apply(gpuI1y_s, gpuI1y_g);

  gpuI0x_g.download(gpuI0x);
  gpuI1x_g.download(gpuI1x);
  gpuI0y_g.download(gpuI0y);
  gpuI1y_g.download(gpuI1y);


  // // Uncomment for checking sobel phase
  // gpuI0x_s.download(gpuI0x);
  // gpuI1x_s.download(gpuI1x);
  // gpuI0y_s.download(gpuI0y);
  // gpuI1y_s.download(gpuI1y);

  // if (correctness_mode) {
  //   // Compare correctness
  //   cv::Mat diff1, diff2, diff3, diff4;
  //   cv::bitwise_xor(I0x, gpuI0x, diff1);
  //   cv::bitwise_xor(I0y, gpuI0y, diff2);
  //   cv::bitwise_xor(I1x, gpuI1x, diff3);
  //   cv::bitwise_xor(I1y, gpuI1y, diff4);
  //   bool eq1 = (cv::countNonZero(diff1) == 0);
  //   bool eq2 = (cv::countNonZero(diff2) == 0);
  //   bool eq3 = (cv::countNonZero(diff3) == 0);
  //   bool eq4 = (cv::countNonZero(diff4) == 0);
  //   if (eq1 && eq2 && eq3 && eq4) {
  //     printf ("Sobel correctness passed\n");
  //   } else {
  //     printf ("Sobel correctness failed: %d %d %d %d\n", eq1,eq2,eq3,eq4);
  //   }
  // }

  // Till this point, code is for Sobel & Gaussian. But Gaussian not passing correctness

  // const cv::Size imgSize = I0.size();  

  //   for (int y = 0; y < imgSize.height; ++y) {
  //     for (int x = 0; x < imgSize.width; ++x) {
  //       if (alpha0.at<float>(y, x) > kUpdateAlphaThreshold && alpha1.at<float>(y, x) > kUpdateAlphaThreshold) {
  //         
  //       }
  //     }
  //   }

}
