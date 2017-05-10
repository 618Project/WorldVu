#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaarithm.hpp"
#include "MathUtil.h"

using namespace cv;
using namespace cv::cuda;

cv::Mat sharpenGPU (cv::Mat Input)
{

  cv::cuda::GpuMat gpu_Input;  
  gpu_Input.upload(Input);
  cv::cuda::GpuMat gpu_Temp, gpu_Result;
  cv::Mat Result;
  cv::Ptr<cv::cuda::Filter> LaplacianFilter = cv::cuda::createLaplacianFilter(gpu_Input.type(),gpu_Temp.type(), 3, 1, cv::BORDER_DEFAULT);
  LaplacianFilter->apply(gpu_Input, gpu_Temp);
  cv::cuda::subtract(gpu_Input, gpu_Temp, gpu_Result);
  //LaplacianFilter->apply(gpu_Result, gpu_Temp);
  //cv::cuda::add(gpu_Input, gpu_Temp, gpu_Result);
  gpu_Result.download(Result);
  //gpu_Temp.download(Result);


  return Result;
}
///template <typename H, typename V, typename P>
///cv::Mat IIRFilter (cv::Mat Input,
///                   float amount,
///                   cv::Mat& lpImage,
///                   const H& hBoundary,
///                   const V& vBoundary,
///                   const float maxVal = 255.0f
///                  )
///{
///
///  const float alpha = powf(amount, 1.0f, 4.0f);
///  const int width = Input.cols;
///  const int height = Input.rows;
/// 
///  size_t sizeInput   = Input.total() * Input.elemSize();
///  size_t sizeBuffer  = Buffer.total() * Buffer.elemSize();
///  size_t sizelpImage = lpImage.total() * lpImage.elemSize();
///  Mat buffer(std::max(width, height), 1, CV_32FC3);
///  const Vec3f zf(0,0,0);
///  assert(width = lpImage.cols && height == lpImage.rows);
///
///  //unsigned char* d_inputimage, *d_buffer, *d_lpImage;
///  //cudaMalloc(&d_inputimage, sizeInput); 
///  //cudaMalloc(&d_buffer, sizeInput); 
///  //cudaMalloc(&d_lpimage, sizeInput); 
///  //
///  //cudaMemcpy(d_inputimage, Input.ptr(), sizeInput, cudaMemcpyHostToDevice); 
///  //cudaMemcpy(d_buffer, Buffer.ptr(), sizeBuffer, cudaMemcpyHostToDevice); 
///  //cudaMemcpy(d_lpImage, lpImage.ptr(), sizelpImage, cudaMemcpyHostToDevice);
///
///  GpuMat d_inputimage, d_buffer, d_lpImage;
///  d_inputimage.upload(Input);
///  d_buffer.upload(Buffer);
///  d_lpImage.upload(lpImage);
///  //Reflect Boundary Code here.
///  //Wrap Boundary Code here.
///
///
///  // Write CUDA Code for Horizontal Pass. 
///  HorizontalPass<<<1, height>> ( d_inputimage, width, alpha, d_buffer, d_lpImage );
///
///
/// //VerticalPass<<width, 1>> ( );
///
///
///}
///float __device__ hBoundary ( float x, float r ) {
///  return x < 0 ? r + x : x >= r ? x - r : x;
///}
///
///
///Vec3f __device__ lerp (Vec3f x0, Vec3f x1, float alpha ) { 
///  return x0 * (1.0 - alpha) + x1 * alpha;
///
///}
///
///float __device__ vBoundary ( float x, float r) { 
///  return x < 0 ? -x : x >= r ? 2*r - x - 2 : x;
///}
///
///void __global__ HorizontalPass ( PtrStepSz<Vec3f>& inputImage, int width, float alpha, PtrStepSz<Vec3f>& buffer, PtrStepSz<Vec3f>& lpImage ) {
///
///    int idx = blockIdx.x * blockDim.x + threadIdx.x;
///    Vec3f v = inputImage(idx,0);
///    for (int j = 0; j < width; j++) {        
///      Vec3f ip = inputImage(idx, hBoundary(j, width));
///      v = lerp(ip, v, alpha);
///      buffer(hBoundary(j - 1, width)) = v;
///    }
///    __syncthreads();
///}
///

