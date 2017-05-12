#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaarithm.hpp"
#include "MathUtil.h"
#include "SystemUtil.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

using namespace cv;
using namespace cv::cuda;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


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

float __device__ hBoundary ( float x, float r ) {
  return x < 0 ? r + x : x >= r ? x - r : x;
}


float __device__ lerp (float x0, float x1, float alpha ) { 
  return x0 * (1.0 - alpha) + x1 * alpha;

}

float __device__ vBoundary ( float x, float r) { 
  return x < 0 ? -x : x >= r ? 2*r - x - 2 : x;
}

float __device__ clamp(float x, float a, float b)
{
    return x < a ? a : (x > b ? b : x);
}

float __device__ square(float x)
{
    return x*x;
}

void __global__ HorizontalPass ( 
                                unsigned char *d_b_ch,
                                unsigned char *d_g_ch,
                                unsigned char *d_r_ch,
                                unsigned char *d_b_buff,
                                unsigned char *d_g_buff,
                                unsigned char *d_r_buff,
                                unsigned char *d_b_lp,
                                unsigned char *d_g_lp,
                                unsigned char *d_r_lp,
                                int width,
                                int height,
                                float maxVal,                    
                                float alpha
                               ) {


   int idx = blockDim.x*blockIdx.x + threadIdx.x;
   
   if ( idx < height )
   {

   int i = idx*width;

   float v0, v1, v2;
   v0 = (float)d_b_ch[i];
   v1 = (float)d_g_ch[i];
   v2 = (float)d_r_ch[i];
   
   float i0, i1, i2;
   for (int j = 1; j <= width; ++j) {
      int hb1 = (int)hBoundary(j,width);
      i0 = (float)d_b_ch[i+hb1]; 
      i1 = (float)d_g_ch[i+hb1];
      i2 = (float)d_r_ch[i+hb1];

      v0 = lerp(i0, v0, alpha);
      v1 = lerp(i1, v1, alpha);
      v2 = lerp(i2, v2, alpha);

      int hb2 = (int)hBoundary(j-1,width);
      d_b_buff[hb2] = (unsigned char)v0;
      d_g_buff[hb2] = (unsigned char)v1;
      d_r_buff[hb2] = (unsigned char)v2;
    }

    for (int j = width - 2; j >= -1; --j) {
      int hb1 = (int)hBoundary(j,width);
      i0 = (float)d_b_buff[hb1]; 
      i1 = (float)d_g_buff[hb1]; 
      i2 = (float)d_r_buff[hb1];

      v0 = lerp(i0, v0, alpha);
      v1 = lerp(i1, v1, alpha);
      v2 = lerp(i2, v2, alpha);

      d_b_lp[i+j+1] = (unsigned char)clamp(v0, 0.0f, maxVal);
      d_g_lp[i+j+1] = (unsigned char)clamp(v1, 0.0f, maxVal);
      d_r_lp[i+j+1] = (unsigned char)clamp(v2, 0.0f, maxVal);
    }
   }
}

void __global__ VerticalPass ( 
                                unsigned char *d_b_ch,
                                unsigned char *d_g_ch,
                                unsigned char *d_r_ch,
                                unsigned char *d_b_buff,
                                unsigned char *d_g_buff,
                                unsigned char *d_r_buff,
                                unsigned char *d_b_lp,
                                unsigned char *d_g_lp,
                                unsigned char *d_r_lp,
                                int width,
                                int height,
                                float maxVal,                    
                                float alpha
                               ) {


   unsigned char idx = blockDim.x*blockIdx.x + threadIdx.x;

   if ( idx < width )
   {
   unsigned char i = idx*height;

   float v0, v1, v2;
   v0 = (float)d_b_ch[i];
   v1 = (float)d_g_ch[i];
   v2 = (float)d_r_ch[i];
   
   float i0, i1, i2;
   for (int j = 1; j <= height; ++j) {
      unsigned char hb1 = (unsigned char)vBoundary(j,width);
      i0 = (float)d_b_ch[i+hb1]; 
      i1 = (float)d_g_ch[i+hb1];
      i2 = (float)d_r_ch[i+hb1];

      v0 = lerp(i0, v0, alpha);
      v1 = lerp(i1, v1, alpha);
      v2 = lerp(i2, v2, alpha);

      int hb2 = (int)vBoundary(j-1,width);
      d_b_buff[hb2] = (unsigned char)v0;
      d_g_buff[hb2] = (unsigned char)v1;
      d_r_buff[hb2] = (unsigned char)v2;
    }

    for (int j = height - 2; j >= -1; --j) {
      unsigned char hb1 = (unsigned char)vBoundary(j,width);
      i0 = (float)d_b_buff[hb1]; 
      i1 = (float)d_g_buff[hb1]; 
      i2 = (float)d_r_buff[hb1];

      v0 = lerp(i0, v0, alpha);
      v1 = lerp(i1, v1, alpha);
      v2 = lerp(i2, v2, alpha);

      d_b_lp[i+j+1] = (unsigned char)clamp(v0, 0.0f, maxVal);
      d_g_lp[i+j+1] = (unsigned char)clamp(v1, 0.0f, maxVal);
      d_r_lp[i+j+1] = (unsigned char)clamp(v2, 0.0f, maxVal);
    }
  printf("\n Contents of B low pass are %u", d_b_lp[i]);
  printf("\n Contents of G low pass are %u", d_g_lp[i]);
  printf("\n Contents of R low pass are %u", d_r_lp[i]);
  }
}

void __global__ sharpenWithIirLowPass( 
                                unsigned char *d_b_ch,
                                unsigned char *d_g_ch,
                                unsigned char *d_r_ch,
                                unsigned char *d_b_lp,
                                unsigned char *d_g_lp,
                                unsigned char *d_r_lp,
                                float rAmount,
                                float gAmount,
                                float bAmount,
                                float noiseCore,
                                int height,
                                int width,
                                float maxVal)
{
   int idx = blockDim.x*blockIdx.x + threadIdx.x;
   if ( idx < height )
   {
  
    int i = idx*width;
      for (int j = 0; j < width; ++j) {
        //const Vec3f lp = lpImage.at<P>(i, j);

        float lp0 = (float)d_b_lp[i+j];
        float lp1 = (float)d_g_lp[i+j];
        float lp2 = (float)d_r_lp[i+j];
        
        float p0 = (float)d_b_ch[i+j];
        float p1 = (float)d_g_ch[i+j];
        float p2 = (float)d_r_ch[i+j];
       
        float hp0 = p0 - lp0; 
        float hp1 = p1 - lp1; 
        float hp2 = p2 - lp2;

        float ng0 = 1.0f - expf(-(square(hp0 * noiseCore)));
        float ng1 = 1.0f - expf(-(square(hp1 * noiseCore)));
        float ng2 = 1.0f - expf(-(square(hp2 * noiseCore)));

        p0 = clamp(lp0 + hp0 * ng0 * bAmount, 0.0f, maxVal);
        p1 = clamp(lp1 + hp1 * ng1 * bAmount, 0.0f, maxVal);
        p2 = clamp(lp2 + hp2 * ng2 * bAmount, 0.0f, maxVal);

        d_b_ch[i+j] = (char)p0;
        d_g_ch[i+j] = (char)p1;
        d_r_ch[i+j] = (char)p2;
        //P& p = inputImage.at<P>(i, j);
        // High pass signal - just the residual of the low pass
        // subtracted from the original signal.
    }
   }
}
//template <typename H, typename V, typename P>
void IIRFilter(cv::Mat Input,
                   float amount,
                   cv::Mat& lpImage,
                   const float maxVal = 255.0f
                  )
{

  const float alpha = powf(amount, 1.0f/4.0f);
  const int width = Input.cols;
  const int height = Input.rows;
  Mat bgr[3];
  cv::split(Input, bgr); 
  unsigned char *b_ch = bgr[0].data;
  unsigned char *g_ch = bgr[1].data;
  unsigned char *r_ch = bgr[2].data;
  unsigned char *d_b_ch ;
  unsigned char *d_g_ch ;
  unsigned char *d_r_ch ;
  
  //Mat buffer[3];
  unsigned char *b_buf = new unsigned char[std::max(width,height)];//= buffer[0].data;
  unsigned char *g_buf = new unsigned char[std::max(width,height)];//= buffer[1].data;
  unsigned char *r_buf = new unsigned char[std::max(width,height)];//= buffer[2].data;
  unsigned char *d_b_buf ;
  unsigned char *d_g_buf ;
  unsigned char *d_r_buf ;
  
  //Mat lpim[3];
  unsigned char *b_lp  = new unsigned char[width*height];// = lpim[0].data;
  unsigned char *g_lp  = new unsigned char[width*height];// = lpim[1].data;
  unsigned char *r_lp  = new unsigned char[width*height];// = lpim[2].data;
  unsigned char *d_b_lp ;
  unsigned char *d_g_lp ;
  unsigned char *d_r_lp ;


gpuErrchk(  cudaMalloc((void **)&d_b_ch,  sizeof(unsigned char) * width * height)); 
gpuErrchk(  cudaMalloc((void **)&d_g_ch,  sizeof(unsigned char) * width * height)); 
gpuErrchk(  cudaMalloc((void **)&d_r_ch,  sizeof(unsigned char) * width * height)); 
gpuErrchk(  cudaMalloc((void **)&d_b_buf, sizeof(unsigned char) * std::max(width, height))); 
gpuErrchk(  cudaMalloc((void **)&d_g_buf, sizeof(unsigned char) * std::max(width, height))); 
gpuErrchk(  cudaMalloc((void **)&d_r_buf, sizeof(unsigned char) * std::max(width, height))); 
gpuErrchk(  cudaMalloc((void **)&d_b_lp,  sizeof(unsigned char) * width * height)); 
gpuErrchk(  cudaMalloc((void **)&d_g_lp,  sizeof(unsigned char) * width * height)); 
gpuErrchk(  cudaMalloc((void **)&d_r_lp,  sizeof(unsigned char) * width * height)); 
gpuErrchk(  cudaMemcpy     ( d_b_ch, b_ch,   sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_g_ch, g_ch,   sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_r_ch, r_ch,   sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_b_buf, b_buf, sizeof(unsigned char)*std::max(width,height), cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_g_buf, g_buf, sizeof(unsigned char)*std::max(width,height), cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_r_buf, r_buf, sizeof(unsigned char)*std::max(width,height), cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_b_lp, b_lp,   sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_g_lp, g_lp,   sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice ));
gpuErrchk(  cudaMemcpy     ( d_r_lp, r_lp,   sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice ));

  dim3 blocks1D(height, 1);
  
  double startTime = surround360::util::getCurrTimeSec() ;

  LOG(INFO) << startTime ;
  LOG(INFO) << "Entering LowH Pass Kernel here" ;
  HorizontalPass<<<height, 1>>>(d_b_ch, d_g_ch, d_r_ch, d_b_buf, d_g_buf, d_r_buf,  d_b_lp, d_g_lp, d_r_lp,  width, height, maxVal, alpha  );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  LOG(INFO) << "Exiting LowH Pass Kernel here" ;
  LOG(INFO) << "Entering LowV Pass Kernel here" ;
  HorizontalPass<<<width, 1>>>(d_b_ch, d_g_ch, d_r_ch, 
                                d_b_buf, d_g_buf, d_r_buf,
                                d_b_lp, d_g_lp, d_r_lp,
                                width, height, maxVal, alpha  );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  LOG(INFO) << "Exiting LowV Pass Kernel here" ;
  float rAmount = amount;
  float gAmount = amount;
  float bAmount = amount;
  float noiseCore = 100.0f;
  sharpenWithIirLowPass<<<height,1>>> 
                                (d_b_ch,
                                d_g_ch,
                                d_r_ch,
                                d_b_lp,
                                d_g_lp,
                                d_r_lp,
                                rAmount,
                                gAmount,
                                bAmount,
                                noiseCore,
                                height,
                                width,
                                maxVal);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  //LOG(INFO) << surround360::util::getCurrTimeSec() ;
  double stopTime =  surround360::util::getCurrTimeSec() - startTime; 
  gpuErrchk(  cudaMemcpy     ( b_ch, d_b_ch,   sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost ));
  gpuErrchk(  cudaMemcpy     ( g_ch, d_g_ch,   sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost ));
  gpuErrchk(  cudaMemcpy     ( r_ch, d_r_ch,   sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost ));
  LOG(INFO) << "Time between Kernels is" << stopTime;

//lpim[0].data = b_lp;
//lpim[1].data = g_lp;
//lpim[2].data = r_lp;

  //for (int i = 0; i < height/1000 ; i++ )
  //{
  //    LOG(INFO) << " Index is " << i ;
  //    LOG(INFO) << i << &b_ch[i] ;
  //    LOG(INFO) << i << &g_ch[i] ;
  //    LOG(INFO) << i << &r_ch[i] ;
  //}
  
  Mat lpim[3];
  lpim[0] = cv::Mat(height, width, CV_8UC1, b_ch);
  lpim[1] = cv::Mat(height, width, CV_8UC1, g_ch);
  lpim[2] = cv::Mat(height, width, CV_8UC1, r_ch);

  if ( lpim[0].empty() ) 
    LOG(INFO) << "Blue Channel Empty" ;
  if ( lpim[1].empty() ) 
    LOG(INFO) << "Red Channel Empty" ;
  if ( lpim[2].empty() ) 
    LOG(INFO) << "Green Channel Empty" ;

  std::vector<cv::Mat> array_to_merge;

  array_to_merge.push_back(lpim[0]);
  array_to_merge.push_back(lpim[1]);
  array_to_merge.push_back(lpim[2]);
  cv::merge(array_to_merge, lpImage);
}

