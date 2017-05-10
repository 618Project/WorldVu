<span>
Team: Sampath Chanda (schanda) and Harish Dixit (hdixit) <br/>
</span>
Please refer to: <br/>
<a href="https://618project.github.io/WorldVu/proposal" style="background:#D8D8D8;">Proposal</a>
<a href="https://618project.github.io/WorldVu/checkpoint" style="background:#D8D8D8">Checkpoint</a>

<h1>Overview</h1>
We have accelerated the frame processing of video feed from Facebook 3D-360 camera system to produce a fully stitched image for stereo vision. Our acceleration techniques utilized CPU Multithreading and GPU acceleration to exploit pixel level parallelism for stitching frames together to produce a complete 360 degree vision.

<h1>Motivation and Background</h1>
<body>
Virtual reality has the power of changing the way that we see and interact with the world. The very concept of being able to perceive a different environment other than the one we are actually in, opens up a huge space of possibilities for innovative applications and technologies. One key method of generating a VR environment is using a 360 degree camera to record, process and buffer a 360 degree video. But rendering a high quality video would require processing single camera frames to produce a stitched larger frame and this process is highly compute intensive.

Facebook Surround360 camera rig is a 3D-360 camera system that captures frames from 17 cameras at 30-60 fps to cover the overall 360 degree vision. Just to render a properly stitched image from one time instance (17 frames), current multithreaded implementation of Facebook code takes around 66 seconds on AWS (8 core Intel Xeon E5-2670, 15GB RAM). This converts into a whopping <strong>1980 minutes to render a 1 minute video</strong> at 30 fps. Such a high compute time could discourage a wider adaptability of this technology. With a goal of alleviating such dramatic compute latencies, we targeted to accelerate the frame processing of a 360 degree video.
</body>

<h1>Approach</h1>

<h2>Profiling</h2>

With a view of accelerating the components of the current algorithm that are primary hotspots of performance bottle neck, we started with profiling the current implementation. After getting an initial phase wise break down of processing time of the algorithm, we noticed that computing optical flow and sharpening are taking up almost 85-90% of the overall processing time.

Computing optical flow being the highest, we dived deep into the code and identified that optical flow generation is computed 28 times with different sizes for each adjacent pair of frames. Also, computation of flow at each pixel is dependent on flow value of pixel at previous position, making that phase of the algorithm fully sequential.Identifying possible pixel level parallelism, we felt that SIMD parallelism of GPU would provide us a lot of performance improvement.

<h2>CUDA integration</h2>
Original Surround360 code uses a multi-threaded implementation to compute flow at image level. Flow for each pair of adjacent images is computed by a separate thread on a multi-core processor. However, computation within each thread can still be parallelized at pixel level. GPU being an ideal candidate for such a case, we decided to integrate CUDA into the current flow. To keep things simple with CUDA integration, we converted the multi-threaded implementation into a single threaded one first. 

Current code has lot of dependencies like folly, CMake, OpenCV, glog, etc., For CUDA to be integrated, it has to be made sure that it is compatible and stable with each one of these dependencies. First, we tried to integrate a simple CUDA kernel into the current execution flow. Since only g++ compiler is being used currently, including nvcc and directing CMake to handle picking up different compilers for different files was a tough task. Lot of linking errors were seen and fixed to integrate a simple CUDA kernel into the current flow.

Later, when trying to use OpenCV CUDA API, we re-built OpenCV with CUDA support. After fixing a series of comptability and linking errors, we were finally able to successfully integrate CUDA into the existing flow. 

<h1>Performance Optimizations</h1>

<h2>Optical Flow</h2>
To exploit the pixel level parallelism, we resorted to using GPU CUDA cores. OpenCV CUDA provides an API for computing optical flow using dense pyramidal lucas kanade algorithm, which is similar algorithm being used by the current Facebook implementation. We identified the levels to which the algorithm from Facebook was using optical flow and provided appropriate paramters to the CUDA flow.  We replaced the current sequential implementation with parallel Opencv CUDA implementation and observed more than 3x speedup, just for the optical flow phase.

We know that GPU call incurs a significant overhead and will be amortized only when having considerable amount of computation in the GPU kernel. Current algorithm uses a pyramidal structure of computing flow on different sized images to compute an overall optical flow for a pair of adjacent images. For smaller sized images, this GPU kernel launch overhead cannot be amortized and hence it would be lot better to do such computation on CPU itself. To take care of this, we compute flow corresponding to smaller images, on CPU and utilize GPU only for larger compute intensive image pairs.

While just exploiting pixel level parallelism of an frame pair, a modern GPU would be left under utilized. To improve GPU utilization, we re-enabled image level multi-threading. With this, each adjacent pair of images would be processed by a separate thread, which would in turn make calls to GPU to exploit SIMD parallelism on GPU.

Careful fine tuning of the argument parameters provided us the desired speedup and the current state of accuracy. We are still exploring further optimizations to the optical flow so that we have improvements in accuracy.

<h2>Sharpening</h2>
One other hotspot of performance is the final sharpening of the fully stitched images (for left and right eye). The current algorithm uses an iirLowPass filter that computes horizontal and vertical pass separately and subsequently sends the filtered image to sharpening block. Each pass in turn has causal and anti-causal passes each of which iterate through every pixel of the final image. Sharpening a pair of 8k images with such a  sequential algorithm takes around 28 seconds. To accelerate this, we again resort to OpenCV CUDA API. Since the API doesn't already have an iirLowPass filter and sharpening filters. For sharpening, we follow the usual approach of sharpening an image by calculating Laplacian of the image (that detects edges) and then subtracting/adding it to the image itself. This being a CUDA implementation, gave us an acceleration of around 24.6x by taking only around 1.1 seconds to sharpen the final image pair. However, we see some inaccuracies in the images when compared to the one that is produced by original sequencec of iirLowPass filter + sharpening. We are currently looking into the ways to minimize these inaccuracies. We implemented our own CUDA kernel for iirLowPass filter but are seeing some issues with conversion and handling of abstract data types like cv::Mat that are used in current iir filter. With porting the IIR Low Pass Filter to custom CUDA Kernel and then subsequently using Laplacian Sharpening, we expect the image quality to improve further. 

<h1>Current Result</h1>
Below you can see the resultant renderings from Facebook and our accelerated flow. As visible from the images, we have not 
significantly lost image perception in spite of our speedup. We see minimal overlaps in the stitching framework implementation.

<h2> Image Renderings </h2> 
<h3> Facebook Rendering </h3> 
![Flickr](https://c1.staticflickr.com/5/4173/33772131443_3965ea2578_k.jpg)

<h3> Our Accelerated Rendering </h3> 
![Imgur](http://i.imgur.com/vK583d0.jpg)

<h2>Comparison</h2>
To provide a comparision of the speedup, 
1. The original code takes **66 seconds per frame**, thus resulting in **1980 minutes for rendering 1 min of video at 30fps**.
2. Our accelerated code, if we replace only the blocks that we have worked on ( optical flow + sharpening ) in the original code, the rendering time is **19 seconds per frame**, thus resulting in **573 minutes for rendering 1 min of video at 30 fps**. 

<h2> Speedup Breakdown </h2> 
We see mainly 6 different stages in the computation of the Facebook Stereo Rendering Panorama Pipeline.
1. Spherical projection of the images for each camera.
2. Side optical flow for stereo.
3. Novel view panorama.
4. Flow top+bottom with sides.
5. Sharpening
6. Equirectangular to Cubemap transformation.

As part of this project, we have accelerated the 2 primary hotspots, Side optical flow and Sharpening and the results are 
as follows. All our results have been run on AWS machine (g2.2xlarge: CPU - 8 core Intel Xeon E5-2670 (Sandy Bridge) Processors and GPU - NVIDIA GPU, with 1,536 CUDA cores and 4GB video memory). Results are for a run where CPU rendering of all phases included take 66.3 seconds. <br/>

In the below table, accuracy is measured as the number of pixels that are having exact pixel intensities when compared to that of the original implementation. Also, speedup is the ratio by which our accelerated implementation is faster than that of the original implementation.
<table>
<thead>
  <tr>
    <th style="text-align: center"> Phase </th>
    <th style="text-align: center"> Original Time </th>
    <th style="text-align: center"> Accelerated time </th>
    <th style="text-align: center"> Accuracy </th>
    <th style="text-align: center"> Speedup </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align: center"> Optical Flow </td>
    <td style="text-align: center"> 29.7 secs </td>
    <td style="text-align: center"> 9.1 secs </td>
    <td style="text-align: center"> 63.5% </td>
    <td style="text-align: center"> 3.26x </td>
  </tr>
  <tr>
    <td style="text-align: center"> Sharpening </td>
    <td style="text-align: center"> 27.7 secs </td>
    <td style="text-align: center"> 1.1 secs </td>
    <td style="text-align: center"> 77% </td>
    <td style="text-align: center"> 25.2x </td>
  </tr>
</tbody>
<table>

<h1>Demo and Current work</h1>
We plan to show a demo of runs on original and accelerated implementation and contrast the times taken and similarity of the results. Now that we have good speed up as proposed, we are currently working towards improving accuracy of the computations.

<h1>References</h1>
1. http://docs.opencv.org/2.4/modules/gpu/doc/introduction.html
