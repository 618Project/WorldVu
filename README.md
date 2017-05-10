<span>
Team: Sampath Chanda (schanda) and Harish Dixit (hdixit)
</span>
Please refer to: <br/>
<a href="https://618project.github.io/WorldVu/proposal" style="background:#D8D8D8;">Proposal</a>
<a href="https://618project.github.io/WorldVu/checkpoint" style="background:#D8D8D8">Checkpoint</a>

<h1>Overview</h1>
We have accelerated the frame processing of video feed from Facebook 3D-360 camera system to produce a fully stitched image for stereo vision. Our acceleration techniques utilized CPU Multithreading, GPU acceleration for 

<h1>Motivation and Background</h1>
<body>
*ADD a good stat fact* Virtual reality has the power of changing the way that we see and interact with the world. The very concept of being able to perceive a different environment other than the one we are actually in, opens up a huge space of possibilities for innovative applications and technologies. One key method of generating a VR environment is using a 360 degree camera to record, process and buffer a 360 degree video. But rendering a high quality video would require processing single camera frames to produce a stitched larger frame and this process is highly compute intensive.

Facebook Surround360 camera rig is a 3D-360 camera system that captures frames from 17 cameras at 30-60 fps. Just to render a properly stitched image from one time instances (17 frames), current multithreaded implementation of Facebook code takes around 22.5 seconds *Insert machine characteristics*. This converts into a whopping 675 minutes to render a 1 minute video at 30 fps. Such a high compute time could discourage a wider adaptability of this technology. With a goal of alleviating such dramatic compute latencies, we targetted to accelerate the frame processing of a 360 degree video.
</body>

<h1>Approach</h1>

<h2>Profiling</h2>

With a view of accelerating the components of the current algorithm that are primary hotspots of performance bottle neck, we started with profiling the current implementation. After getting an initial phase wise break down of processing time of the algorithm, we noticed that computing optical flow, sharpening and novel view generation are taking up almost *ADD how much percent of the overall time* of the overall processing time.

Computing optical flow being the highest, we dived deep into the code and identified that optical flow generation is computed 28 times with different sizes for each adjacent pair of frames. Also, computation of flow at each pixel is dependent on flow value of pixel at previous position, making that phase of the algorithm fully sequential. However, this computation is similar to that of a 2D grid solver and can be made parallel. Identifying possible pixel level parallelism, we felt that SIMD parallelism of GPU would provide us a lot of performance improvement.

<h2>CUDA integration</h2>
Original Surround360 code uses a multi-threaded implementation to compute flow at image level. Flow for each pair of adjacent images is computed by a separate thread on a multi-core processor. However, computation within each thread can still be parallelized at pixel level. GPU being an ideal candidate for such a case, we decided to integrate CUDA into the current flow. To keep things simple with CUDA integration, we converted the multi-threaded implementation into a single threaded one. *May include single threaded implementation performance stats*. 

Current code has lot of dependencies like folly, CMake, OpenCV, glog, etc., For CUDA to be integrated, it has to be made sure that it is compatible and stable with each one of these dependencies. First, we tried to integrate a simple CUDA kernel into the current execution flow. Since only g++ is being used currently, including nvcc and directing CMake to handle picking up different compilers for different files was a tough task. Lot of linking errors are seen and fixed to integrate a simple CUDA kernel into the current flow.

Later, when trying to use OpenCV CUDA API, we re-built OpenCV with CUDA support. After fixing a series of comptability and linking errors, we were finally able to have CUDA successfully integrated into the existing flow. 

<h1>Performance Optimizations</h1>

<h2>Optical Flow</h2>
To exploit the pixel level parallelism, .... OpenCV cuda provides an API for computing optical flow using dense pyramidal lucas kanade algorithm, which is the same algorithm being used by the current Facebook implementation. We replaced the current sequential implementation with parallel opencv cuda implementation. *INSERT EXACT PERF IMPROVEMENT NUMBERS*

We know that GPU call incurs a significant overhead and will be amortized only when having considerable amount of computation in the GPU kernel. Current algorithm uses a pyramidal structure of computing flow on different sized images to compute an overall optical flow for a pair of adjacent images. For smaller sized images, this GPU kernel launch overhead cannot be amortized and hence it would be lot better to do such computation on CPU itself. To take care of this, we compute flow corresponding to smaller images, on CPU and utilize GPU only for larger compute intensive image pairs.

While just exploiting pixel level parallelism of an frame pair, a modern GPU would be left under utilized. To improve GPU utilization, we re-enabled image level multi-threading. With this, each adjacent pair of images would be processed by a separate thread, which would in turn make calls to GPU to exploit SIMD parallelism on GPU. *May insert memory access issue with multi-threading + GPU*

<h2>Sharpening</h2>
One other hotspot of performance is the final sharpening of the fully stitched images (for left and right eye). The current algorithm uses an iirLowPass filter that computes horizontal and vertical pass separately. Each pass in turn has casual and anti-casual passes each of which iterate through every pixel of the final image. Sharpening a pair of 8k images with such a  sequential algorithm takes around 8 seconds. To accelerate this, we again resort to OpenCV CUDA API. Since the API doesn't already have an iirLowPass filter, we follow the usual approach of sharpening an image by calculating Laplacian of the image (that detects edges) and then subtracting/adding it to the image itself. This being a CUDA implementation, gave us an acceleration of around 16x by taking only around 0.5 seconds to sharpen the final image pair. However, we see some inaccuracies in the images when compared to the one that is produced by original iirLowPass filter. We are currently looking into the ways to minimize these inaccuracies. We implemented our own CUDA kernel for iirLowPass filter but are seeing some issues with conversion and handling of abstract data types like cv::Mat that are used in current iir filter. 

<h1>Current Result</h1>
<h2> Image Renderings </h2> 
<h3> Facebook Rendering </h3> 
![Facebook Rendering]("https://drive.google.com/open?id=0B_ThtGsKhnxNX3JzcGpndXEzSlE")

<h3> Our Accelerated Rendering </h3> 
![Our Rendering]("https://drive.google.com/open?id=0B_ThtGsKhnxNbFdtMVNXdlVwSWc")

- Mention speedup
<h2> Speedup Breakdown </h2> 
We see mainly 6 different stages in the computation of the Facebook Stereo Rendering Panorama Pipeline.
1. Spherical projection of the images for each camera.
2. Side optical flow for stereo.
3. Novel view panorama.
4. Flow top+bottom with sides.
5. Sharpening
6. Equirectangular to Cubemap transformation.

As part of this project, we have accelerated the 2 primary hotspots, Side optical flow and Sharpening and the results are 
as follows.

- Inaccuracies with performance rise
- further analysis
- 
6.4x overall speedup seen. *ELABORATE*

<h1>References</h1>
*ADD references*
