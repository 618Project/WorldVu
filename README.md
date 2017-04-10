
# FB_360_Accelerator

<span style="color:lightgray">
Please provide the title of your project, followed by the names of all team members. Teams may include up to two students. There are no exceptions to this rule.
</span>

<span style="color:black">
The title of our project is "Insert title here".

Team Members(in alphabetical order): Harish Dattatraya Dixit , Sampath Chanda. 
</span>


# Summary

<span style="color:lightgray">
Summarize your project in no more than 2-3 sentences. Describe what you plan to do and what parallel systems you will be working with. Example one-liners include (you should add a bit more detail):
</span>

<span style="color:black">
Our project goal is to accelerate the (360 degree) video processing using source video feed from Facebook Surround 360
and also referencing the correctness of the code with the Facebook implementation. We intend to accelerate the processing
using GPUs as part of the effort.
</span>
<!---
<span style="color:lightgray">
We are going to implement an optimized Smoothed Particle Hydrodynamics fluid solver on the NVIDIA GPUs in the lab.
We are going port the Go runtime to Blacklight.
We are going to create optimized implementations of sparse-matrix multiplication on both GPU and multi-core CPU platforms, and perform a detailed analysis of both systems' performance characteristics.
We are going to back-engineer the unpublished machine specifications of the GPU in the tablet my partner just purchased.
We are going to implement two possible algorithms for a real-time computer vision application on a mobile device and measure their energy consumption in the lab.
</span>
-->
# Background
<span style="color:black">
Our project involves the acceleration of the 360 degree video processing application. It takes the source set of frames
from Facebook's Surround360 Cameras, which involves a rig of 14 cameras taking images in different directions, and finally 
constructing the complete image from each of the 14 cameras and preserving the quality. The current state of the art 
process implemented by Facebook takes about (45 seconds on Alex's Mac) to implement this. We intend to accelerate the same.
We would also be using the current implementation of the same algorithm in scanner as the starting point. The current implementation in python using the Scanner API performs the said computation from Facebook in a better way. We intend to 
further accelerate the process by attempting paralellism across images/ redundancy between data from images and also utilsing
the effective system architecture of the GPU in obtaining the best possible performance.

</span>
<!---
<span style="color:lightgray">
If your project involves accelerating a compute-intensive application, describe the application or piece of the application you are going to implement in more detail. This description need only be a few paragraphs. It might be helpful to include a block diagram or pseudocode of the basic idea. An important detail is what aspects of the problem might benefit from parallelism? and why?
</span>
-->

# The Challenge
<!---
<span style="color:lightgray">
Describe why the problem is challenging. What aspects of the problem might make it difficult to parallelize? In other words, what to you hope to learn by doing the project?
</span>
<span style="color:lightgray">
Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?
Describe constraints: What are the properties of the system that make mapping the workload to it challenging?
</span>
-->
<span style="color:black">
The problem poses many challenges in accelerating the implementation. The sheer computation invloved in ensuring the generation
of 360 degree frames would need the processing of the video from 14 high resolution cameras. This would also mean that the 
memories storing the images would be completely stressed due to the size of the image from one individual camera not fitting in
cache. Also cache thrashing may occur due to the fact that each image may be contending for the same cache line based on what 
stage of the computation we are in. Also the amount of time spent in processing needs to be carefully tuned. Given that there are 14 cameras, there will be redundancy in images for sure, but identifying and tuning the performance of the GPU for these redundancies ( to avoid repeated computations of no value ) is also a hard problem as workload characterisation cannot be done on just a single sample of data/image frame. So an efficient pipeline considering all these architectural disciplines would 
be a challenging as well as a rewarding project utilizing variety of aspects that were taught during the course.
</span>

# Resources
<span style="color:lightgray">
Describe the resources (type of computers, starter code, etc.) you will use. What code base will you start from? Are you starting from scratch or using an existing piece of code? Is there a book or paper that you are using as a reference (if so, provide a citation)? Are there any other resources you need, but haven't figured out how to obtain yet? Could you benefit from access to any special machines?
</span>

<span style="color:black">
We plan to utilize the GPUs present in the Gates Cluster for our development since we anticipate our code to have longer runtimes and the iterations also to be on the higher side. We would be starting from the scanner code base found at :
"insert link here" and also we would be referring to the original implementation of the facebook surround 360 pipeline found here " ". We would also be referring the documentation and overall implementaiton of the scanner as well as the draft documentation from Facebook describing challenges of the same. We would also be using CUDA documentations for our coding 
references and looking into OpenCV and Vision Pipeline accelerator related papers to aid our efforts in designing the best
possible accelerator for the said problem.
</span>

# Goals and Deliverables
<span style="color:lightgray">
Describe the deliverables or goals of your project.
</span>

<span style="color:lightgray">
Separate your goals into what you PLAN TO ACHIEVE (what you believe you must get done to have a successful project and get the grade you expect) and an extra goal or two that you HOPE TO ACHIEVE if the project goes really well and you get ahead of schedule. It may not be possible to state precise performance goals at this time, but we encourage you be as precise as possible. If you do state a goal, give some justification of why you think you can achieve it. (e.g., I hope to speed up my starter code 10x, because if I did it would run in real-time.)
If applicable, describe the demo you plan to show at the parallelism computation (will it be an interactive demo? will you show an output of the program that is really neat? will you show speedup graphs?). Specifically, what will you show us that will demonstrate you did a good job?
If your project is an analysis project, what are you hoping to learn about the workload or system being studied? What question(s) do you plan to answer in your analysis?
Systems project proposals should describe what the system will be capable of and what performance is hoped to be achieved.
IN GENERAL: Imagine that I didn't give you a grading script on assignments 2, 3, or 4. Imagine you did the entire assignment, made it as fast as you could, and then turned it in. You wouldn't have any idea if you'd done a good job!!! That's the situation you are in for the final project. And that's the situation I'm in when grading your final project. As part of your project plan, and ONE OF THE FIRST THINGS YOU SHOULD DO WHEN YOU GET STARTED WORKING is implement the test harnesses and/or baseline "reference" implementations for your project. Then, for the rest of your project you always have the ability to run your optimized code and obtain a comparison.
</span>

<span style="color:black">
We plan to achieve a significant performance improvement (at least a 2.5X) over the current Facebook Surround360 pipeline in the order.
Our stretch goal is to be able to bring down the frame processing time to 1 second from 45 seconds. Being able to process videos of duration (1 min) currently takes around 1350 minutes. Reducing the processing time from 45 seconds to 15 seconds will reduce the 1 min video processing duration to 450 minutes. So this is the achievable goal that we are setting ourselves with. 
Our stretch goal is to bring down the processing from 45 seconds to 1 second and reduce the processing time from 1350 minutes to 30 minutes to process the video. 

Demo: 
In the demo, we would be comparing our implementation with the implementation from Facebook and perform a run-time 
comparision of both of the video processing pipelines.Specifically, we would be taking a 360 degree frame seqeunce and feed it to both the pipelines and time the output generation sequence. In addition, we would also be generating some graphs to indicate
the scalability and the performance bottlenecks identified in the implementation. 
</span>

# Platform Choice
<!---
<span style="color:lightgray">
Describe why the platform (computer and/or language) you have chosen is a good one for your needs. Why does it make sense to use this parallel system for the workload you have chosen?
</span>
-->
<span style="color:black">
We would be implementing our algorithms for acceleration on GPUs. Given that the input consists of a number of frames and GPUs are predominantly good at utilizing data parallelism at an impressive pace, we would be using the GPU platform for our implementation. We do notice that to be able to perform a fair comparision, we would have to relatively estimate an improved reference scale against our implementation for which we will be replicating a naive kernel that translates Facebook source code into CUDA and executes it. We will be improving the code with our novel algorithms and optimizations.
</span> 

# Schedule
<span style="color:black">
1.	Get the Surround360 pipeline compiling (We will be using this repo to setup the dependencies. https://github.com/holgr/build_surround360)
2.	Get the pipeline running over the 2 frame sample dataset they provide (we are working on getting a larger dataset)
3.	Do a deep dive into the performance of the pipeline, determining timing and where the bottlenecks are.
4.	Figure out some small experiments to run that you think might improve performance in order to verify your hypothesis about what the bottlenecks are.
5.	Fully implement the intended optimizations (for example, making better use of cache locality/using GPUs)
</span>

<!---
<span style="color:lightgray">
Produce a schedule for your project. Your schedule should have at least one item to do per week. List what you plan to get done each week from now until the parallelism competition in order to meet your project goals. Keep in mind that due to other classes, you'll have more time to work some weeks than others (work that into the schedule). You will need to re-evaluate your progress at the end of each week and update this schedule accordingly. Note the intermediate checkpoint deadline is April 25th. In your schedule we encourage you to be precise as precise as possible. It's often helpful to work backward in time from your deliverables and goals, writing down all the little things you'll need to do (establish the dependencies!).
</span>
-->
