# Checkpoint Report

Team: Sampath Chanda (schanda) and Harish Dattatreya Dixit (hdixit)

## Current Status
We have compiled the Surround360 pipeline and setup all the dependencies required on local machine to run the code from Facebook Surround360 [repo](https://github.com/facebook/Surround360).
Also, since one of us don't have an NVIDIA GPU on local machine (schanda), we worked with a TA to setup all the dependencies to get the code from Facebook compiled and running on a GHC machine.
However, due to the large amount of intermediate data produced, and with current AFS storage limitations, we are unable to render the frames on GHC machines.
We got a ticket raised (by TA) for extension of AFS storage limit and are waiting for a reply on that.
Further, we got the pipeline to be running over the 2 frame dataset provided in the same repository using local machine. The running time is observed to be around 45-55 seconds for rendering 2 frames.

## Goals and Deliverables
We have started with the goal of looking into optimizations to the current available code of Surround360 and identified few potential performance hotspots.
We are taking a deeper dive into the code to understand, analyze and time the performance hotspots which could help us evaluate and make optimizations to accelerate the rendering.

For the next half of the project, we will concentrate on making a GPU implementation of the rendering algorithm.
We plan to use the current running CPU implementation as our Test harness to check the correctness of our GPU implementation.
Later, we would stress on performance optimizations to accelerate the GPU implementation.

During the final demo and in the parallelism competition, we plan to contrast the performance of the current CPU version of Surround360 rendering against our accelerated CUDA implementation.
We aim to achieve at least a 3X speedup for rendering. 

## Schedule
<table>
<thead>
  <tr>
    <th style="text-align: center"> Timeline </th>
    <th style="text-align: left"> Tasks </th>
    <th style="text-align: center"> Status </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align: center"> Apr 2  - Apr 8 </td>
    <td style="text-align: left"> Research on project ideas, analyze and understand feasibility of implementation in project timeframe, draft project proposal for submission </td>
    <td style="text-align: center"> DONE </td>
  </tr>
  <tr>
    <td style="text-align: center"> Apr 9  - Apr 15 </td>
    <td style="text-align: left"> Revise the project proposal and submit. Setting up all dependencies required for Facebook Surround360 </td>
    <td style="text-align: center"> DONE </td>
  </tr>
  <tr>
    <td style="text-align: center"> Apr 16 - Apr 22 </td>
    <td style="text-align: left"> Get the available code from Surround360 repo to compile and run on the 2 frames dataset availabl </td>
    <td style="text-align: center"> DONE </td>
  </tr>
  <tr>
    <td style="text-align: center"> Apr 23 - Apr 29 </td>
    <td style="text-align: left"> Deep dive into the current CPU implementation, identify performance bottlenecks and potential sources of optimizations </td>
    <td style="text-align: center"> In-Progress </td>
  </tr>
  <tr>
    <td style="text-align: center"> Apr 30 - May 6 </td>
    <td style="text-align: left"> Time and evaluate the performance hotspots. Start with implementing the renderer counterpart in CUDA </td>
    <td style="text-align: center">  </td>
  </tr>
  <tr>
    <td style="text-align: center"> May 7  - May 12 </td>
    <td style="text-align: left"> Evaluate correctness of GPU implementation against the already available CPU implementation. Making optimizations for reaching our performance target </td>
    <td style="text-align: center">  </td>
  </tr>
</tbody>

</table>
