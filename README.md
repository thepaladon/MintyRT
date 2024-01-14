# MintyRT
A CUDA Ray Tracer. 

## Most Recent Screenshot :
<img width="1627" alt="image" src="https://github.com/thepaladon/MintyRT/assets/44022509/0140fe24-4308-418b-9987-49ba8c70d5c4">

## Overview
My goals with this personal project were:
- Learn how to write efficient CUDA code
- Learn more about ray/path tracing and the BVH acceleration structures which make it possible in real-time.
- Create a well-structured easy-to-debug environment where I can test and apply new knowledge that I've gathered from papers/lectures online.

So far I've gotten started with the basics but due to heavy work in my school project which is also a ray tracing renderer, I've very little time to dedicate here. 

## Features
- Both CPU and GPU rendering modes for easy debugging
- Robust glTF Model Loading
- BVH API (WIP)


## Getting Started
1. Make sure you've downloaded the [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Clone the project `git clone https://github.com/thepaladon/MintyRT`
3. Open `MintyRT.sln` in VS22
4. Run the project

Look into the code, if you want to run it on the GPU. There's a `USE_GPU` define in `CudaUtils.cuh`.

**NOTE:** Using GPU on Debug is broken for some reason so don't do that!

## Controls:
W A S D - Move the Camera Forward / Left / Back / Right
R / F - Move the Camera Up / Down
Arrow Keys - Adjust the camera's pitch and yaw
