#pragma once

#include <cuda_runtime.h>
#include "glm/vec3.hpp"

__host__ __device__ inline uchar3 to_uchar3(glm::vec3 val)
{
	uchar3 rgb;
	rgb.z = static_cast<unsigned char>(val[0] * 255.99f);
	rgb.y = static_cast<unsigned char>(val[1] * 255.99f);
	rgb.x = static_cast<unsigned char>(val[2] * 255.99f);
	return rgb;
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
static void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		printf("CUDA error %i = %s at %s : %i \n", result, func, file, line);
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


