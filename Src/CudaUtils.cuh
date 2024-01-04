#pragma once


#include <cuda_runtime.h>

#include "Ray.cuh"
#include "glm/vec3.hpp"

struct GPUTriData
{
    const float* vertex_buffer;
    const unsigned* index_buffer;
};

struct Triangle
{
    glm::vec3 vertex0;
    glm::vec3 vertex1;
    glm::vec3 vertex2;
};

__device__ inline void intersect_tri(Ray& ray, const Triangle& tris)
{
    const glm::vec3 edge1 = tris.vertex1 - tris.vertex0;
    const glm::vec3 edge2 = tris.vertex2 - tris.vertex0;
    const glm::vec3 h = cross(ray.dir, edge2);
    const float a = dot(edge1, h);
    if (fabs(a) < .0001f) return; // ray parallel to triangle
    const float f = 1 / a;
    const glm::vec3 s = ray.org - tris.vertex0;
    const float u = f * dot(s, h);
    if (u < 0 || u > 1) return;
    const glm::vec3 q = cross(s, edge1);
    const float v = f * dot(ray.dir, q);
    if (v < 0 || u + v > 1) return;
    const float t = f * dot(edge2, q);
    if (t > 0.0001f) { // T Has to be positive
        if (ray.t > t)
        {
            ray.t = t;
            ray.hit = true;
            //ray->intersection.tri_hit = triIdx;
            //ray->intersection.u = u;
            //ray->intersection.v = v;
            ray.normal = cross(edge1, edge2);
        }
    }
}

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
		const auto errorMessage = cudaGetErrorString(result);
		printf("Error Message: %s \n", errorMessage);

		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


