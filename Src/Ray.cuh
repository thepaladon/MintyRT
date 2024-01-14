

#pragma once

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

class Ray
{
public:

    __host__ __device__ Ray(const glm::vec3& o_i, const glm::vec3& d_i)
    {
	    org = o_i;
    	dir = d_i;
        t = 1e30f;
        normal = glm::vec3(0.0f);
        hit = false;
    }

    float t;
	bool hit;
    glm::vec3 org;
    glm::vec3 dir;
    glm::vec3 normal;
};