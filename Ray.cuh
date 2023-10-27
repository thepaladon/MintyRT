

#pragma once

#include <glm/glm.hpp>

class Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const glm::vec3& o_i, const glm::vec3& d_i) { o = o_i; d = d_i; }
    __device__ glm::vec3 origin() const { return o; }
    __device__ glm::vec3 direction() const { return d; }
    __device__ glm::vec3 point_at_parameter(float t) const { return o + t * d; }

    glm::vec3 o;
    glm::vec3 d;
};