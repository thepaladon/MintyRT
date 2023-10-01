#pragma once
#include <crt/host_defines.h>

#include "Ray.cuh"
#include "Vec3.cuh"

constexpr float PI = 3.14159265359f;

__host__ __device__ float degrees_to_radians(double degrees) {
    return degrees * (PI / 180.0);
}

class Camera {

public:
    __host__ __device__ Camera(Vec3 org, Vec3 up, Vec3 view, float fov, float ratio)
        : m_org(org), m_up(up), m_view(view), m_fov(fov), m_ratio(ratio)
    {
        const auto theta = (fov);
        const auto h = tanf(theta / 2.f);
        const auto viewport_height = 2.0f * h;
        const auto viewport_width = ratio * viewport_height;
        
        //complete orthonormal basis
        const Vec3 w = unit_vector(org - view);
        const Vec3 u = unit_vector(cross(up, w));
        const Vec3 v = cross(w, u);
        
        m_x_axis = viewport_width * u;
        m_z_axis = viewport_height * v;
        m_image_plane_org = org - m_x_axis / 2 - m_z_axis / 2 * w;

        /*auto const left = -m_ratio * 0.5f;
        auto const top = 0.5f;
        auto const pi = 3.14159265358979323846f;
        m_dist = 0.5f / tan(m_fov * pi / 360.f);
        m_view = m_view.normalize();
        m_up = m_up.normalize();
        m_x_axis = cross(m_view, m_up).normalize();
        //m_x_axis = m_x_axis.normalize();
        m_z_axis = cross(view, m_x_axis).normalize();
        //m_z_axis = m_z_axis.normalize();
        m_image_plane_org = m_dist * m_view + left * m_x_axis - top * m_z_axis;
        m_x_axis *= m_ratio;*/
    }

    // Generate Ray for pixel located a {x,y} on a image dimension {w,h}
    __device__ Ray generate(float w, float h, float x, float y) const {
        

        return Ray(m_org, m_image_plane_org + x * m_x_axis + y * m_z_axis - m_org);

    }


    Vec3 m_org;
    Vec3 m_up;
    Vec3 m_view;
    Vec3 m_image_plane_org;
    Vec3 m_x_axis;
    Vec3 m_z_axis;
    float m_fov;
    float m_ratio;
    float m_dist;
};
