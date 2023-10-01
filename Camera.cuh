#pragma once
#include <crt/host_defines.h>

#include "Ray.cuh"
#include "Vec3.cuh"


class Camera {

public:
    __host__ __device__ Camera(Vec3 org, Vec3 up, Vec3 view, float fov, float ratio)
        : m_org(org), m_up(up), m_view(view), m_fov(fov), m_ratio(ratio)
    {
        auto const left = -m_ratio * 0.5f;
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
        m_x_axis *= m_ratio;
    }

    // Generate Ray for pixel located a {x,y} on a image dimension {w,h}
    __device__ Ray generate(float w, float h, float x, float y) const {
        
        const float rw = 1.0f / w;
        const float rh = 1.0f / h;
        const Vec3 sx_axis = m_x_axis * rw;
        const Vec3 sz_axis = m_z_axis * rh;
        const Vec3 dir = m_image_plane_org + x * sx_axis + y * sz_axis;
        const Ray r = { m_org, dir };
        return r;
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
