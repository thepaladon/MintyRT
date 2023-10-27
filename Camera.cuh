#pragma once
#include <crt/host_defines.h>

#include "Ray.cuh"

#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>

constexpr float PI = 3.14159265359f;

__host__ __device__ float degrees_to_radians(double degrees) {
    return degrees * (PI / 180.0);
}

class Camera {

public:
    __host__ Camera(glm::vec3 org, glm::vec3 pitchYawRoll, float fov, float ratio)
        : m_Pos(org), m_FOV(fov), m_Ratio(ratio), m_PitchYawRoll(pitchYawRoll)
    {
        UpdateCamera();
    }

    __host__ void UpdateCamera()
    {
        // Create a glm::mat4 for the view matrix
        glm::mat4 view_matrix(1.0f);

        // Apply rotations to the view matrix
        view_matrix = rotate(view_matrix, m_PitchYawRoll.y, glm::vec3(0.0f, 1.0f, 0.0f));   // Yaw
        view_matrix = rotate(view_matrix, m_PitchYawRoll.x, glm::vec3(1.0f, 0.0f, 0.0f)); // Pitch
        view_matrix = rotate(view_matrix, m_PitchYawRoll.z, glm::vec3(0.0f, 0.0f, 1.0f));  // Roll

        // Get Direction Vectors from View Matrix
        m_View = glm::normalize(-glm::vec3(view_matrix[2]));	// Negative Z-axis in view matrix
        m_Up = glm::normalize(glm::vec3(view_matrix[1]));		// Y-axis in view matrix

        auto const left = -m_Ratio * 0.5f;
        auto constexpr top = 0.5f;
        auto constexpr pi = 3.14159265358979323846f;
        m_Dist = 0.5f / tan(m_FOV * pi / 360.f);
        m_View = normalize(m_View);
        m_Up = normalize(m_Up);
        m_xAxis = cross(m_View, m_Up);
        m_xAxis = normalize(m_xAxis);
        m_yAxis = cross(m_View, m_xAxis);
        m_yAxis = normalize(m_yAxis);
        m_ImagePlanePos = m_Dist * m_View + left * m_xAxis - top * m_yAxis;
        m_xAxis *= m_Ratio;
    }

    __host__ void MoveFwd(float input)
    {
        m_Pos += -input * m_View * m_MoveScalar;
    }

    __host__ void MoveUp(float input)
    {
        m_Pos += glm::vec3(0.0f, 1.0f, 0.0f) * input * m_MoveScalar;
    }

    __host__ void MoveRight(float input)
    {
        const glm::vec3 right = glm::normalize(glm::cross(m_View, m_Up));
        m_Pos += input * right * m_MoveScalar;

    }

    __host__ void SetPitch(float input)
    {
        m_PitchYawRoll.x += glm::radians(-input * m_ViewScalar);
        m_PitchYawRoll.x = glm::clamp(m_PitchYawRoll.x, -glm::radians(90.0f), glm::radians(90.0f));
    }

    __host__ void SetYaw(float input)
    {
        m_PitchYawRoll.y += glm::radians(-input * m_ViewScalar);
    }


    // Generate Ray for pixel located a {x,y} on a image dimension {w,h}
    __device__ Ray generate(float w, float h, float x, float y) const {

        auto const rw = 1.f / float(w);
        auto const rh = 1.f / float(h);
        auto const sx_axis = m_xAxis * rw;
        auto const sz_axis = m_yAxis * rh;
        auto const dir = m_ImagePlanePos + x * sx_axis + y * sz_axis;
        Ray const r = { m_Pos, dir };
        return r;

    }

	glm::vec3 m_Pos;
    glm::vec3 m_Up;
    glm::vec3 m_View;
    glm::vec3 m_ImagePlanePos;
    glm::vec3 m_xAxis;
    glm::vec3 m_yAxis;
    float m_FOV;
    float m_Ratio;
    float m_Dist;

    glm::vec3 m_PitchYawRoll; // in Radians

    float m_ViewScalar = 2.f;
    float m_MoveScalar = 0.6f;
};
