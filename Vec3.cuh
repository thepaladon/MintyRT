#pragma once
#include <cmath>
#include <crt/host_defines.h>

struct Vec3 {
    float x, y, z;

    __device__ __host__ Vec3() : x(0), y(0), z(0) {}
    __device__ __host__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __device__ __host__ Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    __device__ __host__ Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    __device__ __host__ Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    __device__ __host__ Vec3 operator/(float scalar) const {
        float invScalar = 1.0f / scalar;
        return Vec3(x * invScalar, y * invScalar, z * invScalar);
    }

    __device__ __host__ float length() const {
        return sqrt(x * x + y * y + z * z);
    }

    __device__ __host__ float squared_length() const {
        return x * x + y * y + z * z;
    }
};


