#pragma once
#include "glm/vec3.hpp"

template <typename T>
T alignUp(T value, T alignment) {
	if (alignment == 0) {
		// Avoid division by zero; you may want to handle this differently depending on your needs
		return value;
	}

	T remainder = value % alignment;

	if (remainder != 0) {
		return value + (alignment - remainder);
	}

	return value;
}

__host__ __device__ inline uchar3 to_uchar3(glm::vec3 val)
{
	uchar3 rgb;
	rgb.z = static_cast<unsigned char>(val[0] * 255.99f);
	rgb.y = static_cast<unsigned char>(val[1] * 255.99f);
	rgb.x = static_cast<unsigned char>(val[2] * 255.99f);
	return rgb;
}