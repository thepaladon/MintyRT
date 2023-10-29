#pragma once
#include <cstdint>

#include "GpuModelStruct.h"

namespace tinygltf
{
	struct Primitive;
}

namespace Ball {
	class Primitive
	{
	public:
		Primitive(const tinygltf::Primitive& primitive);

		uint32_t GetPositionIndex() const { return m_Data.m_PositionIndex; }
		uint32_t GetIndexBufferIndex() const { return m_Data.m_IndexBufferId; }

	private:
		PrimitiveGPU m_Data;
	};

	static_assert(sizeof(Primitive) == sizeof(PrimitiveGPU), "GPU and CPU buffers must match in size"
		" to be uploaded and used correctly on CPU and GPU");

}