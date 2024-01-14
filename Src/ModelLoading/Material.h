#pragma once
#include "GpuModelStruct.h"

namespace tinygltf
{
	class Model;
}

namespace bml {
	// Struct is Used to upload data to GPU Material Buffer as well
	// Total: 64 bytes
	struct Material
	{

		Material(const tinygltf::Model& model, int index);
		Material() = delete;
		~Material() = default;

		// TODO (Would): Support more than 1 Tex Coord
		// TODO (Would): Support Alpha Modes and Cutoff
		// TODO (Would): Support "Double Sided"

		MaterialGPU m_Data;
	};

	static_assert(sizeof(Material) == sizeof(MaterialGPU),
		"The Material class must be exactly 64 bytes in size");
}