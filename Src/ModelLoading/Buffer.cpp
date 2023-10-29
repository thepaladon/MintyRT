#include "Buffer.h"

namespace bml {

	Buffer::Buffer(const tinygltf::Model& document, const tinygltf::Accessor& accessor, std::string name)
	{
		printf("Add CUDA code for creating a buffer here:  %s\n", name.c_str());
	}

	Buffer::Buffer(const void* data, const size_t stride, const size_t count, const std::string& name)
	{
		printf("Add CUDA code for creating a buffer here:  %s\n", name.c_str());
	}


}