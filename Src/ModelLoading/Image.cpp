#include "Image.h"

#include <cstdint>

namespace bml {

Image::Image(const tinygltf::Model& model, int index, const std::string& filepath)
{
	printf("Add CUDA code for creating a texture here:  %s\n", filepath.c_str());

}


int32_t Image::CreateTextureWithData(uint8_t* data)
{
	return 0;
}
}



