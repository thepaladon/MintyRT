#pragma once
#include <string>

namespace tinygltf
{
	class Model;
}

namespace bml {
	struct Image
	{
		Image() = delete;

		//Constructing a texture from TinyGLTF model
		Image(const tinygltf::Model& model, int index, const std::string& filepath);

		//ToDo: Constructing a texture from raw data

		//ToDo: Texture GPU Deallocation

		int32_t CreateTextureWithData(uint8_t* data);
		void* GetGPUHandle() const { return m_texture; }

		void* m_texture = {};
		int m_width = -1;
		int m_height = -1;
		int m_channels = -1;
	};

}