#pragma once
#include <string>

#include <cuda_runtime_api.h>

namespace tinygltf
{
	struct Accessor;
	class Model;
}

namespace bml {
	enum class BufferType;
	enum class ComponentType;

	class Buffer
	{

	public:
		Buffer() = delete;

		//Constructing a Buffer from CPU Data
		Buffer(const void* data, const size_t stride, const size_t count, const std::string& name);

		//Constructing a Buffer from TinyGLTF Accessor
		Buffer(const tinygltf::Model& document, const tinygltf::Accessor& accessor, const std::string& name);

		// Todo: Handle Buffer Deallocation

		const void* GetBufferDataPtr() const;
		uint32_t GetNumElements() const;
		uint32_t GetStride() const;			// in Bytes
		uint32_t GetSizeBytes() const;		// in Bytes
		std::string GetName() const { return m_Name; }
		void* GetGPUHandle() const { return m_BufferHandle; }
	private:

		void* m_BufferHandle;
		std::string m_Name = "DEFAULT_NAME_FOR_BUFFER";
		uint32_t m_Stride = -1;		 // in Bytes
		uint32_t m_NumElements = 0;
		uint32_t m_SizeBytes = 0;

	};


}
