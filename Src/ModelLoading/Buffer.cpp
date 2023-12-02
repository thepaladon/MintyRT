#include "Buffer.h"

#include <cuda_runtime.h>
#include <TinyglTF/tiny_gltf.h>
#include "../CudaUtils.cuh"

namespace bml {

	Buffer::Buffer(const tinygltf::Model& document, const tinygltf::Accessor& accessor, const std::string& name)
	{
        printf("Created CUDA Buffer:  %s\n", name.c_str());

        m_Name = name;
        const auto& view = document.bufferViews[accessor.bufferView];
        const auto& buffer = document.buffers[view.buffer];

        const size_t comp_size_in_bytes  = tinygltf::GetComponentSizeInBytes(accessor.componentType);
        const size_t num_elements_in_stride = tinygltf::GetNumComponentsInType(accessor.type);
        const size_t stride = comp_size_in_bytes * num_elements_in_stride;
        const size_t total_size_in_bytes  = stride * accessor.count;

        m_Stride = stride;
        m_NumElements = accessor.count;
		m_SizeBytes = total_size_in_bytes;

        checkCudaErrors(cudaMalloc(&m_BufferHandle, m_SizeBytes));
        checkCudaErrors(cudaMemcpy(m_BufferHandle, &buffer.data.at(view.byteOffset + accessor.byteOffset), m_SizeBytes, cudaMemcpyHostToDevice));

    }

	Buffer::Buffer(const void* data, const size_t stride, const size_t count, const std::string& name)
	{

        m_Name = name;
        m_Stride = static_cast<uint32_t>(stride);
        m_NumElements = static_cast<uint32_t>(count);
        m_SizeBytes = m_Stride * m_NumElements;

        // Allocate a CUDA buffer and copy data to it
        cudaMalloc(&m_BufferHandle, m_SizeBytes);
        cudaMemcpy(m_BufferHandle, data, m_SizeBytes, cudaMemcpyHostToDevice);

        printf("Created CUDA Buffer:  %s\n", name.c_str());

    }

    const void* Buffer::GetBufferDataPtr() const {
        return m_BufferHandle;
    }

    uint32_t Buffer::GetNumElements() const {
        return m_NumElements;
    }

    uint32_t Buffer::GetStride() const {
        return m_Stride;
    }

    uint32_t Buffer::GetSizeBytes() const {
        return m_SizeBytes;
    }

}
