#include "Buffer.h"

#include <cuda_runtime.h>

namespace bml {

	Buffer::Buffer(const tinygltf::Model& document, const tinygltf::Accessor& accessor, std::string name)
	{
        printf("Add CUDA code for creating a buffer here:  %s\n", name.c_str());
    }

	Buffer::Buffer(const void* data, const size_t stride, const size_t count, const std::string& name)
	{
        printf("Add CUDA code for creating a buffer here:  %s\n", name.c_str());
        m_Name = name;
        m_Stride = static_cast<uint32_t>(stride);
        m_NumElements = static_cast<uint32_t>(count);
        m_SizeBytes = m_Stride * m_NumElements;

        // Allocate a CUDA buffer and copy data to it
        cudaMalloc(&m_BufferHandle, m_SizeBytes);
        cudaMemcpy(m_BufferHandle, data, m_SizeBytes, cudaMemcpyHostToDevice);
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
