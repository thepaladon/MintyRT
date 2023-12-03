#include "Buffer.h"

#include <cuda_runtime.h>
#include <TinyglTF/tiny_gltf.h>
#include "../CudaUtils.cuh"

namespace bml {

    // Indicies are one after another
    // We group them by 3 manually
    template <typename T>
    std::vector<uint32_t> ConvertTo32BitIndices(const T* indices, size_t numIndices) {
        std::vector<uint32_t> intIndices;
        intIndices.reserve(numIndices);

        for (size_t i = 0; i < numIndices; ++i) {
            intIndices.push_back(static_cast<uint32_t>(indices[i]));
        }

        return intIndices;
    }

    
	Buffer::Buffer(const tinygltf::Model& document, const tinygltf::Accessor& accessor, const std::string& name)
	{

        m_Name = name;
        const auto& view = document.bufferViews[accessor.bufferView];
        const auto& buffer = document.buffers[view.buffer];

        const size_t comp_size_in_bytes  = tinygltf::GetComponentSizeInBytes(accessor.componentType);
        const size_t num_elements_in_stride = tinygltf::GetNumComponentsInType(accessor.type);

        // Dumb assumption for now
    	//assert(comp_size_in_bytes == 4);

    	const size_t stride = 4 * num_elements_in_stride;
        const size_t total_size_in_bytes  = stride * accessor.count;

        const void* data_loc = &buffer.data.at(view.byteOffset + accessor.byteOffset);

        m_Stride = stride;
        m_NumElements = accessor.count;
		m_SizeBytes = total_size_in_bytes;

        
        checkCudaErrors(cudaMalloc(&m_BufferHandle, m_SizeBytes));

        // If buffer is not sizeof(uint32_t) then convert it to that
        if (comp_size_in_bytes == 2)
        {
            printf("[Warning]: Resizing Index Buffer from uint16_t to uint32_t! \n");
            const auto u32_from_u16 = ConvertTo32BitIndices((uint16_t*)data_loc, accessor.count);
            checkCudaErrors(cudaMemcpy(m_BufferHandle, u32_from_u16.data(), m_SizeBytes, cudaMemcpyHostToDevice));
        }
        else if (comp_size_in_bytes == 1)
        {
            printf("[Warning]: Resizing Index Buffer from uint8_t to uint32_t! \n");
            const auto u32_from_u8 = ConvertTo32BitIndices((uint8_t*)data_loc, accessor.count);
            checkCudaErrors(cudaMemcpy(m_BufferHandle, u32_from_u8.data(), m_SizeBytes, cudaMemcpyHostToDevice));
        }
        else {
            checkCudaErrors(cudaMemcpy(m_BufferHandle, data_loc, m_SizeBytes, cudaMemcpyHostToDevice));
        }

        printf("Created CUDA Buffer:  %s\n", name.c_str());
        printf("    - GPU Location : 0x%p \n", m_BufferHandle);
        printf("    - Stride : %llu \n", m_Stride);
        printf("    - Elements : %llu \n", m_NumElements);
        printf("    - Size : %llu \n \n", m_SizeBytes);

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

    size_t  Buffer::GetNumElements() const {
        return m_NumElements;
    }

    size_t  Buffer::GetStride() const {
        return m_Stride;
    }

    size_t  Buffer::GetSizeBytes() const {
        return m_SizeBytes;
    }

}
