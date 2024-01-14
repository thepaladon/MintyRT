#pragma once

// This file is used to specify alloc, dealloc and copy functions so my classes can run
// on both CPU and GPU 

#include "cuda_runtime.h"
#include "CudaUtils.cuh"

// Agnostic CUDA renderer
namespace acr {
#ifdef USE_GPU

    const cudaMemcpyKind cudaMemcpyHostToSpecified = cudaMemcpyHostToDevice;
    const cudaMemcpyKind cudaMemcpySpecifiedTiHost = cudaMemcpyDeviceToHost;

    // GPU allocation and deallocation
    template <typename T>
    inline void  allocate(T** ptr, size_t size) {
        //Not sure I need to cast it to void**
        checkCudaErrors(cudaMalloc((void**)ptr, size));
    }

    inline void deallocate(void* ptr) {
        checkCudaErrors(cudaFree(ptr));
    }

#else

    const cudaMemcpyKind cudaMemcpyHostToSpecified = cudaMemcpyHostToHost;
    const cudaMemcpyKind cudaMemcpySpecifiedTiHost = cudaMemcpyHostToHost;

    // CPU allocation and deallocation
    template <typename T>
    inline void allocate(T** ptr, size_t size) {
        *ptr = static_cast<T*>(malloc(size));
    }

    inline void deallocate(void* ptr) {
        free(ptr);
    }


#endif

}
