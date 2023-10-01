
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Window.h"



constexpr int FB_WIDTH = 943; 
constexpr int FB_HEIGHT= 540;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(uchar3* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i ;
    fb[pixel_index].z = i * 255 / max_x;
    fb[pixel_index].y = j * 255 / max_y;
    fb[pixel_index].x = 10;
}


int main()
{
	auto* m_Window = new Window(FB_WIDTH, FB_HEIGHT, "Minty Cuda RT");

    uchar3* gpu_fb;
    uchar3* cpu_fb = nullptr;

    uint32_t alignedX = m_Window->GetAlignedWidth();
    uint32_t alignedY = m_Window->GetAlignedHeight();

	// Initial Allocate Frame Buffer
	{
        const int num_pixels = alignedX * alignedY;
        const size_t fb_size = num_pixels * sizeof(uchar3);
        checkCudaErrors(cudaMallocManaged((void**)&gpu_fb, fb_size));
        cpu_fb = new uchar3[num_pixels];
    }


    // Output FB
    bool running = true;
    while (running)
    {
        running = m_Window->OnUpdate();

        if (m_Window->GetIsResized()) {
            m_Window->CreateSampleDIB();

            alignedX = m_Window->GetAlignedWidth();
            alignedY = m_Window->GetAlignedHeight();

        	checkCudaErrors(cudaFree(gpu_fb));
            delete cpu_fb;

            const int num_pixels = alignedX * alignedY;
            const size_t fb_size = num_pixels * sizeof(uchar3);
            checkCudaErrors(cudaMallocManaged((void**)&gpu_fb, fb_size));
            cpu_fb = new uchar3[alignedX * alignedY];

            printf("Resized : %i : %i \n", alignedX, alignedY);
        }

        // Thread Groups
        int tx = 8;
        int ty = 8;

        // Render our buffer
        dim3 blocks(alignedX / tx + 1, alignedY / ty + 1);
        dim3 threads(tx, ty);
        render << <blocks, threads >> > (gpu_fb, alignedX, alignedY);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMemcpy(cpu_fb, gpu_fb, alignedX * alignedY * sizeof(uchar3), cudaMemcpyDeviceToHost);

        m_Window->RenderFb(cpu_fb);
    }


    m_Window->Shutdown();
    delete m_Window;
    delete cpu_fb;
    checkCudaErrors(cudaFree(gpu_fb));

    return 0;
}


