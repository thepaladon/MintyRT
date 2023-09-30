
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Window.h"

constexpr int FB_WIDTH = 800; 
constexpr int FB_HEIGHT= 420;

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
    fb[pixel_index].x = i * 255 / max_x;
    fb[pixel_index].y = j * 255 / max_y;
    fb[pixel_index].z = 10;
}


int main()
{
	Window* m_Window = new Window(FB_WIDTH, FB_HEIGHT, "Cuda RT");

	const int num_pixels = FB_WIDTH * FB_HEIGHT;
	const size_t fb_size = num_pixels * sizeof(uchar3);

    // allocate FB
    uchar3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Thread Groups
    int tx = 8;
    int ty = 8;


    // Render our buffer
    dim3 blocks(FB_WIDTH / tx + 1, FB_HEIGHT / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads>>> (fb, FB_WIDTH, FB_HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    uchar3* h_framebuffer = new uchar3[FB_WIDTH * FB_HEIGHT];
    cudaMemcpy(h_framebuffer, fb, FB_WIDTH * FB_HEIGHT * sizeof(uchar3), cudaMemcpyDeviceToHost);


    // Output FB
    std::cout << "P3\n" << FB_WIDTH << " " << FB_HEIGHT << "\n255\n";
    for (int j = FB_HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < FB_WIDTH; i++) {
            const size_t pixel_index = j * FB_WIDTH + i;
            const uint8_t r = fb[pixel_index].x;
            const uint8_t g = fb[pixel_index].y;
            const uint8_t b = fb[pixel_index].z;
            //printf("Value: %hhu  %hhu %hhu \n", r, g, b);
        }
    }

    bool running = true;
    while (running)
    {
        running = m_Window->OnUpdate(h_framebuffer);
    }

    //checkCudaErrors(cudaFree(fb));


    return 0;
}


