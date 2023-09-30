
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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


__global__ void render(float* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}


int main()
{
	const int num_pixels = FB_WIDTH * FB_HEIGHT;
	const size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate FB
    float* fb;
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


    // Output FB
    // Output FB as Image
    std::cout << "P3\n" << FB_WIDTH << " " << FB_HEIGHT << "\n255\n";
    for (int j = FB_HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < FB_WIDTH; i++) {
            size_t pixel_index = j * 3 * FB_WIDTH + i * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));


    return 0;
}


