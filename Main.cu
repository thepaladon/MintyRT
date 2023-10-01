
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "Camera.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ray.cuh"
#include "Vec3.cuh"

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

struct Triangle
{
public:
    Vec3 vertex0;
    Vec3 vertex1;
    Vec3 vertex2;
};

__device__ bool RayIntersectsTriangle(
    Vec3 rayOrigin,
    Vec3 rayVector,
    Triangle* inTriangle,
    Vec3& outIntersectionPoint,
    Vec3& outNormal
)
{
    const Vec3 edge1 = inTriangle->vertex1 - inTriangle->vertex0;
    const Vec3 edge2 = inTriangle->vertex2 - inTriangle->vertex0;
    const Vec3 h = cross(rayVector, edge2);
    const float a = dot(edge1, h);
    if (a > -0.0001f && a < 0.0001f) return false; // ray parallel to triangle

    const float f = 1 / a;
    const Vec3 s = rayOrigin - inTriangle->vertex0;
    const float u = f * dot(s, h);
    if (u < 0 || u > 1) return false;
    const Vec3 q = cross(s, edge1);
    const float v = f * dot(rayVector, q);
    if (v < 0 || u + v > 1) return false ;

	const float t = f * dot(edge2, q);

    if (t > 0.0001f) outIntersectionPoint = t;
    return true;
}


__device__ Vec3 color(const Ray& r) {

    Vec3 v0 = Vec3(0.0f);
    Vec3 v1 = Vec3(-1.f);
    Vec3 v2 = Vec3(-1.f, 0.f, -1.0f);

    Triangle tri{ v0, v1, v2 } ;

    Vec3 point(99000.0f);
    Vec3 normal(420.420f);


    if (RayIntersectsTriangle(r.origin(), r.direction(), &tri, point, normal))
    {
    	return { 1.0f, 0.0f, 0.0f };
    }
    else {
        Vec3 unit_direction = r.direction().normalize();
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);

        return r.direction();
    }
}

__global__ void render(uchar3* fb, int max_x, int max_y, Camera cam) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    Ray r = cam.generate((float)max_x, (float)max_y, u, v);
    fb[pixel_index] = color(r).to_uchar3();
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

    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
	auto end_time = std::chrono::high_resolution_clock::now();
    float run_timer_s = 0.0f;

    // Output FB
    bool running = true;
    while (running)
    {
        // Note, resizing and moving the window won't be caught in DT because it happens in m_Window->Update()
        // This is desired behavior because nobody likes things to jump around 
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta_time_s = end_time - start_time; // in seconds
        run_timer_s += delta_time_s.count();
    	start_time = std::chrono::high_resolution_clock::now();

        static Vec3 view = Vec3(0.0f, -1.0f, -2.f);
    	printf("%f, %f, %f \n", view.x(), view.y(), view.z());

        const float sensitivity = 0.001f;

        view.x() += m_Window->GetMouseDeltaX() * sensitivity;
        view.z() += m_Window->GetMouseDeltaY() * sensitivity;

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

        const Camera cam(Vec3(0.0f, 0.0f, -5.0f), Vec3(0.0f, 1.0f, 0.0f), view, 50.0f, float(alignedX) / float(alignedY));

        // Render our buffer
        dim3 blocks(alignedX / tx + 1, alignedY / ty + 1);
        dim3 threads(tx, ty);
        render << <blocks, threads >> > (
            gpu_fb, 
            alignedX, 
            alignedY, 
            cam
            );
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


