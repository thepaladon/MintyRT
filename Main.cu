// GLM Defines
#define CUDA_VERSION 12020
#define GLM_FORCE_CUDA

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "Camera.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ray.cuh"
#include "Utils.h"

#include "Window.h"


constexpr int FB_WIDTH = 1200; 
constexpr int FB_HEIGHT= 800;

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
    glm::vec3 vertex0;
    glm::vec3 vertex1;
    glm::vec3 vertex2;
};

 __device__ bool intersect_tri(Ray& ray, const Triangle* tris, glm::uint triIdx)
{
    const glm::vec3 edge1 = tris[triIdx].vertex1 - tris[triIdx].vertex0;
    const glm::vec3 edge2 = tris[triIdx].vertex2 - tris[triIdx].vertex0;
    const glm::vec3 h = cross(ray.d, edge2);
    const float a = dot(edge1, h);
    if (fabs(a) < 0.0001) return false; // ray parallel to triangle
    const float f = 1 / a;
    const glm::vec3 s = ray.o - tris[triIdx].vertex0;
    const float u = f * dot(s, h);
    if (u < 0 || u > 1) return false;
    const glm::vec3 q = cross(s, edge1);
    const float v = f * dot(ray.d, q);
    if (v < 0 || u + v > 1) return false;
    const float t = f * dot(edge2, q);
    if (t > 0.0001f)
    if (ray.t > t)
    {
        ray.t = t;
        //ray->intersection.tri_hit = triIdx;
        //ray->intersection.u = u;
        //ray->intersection.v = v;
        //ray->intersection.header_tri_count = header[0].tris_count;
        //ray->intersection.geo_normal = cross(edge1, edge2);
    }
    return true;
}






__device__ glm::vec3 color(Ray& r) {

    glm::vec3 v0 = glm::vec3(0.0f, 1.0f, 1.0f);
    glm::vec3 v1 = glm::vec3(.5f, 0.0f, 1.0f);
    glm::vec3 v2 = glm::vec3(-.5f, 0.0f, 1.0f);

    Triangle tri{ v0, v1, v2 } ;

    glm::vec3 point(99000.0f);
    glm::vec3 normal(420.420f);
    float t;


    if (intersect_tri(r, &tri, 0))
    {
    	return { 1.0f, 0.0f, 0.0f };
    }
    else {
        glm::vec3 unit_direction = normalize(r.direction());
        float t = 0.5f * (unit_direction.y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);

    }
}

__global__ void render(uchar3* fb, int max_x, int max_y, Camera cam) {

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    Ray r = cam.generate((float)max_x, (float)max_y, (float)i, (float)j);
    fb[pixel_index] = to_uchar3(color(r));
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

	Camera cam(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, glm::radians(-224.f), 0.0f), 75.f, float(alignedX) / float(alignedY));

    while (running)
    {
        // Note, resizing and moving the window won't be caught in DT because it happens in m_Window->Update()
        // This is desired behavior because nobody likes things to jump around 
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta_time_s = end_time - start_time; // in seconds
        run_timer_s += delta_time_s.count();
    	start_time = std::chrono::high_resolution_clock::now();


        const float sensitivity = .5f;

        float hor_inp = 0;
        float ver_inp = 0;
        if (m_Window->GetKey(VK_LEFT)) { hor_inp = 1.0; }
        if (m_Window->GetKey(VK_RIGHT)) { hor_inp = -1.0; }
        if (m_Window->GetKey(VK_UP)) { ver_inp = 1.0; }
        if (m_Window->GetKey(VK_DOWN)) { ver_inp = -1.0; }


        float m_dtx = hor_inp ;// m_Window->GetMouseDeltaX();
        float m_dty = ver_inp ;// m_Window->GetMouseDeltaY();


        if (m_Window->GetKey('W'))
        {
            cam.MoveFwd(1.0f);
        }

        if (m_Window->GetKey('S'))
        {
            cam.MoveFwd(-1.0f);
        }

        if (m_Window->GetKey('D'))
        {
            cam.MoveRight(1.0f);
        }

        if (m_Window->GetKey('A'))
        {
            cam.MoveRight(-1.0f);
        }

        if (m_Window->GetKey('R'))
        {
            cam.MoveUp(1.0f);
        }

    	if (m_Window->GetKey('F'))
        {
            cam.MoveUp(-1.0f);
        }

        cam.SetPitch(m_dty);
        cam.SetYaw(m_dtx);
    	cam.UpdateCamera();

        //printf(" %f          %f \n", m_dtx, m_dty );

        
        printf("Pos - X: %f, Y: %f, Z : %f \n", cam.m_Pos.x, cam.m_Pos.y, cam.m_Pos.z );

        auto rad = glm::degrees(cam.m_PitchYawRoll);
    	printf("Pitch: %f, Yaw: %f, Roll: %f \n \n", rad.x, rad.y, rad.z );

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


