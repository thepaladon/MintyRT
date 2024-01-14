// GLM Defines
#define CUDA_VERSION 12020
#define GLM_FORCE_CUDA
#define CUDA_LAUNCH_BLOCKING = 1

#include <chrono>
#include <cstdlib>
#include <thread>

#include "BLAS.h"
#include "device_launch_parameters.h"
#include "Ray.cuh"

#include "Window.h"
#include "ModelLoading/Buffer.h"
#include "ModelLoading/Model.h"

#include "Camera.cuh"
#include "CudaUtils.cuh"

constexpr int FB_INIT_WIDTH = 1200; 
constexpr int FB_INIT_HEIGHT= 800;

#define MODEL_FP(model) (std::string("Resources/Models/") + model + "/" + model + ".gltf")

struct Square {
    int minX, minY; // Top-left corner
    int maxX, maxY; // Bottom-right corner
};

std::vector<Square> divideScreenIntoSquares(int screenWidth, int screenHeight, int squareSize = 32) {
    std::vector<Square> squares;

    for (int y = 0; y < screenHeight; y += squareSize) {
        for (int x = 0; x < screenWidth; x += squareSize) {
            squares.push_back({ x, y, x + squareSize - 1, y + squareSize - 1 });
        }
    }

    return squares;
}

__host__ __device__ glm::vec3 color(Ray& r, BLAS blas, GPUTriData model) {

	blas.IntersectBVH(r, 0, model);

    //printf("[2] Index Test %p, %i \n", model.index_buffer, model.index_buffer[0]);
    //printf("[2] Vertex Test %p, %f \n", model.vertex_buffer, model.vertex_buffer[0]);

    /*for (int i = 0; i < 12; i++)
    {
        const auto& i0 = model.index_buffer[i * 3 + 0];
        const auto& i1 = model.index_buffer[i * 3 + 1];
        const auto& i2 = model.index_buffer[i * 3 + 2];
        
        const auto& v0x = model.vertex_buffer[i0 * 3 + 0];
        const auto& v0y = model.vertex_buffer[i0 * 3 + 1];
        const auto& v0z = model.vertex_buffer[i0 * 3 + 2];
    	const auto& v1x = model.vertex_buffer[i1 * 3 + 0];
        const auto& v1y = model.vertex_buffer[i1 * 3 + 1];
        const auto& v1z = model.vertex_buffer[i1 * 3 + 2];
        const auto& v2x = model.vertex_buffer[i2 * 3 + 0];
        const auto& v2y = model.vertex_buffer[i2 * 3 + 1];
        const auto& v2z = model.vertex_buffer[i2 * 3 + 2];

        const glm::vec3& v0 = glm::vec3(v0x, v0y, v0z);
        const glm::vec3& v1 = glm::vec3(v1x, v1y, v1z);
        const glm::vec3& v2 = glm::vec3(v2x, v2y, v2z);
        
        Triangle tri{ v0, v1, v2 };

        intersect_tri(r, tri);
    }*/

    if(r.hit == true)
    {
        return { r.normal * glm::vec3(0.5) + 0.5f } ;
    }

    const glm::vec3 unit_direction = normalize(r.dir);
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(uchar3* fb, int max_x, int max_y, Camera cam, BLAS blas, GPUTriData model) {

    const int u = threadIdx.x + blockIdx.x * blockDim.x;
    const int v = threadIdx.y + blockIdx.y * blockDim.y;
    if ((u >= max_x) || (v >= max_y)) return;

    int pixel_index = v * max_x + u;
 	Ray r = cam.generate((float)max_x, (float)max_y, (float)u, (float)v);
    fb[pixel_index] = to_uchar3(color(r, blas, model));
}


void CPURender(uchar3* fb, glm::ivec2 min, glm::ivec2 max, glm::ivec2 total, Camera cam, BLAS blas, GPUTriData model)
{
    for (int u = min.x; u <= max.x; u++) {
        for (int v = min.y; v <= max.y ; v++) 
        {
            if ((u >= total.x) || (v >= total.y)) return;
        	int pixel_index = v * total.x + u;
        	Ray r = cam.generate((float)total.x, (float)total.y, (float)u, (float)v);
            fb[pixel_index] = to_uchar3(color(r, blas, model));
        }
    }
}

void CPURenderOpenMP(uchar3* fb,  glm::ivec2 total, Camera cam, BLAS blas, GPUTriData model)
{
	for (int i = 0; i <= total.x * total.y ;i++) 
    {
        const int u = i % total.x;
        const int v = i / total.x;
        Ray r = cam.generate((float)total.x, (float)total.y, (float)u, (float)v);
        fb[i] = to_uchar3(color(r, blas, model));
    }
}


int main()
{
	auto* m_Window = new Window(FB_INIT_WIDTH, FB_INIT_HEIGHT, "Minty Cuda RT");

    uchar3* gpu_fb;
    uchar3* cpu_fb = nullptr;

    uint32_t alignedX = m_Window->GetAlignedWidth();
    uint32_t alignedY = m_Window->GetAlignedHeight();
    std::vector<Square> screenSquares = divideScreenIntoSquares(alignedX, alignedY, 150);

	// Initial Allocate Frame Buffer
	{
        const int num_pixels = alignedX * alignedY;
        const size_t fb_size = num_pixels * sizeof(uchar3);
        checkCudaErrors(cudaMallocManaged((void**)&gpu_fb, fb_size));
        cpu_fb = new uchar3[num_pixels];
    }

    // MAKE SURE TO CHANGE 'index' and 'vertex' to the same 
    // as the ones specified in the glTF spec below!!!!!!!!!!!!
	// const auto model = new bml::Model(MODEL_FP("CesiumMilkTruck"));
	// const auto model = new bml::Model(MODEL_FP("Cube"));
	// const auto model = new bml::Model(MODEL_FP("DamagedHelmet"));
	const auto model = new bml::Model(MODEL_FP("sah_test"));
	// const auto model = new bml::Model(MODEL_FP("SciFiHelmet"));

    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
	auto end_time = std::chrono::high_resolution_clock::now();
    float run_timer_s = 0.0f;

    // Output FB
    bool running = true;

	Camera cam(glm::vec3(0.0f, 0.0f, -15.0f), glm::vec3(0.0f, glm::radians(-224.f), 0.0f), 75.f, float(alignedX) / float(alignedY));

    auto vertex = model->GetBuffers()[0];
    auto index = model->GetBuffers()[3];

    BLASInput blasInput;
    blasInput.vertex = vertex;
    blasInput.index = index;
    blasInput.transform = glm::identity<glm::mat4>();
    std::vector<BLASInput> blastestis;
	blastestis.push_back(blasInput);

    BLAS blastest(blastestis);

	const GPUTriData model_data {
        (const float*)vertex->GetBufferDataPtr(),
        (const unsigned*)index->GetBufferDataPtr(),
    };

    // Make sure everything is available before start of Render
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    while (running)
    {
        // Note, resizing and moving the window won't be caught in DT because it happens in m_Window->Update()
        // This is desired behavior because nobody likes things to jump around 
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta_time_s = end_time - start_time; // in seconds
        run_timer_s += delta_time_s.count();
        float delta_time_ms = delta_time_s.count() * 1000;
    	start_time = std::chrono::high_resolution_clock::now();

        //Camera
        cam.dt = delta_time_ms;
        CameraInput(cam, m_Window);
    	cam.UpdateCamera();


        // Update & Resize
        running = m_Window->OnUpdate(delta_time_ms);
        if (m_Window->GetIsResized()) {
            m_Window->CreateSampleDIB();

            alignedX = m_Window->GetAlignedWidth();
            alignedY = m_Window->GetAlignedHeight();
            screenSquares = divideScreenIntoSquares(alignedX, alignedY, 150);

        	checkCudaErrors(cudaFree(gpu_fb));
            delete cpu_fb;

            const int num_pixels = alignedX * alignedY;
            const size_t fb_size = num_pixels * sizeof(uchar3);
            checkCudaErrors(cudaMallocManaged((void**)&gpu_fb, fb_size));
            cpu_fb = new uchar3[alignedX * alignedY];

            printf("Resized : %i : %i \n", alignedX, alignedY);
        }

        // Render
#ifdef USE_GPU
        // Thread Groups
        constexpr int tx = 8;
        constexpr int ty = 8;

        // Render our buffer
        const dim3 blocks(alignedX / tx + 1, alignedY / ty + 1);
        const dim3 threads(tx, ty);


    	render << < blocks, threads >> > (
            gpu_fb,
            alignedX,
            alignedY,
            cam,
            blastest,
            model_data
        );

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMemcpy(cpu_fb, gpu_fb, alignedX * alignedY * sizeof(uchar3), cudaMemcpyDeviceToHost);
        
#else
        glm::ivec2 total = { alignedX, alignedY };

        constexpr bool shittyMultithreading = true;
        if (shittyMultithreading) {
            // Worst multithreading on the planet,
            // but aye it works
            std::vector<std::thread> threads;

            for (const auto& group : screenSquares) {
                glm::ivec2 min = { group.minX, group.minY };
                glm::ivec2 max = { group.maxX, group.maxY };

                threads.emplace_back(CPURender, cpu_fb,
                    min,
                    max,
                    total,
                    cam,
                    blastest,
                    model_data);
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }
        else {
            CPURenderOpenMP(cpu_fb,
                total,
                cam,
                blastest,
                model_data);
        }
#endif

        // Copy to FB (End Frame)
        m_Window->RenderFb(cpu_fb);
    }

    m_Window->Shutdown();
    delete m_Window;
    delete cpu_fb;
    checkCudaErrors(cudaFree(gpu_fb));

    return 0;
}


