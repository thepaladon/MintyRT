#pragma once
#include <vector>

#include "glm/vec3.hpp"
#include "glm/matrix.hpp"
#include "CudaUtils.cuh"
#include "Ray.cuh"
#include "ModelLoading/Buffer.h"

struct BLASInput
{
	bml::Buffer* vertex;
	bml::Buffer* index;
	glm::mat4 transform;
};

struct AABB
{
	glm::vec3 min;
	glm::vec3 max;
	__host__ glm::vec3 centroid() const { return (max + min) / 2.f;  }
};

struct BLASNode
{
	AABB aabb;
	int leftFirst;
	int count;
	__device__ bool isLeaf() { return count > 0; }
};


class BLAS
{
public:
	// Only works with 1 BLASInput Currently
	__host__ BLAS(std::vector<BLASInput>& blas_build_data);
	~BLAS() = default;
	

private:

	__host__ void Subdivide(glm::uint nodeIdx, std::vector<BLASNode>& out_nodes, const std::vector<AABB>& in_temp_aabb, std::vector<int>&
	                        in_tri_indices);
	__host__ void PrecomputeAABB(std::vector<AABB>& temp_aabb, const std::vector<glm::vec3>& vertices, const  std::vector<glm::ivec3>& indices);

	int m_NodesUsed = 0;
	BLASNode* m_GPUNodes;
	int* m_GPUSortedTriIndices;

public:
	__device__ bool IntersectAABB(const Ray& ray, const glm::vec3 bmin, const glm::vec3 bmax)
	{
		float tx1 = (bmin.x - ray.org.x) / ray.dir.x, tx2 = (bmax.x - ray.org.x) / ray.dir.x;
		float tmin = glm::min(tx1, tx2), tmax = glm::max(tx1, tx2);
		float ty1 = (bmin.y - ray.org.y) / ray.dir.y, ty2 = (bmax.y - ray.org.y) / ray.dir.y;
		tmin = glm::max(tmin, glm::min(ty1, ty2)), tmax = glm::min(tmax, glm::max(ty1, ty2));
		float tz1 = (bmin.z - ray.org.z) / ray.dir.z, tz2 = (bmax.z - ray.org.z) / ray.dir.z;
		tmin = glm::max(tmin, glm::min(tz1, tz2)), tmax = glm::min(tmax, glm::max(tz1, tz2));
		return tmax >= tmin && tmin < ray.t && tmax > 0;
	}

	// ISSUE IS MOST LIKELY NOT HERE BUT IN THE BVH CREATION 
	__device__ void IntersectBVH(Ray& ray, glm::uint nodeIdx, GPUTriData model)
	{
		const auto node = &m_GPUNodes[nodeIdx];
		//printf("Checking Node read data: %i \n", node->leftFirst);

		if (!IntersectAABB(ray, node->aabb.min, node->aabb.max)) return;
		ray.hit = true;
		ray.normal.x += 0.1f;

		if (node->isLeaf())
		{
			//printf("Intersecting primitive of: %i \n", nodeIdx);
			ray.hit = true;
			ray.normal.y = 1;


			// ISSUE IS MOST LIKELY NOT HERE BUT IN THE BVH CREATION 
			//for (int i = 0; i < 120; i++)
			for (glm::uint i = 0; i < node->count; i++)
			{
				//printf("What the fuck is going on? \n" );

				const auto triIdx = m_GPUSortedTriIndices[node->leftFirst + i];
				//printf("Triangle Idx: %i \n", triIdx);

				const auto& i0 = model.index_buffer[triIdx * 3 + 0];
				const auto& i1 = model.index_buffer[triIdx * 3 + 1];
				const auto& i2 = model.index_buffer[triIdx * 3 + 2];

				//printf("Vertex Idx: %i,  %i, %i \n", i0, i1, i2);
				const auto& v0x = model.vertex_buffer[i0 * 3 + 0];
				const auto& v0y = model.vertex_buffer[i0 * 3 + 1];
				const auto& v0z = model.vertex_buffer[i0 * 3 + 2];
				
				const auto& v1x = model.vertex_buffer[i1 * 3 + 0];
				const auto& v1y = model.vertex_buffer[i1 * 3 + 1];
				const auto& v1z = model.vertex_buffer[i1 * 3 + 2];
				
				const auto& v2x = model.vertex_buffer[i2 * 3 + 0];
				const auto& v2y = model.vertex_buffer[i2 * 3 + 1];
				const auto& v2z = model.vertex_buffer[i2 * 3 + 2];

				printf("Vertex Idx: %f, %f, %f \n", v2x, v2y, v2z);


				//const glm::vec3& v0 = glm::vec3(v0x, v0y, v0z);
				//const glm::vec3& v1 = glm::vec3(v1x, v1y, v1z);
				//const glm::vec3& v2 = glm::vec3(v2x, v2y, v2z);
				//Triangle tri{ v0, v1, v2 };

				//intersect_tri(ray, tri);
			}

		}
		else
		{
			//printf("Intersecting BVH children of: %i \n", nodeIdx);
			IntersectBVH(ray, node->leftFirst, model);
			IntersectBVH(ray, node->leftFirst + 1, model);
		}
	}
};


