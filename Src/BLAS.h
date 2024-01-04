#pragma once
#include <vector>

#include "Ray.cuh"
#include "glm/vec3.hpp"
#include "glm/matrix.hpp"
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

	__device__ void IntersectBVH(Ray& ray, glm::uint nodeIdx)
	{
		const auto node = &m_GPUNodes[nodeIdx];
		//printf("Checking Node read data: %i \n", node->leftFirst);

		if (!IntersectAABB(ray, node->aabb.min, node->aabb.max)) return;
		ray.normal.x += 0.1f;

		if (node->isLeaf())
		{
			//printf("Intersecting primitive of: %i \n", nodeIdx);
			ray.hit = true;
			ray.normal.y = 1;

			// for (glm::uint i = 0; i < node->count; i++) {}
			// nada yet //IntersectTri(ray, tri[triIdx[node.firstTriIdx + i]]);
		}
		else
		{
			//printf("Intersecting BVH children of: %i \n", nodeIdx);
			IntersectBVH(ray, node->leftFirst);
			IntersectBVH(ray, node->leftFirst + 1);
		}
	}
};

