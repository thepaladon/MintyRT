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

	glm::vec3 centroid() const { return (max + min) / 2.f;  }
};

struct BLASNode
{
	AABB aabb;
	int leftFirst;
	int count;

	bool isLeaf() { return count > 0; }
};

class BLAS
{
public:
	// Only works with 1 BLASInput Currently
	BLAS(std::vector<BLASInput>& blas_build_data);
	~BLAS() = default;

private:

	void Subdivide(glm::uint nodeIdx);
	void PrecomputeAABB(std::vector<AABB>& temp_aabb, const std::vector<glm::vec3>& vertices, const  std::vector<glm::ivec3>& indices);
	void IntersectBVH(Ray& ray, glm::uint nodeIdx);

	int m_NodesUsed = 0;
	BLASNode* m_RootNode;
	std::vector<BLASNode> m_Nodes;
	std::vector<int> m_TriIndices;

	//Temporal CPU Data
	std::vector<AABB> temp_aabb;
	std::vector<glm::vec3> temp_cpu_vertices;
	std::vector<glm::ivec3> temp_cpu_triangle_idx;

};

