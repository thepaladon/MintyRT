#pragma once
#include <vector>

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "ModelLoading/Buffer.h"

struct BLASInput
{
	bml::Buffer* vertex;
	bml::Buffer* index;
	glm::vec4 transform;
};

struct AABB
{
	glm::vec3 min;
	glm::vec3 max;
};

struct BLASNode
{
	AABB aabb;
	int leftFirst;
	int count;
};

class BLAS
{
	BLAS(std::vector<BLASInput>& blas_build_data);
	AABB UpdateNodeBounds(glm::uint nodeIdx);
	~BLAS();

private:
	BLASNode* m_RootNode;
	std::vector<BLASNode> m_Nodes;
	std::vector<int> m_BvhIndices;

	//Temporal CPU Data
	std::vector<glm::vec3> temp_cpu_vertices;
	std::vector<glm::ivec3> temp_cpu_tri_idx;

};

