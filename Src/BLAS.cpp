#include "BLAS.h"

#include "CudaUtils.cuh"
#include "glm/common.hpp"

// Credit to:
// Jacco Bikker
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/


BLAS::BLAS(std::vector<BLASInput>& blas_build_data)
{

	for (auto& data : blas_build_data)
	{
		// Check for expected formats
		assert(data.vertex->GetStride() == sizeof(glm::vec3));
		assert(data.index->GetStride() == sizeof(int));

		temp_cpu_vertices.resize(data.vertex->GetNumElements());
		auto num_primitivess = data.index->GetNumElements() / 3;
		temp_cpu_tri_idx.resize(num_primitivess * 3);

		// Copy Data from GPU to CPU
		checkCudaErrors(cudaMemcpy(temp_cpu_vertices.data(), data.vertex->GetBufferDataPtr(), data.vertex->GetSizeBytes(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(temp_cpu_tri_idx.data(), data.index->GetBufferDataPtr(), data.index->GetSizeBytes(), cudaMemcpyDeviceToHost));

		//m_BvhIndices.resize(num_primitivess);
		//for (int i = 0; i < m_BvhIndices.size(); i++) m_BvhIndices[i] = temp_cpu_tri_idx[i];

		m_Nodes.resize(num_primitivess * 2 - 1);
		m_RootNode = &m_Nodes.front();

		// Add "Calculate Bounds Function"
		m_RootNode->count = num_primitivess;
		m_RootNode->leftFirst = 0;
		m_RootNode->aabb = UpdateNodeBounds(0);
	}

}

AABB BLAS::UpdateNodeBounds(glm::uint nodeIdx)
{
	using namespace glm;
	struct Tri {
		vec3 vertex0;
		vec3 vertex1;
		vec3 vertex2;
	};

	auto& node = m_Nodes[nodeIdx];
	AABB aabb;
	aabb.min = vec3(1e30f);
	aabb.max = vec3(-1e30f);

	for (uint first = node.leftFirst, i = 0; i < node.count; i++)
	{
		const auto& leafTriIdx = temp_cpu_tri_idx[first + i];

		Tri leafTri;
		leafTri.vertex0 = temp_cpu_vertices[leafTriIdx.x];
		leafTri.vertex1 = temp_cpu_vertices[leafTriIdx.y];
		leafTri.vertex2 = temp_cpu_vertices[leafTriIdx.z];

		aabb.min = min(aabb.min, leafTri.vertex0);
		aabb.min = min(aabb.min, leafTri.vertex1);
		aabb.min = min(aabb.min, leafTri.vertex2);
		aabb.max = max(aabb.max, leafTri.vertex0);
		aabb.max = max(aabb.max, leafTri.vertex1);
		aabb.max = max(aabb.max, leafTri.vertex2);
	}

	return aabb;
}
