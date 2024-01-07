#include "BLAS.h"

#include <algorithm>

#include "CudaPicker.h"

#include "Ray.cuh"
#include "glm/common.hpp"

// Credit to:
// Jacco Bikker
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

AABB combineAABBs(std::vector<AABB>::const_iterator startIt, std::vector<AABB>::const_iterator endIt)
{
	AABB combined;
	combined.min = glm::vec3(std::numeric_limits<float>::max());
	combined.max = glm::vec3(-std::numeric_limits<float>::max());

	for (auto it = startIt; it != endIt; ++it)
	{
		const AABB& aabb = *it;
		combined.min.x = std::min(combined.min.x, aabb.min.x);
		combined.min.y = std::min(combined.min.y, aabb.min.y);
		combined.min.z = std::min(combined.min.z, aabb.min.z);

		combined.max.x = std::max(combined.max.x, aabb.max.x);
		combined.max.y = std::max(combined.max.y, aabb.max.y);
		combined.max.z = std::max(combined.max.z, aabb.max.z);
	}

	return combined;
}


BLAS::BLAS(std::vector<BLASInput>& blas_build_data)
{
	// Works with only one BLAS for the time being
	assert(blas_build_data.size() == 1);

	std::vector<glm::vec3> temp_cpu_vertices;
	std::vector<glm::ivec3> temp_cpu_triangle_idx;
	std::vector<AABB> temp_aabb;
	std::vector<int> tri_indices;
	std::vector<BLASNode> nodes;

	for (auto& data : blas_build_data)
	{

		// Check for expected formats
		assert(data.vertex->GetStride() == sizeof(glm::vec3));
		assert(data.index->GetStride() == sizeof(int));

		
		temp_cpu_vertices.resize(data.vertex->GetNumElements());
		auto num_primitivess = data.index->GetNumElements() / 3;

		temp_cpu_triangle_idx.resize(num_primitivess);

		// Copy Data from GPU to CPU
		checkCudaErrors(cudaMemcpy(temp_cpu_vertices.data(), data.vertex->GetBufferDataPtr(), data.vertex->GetSizeBytes(),	 acr::cudaMemcpySpecifiedTiHost));
		checkCudaErrors(cudaMemcpy(temp_cpu_triangle_idx.data(), data.index->GetBufferDataPtr(), data.index->GetSizeBytes(), acr::cudaMemcpySpecifiedTiHost));


		// Represents the temp_cpu_triangle_idx as we sort it
		tri_indices.resize(num_primitivess);
		for (int i = 0; i < tri_indices.size(); i++) tri_indices[i] = i;

		// Precompute all AABBs for an index + vertex pair
		temp_aabb.resize(num_primitivess);
		PrecomputeAABB(temp_aabb, temp_cpu_vertices, temp_cpu_triangle_idx);

		nodes.resize(num_primitivess * 2 - 1);
		auto& RootNode = nodes.front();

		// Add "Calculate Bounds Function"
		RootNode.count = num_primitivess;
		RootNode.leftFirst = 0;
		RootNode.aabb = combineAABBs(temp_aabb.begin(), temp_aabb.end());
		m_NodesUsed += 2; // we use Node 0 & 1 for Root nodes.

		// Subdivide
		Subdivide(0, nodes, temp_aabb, tri_indices);
	}

	{
		const size_t nodesSizeInBytes = sizeof(BLASNode) * m_NodesUsed;
		acr::allocate(&m_GPUNodes, nodesSizeInBytes);
		//checkCudaErrors(cudaMalloc(&m_GPUNodes, nodesSizeInBytes));
		checkCudaErrors(cudaMemcpy(m_GPUNodes, nodes.data(), nodesSizeInBytes, acr::cudaMemcpyHostToSpecified));
	}

	{
		const size_t triIndicesArrSizeInBytes = sizeof(int) * tri_indices.size();
		//checkCudaErrors(cudaMalloc(&m_GPUSortedTriIndices, triIndicesArrSizeInBytes));
		acr::allocate(&m_GPUSortedTriIndices, triIndicesArrSizeInBytes);
		checkCudaErrors(cudaMemcpy(m_GPUSortedTriIndices, tri_indices.data(), triIndicesArrSizeInBytes, acr::cudaMemcpyHostToSpecified));
	}

}

void BLAS::PrecomputeAABB(std::vector<AABB>& temp_aabb, const std::vector<glm::vec3>& vertices, const  std::vector<glm::ivec3>& indices)
{
	using namespace glm;
	struct Tri {
		vec3 vertex0;
		vec3 vertex1;
		vec3 vertex2;
	};

	for (int i = 0; i < indices.size(); i++)
	{
		AABB aabb;
		aabb.min = vec3(1e30f);
		aabb.max = vec3(-1e30f);

		Tri leafTri;
		leafTri.vertex0 = vertices[indices[i].x];
		leafTri.vertex1 = vertices[indices[i].y];
		leafTri.vertex2 = vertices[indices[i].z];

		aabb.min = min(aabb.min, leafTri.vertex0);
		aabb.min = min(aabb.min, leafTri.vertex1);
		aabb.min = min(aabb.min, leafTri.vertex2);
		aabb.max = max(aabb.max, leafTri.vertex0);
		aabb.max = max(aabb.max, leafTri.vertex1);
		aabb.max = max(aabb.max, leafTri.vertex2);
		temp_aabb[i] =  aabb;
	}

}

void BLAS::Subdivide(glm::uint nodeIdx, std::vector<BLASNode>& out_nodes, 
const std::vector<AABB>& in_temp_aabb, std::vector<int>& in_tri_indices)
{
	// Terminate recursion
	BLASNode& node = out_nodes[nodeIdx];
	if (node.count <= 2) return;

	// determine split axis and position
	auto extent = node.aabb.max - node.aabb.min;
	int axis = 0;							// x axis
	if (extent.y > extent.x) axis = 1;		// y axis
	if (extent.z > extent[axis]) axis = 2;	// z axis
	const float splitPos = node.aabb.min[axis] + extent[axis] * 0.5f;

	// in-place partition
	int start = node.leftFirst;
	int end = start + node.count - 1;
	while (start <= end)
	{
		if (in_temp_aabb[in_tri_indices[start]].centroid()[axis] < splitPos)
			start++;
		else
			std::swap(in_tri_indices[start], in_tri_indices[end--]);
	}

	// abort split if one of the sides is empty
	const int leftCount = start - node.leftFirst;
	if (leftCount == 0 || leftCount == node.count) return;

	// create child nodes
	const int leftChildIdx = m_NodesUsed++;
	const int rightChildIdx = m_NodesUsed++;

	out_nodes[leftChildIdx].leftFirst = node.leftFirst;
	out_nodes[leftChildIdx].count = leftCount;
	out_nodes[leftChildIdx].aabb = combineAABBs(in_temp_aabb.begin() + node.leftFirst, in_temp_aabb.begin() + node.leftFirst + leftCount);

	out_nodes[rightChildIdx].leftFirst = start;
	out_nodes[rightChildIdx].count = node.count - leftCount;
	out_nodes[rightChildIdx].aabb = combineAABBs(in_temp_aabb.begin() + out_nodes[rightChildIdx].leftFirst, in_temp_aabb.begin() + out_nodes[rightChildIdx].leftFirst + out_nodes[rightChildIdx].count);

	node.leftFirst = leftChildIdx;	// Points to the pair of nodes connected to it left & right (left + 1) 
	node.count = 0;					// Set to zero to indicate it points to a child node

	// recurse
	Subdivide(leftChildIdx, out_nodes, in_temp_aabb, in_tri_indices);
	Subdivide(rightChildIdx, out_nodes, in_temp_aabb, in_tri_indices);
}