#include "BLAS.h"

#include <algorithm>

#include "CudaUtils.cuh"
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
	for (auto& data : blas_build_data)
	{
		// Check for expected formats
		assert(data.vertex->GetStride() == sizeof(glm::vec3));
		assert(data.index->GetStride() == sizeof(int));

		temp_cpu_vertices.resize(data.vertex->GetNumElements());
		auto num_primitivess = data.index->GetNumElements() / 3;

		// We'll sort this CPU vector as we build the BVH
		temp_cpu_triangle_idx.resize(num_primitivess);

		// Copy Data from GPU to CPU
		checkCudaErrors(cudaMemcpy(temp_cpu_vertices.data(), data.vertex->GetBufferDataPtr(), data.vertex->GetSizeBytes(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(temp_cpu_triangle_idx.data(), data.index->GetBufferDataPtr(), data.index->GetSizeBytes(), cudaMemcpyDeviceToHost));

		// Represents the temp_cpu_triangle_idx as we sort it
		m_TriIndices.resize(num_primitivess);
		for (int i = 0; i < m_TriIndices.size(); i++) m_TriIndices[i] = i;

		// Precompute all AABBs for an index + vertex pair
		temp_aabb.resize(num_primitivess);
		PrecomputeAABB(temp_aabb, temp_cpu_vertices, temp_cpu_triangle_idx);

		m_Nodes.resize(num_primitivess * 2 - 1);
		m_RootNode = &m_Nodes.front();

		// Add "Calculate Bounds Function"
		m_RootNode->count = num_primitivess;
		m_RootNode->leftFirst = 0;
		m_RootNode->aabb = combineAABBs(temp_aabb.begin(), temp_aabb.end());
		m_NodesUsed += 2; // we use Node 0 & 1 for Root nodes.

		// Subdivide
		Subdivide(0);
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
		leafTri.vertex0 = temp_cpu_vertices[indices[i].x];
		leafTri.vertex1 = temp_cpu_vertices[indices[i].y];
		leafTri.vertex2 = temp_cpu_vertices[indices[i].z];

		aabb.min = min(aabb.min, leafTri.vertex0);
		aabb.min = min(aabb.min, leafTri.vertex1);
		aabb.min = min(aabb.min, leafTri.vertex2);
		aabb.max = max(aabb.max, leafTri.vertex0);
		aabb.max = max(aabb.max, leafTri.vertex1);
		aabb.max = max(aabb.max, leafTri.vertex2);
		temp_aabb[i] =  aabb;
	}

}

bool IntersectAABB(const Ray& ray, const glm::vec3 bmin, const glm::vec3 bmax)
{
	float tx1 = (bmin.x - ray.org.x) / ray.dir.x, tx2 = (bmax.x - ray.org.x) / ray.dir.x;
	float tmin = glm::min(tx1, tx2), tmax = glm::max(tx1, tx2);
	float ty1 = (bmin.y - ray.org.y) / ray.dir.y, ty2 = (bmax.y - ray.org.y) / ray.dir.y;
	tmin = glm::max(tmin, glm::min(ty1, ty2)), tmax = glm::min(tmax, glm::max(ty1, ty2));
	float tz1 = (bmin.z - ray.org.z) / ray.dir.z, tz2 = (bmax.z - ray.org.z) / ray.dir.z;
	tmin = glm::max(tmin, glm::min(tz1, tz2)), tmax = glm::min(tmax, glm::max(tz1, tz2));
	return tmax >= tmin && tmin < ray.t && tmax > 0;
}

void BLAS::IntersectBVH(Ray& ray, const glm::uint nodeIdx)
{
	BLASNode& node = m_Nodes[nodeIdx];
	if (!IntersectAABB(ray, node.aabb.min, node.aabb.max)) return;

	if (node.isLeaf())
	{
		for (glm::uint i = 0; i < node.count; i++) {}
			// nada yet //IntersectTri(ray, tri[triIdx[node.firstTriIdx + i]]);
	}
	else
	{
		IntersectBVH(ray, node.leftFirst);
		IntersectBVH(ray, node.leftFirst + 1);
	}
}

void BLAS::Subdivide(glm::uint nodeIdx)
{
	// Terminate recursion
	BLASNode& node = m_Nodes[nodeIdx];
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
		if (temp_aabb[m_TriIndices[start]].centroid()[axis] < splitPos)
			start++;
		else
			std::swap(m_TriIndices[start], m_TriIndices[end--]);
	}

	// abort split if one of the sides is empty
	const int leftCount = start - node.leftFirst;
	if (leftCount == 0 || leftCount == node.count) return;

	// create child nodes
	const int leftChildIdx = m_NodesUsed++;
	const int rightChildIdx = m_NodesUsed++;

	m_Nodes[leftChildIdx].leftFirst = node.leftFirst;
	m_Nodes[leftChildIdx].count = leftCount;
	m_Nodes[leftChildIdx].aabb = combineAABBs(temp_aabb.begin() + node.leftFirst, temp_aabb.begin() + node.leftFirst + leftCount);

	m_Nodes[rightChildIdx].leftFirst = start;
	m_Nodes[rightChildIdx].count = node.count - leftCount;
	m_Nodes[rightChildIdx].aabb = combineAABBs(temp_aabb.begin() + m_Nodes[rightChildIdx].leftFirst, temp_aabb.begin() + m_Nodes[rightChildIdx].leftFirst + m_Nodes[rightChildIdx].count);

	node.leftFirst = leftChildIdx;	// Points to the pair of nodes connected to it left & right (left + 1) 
	node.count = 0;					// Set to zero to indicate it points to a child node

	// recurse
	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}