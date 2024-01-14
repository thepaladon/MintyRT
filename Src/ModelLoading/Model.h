#pragma once

#include <vector>
#include <string>
#include "glm/glm.hpp"

namespace bml {

	struct Material;
	struct Image;
	class Mesh;
	class Buffer;

	struct Node
	{
		glm::mat4		 m_transform = {};
		std::vector<int> m_children;
		int32_t			 m_mesh = -1;
	};


	class Model
	{
	public:
		Model(const std::string& filepath);
		Model() = delete;
		Model(Model&) = delete;
		Model(Model&&) = delete;
		Model& operator=(const Model&) = delete;
		Model& operator=(Model&&) = delete;

		const std::vector<Mesh*>& GetMeshes() const { return m_Meshes; }
		const std::vector<Buffer*>& GetBuffers() const { return m_Buffers; }
		const std::vector<Node*>& GetNodes() const { return m_Nodes; }
		const std::vector<Image*>& GetImages() const { return m_Images; }
		const std::vector<int>& GetRootNodes() const { return m_RootNodesIndex; }
		//const BLAS& GetBLAS() const { return m_BLAS; }

		const std::string m_Filepath;
	private:
		std::vector<int> m_RootNodesIndex;

		//BLAS m_BLAS;
		std::vector<Mesh*>							m_Meshes;
		std::vector<Buffer*>						m_Buffers;
		std::vector<Node*>							m_Nodes;
		std::vector<Image*>							m_Images;

		Buffer* m_GPUMaterialBuffer;
		std::vector<Material>		m_Materials;

	};
}
