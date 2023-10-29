#include "Model.h"

#include <filesystem>

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Material.h"
#include "Buffer.h"
#include "Image.h"
#include "Mesh.h"

#include <TinyglTF/tiny_gltf.h>

namespace bml {

	bool StringEndsWith(const std::string& subject, const std::string& suffix)
	{
		if (suffix.length() > subject.length())
			return false;

		return subject.compare(
			subject.length() - suffix.length(),
			suffix.length(),
			suffix) == 0;
	}


	inline glm::vec3 to_vec3(std::vector<double> array)
	{
		return glm::vec3((float)array[0], (float)array[1], (float)array[2]);
	}

	inline glm::quat to_quat(std::vector<double> array)
	{
		return glm::quat((float)array[3], (float)array[0], (float)array[1], (float)array[2]);
	}

	void DecomposeMatrix(const glm::mat4& transform, glm::vec3& translation, glm::vec3& scale, glm::quat& rotation)
	{
		auto m44 = transform;
		translation.x = m44[3][0];
		translation.y = m44[3][1];
		translation.z = m44[3][2];


		scale.x = glm::length(glm::vec3(m44[0][0], m44[0][1], m44[0][2]));
		scale.y = glm::length(glm::vec3(m44[1][0], m44[1][1], m44[1][2]));
		scale.z = glm::length(glm::vec3(m44[2][0], m44[2][1], m44[2][2]));

		glm::mat4 myrot(
			m44[0][0] / scale.x, m44[0][1] / scale.x, m44[0][2] / scale.x, 0,
			m44[1][0] / scale.y, m44[1][1] / scale.y, m44[1][2] / scale.y, 0,
			m44[2][0] / scale.z, m44[2][1] / scale.z, m44[2][2] / scale.z, 0,
			0, 0, 0, 1
		);
		rotation = quat_cast(myrot);
	}

	Model::Model(const std::string& filepath)
		:m_Filepath(filepath)
	{
		printf("Loading [%s]: \n", m_Filepath.c_str());
		//START_TIMER(LoadingModel);

		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;
		bool res = false;

		//START_TIMER(tinyGLTF);
		tinygltf::Model model;
		// Check which format to load
		if (StringEndsWith(filepath, ".gltf"))
			res = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
		if (StringEndsWith(filepath, ".glb"))
			res = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);

		if (!warn.empty())
			printf("Warning: %s\n", warn.c_str());

		if (!err.empty())
			printf("Error: %s\n", err.c_str());

		if (!res) {
			printf("Failed to load glTF: %s\n", filepath.c_str());
			assert(res);
		}
		//END_TIMER(tinyGLTF);


		// Load Materials on CPU
		//START_TIMER(materials);
		for (int i = 0; i < static_cast<int>(model.materials.size()); i++)
		{
			auto material = Material(model, i);
			m_Materials.push_back(material);
		}

		// Upload Material Buffer to GPU
		m_GPUMaterialBuffer = new Buffer(m_Materials.data(), sizeof(Material), m_Materials.size(), "Material Buffer: " + m_Filepath);

	//END_TIMER_MSG(materials, " - Materials %lu", model.materials.size());

	// Load Buffers (Accessors) to GPU
	//START_TIMER(accessors);
		for (int i = 0; i < static_cast<int>(model.accessors.size()); i++)
		{
			auto buffer = new Buffer(model, model.accessors[i], "Accessor [" + std::to_string(i) + "]");
			m_Buffers.push_back(buffer);
		}
	//END_TIMER_MSG(accessors, " - Accessors %lu", model.accessors.size());


		// Load Meshes and Primitives Indices to CPU
		//START_TIMER(meshes);
		for (int i = 0; i < static_cast<int>(model.meshes.size()); i++)
		{
			auto mesh = new Mesh(model, i);
			m_Meshes.push_back(mesh);
		}
		//END_TIMER_MSG(meshes, " - Meshes %lu", model.meshes.size());


		// Load Images (texture data)
		//START_TIMER(textures);
		for (int i = 0; i < static_cast<int>(model.images.size()); i++)
		{
			m_Images.push_back(new Image(model, i, filepath));
		}
		//END_TIMER_MSG(textures, " - Textures %lu", model.images.size());


		//START_TIMER(nodes);
		const auto rootNodeIdx = model.scenes[model.defaultScene].nodes;
		m_RootNodesIndex = rootNodeIdx;

		// Load Nodes
		// Also handle hierarchy model matrix multiplication
		for (int i = 0; i < static_cast<int>(model.nodes.size()); i++)
		{
			glm::vec3 position = glm::vec3(0.f);
			glm::vec3 rotation = glm::vec3(0.f);
			glm::vec3 scale = glm::vec3(1.f);
			glm::quat quat{};

			const auto& node = model.nodes[i];
			auto my_node = new Node();

			if (!node.matrix.empty())
			{
				glm::mat4 mat = glm::make_mat4(node.matrix.data());
				DecomposeMatrix(mat, position, scale, quat);
				rotation = glm::rotate(quat, rotation);

			}
			else
			{
				if (!node.translation.empty())
					position = to_vec3(node.translation);

				if (!node.rotation.empty() && node.rotation.size() == 3)
					rotation = to_vec3(node.rotation);

				if (!node.rotation.empty() && node.rotation.size() == 4)
				{
					quat = to_quat(node.rotation);
					rotation = rotate(quat, rotation);
				}

				if (!node.scale.empty())
					scale = to_vec3(node.scale);
			}

			glm::mat4 mat_rot = glm::toMat4(glm::quat(rotation));
			glm::mat4 mat_trans = glm::translate(glm::mat4(1.0f), position);
			glm::mat4 mat_scale = glm::scale(glm::mat4(1.0f), scale);
			glm::mat4 final = mat_trans * mat_rot * mat_scale;

			my_node->m_transform = final;
			my_node->m_mesh = model.nodes[i].mesh;
			my_node->m_children = model.nodes[i].children;

			m_Nodes.push_back(my_node);
		}
		//END_TIMER_MSG(nodes, " - Nodes %lu", model.nodes.size());


		//END_TIMER_MSG(LoadingModel, "Finished Loading: %s", m_Filepath.c_str());
		printf(" ------------------------------ \n \n");
	}

}

