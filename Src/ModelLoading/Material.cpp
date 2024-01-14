#include "Material.h"

#include <TinyglTF/tiny_gltf.h>

namespace bml
{

	Material::Material(const tinygltf::Model& model, int index)
	{
		auto& mat = model.materials[index];
		auto& textures = model.textures;

		if (mat.normalTexture.index != -1)
			m_Data.m_NormalMapIndex = textures[mat.normalTexture.index].source;

		m_Data.m_NormalScale = static_cast<float>(mat.normalTexture.scale);

		// PBR
		m_Data.m_BaseColorFactor = glm::vec4(mat.pbrMetallicRoughness.baseColorFactor[0], mat.pbrMetallicRoughness.baseColorFactor[1], mat.pbrMetallicRoughness.baseColorFactor[2], mat.pbrMetallicRoughness.baseColorFactor[3]);
		m_Data.m_MetallicFactor = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
		m_Data.m_RoughnessFactor = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);

		if (mat.pbrMetallicRoughness.baseColorTexture.index != -1)
			m_Data.m_BaseColorTextureIndex = textures[mat.pbrMetallicRoughness.baseColorTexture.index].source;

		if (mat.pbrMetallicRoughness.metallicRoughnessTexture.index != -1)
			m_Data.m_MetallicRoughnessTextureIndex = textures[mat.pbrMetallicRoughness.metallicRoughnessTexture.index].source;

		// Emissive
		if (mat.emissiveTexture.index != -1)
			m_Data.m_EmssiveTextureIndex = textures[mat.emissiveTexture.index].source;
		m_Data.m_EmissiveFactor = glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);

		// Ao
		if (mat.occlusionTexture.index != -1)
			m_Data.m_AOTextureIndex = textures[mat.occlusionTexture.index].source;
		m_Data.m_AOStrength = static_cast<float>(mat.occlusionTexture.strength);

	}

}