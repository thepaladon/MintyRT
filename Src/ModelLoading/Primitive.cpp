#include "Primitive.h"

#include <string>
#include <glm/ext/matrix_transform.hpp>
#include <TinyglTF/tiny_gltf.h>

namespace Ball {
	Primitive::Primitive(const tinygltf::Primitive& primitive)
	{
		// Note: This gets set during the BLAS creation step from
		// multiplying all Nodes to get into world space from vertex space
		m_Data.m_Model = glm::identity<glm::mat4x4>();
		m_Data.m_MaterialIndex = primitive.material;
		m_Data.m_IndexBufferId = primitive.indices;

		// Buffers relevant to Material
		for (auto& attribute : primitive.attributes)
		{

			std::string attributeName = attribute.first;
			if (attributeName == "POSITION") {
				m_Data.m_PositionIndex = attribute.second;
			}
			else if (attributeName == "NORMAL") {
				m_Data.m_NormalIndex = attribute.second;
			}
			else if (attributeName == "TANGENT") {
				m_Data.m_TangentIndex = attribute.second;
			}
			else if (attributeName == "TEXCOORD_0") {
				m_Data.m_TexCoordIndex = attribute.second;
			}
			else if (attributeName == "COLOR_0") {
				m_Data.m_ColorIndex = attribute.second;
			}
			else {
				printf("LOG: [WARNING] No support for %s yet. \n", attribute.first.c_str());
			}
		}
	}
}