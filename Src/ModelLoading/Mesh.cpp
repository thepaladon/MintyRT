#include "Mesh.h"

#include <cassert>
#include <TinyglTF/tiny_gltf.h>
#include "Primitive.h"

namespace bml
{

Mesh::Mesh(const tinygltf::Model& model, int index)
{
	const auto mesh = model.meshes[index];

	assert(!mesh.primitives.empty());
	for (const auto& tinyPrimitive : mesh.primitives)
	{
		m_Primitives.push_back(Primitive(tinyPrimitive));
	}
}

Mesh::~Mesh()
{
	// TODO Primitive Resource Deallocation
}

}