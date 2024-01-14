#pragma once
#include <vector>

namespace tinygltf
{
	class Model;
}

namespace bml {
	class Primitive;

	class Mesh
	{
	public:
		Mesh() = delete;
		Mesh(Mesh&) = delete;
		Mesh(Mesh&&) = default;
		Mesh(const tinygltf::Model& model, int index);
		Mesh& operator=(const Mesh&) = delete;
		Mesh& operator=(Mesh&&) = default;
		~Mesh();

		const std::vector<Primitive>& GetPrimitives() const { return m_Primitives; }

	private:
		std::vector<Primitive> m_Primitives;
	};

}