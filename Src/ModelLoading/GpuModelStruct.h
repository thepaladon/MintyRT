#pragma once

//Math Types
#include "glm/glm.hpp"
typedef glm::mat4 float4x4;
typedef glm::vec4 float4;
typedef glm::vec3 float3;

struct ModelInfo
{
	int m_ModelStart = -1;
	// Index of the Model in Resource Descriptor
	// m_ModelStart = Primitive Info Buffer
	// m_ModelStart + 1 => Material Info Buffer

	int m_BufferStart = -1;		// First Buffer this index
	int m_TextureStart = -1;	// First Texture this index
	int padding;				// Padding for 16 byte aligned
};

struct PrimitiveGPU
{
	// Note: This gets set during the BLAS creation step from
	// multiplying all Nodes to get into world space from vertex space
	// It exists here because we need it on the GPU.
	float4x4 m_Model = {};

	// Material Index
	int m_MaterialIndex = -1;

	// Accessor Index
	int m_IndexBufferId = -1;
	int m_PositionIndex = -1;
	int m_TexCoordIndex = -1;
	int m_TangentIndex = -1;
	int m_NormalIndex = -1;
	int m_ColorIndex = -1;
	int padding = -1;

};

struct MaterialGPU
{
	//Indices
	int m_BaseColorTextureIndex = -1;												// 4 bytes
	int m_MetallicRoughnessTextureIndex = -1;										// 4 bytes
	int m_EmssiveTextureIndex = -1;													// 4 bytes
	int m_AOTextureIndex = -1;														// 4 bytes
	int m_NormalMapIndex = -1;														// 4 bytes


	// PBR Workflow (32 bytes)
	float4 m_BaseColorFactor = float4(1.0f, 1.0f, 1.0f, 1.0f);				// 16 bytes
	float m_MetallicFactor = 1.f;														// 4 bytes
	float m_RoughnessFactor = 1.f;														// 4 bytes

	// Emissive (16 bytes)
	float3 m_EmissiveFactor = float3(0.0f, 0.0f, 0.0f);						// 12 bytes

	// Ao Scale and Normal Scale
	float m_AOStrength = 1.0f;															// 4 bytes
	float m_NormalScale = 1.0f;															// 4 bytes
};