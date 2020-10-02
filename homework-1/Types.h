#pragma once
#include <vector_types.h>

#define EPSILON 0.001f

enum RayType
{
	RAY_TYPE_RADIANCE = 0,  // for shading & reflections
	RAY_TYPE_OCCLUSION = 1, // for shadow calculation
	RAY_TYPE_COUNT			// total
};

enum PrimativeType 
{
	SPHERE = 0,
	TRIANGLE = 1,
	QUADLIGHT = 2
};

enum IntegratorType 
{
	RAYTRACER,
	ANALYTICDIRECT,
	DIRECT,
	PATHTRACER
};

enum ImportanceSampling
{
	HEMISPHERE = 0,
	COSINE = 1,
	BRDF = 2
};

enum BRDFAlgorithm
{
	PHONG,
	GGX
};

enum NextEventEstimation
{
	OFF = 0,
	ON = 1,
	MIS = 2
};

struct Params
{
	uchar4* image;
	uchar4* lights;
	uchar4* quadLights;
	unsigned int light_count;
	unsigned int quad_light_count;
	unsigned int image_width;
	unsigned int image_height;
	float3 cam_eye;
	float3 cam_u, cam_v, cam_w;
	OptixTraversableHandle handle;
	unsigned int depth;
	unsigned int integrator;
	unsigned int light_samples;
	unsigned int light_stratify;
	unsigned int nee;
	unsigned int spp;
	unsigned int russian_roulette;
	unsigned int importance_sampling;
	float gamma;
};


struct RayGenData
{
	// No data needed
};


struct MissData
{
	float3 bg_color;
};

struct DSphere 
{
	float3 center;
	float radius;
};

struct HitGroupData
{
	int primativeType;
	int index;
	float3 diffuse, specular, emission;
	float3 ambient;
	float3 verticies[3];
	float shininess;
	float roughness;
	unsigned int brdf_algorithm;
	float transform[16];
	float inverseTransform[16];
	float inverseWithoutTranslate[16];

	DSphere sphere;

};

struct DLight {
	int type;
	float3 position;
	float3 color;
	float atten0, atten1, atten2;
};

struct DQuadLight {
	float3 a, ab, ac;
	float3 intensity;
};
