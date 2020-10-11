#include <optix.h>

#include "Types.h"
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <sutil/vec_math.h>
#include "helpers.h"



bool intersect_triangle(HitGroupData* hg_data, float3 orig, float3 dir, float3& normal, float& t)
{
	float3 a = hg_data->verticies[0];
	float3 b = hg_data->verticies[1];
	float3 c = hg_data->verticies[2];
	//normal = normalize(cross(c - a, b - a));
	normal = normalize(cross(b - a, c - a));

	float dirDotN = dot(dir, normal);

	// No ray-plane intersection (orthogonal)
	if (abs(dirDotN) < EPSILON)
		return false;

	// Calculate distance along ray to intersection point
	t = (dot(a, normal) - dot(orig, normal)) / dirDotN;

	// Intersections from behind origin are invalid
	if (t < 0)
		return false;

	float3 P = orig + dir * t;

	// Intersection with the triangle's plane, determine if it's inside or outside
	float3 beta = cross(b - a, P - a);
	if (dot(normal, beta) < 0)
		return false;

	float3 gamma = cross(c - b, P - b);
	if (dot(normal, gamma) < 0)
		return false;

	float3 alpha = cross(a - c, P - c);
	if (dot(normal, alpha) < 0)
		return false;


	return true;
}


bool intersect_sphere(HitGroupData* hg_data, float3 orig, float3 dir, float3& normal, float& t)
{

	float3 C = hg_data->sphere.center;
	float r = hg_data->sphere.radius;

	float3 p0MinusC = orig - C;
	float a = dot(dir, dir);
	float b = dot(dir * 2, p0MinusC);
	float c = dot(p0MinusC, p0MinusC) - r * r;

	float discriminant = b * b - 4 * a * c;
	if (discriminant > 0.0f) {
		int hit = 1;
		float sDiscriminant = sqrt(discriminant);
		float t1 = (-b + sDiscriminant) / (2.0f * a);
		float t2 = (-b - sDiscriminant) / (2.0f * a);

		if (t1 > 0.0f && t2 > 0.0f)
			t = t1 < t2 ? t1 : t2; // min(t1, t2);
		else if (t1 > 0.0f)
			t = t1;
		else if (t2 > 0.0f)
			t = t2;
		else
			hit = 0;

		if (hit > 0) {
			float3 hitPosition = orig + dir * t;
			normal = normalize(hitPosition - C);
			return true;
		}
	}

	return false;
}


extern "C" __global__ void __intersection__primative()
{
	HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

	float3 orig = optixGetObjectRayOrigin();
	float3 dir = optixGetObjectRayDirection();
	unsigned int flags = optixGetRayFlags();

	// apply inverse transform
	orig = transform(orig, hg_data->inverseTransform);
	dir = transform(dir, hg_data->inverseWithoutTranslate);


	float3 normal;
	float t;
	bool hit = false;
	TraceData* td = getTraceData();


	if (hg_data->primativeType == SPHERE)
	{
		hit = intersect_sphere(hg_data, orig, dir, normal, t);
	}
	else if (hg_data->primativeType == TRIANGLE)
	{
		hit = intersect_triangle(hg_data, orig, dir, normal, t);
	}
	else if (hg_data->primativeType == QUADLIGHT)
	{
		hit = intersect_triangle(hg_data, orig, dir, normal, t);

	}

	if (hit)
	{
		td->hitT = t;

		// and apply the transpose inverse matrix to the normal
		float transp[16];
		copy(hg_data->inverseTransform, transp);
		transpose(transp);
		normal = transform(normal, transp);
		normal = normalize(normal);

		optixReportIntersection(
			t,      // t hit
			0,          // user hit kind
			float_as_int(normal.x),
			float_as_int(normal.y),
			float_as_int(normal.z)
		);
	}
}


