//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "Types.h"
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <sutil/vec_math.h>

const float  PI = 3.1415927f;

extern "C" {
	__constant__ Params params;
}

struct TraceData
{
	float3 color;
	float3 origin;
	float3 normal;
	float3 throughput;
	int depth;
	unsigned int seed;
};


static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
	const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
	void*           ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
	const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ TraceData* getTraceData()
{
	const unsigned int u0 = optixGetPayload_0();
	const unsigned int u1 = optixGetPayload_1();
	return reinterpret_cast<TraceData*>(unpackPointer(u0, u1));
}


static __forceinline__ __device__ void setPayload(float3 p)
{
	optixSetPayload_0(float_as_int(p.x));
	optixSetPayload_1(float_as_int(p.y));
	optixSetPayload_2(float_as_int(p.z));
}


static __forceinline__ __device__ void computeRay(uint3 idx, float2 subpixel_jitter, uint3 dim, float3& origin, float3& direction)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;
	const float2 d = 2.0f * make_float2(
		(static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(dim.x),
		(static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(dim.y)
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U - d.y * V + W);
}

static __forceinline__ __device__ void copy(float t[16], float (& t2)[16])
{
	for (int i = 0; i < 16; i++)
	{
		t2[i] = t[i];
	}
}

static __forceinline__ __device__ float3 transform(float3 ray, float(&t)[16])
{
	float3 ray2;
	ray2.x = (t[0] * ray.x) + (t[1] * ray.y) + (t[2] * ray.z) + t[3];
	ray2.y = (t[4] * ray.x) + (t[5] * ray.y) + (t[6] * ray.z) + t[7];
	ray2.z = (t[8] * ray.x) + (t[9] * ray.y) + (t[10] * ray.z) + t[11];
	return ray2;
}


static __forceinline__ __device__ float dot0(float3 f1, float3 f2)
{
	return fmax(dot(f1, f2), 0.0f);
}


static __forceinline__ __device__ void transpose(float (& t)[16])
{
	float t1[16];
	copy(t, t1);

	t[1] = t1[4];
	t[2] = t1[8];
	t[3] = t1[12];
	t[4] = t1[1];
	t[6] = t1[9];
	t[7] = t1[13];
	t[8] = t1[2];
	t[9] = t1[6];
	t[11] = t1[14];
	t[12] = t1[3];
	t[13] = t1[7];
	t[14] = t1[11];
}


static __forceinline__ __device__ bool traceOcclusion(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	float                  tmin,
	float                  tmax)
{
	unsigned int occluded = 0u;
	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		tmin,
		tmax,
		0.0f,                    // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		RAY_TYPE_OCCLUSION,      // SBT offset
		RAY_TYPE_COUNT,          // SBT stride
		RAY_TYPE_OCCLUSION,      // missSBTIndex
		occluded);
	return occluded;
}



static __forceinline__ __device__ float3 traceRadiance(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	int					   depth,
	unsigned int&		   seed,
	float3&				   throughput)
{
	TraceData td;
	td.depth = depth;
	td.seed = seed;
	td.throughput = throughput;

	unsigned int u0, u1;
	packPointer(&td, u0, u1);

	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		0.0f,					// tmin
		1e16f,					// tmax
		0.0f,                    // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,      // SBT offset
		RAY_TYPE_COUNT,         // SBT stride
		RAY_TYPE_RADIANCE,      // missSBTIndex
		u0, u1);
	
	seed = td.seed;
	throughput = td.throughput;
	return td.color;
}




static __forceinline__ __device__ uchar4 make_color_no_gamma(float3 c)
{
	return make_uchar4(quantizeUnsigned8Bits(c.x), quantizeUnsigned8Bits(c.y), quantizeUnsigned8Bits(c.z), 255u);
}



extern "C"
__global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	
	// First ray goes through the center of the pixel
	float2 subpixel_jitter = make_float2(0.5f);

	unsigned int seed = tea<4>(idx.y*params.image_width + idx.x, 0);

	// Map our launch idx to a screen location and create a ray from the camera
	// location through the screen
	float3 ray_origin, ray_direction;
	float3 throughput = make_float3(1);

	// Iterate over all samples-per-pixel
	float3 result = make_float3(0);
	for (int p = 0; p < params.spp; p++)
	{
		computeRay(idx, subpixel_jitter, dim, ray_origin, ray_direction);
		result += traceRadiance(params.handle, ray_origin, ray_direction, 0, seed, throughput);

		// Subsequent rays go through random point in pixel
		subpixel_jitter = make_float2(rnd(seed), rnd(seed));
	}

	// Take the average of the samples for this pixel
	result /= params.spp;

	// Record results in our output raster
	params.image[idx.y * params.image_width + idx.x] = make_color_no_gamma(result);
}



extern "C" __global__ void __miss__ms() 
{
	MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

	TraceData* td = getTraceData();
	td->color = miss_data->bg_color;
}

float3 rayTracerShade(float3 N, HitGroupData* hit_data) 
{
	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float  t = optixGetRayTmax();
	const float3 P = orig + t * dir; // hit point

	DLight* dl = (DLight*)params.lights;

	float3 c = make_float3(0);
	c += hit_data->ambient + hit_data->emission;

	for (int i = 0; i < params.light_count; i++)
	{
		DLight light = dl[i];
		float3 L = make_float3(0);

		float intensity = 1.0f;
		float LDist;
		if (light.type == 0) // point
		{
			L = normalize(light.position - P);
			LDist = length(light.position - P);
			intensity = 1.0f / (light.atten0 + light.atten1 * LDist + light.atten2 * LDist * LDist);
		}
		else // directional
		{
			L = normalize(light.position);
			LDist = 1e16f;
		}

		//float LdotN = max( dot(L, N), 0.0f);
		float LdotN = dot(L, N);
		LdotN = LdotN < 0 ? 0 : LdotN;
		float3 lambert = hit_data->diffuse * LdotN;

		float3 H = normalize(L + (normalize(orig - P)));
		float HdotN = dot(H, N);
		HdotN = (HdotN < 0) ? 0 : HdotN;
		float3 phong = make_float3(0);
		if (hit_data->shininess > 0)
			phong = hit_data->specular * pow(HdotN, hit_data->shininess);


		// calculate shadow
		float V = 1.0f;
		float3 occlusionOrig = P + N * EPSILON;
		float3 occlusionDir = L;

		const bool occluded = traceOcclusion(
			params.handle, occlusionOrig, occlusionDir,
			0.0001f, LDist);

		if (occluded)
		{
			V = 0;
		}

		c += V * intensity * light.color * (lambert + phong);
	}

	TraceData* td = getTraceData();


	// specularity
	float s = hit_data->specular.x + hit_data->specular.y + hit_data->specular.x;
	if (s > 0 && td->depth < params.depth)
	{
		td->origin = P;

		float3 r = normalize(orig - P);
		float3 refl = N * 2.0f * dot(r, N) - r;
		refl = normalize(refl);

		float3 reflOrigin = P + N * EPSILON;

		// trace reflection
		float3 reflColor = traceRadiance(params.handle, reflOrigin, refl, td->depth + 1, td->seed, td->throughput);
		c += hit_data->specular * reflColor;
	}

	return c;
}

float3 analyticDirectShade(float3 N, HitGroupData* hit_data) 
{
	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float  t = optixGetRayTmax();
	const float3 P = orig + t * dir; // hit point

	DQuadLight* dql = (DQuadLight*)params.quadLights;
	float3 c = make_float3(0);

	c += hit_data->ambient + hit_data->emission;

	for (int j = 0; j < params.quad_light_count; j++)
	{
		DQuadLight ql = dql[j];
		const int vertexCount = 4;
		float3 verticies[4];
		verticies[0] = ql.a;
		verticies[1] = ql.a + ql.ab;
		verticies[2] = ql.a + ql.ab + ql.ac;
		verticies[3] = ql.a + ql.ac;



		float3 irradiance = make_float3(0);

		for (int i = 0; i < vertexCount; i++) {
			float3 v1 = verticies[i];
			float3 v2 = verticies[(i + 1) % vertexCount];
			float theta = acosf(dot(normalize(v1 - P), normalize(v2 - P)));
			float3 gamma = normalize(cross(v1 - P, v2 - P));

			irradiance += theta * gamma;
		}

		irradiance *= 0.5f;

		c += (hit_data->diffuse / PI) * ql.intensity * dot(irradiance, N);
	}

	return c;
}

float3 brdf(float3 omegaI, float3 dir, float3 N, float3 kd, float3 ks, float s) 
{
	// reflection vector of sample
	float3 refl = normalize((2.0f * dot0(-dir, N) * N) + dir);

	float3 result = kd / PI;						// Lambert shading

	float rDotWi = fmax(0, dot0(refl, omegaI));		// Specular shading
	if (length(ks) > 0 && rDotWi > 0)
		result += ks * ((s + 2.0f) / (2.0f*PI)) * pow(rDotWi, s);

	return result;
}

float3 directShade(float3 N, HitGroupData* hit_data) 
{
	const int light_samples = params.light_samples;
	const int strat_grid = params.light_stratify ? sqrtf(light_samples) : 1;
	
	TraceData* td = getTraceData();

	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float  t = optixGetRayTmax();
	const float3 P = orig + t * dir; // hit point

	DQuadLight* dql = (DQuadLight*)params.quadLights;
	float3 fc = make_float3(0);

	if (params.nee == 0)
		fc += hit_data->ambient + hit_data->emission;

	for (int j = 0; j < params.quad_light_count; j++) 
	{
		float3 col = make_float3(0);
		DQuadLight ql = dql[j];

		float3 a = ql.a;							// verticies of quad light
		float3 b = ql.a + ql.ab;
		float3 c = ql.a + ql.ac;
		float3 nl = normalize(cross(c - a, b - a));	// surface normal of the area light

		float A = length(ql.ab) * length(ql.ac);	// area of parallelogram
		//float A = length(cross(ql.ab, ql.ac));

		for (int k = 0; k < light_samples; k++) {

			float u1 = rnd(td->seed);
			float u2 = rnd(td->seed);

			int si, sj;
			if (params.light_stratify)
			{
				si = k / strat_grid;						// Stratified grid cell i
				sj = k % strat_grid;						// Stratified grid cell j
			}
			else {
				si = sj = 0;
			}
			
			float3 x1 = ql.a								// sampled point in light source
				+ ((sj + u1)/strat_grid) * ql.ab
				+ ((si + u2)/strat_grid) * ql.ac;		

			float3 omegaI = normalize(x1 - P);				// direction vector from hit point to light sample
			float R = length(x1 - P);						// distance from hit point to light sample
			float nDotWi = dot0(N, omegaI);					// Cosine component
			float LnDotWi = dot0(-nl, omegaI);				// differential omegaI
			
			// Visibility of sample
			const bool occluded = traceOcclusion(params.handle, P, omegaI,
				0.0001f, R);
			float V = occluded ? 0 : 1;

			// BRDF
			float3 f = brdf(omegaI, dir, N, hit_data->diffuse, hit_data->specular, hit_data->shininess);

			col += V * f * nDotWi * LnDotWi / (R * R);	// Put it all together
		}

		fc += (col * ql.intensity * A) / light_samples;
	}

	return fc;
}

float3 pathTraceShade(float3 N, HitGroupData* hit_data)
{
	TraceData* td = getTraceData();

	// Exit on maximum recusion depth
	int max_depth = params.depth;
	if (params.nee == 1)
		max_depth--;

	if (td->depth >= max_depth)
	{
		if (params.nee == 1)
			return make_float3(0);
		else
			return hit_data->emission;
	}

	// Exit if we intersect the light source
	if (hit_data->primativeType == QUADLIGHT)
	{
		if (params.nee == 1)
			return make_float3(0);
		else
			return hit_data->emission;
	}

	// 
	// If using Russian Roulette terminate based on probability q
	//
	float q = 1;						// termination probability 
	float rrBoost = 1.0f;
	if (params.russian_roulette == 1)
	{
		// choose paths with lower throughput to terminate more frequently 
		q = 1.0f - fmin(fmax(fmax(td->throughput.x, td->throughput.y), td->throughput.z), 1.0f);
		float p = rnd(td->seed);
		if (p < q)
		{
			return make_float3(0);		// terminate
		}
		else
		{
			if(q < 1.0f)
				rrBoost = 1.0f / (1.0f - q);	// boost paths that are not terminated 
		}
	}

	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float  t = optixGetRayTmax();
	const float3 P = orig + t * dir; // hit point

	// randomly generate hemisphere sample
	float psi1 = rnd(td->seed);
	float psi2 = rnd(td->seed);

	float theta = acos(clamp(psi1, 0.0f, 1.0f));		// random number between pi/2 and 0
	float phi = 2.0f * PI * psi2;						// random number between 0 and 2*pi

	// calcualte sample in cartesian coordinates 
	float3 s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

	// rotate sample to be centered about normal N
	float3 a = make_float3(0, 1, 0);
	if (dot(a, N) > 0.9)
		a = make_float3(1, 0, 0);
	float3 w = normalize(N);
	float3 u = normalize(cross(a, w));
	float3 v = cross(w, u);

	// direction vector to next sample
	float3 omegaI = normalize(s.x * u + s.y * v + s.z * w);

	// BRDF
	float3 f = brdf(omegaI, dir, N, hit_data->diffuse, hit_data->specular, hit_data->shininess);

	float nDotWi = dot0(N, omegaI);				// Cosine component

	// Apply throughput for next hop in path
	td->throughput *= 2.0f * PI * f * nDotWi;

	// Recursively sample next color along path
	float3 nextColor = traceRadiance(params.handle, P + EPSILON * N, omegaI, td->depth + 1, td->seed, td->throughput);
	//nextColor = clamp(nextColor, 0.0f, 1.0f);

	float3 Lo = 2.0f * PI * f * nextColor * nDotWi * rrBoost;
	
	//return clamp(Lo, 0, 6);
	return Lo;
}


void shade(float3 N, HitGroupData* hit_data)
{
	TraceData* td = getTraceData();

	if (params.integrator == RAYTRACER) {
		td->color = rayTracerShade(N, hit_data);
	}
	else if (params.integrator == ANALYTICDIRECT) {
		td->color = analyticDirectShade(N, hit_data);
	}
	else if (params.integrator == DIRECT) {
		td->color = directShade(N, hit_data);
	}
	else if (params.integrator == PATHTRACER) {
		if (params.nee == 1) {
			td->color = make_float3(0);

			if(td->depth == 0)
				td->color += hit_data->ambient + hit_data->emission;

			td->color += directShade(N, hit_data);
			td->color += pathTraceShade(N, hit_data);
		}
		else
		{
			// NEE off, do indirect lighting only 
			td->color = pathTraceShade(N, hit_data);
		}
	}
}


extern "C" __global__ void __closesthit__primative()
{
	// When built-in triangle intersection is used, a number of fundamental
	// attributes are provided by the OptiX API, indlucing barycentric coordinates.
	//const float2 barycentrics = optixGetTriangleBarycentrics();


	HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

	const float3 normal =
		make_float3(
			int_as_float(optixGetAttribute_0()),
			int_as_float(optixGetAttribute_1()),
			int_as_float(optixGetAttribute_2())
		);

	shade(normal, hit_data);

}



extern "C" __global__ void __closesthit__occlusion()
{
	optixSetPayload_0(static_cast<unsigned int>(true));
}


bool intersect_triangle(HitGroupData* hg_data, float3 orig, float3 dir, float3 &normal, float& t)
{
	float3 a = hg_data->verticies[0];
	float3 b = hg_data->verticies[1];
	float3 c = hg_data->verticies[2];
	normal = normalize(cross(c - a, b - a));

	float dirDotN(dot(dir, normal));

	// No ray-plane intersection (orthogonal)
	if (dirDotN == 0)
		return false;


	// Calculate distance along ray to intersection point
	t = (dot(a, normal) - dot(orig, normal)) / dirDotN;

	// Intersections from behind origin are invalid
	if (t < 0)
		return false;

	float3 P = orig + dir * t;
	

	// Intersection with the triangle's plane, determine if it's inside or outside
	float3 beta = cross(P - a, b - a);
	if (dot(normal, beta) < 0)
		return false;

	float3 gamma = cross(P - b, c - b);
	if (dot(normal, gamma) < 0)
		return false;

	float3 alpha = cross(P - c, a - c);
	if (dot(normal, alpha) < 0)
		return false;

	// TODO: This I don't fully understand.  But I know that the normal is facing the wrong direction
	// if it is not reversed here.   
	normal *= -1;

	return true;
}

bool intersect_sphere(HitGroupData* hg_data, float3 orig, float3 dir, float3 &normal, float& t)
{

	float3 C = hg_data->sphere.center;
	float r = hg_data->sphere.radius;

	float3 p0MinusC = orig - C;
	float a = dot(dir, dir);
	float b = dot(dir * 2, p0MinusC);
	float c = dot(p0MinusC, p0MinusC) - r * r;

	float discriminant = b * b - 4 * a*c;
	if (discriminant > 0.0f) {
		int hit = 1;
		float sDiscriminant = sqrt(discriminant);
		float t1 = (-b + sDiscriminant) / (2.0f*a);
		float t2 = (-b - sDiscriminant) / (2.0f*a);

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


void printRay(float3 orig, float3 dir)
{
	printf("origin: %f, %f, %f | dir: %f, %f, %f \n", orig.x, orig.y, orig.z, dir.x, dir.y, dir.z);
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


	if(hg_data->primativeType == SPHERE) 
	{
		hit = intersect_sphere(hg_data, orig, dir, normal, t);
	}
	else if(hg_data->primativeType == TRIANGLE)
	{
		hit = intersect_triangle(hg_data, orig, dir, normal, t);
	}
	// Quad lights should not participate in occlusion checks.
	// Assume that we are doing an occlusion trace based on flags
	else if (hg_data->primativeType == QUADLIGHT && flags != OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT)
	{
		hit = intersect_triangle(hg_data, orig, dir, normal, t);
	}

	if(hit) 
	{
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


