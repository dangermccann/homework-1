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

#define MODE_RADIANCE 1
#define MODE_OCCLUSION 2
#define MODE_BRDF_DIRECT 3
#define NO_INDEX 0xffff

extern "C" {
	__constant__ Params params;
}

struct TraceData
{
	float3 color;
	float3 throughput;
	int depth;
	unsigned int seed;
	unsigned int mode;
	float hitT;
	unsigned int primativeIndex;
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

static __forceinline__ __device__ float dotC(float3 f1, float3 f2)
{
	return clamp(dot(f1, f2), 0.0f, 1.0f);
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

static __forceinline__ __device__ uchar4 make_color_no_gamma(float3 c)
{
	return make_uchar4(quantizeUnsigned8Bits(c.x), quantizeUnsigned8Bits(c.y), quantizeUnsigned8Bits(c.z), 255u);
}


float __forceinline__ __device__ avgf3(float3 f3)
{
	return (f3.x + f3.y + f3.z) / 3.0f;
}


static __forceinline__ __device__ bool traceOcclusion(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	float                  tmin,
	float                  tmax)
{

	TraceData td;
	td.mode = MODE_OCCLUSION;
	td.color = make_float3(0);

	unsigned int u0, u1;
	packPointer(&td, u0, u1);

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
		u0, u1);

	return td.color.x;
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
	td.mode = MODE_RADIANCE;
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

static __forceinline__ __device__ float3 traceLightSource(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	unsigned int&		   primativeIndex,
	float&				   hitT,
	unsigned int&		   seed)
{
	TraceData td;
	td.mode = MODE_BRDF_DIRECT;
	td.seed = seed;
	td.primativeIndex = primativeIndex;

	unsigned int u0, u1;
	packPointer(&td, u0, u1);

	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		EPSILON,				// tmin
		1e16f,					// tmax
		0.0f,                    // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,      // SBT offset
		RAY_TYPE_COUNT,         // SBT stride
		RAY_TYPE_RADIANCE,      // missSBTIndex
		u0, u1);

	seed = td.seed;
	primativeIndex = td.primativeIndex;
	hitT = td.hitT;
	return td.color;
}



extern "C"
__global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	
	// First ray goes through the center of the pixel
	float2 subpixel_jitter = make_float2(0.5f);

	unsigned int seed = tea<8>(idx.y*params.image_width + idx.x, idx.x ^ idx.y);

	// Map our launch idx to a screen location and create a ray from the camera
	// location through the screen
	float3 ray_origin, ray_direction;

	// Iterate over all samples-per-pixel
	float3 result = make_float3(0);
	for (int p = 0; p < params.spp; p++)
	{
		float3 throughput = make_float3(1.0f);

		computeRay(idx, subpixel_jitter, dim, ray_origin, ray_direction);
		result += traceRadiance(params.handle, ray_origin, ray_direction, 0, seed, throughput);

		// Subsequent rays go through random point in pixel
		subpixel_jitter = make_float2(rnd(seed), rnd(seed));
	}

	// Take the average of the samples for this pixel
	result /= params.spp;

	// gamma correction
	result.x = pow(result.x, 1.0f / params.gamma);
	result.y = pow(result.y, 1.0f / params.gamma);
	result.z = pow(result.z, 1.0f / params.gamma);

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

float3 rotate2(const float3 s, const float3 sRot)
{
	float3 a = make_float3(0, 1, 0);
	if (dot(a, sRot) > 0.9)
		a = make_float3(1, 0, 0);
	float3 w = normalize(sRot);
	float3 u = normalize(cross(a, w));
	float3 v = cross(w, u);

	return s.x * u + s.y * v + s.z * w;
}

float3 cartesian(float theta, float phi)
{
	return make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
}

float specularRatioT(HitGroupData* hit_data)
{
	// Calculate t: the relative specular and to diffuse
	// of the material
	float ad = avgf3(hit_data->diffuse);
	float as = avgf3(hit_data->specular);
	float t;
	if (ad + as > 0)
		t = as / (ad + as);
	else
		t = 1;

	return t;
}

float3 quadLightNormal(DQuadLight& ql)
{
	float3 a = ql.a;								// verticies of quad light
	float3 b = ql.a + ql.ab;
	float3 c = ql.a + ql.ac;
	return normalize(cross(c - a, b - a));			// surface normal of the area light
}


float3 phong(const float3 omegaI, const float3 dir, const float3 N, const float3 kd, const float3 ks, const float s)
{
	// reflection vector of sample
	float3 refl = normalize(reflect(dir, N));

	float3 result = kd / PI;						// Lambert shading

	float rDotWi = dot0(refl, omegaI);				// Specular shading
	if (length(ks) > 0 && rDotWi > 0)
		result += ks * ((s + 2.0f) / (2.0f*PI)) * pow(rDotWi, s);

	return result;
}


float microfacetDF(const float3 h, const float3 N, const float roughness)
{
	/*
	float a2 = roughness * roughness;
	float NdotH = dot(N, h);
	float d = ((NdotH * a2 - NdotH) * NdotH + 1);
	return a2 / (d * d * PI);
	*/

	
	float thetaH = acos(dot0(h, N));

	if (isnan(thetaH))
		thetaH = 0;

	float a2 = roughness * roughness;

	float D = (a2) /
		(PI * pow(cos(thetaH), 4.0f) * pow(a2 + pow(tan(thetaH), 2.0f), 2.0f));
	
	return D;
	
}

float smithG(const float3 v, const float3 N, const float roughness)
{
	if (dot(v, N) > 0)
	{
		float thetaV = acos(dot0(v, N));
		return 2.0f / (1.0f + sqrt( 1.0f + pow(roughness, 2.0f) * pow(tan(thetaV), 2.0f)));
	}
	else
	{
		return 0;
	}
}

float3 ggx(const float3 omegaI, const float3 dir, const float3 N, const float3 kd, const float3 ks, const float roughness)
{
	float omegaIDotN = dot(omegaI, N);
	float omegaODotN = dot(-dir, N);

	if (omegaIDotN <= 0 || omegaODotN <= 0)
		return make_float3(0);

	float3 h = normalize(omegaI - dir);

	// microfacet distribution function
	float D = microfacetDF(h, N, roughness);

	// shadowing-masking function
	float G = smithG(omegaI, N, roughness) * smithG(-dir, N, roughness);

	// Fresnel estimation 
	float3 F = ks + (1.0f - ks) * pow(1.0f - dot(omegaI, h), 5.0f);


	if (isnan(D))
	{
		float thetaH = acos(dot0(h, N));
		printf("ggx %f, %f, %f, %f, %f\n", D, dot0(h, N), acos(dot0(h, N)), N.z, thetaH);
	}

	// complete BRDF including diffuse component 
	return (kd / PI) + (F * G * D) / (4 * omegaIDotN * omegaODotN);
}


float phongPDF(float3 dir, float3 N, float3 omegaI, float t, float shininess)
{
	float3 refl = normalize(reflect(dir, N));
	float rDotWi = dot0(refl, omegaI);
	float nDotWi = dot0(N, omegaI);
	float rDotWiPow = shininess > 0 ? pow(rDotWi, shininess) : 1.0f;

	float pdf = (1.0f - t) * (nDotWi / PI) +
		t * (shininess + 1.0f) * rDotWiPow / (2.0f * PI);

	if (isnan(pdf) || isinf(pdf))
	{
		float3 refl = normalize(reflect(dir, N));
		float rDotWi = dot0(refl, omegaI);
		printf("phong %f | %f \n", pdf, rDotWi);
	}

	return pdf;
}

float ggxPDF(float3 dir, float3 N, float3 omegaI, float t, float roughness)
{
	t = fmax(0.25f, t);

	float3 h = normalize(omegaI - dir);
	float nDotH = dot0(N, h);
	float nDotWi = dot0(N, omegaI);
	float hDotWi = dot0(h, omegaI);

	if (abs(nDotH) < 0.001)
	{
		return 0;
	}
	else {
		float D = microfacetDF(h, N, roughness);
		float pdf = ((1.0f - t) * nDotWi / PI);
		if (hDotWi > 0)
			pdf += (t * D * nDotH / (4.0f * hDotWi));

		if (isnan(pdf) || isinf(pdf))
		{
			float thetaH = acos(dot0(h, N));
			printf("GGX PDF is NaN: %f, %f, %f, %f, %f\n", pdf, nDotH, nDotWi, hDotWi, t);
		}

		return pdf;
	}
}

float neePDF(float3 omegaI, HitGroupData* hit_data, unsigned int primativeIndex)
{
	TraceData* td = getTraceData();

	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float  t = optixGetRayTmax();
	const float3 P = orig + t * dir;					// hit point

	DQuadLight* dql = (DQuadLight*)params.quadLights;
	float pdf = 0;										// final result

	DQuadLight ql = dql[primativeIndex];
	float3 nl = quadLightNormal(ql);				// surface normal of the area light

	float A = length(ql.ab) * length(ql.ac);		// area of parallelogram
	//float A = length(cross(ql.ab, ql.ac));


	float hitT;
	float3 result = traceLightSource(params.handle, P, omegaI, primativeIndex, hitT, td->seed);
	float V = length(result);
	if (V > 0)
	{
		float3 x1 = P + omegaI * hitT;				// sampled point in light source
		float R = hitT;								// distance from hit point to light sample

		float nlDotWi = dot0(-nl, omegaI);

		if (nlDotWi > 0)
			pdf += (R*R) / (A * nlDotWi);
	}

	return pdf;
}



float3 directShade(float3 N, HitGroupData* hit_data, float& neePdfAvg, float& brdfPdfAvg)
{
	neePdfAvg = 0;
	brdfPdfAvg = 0;

	const int light_samples = params.light_samples;
	
	// Width and height of stratified grid
	const int strat_grid = params.light_stratify ? sqrtf(light_samples) : 1;
	
	TraceData* td = getTraceData();

	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float  t = optixGetRayTmax();
	const float3 P = orig + t * dir;					// hit point

	DQuadLight* dql = (DQuadLight*)params.quadLights;
	float3 fc = make_float3(0);							// final color

	if (params.nee == OFF)
		fc += hit_data->emission;

	for (int j = 0; j < params.quad_light_count; j++) 
	{
		float3 col = make_float3(0);
		DQuadLight ql = dql[j];
		float3 nl = quadLightNormal(ql);				// surface normal of the area light

		float A = length(ql.ab) * length(ql.ac);		// area of parallelogram
		//float A = length(cross(ql.ab, ql.ac));

		for (int k = 0; k < light_samples; k++) {

			float u1 = rnd(td->seed);
			float u2 = rnd(td->seed);

			int si, sj;
			if (params.light_stratify)
			{
				si = k / strat_grid;					// Stratified grid cell i
				sj = k % strat_grid;					// Stratified grid cell j
			}
			else {
				si = sj = 0;
			}
			
			float3 x1 = ql.a							// sampled point in light source
				+ ((sj + u1)/strat_grid) * ql.ab
				+ ((si + u2)/strat_grid) * ql.ac;		

			float3 omegaI = normalize(x1 - P);			// direction vector from hit point to light sample
			float R = length(x1 - P);					// distance from hit point to light sample
			float nDotWi = dot0(N, omegaI);				// Cosine component
			float LnDotWi = dot0(-nl, omegaI);			// differential omegaI
			
			// Visibility of sample
			const bool occluded = traceOcclusion(params.handle, 
				P, omegaI, EPSILON, R - 3.0f * EPSILON);
			float V = occluded ? 0 : 1;

			// BRDF
			float3 f;
			if (hit_data->brdf_algorithm == PHONG)
			{
				f = phong(omegaI, dir, N, hit_data->diffuse, hit_data->specular, hit_data->shininess);
			}
			else if (hit_data->brdf_algorithm == GGX)
			{
				f = ggx(omegaI, dir, N, hit_data->diffuse, hit_data->specular, hit_data->roughness);
			}

			col += V * f * nDotWi * LnDotWi / (R * R);	// Put it all together


			if (params.nee == MIS && V > 0)
			{
				if (hit_data->brdf_algorithm == PHONG)
					brdfPdfAvg += phongPDF(dir, N, omegaI, specularRatioT(hit_data), hit_data->shininess);
				else if (hit_data->brdf_algorithm == GGX)
					brdfPdfAvg += ggxPDF(dir, N, omegaI, specularRatioT(hit_data), hit_data->roughness);


				if (LnDotWi > 0)
					neePdfAvg += (R*R) / (A * LnDotWi);


				//float tt = neePDF(omegaI, hit_data);
				//if (tt < 0 || tt != ttt)
				//{
				//	printf("%f %f | %f, %f, %f\n", tt, ttt, omegaI.x, omegaI.y, omegaI.z);
				//}
				//neePdfAvg += ttt;
				//neePdfAvg += ttt;
			}
		}

		fc += (col * ql.intensity * A) / light_samples;	// Accumulate final color
	}

	brdfPdfAvg = brdfPdfAvg / params.quad_light_count;
	neePdfAvg = neePdfAvg / params.quad_light_count;

	/*
	uint3 li = optixGetLaunchIndex();
	if (li.x % 5 == 0 && li.y % 5 == 0)
	{
		printf("(%03d, %03d) brdf pdf: %f, nee pdf: %f \n", li.x, li.y, brdfPdfAvg, neePdfAvg);
	}
	*/

	return fc;
}

float3 sampleHemisphere(const float3 N, const float3 dir, const HitGroupData* hit_data, TraceData* td, const float t)
{
	// randomly generate hemisphere sample
	float psi1 = rnd(td->seed);
	float psi2 = rnd(td->seed);

	float theta = acos(clamp(psi1, 0.0f, 1.0f));		// random number between pi/2 and 0
	float phi = 2.0f * PI * psi2;						// random number between 0 and 2*pi

	// calcualte sample in cartesian coordinates 
	float3 s = cartesian(theta, phi);

	// Rotate sample and normalize
	return normalize(rotate2(s, N));
}

float3 sampleCosine(const float3 N, const float3 dir, const HitGroupData* hit_data, TraceData* td, const float t)
{
	// generate sample based on cosine
	float psi1 = sqrt(rnd(td->seed));
	float psi2 = rnd(td->seed);

	float theta = acos(clamp(psi1, 0.0f, 1.0f));		// random number between pi/2 and 0
	float phi = 2.0f * PI * psi2;						// random number between 0 and 2*pi

	// calcualte sample in cartesian coordinates 
	float3 s = cartesian(theta, phi);

	// Rotate sample and normalize
	return normalize(rotate2(s, N));
}



float3 samplePhong(const float3 N, const float3 dir, const HitGroupData* hit_data, TraceData* td, const float t)
{
	// sample based on phong BRDF
	float psi0;
	float psi1 = rnd(td->seed);
	float psi2 = rnd(td->seed);
	// rotate sample to be centered about normal N or reflection vector
	float3 sRot = N;

	psi0 = rnd(td->seed);
		
	if (psi0 <= t)
	{
		psi1 = pow(psi1, (1.0f / (hit_data->shininess + 1)));
		sRot = normalize(reflect(dir, N)); 
	}
	else
	{
		psi1 = sqrt(psi1);
	}

	float theta = acos(clamp(psi1, 0.0f, 1.0f));		// random number between pi/2 and 0
	float phi = 2.0f * PI * psi2;						// random number between 0 and 2*pi

	// calcualte sample in cartesian coordinates 
	float3 s = cartesian(theta, phi);

	// Rotate sample and normalize
	return normalize(rotate2(s, sRot));
}


float3 sampleGGX(const float3 N, const float3 dir, const HitGroupData* hit_data, TraceData* td, const float t)
{
	float psi0 = rnd(td->seed);
	float psi1 = clamp(rnd(td->seed), 0.0f, 1.0f);
	float psi2 = clamp(rnd(td->seed), 0.0f, 1.0f);

	float3 omegaI;			// direction of sample

	if (psi0 <= t)
	{
		// specular 
		float theta = atan((hit_data->roughness * sqrt(psi1)) / sqrt(1.0f - psi1));
		float phi = 2.0f * PI * psi2;

		// produce half vector in cartesian coordinates 
		float3 h = cartesian(theta, phi);
		h = rotate2(h, N);

		omegaI = normalize(reflect(dir, h));

		//if (isnan(omegaI.x) || isnan(omegaI.y) || isnan(omegaI.z))
		//uint3 li = optixGetLaunchIndex();
		//if (li.x % 50 == 0 && li.y % 50 == 0)
		//{
			//printf("(%04d, %03d) x %d : [%f, %f, %f] %f [%f, %f]\n",
			//	li.x, li.y, td->depth, omegaI.x, omegaI.y, omegaI.z, 
			//	hit_data->roughness, theta, phi);
		//}
	}
	else
	{
		// diffuse 
		float theta = acos(clamp(sqrt(psi1), 0.0f, 1.0f));
		float phi = 2.0f * PI * psi2;

		// calcualte sample in cartesian coordinates 
		float3 s = cartesian(theta, phi);

		// Rotate sample and normalize
		omegaI = normalize(rotate2(s, N));
	}


	return omegaI;
}

float3 brdfImportanceSample(HitGroupData* hit_data, TraceData* td, float3 N, float3 dir,
	int& terminate, float3& omegaI)
{
	float t = specularRatioT(hit_data);

	// Calculate sample
	if (params.importance_sampling == HEMISPHERE)
	{
		omegaI = sampleHemisphere(N, dir, hit_data, td, t);
	}
	else if (params.importance_sampling == COSINE)
	{
		omegaI = sampleCosine(N, dir, hit_data, td, t);
	}
	else if (params.importance_sampling == BRDF)
	{
		if (hit_data->brdf_algorithm == PHONG)
		{
			omegaI = samplePhong(N, dir, hit_data, td, t);
		}
		else if (hit_data->brdf_algorithm == GGX)
		{
			omegaI = sampleGGX(N, dir, hit_data, td, t);
		}
	}

	float nDotWi = dot0(N, omegaI);				// Cosine component


	// Evaluate BRDF
	float3 f;
	if (hit_data->brdf_algorithm == PHONG)
	{
		f = phong(omegaI, dir, N, hit_data->diffuse, hit_data->specular, hit_data->shininess);
	}
	if (hit_data->brdf_algorithm == GGX)
	{

		f = ggx(omegaI, dir, N, hit_data->diffuse, hit_data->specular, hit_data->roughness);
	}

	
	// Calculate throughput for current hop in path
	// At the highest level this is calculated as : f / pdf
	// Or: the BRDF evaluation divided by the probability distribution function result
	// We have two BRDF algoritms: Modified Phong and GGX
	float3 throughput = make_float3(0);


	if (params.importance_sampling == HEMISPHERE)
	{
		throughput = (2.0f * PI * f * nDotWi);
	}
	else if (params.importance_sampling == COSINE)
	{
		throughput = (PI * f);	// here the cosine term has canceled out
	}
	else if (params.importance_sampling == BRDF)
	{
		if (hit_data->brdf_algorithm == PHONG)
		{
			float pdf = phongPDF(dir, N, omegaI, t, hit_data->shininess);
			throughput = (f / pdf) * nDotWi;
		}
		else if (hit_data->brdf_algorithm == GGX)
		{
			float pdf = ggxPDF(dir, N, omegaI, t, hit_data->roughness);
			if (pdf != 0)
				throughput = (f / pdf) * nDotWi;
			else
				throughput = make_float3(0);
		}
	}

	// 
	// If using Russian Roulette terminate based on probability q
	//
	if (params.russian_roulette == 1)
	{
		float q = 1;						// termination probability 
		float rrBoost = 1.0f;				// boost factor for rays that continue

		// choose paths with lower throughput to terminate more frequently 
		q = 1.0f - fmin(fmax(fmax(td->throughput.x, td->throughput.y), td->throughput.z), 1.0f);
		float p = rnd(td->seed);
		if (p < q || q >= 1.0f)
		{
			terminate = 1;
			return make_float3(0);			// terminate
		}
		else
		{
			rrBoost = 1.0f / (1.0f - q);	// boost paths that are not terminated 
		}

		throughput *= rrBoost;
	}

	terminate = 0;
	return throughput;
}


float3 indirectShade(float3 N, HitGroupData* hit_data, float3& omegaI, unsigned int& lightIndex)
{
	omegaI = make_float3(0);
	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();

	TraceData* td = getTraceData();

	// Exit on maximum recusion depth
	int max_depth = params.depth;
	if (params.nee == ON)
		max_depth--;

	// Exit if we intersect the light source
	if (hit_data->primativeType == QUADLIGHT)
	{
		if (params.nee == ON)		// This light has been accounted for via direct lighting
		{
			return make_float3(0);
		}
		else 
		{
			// Display the quad light only if we are facing the light's emission surface
			DQuadLight* dql = (DQuadLight*)params.quadLights;
			DQuadLight ql = dql[td->primativeIndex];
			float3 nl = quadLightNormal(ql);

			if (dot(dir, -nl) > 0)
				return ql.intensity;
			else
				return make_float3(0);
		}
	}

	// Exit if we've reached the maximum depth
	if (td->depth >= max_depth)
	{
		return make_float3(0);
	}

	const float3 P = orig + optixGetRayTmax() * dir; // hit point
	int terminate;

	float3 throughput = brdfImportanceSample(hit_data, td, N, dir, terminate, omegaI);

	if (terminate)
		return throughput;

	td->throughput *= throughput;

	if (params.nee == MIS)
	{
		lightIndex = NO_INDEX;
		float hitT;
		float3 hit = traceLightSource(params.handle, P + EPSILON * N, omegaI, lightIndex, hitT, td->seed);
		if (length(hit) > 0)
		{
			DQuadLight* dql = (DQuadLight*)params.quadLights;
			DQuadLight ql = dql[lightIndex];
			return throughput * ql.intensity;
		}
		else
		{
			return make_float3(0);
		}
	}
	else 
	{
		// Recursively sample next color along path
		float3 nextColor = traceRadiance(params.handle, P + EPSILON * N, omegaI, td->depth + 1, td->seed, td->throughput);

		// final illumination function 
		return throughput * nextColor;
	}
}

float misWeight(float pdfI, float pdfK)
{
	float weight = 0;
	float pdfI2 = pdfI * pdfI;
	float pdfK2 = pdfK * pdfK;

	if (pdfI2 > 0 || pdfK2 > 0)
		weight = pdfI2 / (pdfI2 + pdfK2);

	if (isnan(weight))
	{
		printf("!! %f | %f \n", pdfI2, pdfK2);
	}

	return weight;
}

float3 misShade(float3 N, HitGroupData* hit_data, TraceData* td)
{
	const float3 dir = optixGetWorldRayDirection();
	float weightNEE = 0;
	float weightBRDF = 0;
	float3 omegaI;
	float3 brdfColor, neeColor;
	float brdfPdf, neePdf;
	float b = 2.0f;
	float t = specularRatioT(hit_data);
	unsigned int lightIndex;

	brdfColor = indirectShade(N, hit_data, omegaI, lightIndex);
	
	if (hit_data->brdf_algorithm == PHONG)
	{
		brdfPdf = phongPDF(dir, N, omegaI, t, hit_data->shininess);
	}
	else if (hit_data->brdf_algorithm == GGX)
	{
		brdfPdf = ggxPDF(dir, N, omegaI, t, hit_data->roughness);
	}

	neePdf = neePDF(omegaI, hit_data, lightIndex);
	weightBRDF = misWeight(brdfPdf, neePdf);



	//neeColor = directShade(N, hit_data, neePdf, brdfPdf);
	//weightNEE = misWeight(neePdf, brdfPdf);

	/*
	uint3 li = optixGetLaunchIndex();
	if (li.x % 10 == 0 && li.y % 10 == 0)
	{
		printf("(%03d, %03d) : %d : [%f, %f, %f]\n",
			li.x, li.y, td->primativeIndex, brdfPdf, neePdf, 6);
	}
	*/


	//return weightBRDF * brdfColor + weightNEE * neeColor;
	return weightBRDF * brdfColor;
	//return make_float3(neePdf);
	//return weightNEE * neeColor;
}


void shade(float3 N, HitGroupData* hit_data)
{
	TraceData* td = getTraceData();
	unsigned int filterIndex = td->primativeIndex;
	float u1, u2;

	if (hit_data->primativeType == QUADLIGHT)
	{
		td->primativeIndex = hit_data->index;
	}

	if (td->mode == MODE_BRDF_DIRECT)
	{
		if (hit_data->primativeType == QUADLIGHT && 
			(filterIndex == NO_INDEX || filterIndex == hit_data->index))
		{
			const float3 dir = optixGetWorldRayDirection();
			DQuadLight* dql = (DQuadLight*)params.quadLights;
			DQuadLight ql = dql[hit_data->index];
			float3 ln = quadLightNormal(ql);

			// Only count the light if the ray direction is facing the same direction as the light
			if(dot(dir, -ln) > 0)
				td->color = ql.intensity;
			else
				td->color = make_float3(0);
		}
		else
		{
			td->color = make_float3(0);
		}
	}
	else if (params.integrator == RAYTRACER) 
	{
		td->color = rayTracerShade(N, hit_data);
	}
	else if (params.integrator == ANALYTICDIRECT) 
	{
		td->color = analyticDirectShade(N, hit_data);
	}
	else if (params.integrator == DIRECT) 
	{
		td->color = directShade(N, hit_data, u1, u2);
	}
	else if (params.integrator == PATHTRACER) 
	{
		unsigned int uu;
		if (params.nee == OFF) 
		{
			// NEE off, do indirect lighting only 
			float3 omegaI;
			td->color = indirectShade(N, hit_data, omegaI, uu);
		}
		else if (params.nee == ON) 
		{
			float3 omegaI;

			td->color = make_float3(0);

			if(td->depth == 0)
				td->color += hit_data->ambient + hit_data->emission;

			td->color += directShade(N, hit_data, u1, u2);
			td->color += indirectShade(N, hit_data, omegaI, uu);
		}
		else if (params.nee == MIS)
		{
			td->color = misShade(N, hit_data, td);
		}
	}
}


/*

	if (0)
	{
		uint3 li = optixGetLaunchIndex();
		if (li.x % 50 == 0 && li.y % 50 == 0)
		{
			float3 ct = td->throughput;

			printf("(%03d, %03d) x %d : [%f, %f, %f] %d [%f, %f, %f]\n",
				li.x, li.y, td->depth, pt.x, pt.y, pt.z, rrTerminate,
				ct.x, ct.y, ct.z);
		}
	}






void printRay(float3 orig, float3 dir)
{
	printf("origin: %f, %f, %f | dir: %f, %f, %f \n", orig.x, orig.y, orig.z, dir.x, dir.y, dir.z);
}


	*/




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
	TraceData* td = getTraceData();
	td->color.x = 1;
}


bool intersect_triangle(HitGroupData* hg_data, float3 orig, float3 dir, float3 &normal, float& t)
{
	float3 a = hg_data->verticies[0];
	float3 b = hg_data->verticies[1];
	float3 c = hg_data->verticies[2];
	normal = normalize(cross(c - a, b - a));

	float dirDotN = dot(dir, normal);

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

	if(hg_data->primativeType == SPHERE) 
	{
		hit = intersect_sphere(hg_data, orig, dir, normal, t);
	}
	else if(hg_data->primativeType == TRIANGLE)
	{
		hit = intersect_triangle(hg_data, orig, dir, normal, t);
	}
	else if (hg_data->primativeType == QUADLIGHT)
	{
		hit = intersect_triangle(hg_data, orig, dir, normal, t);
		
	}

	if(hit) 
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


