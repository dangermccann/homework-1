
#include <optix.h>

#include "Types.h"
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <sutil/vec_math.h>
#include "helpers.h"



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

float neePDF(float3 omegaI, float3 P)
{
	TraceData* td = getTraceData();

	float pdf = 0;										// final result

	DQuadLight* dql = (DQuadLight*)params.quadLights;

	for (int k = 0; k < params.quad_light_count; k++)
	{
		DQuadLight ql = dql[k];							// current light 
		float3 nl = quadLightNormal(ql);				// surface normal of the area light

		float A = length(ql.ab) * length(ql.ac);		// area of parallelogram
		//float A = length(cross(ql.ab, ql.ac));


		float hitT;
		unsigned int lightIndex = k;
		float3 result = traceLightSource(params.handle, P, omegaI, lightIndex, hitT, td->seed);
		float V = length(result);
		if (V > 0)
		{
			float3 x1 = P + omegaI * hitT;				// sampled point in light source
			float R = hitT;								// distance from hit point to light sample

			float nlDotWi = dot0(-nl, omegaI);

			if (nlDotWi > 0)
				pdf += (R*R) / (A * nlDotWi);
		}
	}

	return pdf / params.quad_light_count;
}

float3 evalBRDF(float3 omegaI, float3 N, float3 dir, HitGroupData* hit_data)
{
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
	return f;
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

			// BRDF evaluation 
			float3 f = evalBRDF(omegaI, N, dir, hit_data);

			col += V * f * nDotWi * LnDotWi / (R * R);	// Put it all together
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

	if (isnan(omegaI.x) || isnan(omegaI.y) || isnan(omegaI.z))
	{
		uint3 li = optixGetLaunchIndex();
		printf("(%04d, %03d) x %d : [%f, %f, %f] %f\n",
				li.x, li.y, td->depth, omegaI.x, omegaI.y, omegaI.z, 
				hit_data->roughness);
	}

	return omegaI;
}

float3 sample(float3 N, float3 dir, HitGroupData* hit_data, TraceData* td, float t)
{
	float3 omegaI;

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

	return omegaI;
}

float brdfPdf(float3 omegaI, float3 N, float3 dir, float t, HitGroupData* hit_data)
{
	float result;
	if (hit_data->brdf_algorithm == PHONG)
	{
		result = phongPDF(dir, N, omegaI, t, hit_data->shininess);
	}
	else if (hit_data->brdf_algorithm == GGX)
	{
		result = ggxPDF(dir, N, omegaI, t, hit_data->roughness);
	}
	return result;
}

int russianRoulette(float& boost, TraceData* td)
{
	int terminate = 0;
	boost = 1;

	if (params.russian_roulette == 1)
	{
		float q = 1;						// termination probability 

		// choose paths with lower throughput to terminate more frequently 
		q = 1.0f - fmin(fmax(fmax(td->throughput.x, td->throughput.y), td->throughput.z), 1.0f);
		float p = rnd(td->seed);
		if (p < q || q >= 1.0f)
		{
			terminate = 1;
		}
		else
		{
			boost = 1.0f / (1.0f - q);	// boost paths that are not terminated 
		}
	}
	return terminate;
}

float misWeight(float pdfI, float pdfK)
{
	float weight = 0;
	float pdfI2 = pdfI * pdfI;
	float pdfK2 = pdfK * pdfK;

	if (pdfI2 > 0 || pdfK2 > 0)
		weight = pdfI2 / (pdfI2 + pdfK2);

	if (isnan(weight) || isinf(weight))
	{
		printf("misWeight %f | %f \n", pdfI2, pdfK2);
	}

	return weight;
}



float3 indirectShade(float3 N, HitGroupData* hit_data)
{
	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float3 P = orig + optixGetRayTmax() * dir; // hit point

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

	float3 throughput = make_float3(0);
	float t = specularRatioT(hit_data);

	// Calculate sample
	float3 omegaI = sample(N, dir, hit_data, td, t);

	// Cosine component
	float nDotWi = dot0(N, omegaI);				

	// Evaluate BRDF
	float3 f = evalBRDF(omegaI, N, dir, hit_data);


	// Calculate throughput for current hop in path
	// At the highest level this is calculated as : f / pdf
	// Or: the BRDF evaluation divided by the probability distribution function result
	// We have two BRDF algorithms: Modified Phong and GGX
	float brdfProbability = 0;

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
		brdfProbability = brdfPdf(omegaI, N, dir, t, hit_data);
		if (brdfProbability != 0)
			throughput = (f / brdfProbability) * nDotWi;
		else
			throughput = make_float3(0);
	}

	// If using Russian Roulette, terminate or boost
	float rrBoost;
	int terminate = russianRoulette(rrBoost, td);
	
	if (terminate)
		return make_float3(0);
	
	throughput *= rrBoost;

	td->throughput *= throughput;
	
	// Recursively sample next color along path
	float3 nextColor = traceRadiance(params.handle, P, omegaI, td->depth + 1, td->seed, td->throughput);

	float weightBRDF = 1;
	if (params.nee == MIS)
	{
		float lightProbability = neePDF(omegaI, P);
		weightBRDF = misWeight(brdfProbability, lightProbability);
	}

	// final illumination function 
	return throughput * nextColor * weightBRDF;
}



/// ---------------------------------- ////



float3 misDirectLight(float3 N, HitGroupData* hit_data, TraceData* td)
{
	const float3 orig = optixGetWorldRayOrigin();
	const float3 dir = optixGetWorldRayDirection();
	const float3 P = orig + optixGetRayTmax() * dir;		// hit point
	DQuadLight* dql = (DQuadLight*)params.quadLights;

	float3 result = make_float3(0);

	for (int j = 0; j < params.quad_light_count; j++)
	{
		float3 col = make_float3(0);
		DQuadLight ql = dql[j];
		float3 nl = quadLightNormal(ql);				// surface normal of the area light

		float A = length(ql.ab) * length(ql.ac);		// area of parallelogram
		//float A = length(cross(ql.ab, ql.ac));

		float u1 = rnd(td->seed);
		float u2 = rnd(td->seed);

		// sampled point in light source
		float3 x1 = ql.a + u1 * ql.ab + u2 * ql.ac;

		float3 omegaI = normalize(x1 - P);			// direction vector from hit point to light sample
		float R = length(x1 - P);					// distance from hit point to light sample
		float nDotWi = dot0(N, omegaI);				// Cosine component
		float LnDotWi = dot0(-nl, omegaI);			// differential omegaI
		if (LnDotWi == 0)
			continue;

		// Visibility of sample
		const bool occluded = traceOcclusion(params.handle,
			P, omegaI, EPSILON, R - 3.0f * EPSILON);
		
		if (occluded)
			continue;

		float3 f = evalBRDF(omegaI, N, dir, hit_data);
		float sampleProbability = (R * R) / (A * LnDotWi);
		float brdfProbability = brdfPdf(omegaI, N, dir, specularRatioT(hit_data), hit_data);
		if (brdfProbability == 0)
			continue;

		float weightBRDF = misWeight(sampleProbability, brdfProbability * params.quad_light_count);

		result += (ql.intensity * f * nDotWi * weightBRDF) / sampleProbability;
	}

	return result;
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
		if (params.nee == OFF) 
		{
			// NEE off, do indirect lighting only 
			td->color = indirectShade(N, hit_data);
		}
		else if (params.nee == MIS)
		{
			td->color = make_float3(0);

			if (td->depth == 0 || params.depth > 1)
			{
				td->color += misDirectLight(N, hit_data, td);
			}

			td->color += indirectShade(N, hit_data);			
			
		}
		else if(params.nee == ON)
		{
			td->color = make_float3(0);

			if(td->depth == 0)
				td->color += hit_data->ambient + hit_data->emission;

			td->color += indirectShade(N, hit_data);
			td->color += directShade(N, hit_data, u1, u2);
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
	TraceData* td = getTraceData();
	td->color.x = 1;
}

