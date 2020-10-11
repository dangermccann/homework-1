#pragma once

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
	void* ptr = reinterpret_cast<void*>(uptr);
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

static __forceinline__ __device__ void copy(float t[16], float(&t2)[16])
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


static __forceinline__ __device__ void transpose(float(&t)[16])
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


static __forceinline__ __device__ float3 traceRadiance(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	int					   depth,
	unsigned int& seed,
	float3& throughput)
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
	throughput = td.throughput;
	return td.color;
}
