#include <optix.h>

#include "Types.h"
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <sutil/vec_math.h>
#include "helpers.h"


extern "C"
__global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	// First ray goes through the center of the pixel
	float2 subpixel_jitter = make_float2(0.5f);

	unsigned int seed = tea<8>(idx.y * params.image_width + idx.x, idx.x ^ idx.y);

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
