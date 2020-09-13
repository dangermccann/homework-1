#pragma once
#include "Scene.h"

#include <optix.h>
#include <cuda_runtime.h>

#include "Types.h"


template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData>	RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


class OptiXTracer
{
public:
	OptiXTracer();
	virtual ~OptiXTracer();

	void Trace(const Scene & scene);
	void Fill(COLORREF* arr);	
	void Cleanup();

	float progress;
	bool isRunning;

private:
	int width, height;
	OptixDeviceContext context = nullptr;
	OptixModule module = nullptr;
	OptixProgramGroup raygen_prog_group = nullptr;
	OptixProgramGroup miss_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_primative = nullptr;
	OptixProgramGroup hitgroup_prog_occlusion = nullptr;
	OptixPipeline pipeline = nullptr;
	OptixShaderBindingTable sbt = {};
	cudaDeviceProp deviceProps;
	CUdeviceptr d_lights, d_quad_lights;
	Params params;
	CUdeviceptr d_gas_output_buffer;
	std::vector<uchar4>  host_pixels;

	void InitProgram();
	void InitSBT(const Scene & scene);
	void BuildTriangleGAS(const Scene & scene);
	void BuildPrimativeGAS(const Scene & scene);
	void SetupCamera(const Scene & scene);
	void SetupLights(const Scene & scene);
};



