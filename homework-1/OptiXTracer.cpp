#include "stdafx.h"



#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>


#include "OptiXTracer.h"

#include <sutil/Exception.h>
#include <nvrtc.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>



#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif
#include <optix_stack_size.h>


// NVRTC compiler options
#define CUDA_NVRTC_OPTIONS  \
  "-std=c++11", \
  "-arch", \
  "compute_60", \
  "-use_fast_math", \
  "-lineinfo", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64", \
  "-G"
std::string g_nvrtcLog;


static float3 vtf3(Vector3 v) {
	float3 f;
	f.x = v.x;
	f.y = v.y;
	f.z = v.z;
	return f;
}

static float3 ctf3(Color3 c) {
	float3 f;
	f.x = c.r;
	f.y = c.g;
	f.z = c.b;
	return f;
}

static void copyToDevice(void* pointer, uint32_t size, CUdeviceptr &d_pointer) 
{
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pointer), size));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_pointer),
		pointer,
		size,
		cudaMemcpyHostToDevice
	));
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

static float minf(float a, float b)
{
	return a < b ? a : b;
}

static float maxf(float a, float b)
{
	return a > b ? a : b;
}


void readCU(const char* filename, std::string& ptx) {
	std::string line;
	std::stringstream contents;
	std::ifstream ptxfile(filename);
	if (ptxfile.is_open())
	{
		while (getline(ptxfile, line))
		{
			contents << line << '\n';
		}	
		ptxfile.close();
	}
	ptx = contents.str().c_str();
}

void compilePTX(const char* name, const char* cuSource, std::string& ptx, const char** log_string) {
	// Create program
	nvrtcProgram prog = 0;
	nvrtcResult code = nvrtcCreateProgram(&prog, cuSource, name, 0, NULL, NULL);

	// Gather NVRTC options
	std::vector<const char*> options;
	options.push_back("-I ./");
	options.push_back("-I C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0/include");
	options.push_back("-I C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include");
	options.push_back("-I C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0/SDK");

	// Collect NVRTC options
	const char*  compiler_options[] = { CUDA_NVRTC_OPTIONS };
	std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

	// JIT compile CU to PTX
	const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

	// Retrieve log output
	size_t log_size = 0;
	code = nvrtcGetProgramLogSize(prog, &log_size);
	g_nvrtcLog.resize(log_size);
	if (log_size > 1)
	{
		code = nvrtcGetProgramLog(prog, &g_nvrtcLog[0]);
		if (log_string)
			*log_string = g_nvrtcLog.c_str();
	}
	if (compileRes != NVRTC_SUCCESS)
		throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

	// Retrieve PTX code
	size_t ptx_size = 0;
	code = nvrtcGetPTXSize(prog, &ptx_size);
	ptx.resize(ptx_size);
	code = nvrtcGetPTX(prog, &ptx[0]);

	// Cleanup
	code = nvrtcDestroyProgram(&prog);
}


OptiXTracer::OptiXTracer()
{
	width = 640;
	height = 480;
	progress = 0;
	InitProgram();
}

void OptiXTracer::InitProgram() {

	//
	// Initialize CUDA and create OptiX context
	//
	char log[2048];
	size_t sizeof_log;


	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	CUcontext cuCtx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));


	cudaGetDeviceProperties(&deviceProps, 0);


	//
	// Create module
	//	
	OptixPipelineCompileOptions pipeline_compile_options = {};
	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipeline_compile_options.numPayloadValues = 3;
	pipeline_compile_options.numAttributeValues = 3;
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
	pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
	//pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	std::string cu, ptx;
	readCU("first.cu", cu);
	compilePTX("first", cu.c_str(), ptx, (const char**)&log);
	sizeof_log = sizeof(log);

	OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
		context,
		&module_compile_options,
		&pipeline_compile_options,
		ptx.c_str(),
		ptx.size(),
		log,
		&sizeof_log,
		&module
	));


	OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

	OptixProgramGroupDesc raygen_prog_group_desc = {}; //
	raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygen_prog_group_desc.raygen.module = module;
	raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&raygen_prog_group_desc,
		1,   // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&raygen_prog_group
	));

	// Leave miss group's module and entryfunc name null
	OptixProgramGroupDesc miss_prog_group_desc = {};
	miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_prog_group_desc.raygen.module = module;
	miss_prog_group_desc.raygen.entryFunctionName = "__miss__ms";
	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&miss_prog_group_desc,
		1,   // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&miss_prog_group
	));

	// Primative hitgroup
	OptixProgramGroupDesc hitgroup_primative_desc = {};
	hitgroup_primative_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroup_primative_desc.hitgroup.moduleCH = module;
	hitgroup_primative_desc.hitgroup.entryFunctionNameCH = "__closesthit__primative";
	hitgroup_primative_desc.hitgroup.moduleIS = module;
	hitgroup_primative_desc.hitgroup.entryFunctionNameIS = "__intersection__primative";
	hitgroup_primative_desc.hitgroup.moduleAH = nullptr;
	hitgroup_primative_desc.hitgroup.entryFunctionNameAH = nullptr;

	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&hitgroup_primative_desc,
		1,   // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&hitgroup_prog_primative
	));

	// Occlusion hit group
	OptixProgramGroupDesc hitgroup_occlusion_desc = {};
	hitgroup_occlusion_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroup_occlusion_desc.hitgroup.moduleCH = module;
	hitgroup_occlusion_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
	hitgroup_occlusion_desc.hitgroup.moduleIS = module;
	hitgroup_occlusion_desc.hitgroup.entryFunctionNameIS = "__intersection__primative";
	hitgroup_occlusion_desc.hitgroup.moduleAH = nullptr;
	hitgroup_occlusion_desc.hitgroup.entryFunctionNameAH = nullptr;

	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&hitgroup_occlusion_desc,
		1,   // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&hitgroup_prog_occlusion
	));





	//
	// Link pipeline
	//
	const uint32_t    max_trace_depth = 0;
	OptixProgramGroup program_groups[] = { 
		raygen_prog_group, miss_prog_group, 
		hitgroup_prog_primative, 
		hitgroup_prog_occlusion 
	}
;

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = max_trace_depth;
	pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixPipelineCreate(
		context,
		&pipeline_compile_options,
		&pipeline_link_options,
		program_groups,
		sizeof(program_groups) / sizeof(program_groups[0]),
		log,
		&sizeof_log,
		&pipeline
	));

	OptixStackSizes stack_sizes = {};
	for (auto& prog_group : program_groups)
	{
		OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
	}

	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
		0,  // maxCCDepth
		0,  // maxDCDEpth
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state, &continuation_stack_size));
	OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state, continuation_stack_size,
		1  // maxTraversableDepth
	));
	
}

void PopulateHitGroupRecord(HitGroupSbtRecord& rec, const Geometry& geometry)
{
	rec.data.diffuse = ctf3(geometry.material.diffuse);
	rec.data.specular = ctf3(geometry.material.specular);
	rec.data.emission = ctf3(geometry.material.emission);
	rec.data.ambient = ctf3(geometry.material.ambient);
	rec.data.shininess = geometry.material.shininess;

	geometry.transform.ToArray16(rec.data.transform);
	geometry.transform.Invert().ToArray16(rec.data.inverseTransform);

	Transform invertNoTranslate = Transform(geometry.transform);
	invertNoTranslate.x4 = 0;
	invertNoTranslate.y4 = 0;
	invertNoTranslate.z4 = 0;
	invertNoTranslate.Invert().ToArray16(rec.data.inverseWithoutTranslate);
}

void OptiXTracer::InitSBT(const Scene & scene) 
{
	//
	// Set up shader binding table
	//

	// Create data that is sent to ray gen function 
	RayGenSbtRecord rg_sbt = {};
	OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));

	CUdeviceptr  raygen_record;
	const size_t raygen_record_size = sizeof(RayGenSbtRecord);

	copyToDevice(&rg_sbt, raygen_record_size, raygen_record);



	// Create data that is sent to miss function 
	MissSbtRecord ms_sbt = {};
	//ms_sbt.data = { 0.3f, 0.1f, 0.2f };
	ms_sbt.data = { 0, 0, 0 };
	OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));

	CUdeviceptr miss_record;
	const size_t miss_record_size = sizeof(MissSbtRecord);

	copyToDevice(&ms_sbt, miss_record_size, miss_record);


	// Create data that is sent to hit group function 
	const size_t numRecs = (scene.tris.size() + scene.spheres.size()) * RAY_TYPE_COUNT;
	CUdeviceptr hitgroup_record;
	size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);



	std::vector<HitGroupSbtRecord> hg_sbts;
	hg_sbts.resize(numRecs);

	int idx = 0;


	std::list<Sphere>::const_iterator it3;
	for (it3 = scene.spheres.begin(); it3 != scene.spheres.end(); ++it3)
	{
		Sphere sphere = *it3;
		
		hg_sbts[idx] = {};
		hg_sbts[idx].data.primativeType = SPHERE;
		hg_sbts[idx].data.sphere.radius = sphere.radius;
		hg_sbts[idx].data.sphere.center = vtf3(sphere.position);
		PopulateHitGroupRecord(hg_sbts[idx], sphere);

		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_primative, &hg_sbts[idx]));
		idx++;

		memcpy(&hg_sbts[idx], &hg_sbts[idx - 1], hitgroup_record_size);
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_occlusion, &hg_sbts[idx]));
		idx++;
	}


	std::list<Tri>::const_iterator it2;
	for (it2 = scene.tris.begin(); it2 != scene.tris.end(); ++it2) 
	{
		Tri tri = *it2;
		
		hg_sbts[idx] = {};
		hg_sbts[idx].data.primativeType = TRIANGLE;
		hg_sbts[idx].data.verticies[0] = vtf3(scene.verticies[tri.one]);
		hg_sbts[idx].data.verticies[1] = vtf3(scene.verticies[tri.two]);
		hg_sbts[idx].data.verticies[2] = vtf3(scene.verticies[tri.three]);
		PopulateHitGroupRecord(hg_sbts[idx], tri);

		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_primative, &hg_sbts[idx]));
		idx++;

		memcpy(&hg_sbts[idx], &hg_sbts[idx - 1], hitgroup_record_size);
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_occlusion, &hg_sbts[idx]));
		idx++;
	}


	copyToDevice(hg_sbts.data(), hitgroup_record_size*numRecs, hitgroup_record);

		
	sbt.raygenRecord = raygen_record;
	sbt.missRecordBase = miss_record;
	sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
	sbt.missRecordCount = 1;
	sbt.hitgroupRecordBase = hitgroup_record;
	sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
	sbt.hitgroupRecordCount = numRecs;
}


OptiXTracer::~OptiXTracer()
{
}

void OptiXTracer::SetupCamera(const Scene & scene)
{
	// set up camera 
	float aspect = (float)scene.width / scene.height;
	float fovRadsY = scene.camera.fieldOfView * PI / 180.0f;

	//fovRadsY *= 1.0085f; // overcome small difference between the grader's camera



	
	float thetaY = fovRadsY / 2.0f;
	float tanThetaY = tan(thetaY);
	float tanThetaX = tan(thetaY) * aspect;

	Vector3 forward, up, right;

	up = scene.camera.up;
	up = up.normalize();

	// vector a = eye - center 
	forward = scene.camera.lookFrom - scene.camera.lookAt;

	// vector w is a normalized, screen is one unit from eye
	forward = forward.normalize();

	// vector u
	right = Vector3::cross(up, forward);

	// vector v
	up = Vector3::cross(forward, right);
	




	Vector3 W = scene.camera.lookAt - scene.camera.lookFrom; // Do not normalize W -- it implies focal length
	float wlen = W.length();
	Vector3 U = Vector3::cross(W, scene.camera.up).normalize();
	Vector3 V = Vector3::cross(U, W).normalize();

	float vlen = wlen * tanf(0.5f * fovRadsY);
	V *= vlen;
	float ulen = vlen * aspect;
	U *= ulen;



	params.cam_u = vtf3(U);
	params.cam_v = vtf3(V);
	params.cam_w = vtf3(W);
	params.cam_eye = vtf3(scene.camera.lookFrom);
}

void OptiXTracer::SetupLights(const Scene & scene)
{
	std::vector<DLight> lights;
	lights.resize(scene.lights.size());

	std::list<Light>::const_iterator it2;
	int idx = 0;
	for (it2 = scene.lights.begin(); it2 != scene.lights.end(); ++it2) {
		Light light = *it2;
		lights[idx].atten0 = light.atten0;
		lights[idx].atten1 = light.atten1;
		lights[idx].atten2 = light.atten2;
		lights[idx].type = light.type;
		lights[idx].position = vtf3(light.position);
		lights[idx].color = ctf3(light.color);

		idx++;
	}
	
	// TODO: cudaFree this?
	CUdeviceptr d_lights;
	copyToDevice(lights.data(), lights.size()*sizeof(DLight), d_lights);


	params.lights = reinterpret_cast<uchar4*>(d_lights);
	params.light_count = scene.lights.size();
}


void PopulateBuildInput(int buildOffset, int count, std::vector<OptixAabb> &aabbs, 
	std::vector<CUdeviceptr> &d_aabb_buffers, std::vector<uint32_t> &sbt_index_offsets,
	std::vector<CUdeviceptr> &d_sbt_index, std::vector<uint32_t> &aabb_input_flags,
	std::vector<OptixBuildInput> &build_inputs,
	int primitiveIndexOffset)
{
	copyToDevice(aabbs.data(), sizeof(OptixAabb) * count, d_aabb_buffers[buildOffset]);
	copyToDevice(sbt_index_offsets.data(), sizeof(uint32_t) * count, d_sbt_index[buildOffset]);

	build_inputs[buildOffset] = {};
	build_inputs[buildOffset].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	build_inputs[buildOffset].customPrimitiveArray.aabbBuffers = &d_aabb_buffers[buildOffset];
	build_inputs[buildOffset].customPrimitiveArray.numPrimitives = count;
	build_inputs[buildOffset].customPrimitiveArray.flags = aabb_input_flags.data();
	build_inputs[buildOffset].customPrimitiveArray.numSbtRecords = count;
	build_inputs[buildOffset].customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_index[buildOffset];
	build_inputs[buildOffset].customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
	build_inputs[buildOffset].customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
	build_inputs[buildOffset].customPrimitiveArray.primitiveIndexOffset = primitiveIndexOffset;
}

void OptiXTracer::BuildPrimativeGAS(const Scene & scene) 
{
	OptixTraversableHandle gas_handle;
	CUdeviceptr            d_gas_output_buffer;
	{
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;


		uint32_t nSpheres = scene.spheres.size();
		uint32_t nTris = scene.tris.size();
		uint32_t nObjects = nSpheres + nTris;
		uint32_t nInputs = scene.sphereInputs + scene.triInputs;

		std::vector<OptixAabb> aabbs;
		aabbs.resize(nObjects);

		std::vector<uint32_t> aabb_input_flags;
		aabb_input_flags.resize(nObjects);

		std::vector<uint32_t> sbt_index_offsets;
		sbt_index_offsets.resize(nObjects);


		std::vector<OptixBuildInput> build_inputs;
		build_inputs.resize(nInputs);

		std::vector<CUdeviceptr> d_aabb_buffers;
		d_aabb_buffers.resize(nInputs);

		std::vector<CUdeviceptr> d_sbt_index;
		d_sbt_index.resize(nInputs);

		std::list<Sphere>::const_iterator it2;
		int idx = 0, count = 0, buildOffset = 0, primitiveIndexOffset = 0;
		Transform prev;


		for (it2 = scene.spheres.begin(); it2 != scene.spheres.end(); ++it2)
		{
			Sphere sphere = *it2;

			// if new build input
			if (idx > 0 && count > 0 && !sphere.transform.Equals(prev)) {

				PopulateBuildInput(buildOffset, count, aabbs, d_aabb_buffers,
					sbt_index_offsets, d_sbt_index, aabb_input_flags, build_inputs, primitiveIndexOffset);

				buildOffset++;
				primitiveIndexOffset += count;
				count = 0;
			}

			aabb_input_flags[count] = OPTIX_GEOMETRY_FLAG_NONE;
			sbt_index_offsets[count] = count;

			float radius = sphere.radius;


			Vector3 boxMin = Vector3(sphere.position.x - radius, sphere.position.y - radius, sphere.position.z - radius);
			Vector3 boxMax = Vector3(sphere.position.x + radius, sphere.position.y + radius, sphere.position.z + radius);


			boxMin = boxMin.ApplyTransformation(sphere.transform);
			boxMax = boxMax.ApplyTransformation(sphere.transform);

			aabbs[count].minX = minf(boxMin.x, boxMax.x);
			aabbs[count].minY = minf(boxMin.y, boxMax.y);
			aabbs[count].minZ = minf(boxMin.z, boxMax.z);
			aabbs[count].maxX = maxf(boxMin.x, boxMax.x);
			aabbs[count].maxY = maxf(boxMin.y, boxMax.y);
			aabbs[count].maxZ = maxf(boxMin.z, boxMax.z);


			count++;
			idx++;
			prev = sphere.transform;
		}

		if (idx > 0 && count != 0) {
			PopulateBuildInput(buildOffset, count, aabbs, d_aabb_buffers,
				sbt_index_offsets, d_sbt_index, aabb_input_flags, build_inputs, primitiveIndexOffset);

			buildOffset++;
		}

		// Triangles
		std::list<Tri>::const_iterator it;
		count = 0;
		primitiveIndexOffset = 0;

		for (it = scene.tris.begin(); it != scene.tris.end(); ++it) 
		{
			Tri tri = *it;

			// if new build input
			if (idx > 0 && count > 0 && !tri.transform.Equals(prev)) {

				PopulateBuildInput(buildOffset, count, aabbs, d_aabb_buffers,
					sbt_index_offsets, d_sbt_index, aabb_input_flags, build_inputs, primitiveIndexOffset);

				buildOffset++;
				primitiveIndexOffset += count;
				count = 0;
			}

			
			aabb_input_flags[count] = OPTIX_GEOMETRY_FLAG_NONE;
			sbt_index_offsets[count] = count;


			Vector3 v1 = scene.verticies[tri.one];
			Vector3 v2 = scene.verticies[tri.two];
			Vector3 v3 = scene.verticies[tri.three];
			Vector3 boxMin, boxMax;

			boxMin.x = minf(minf(v1.x, v2.x), v3.x);
			boxMin.y = minf(minf(v1.y, v2.y), v3.y);
			boxMin.z = minf(minf(v1.z, v2.z), v3.z);
			boxMax.x = maxf(maxf(v1.x, v2.x), v3.x);
			boxMax.y = maxf(maxf(v1.y, v2.y), v3.y);
			boxMax.x = maxf(maxf(v1.z, v2.z), v3.z);


			boxMin = boxMin.ApplyTransformation(tri.transform);
			boxMax = boxMax.ApplyTransformation(tri.transform);

			aabbs[count].minX = minf(boxMin.x, boxMax.x);
			aabbs[count].minY = minf(boxMin.y, boxMax.y);
			aabbs[count].minZ = minf(boxMin.z, boxMax.z);
			aabbs[count].maxX = maxf(boxMin.x, boxMax.x);
			aabbs[count].maxY = maxf(boxMin.y, boxMax.y);
			aabbs[count].maxZ = maxf(boxMin.z, boxMax.z);



			count++;
			idx++;
			prev = tri.transform;
		}

		if (idx > 0 && count != 0) {
			PopulateBuildInput(buildOffset, count, aabbs, d_aabb_buffers,
				sbt_index_offsets, d_sbt_index, aabb_input_flags, build_inputs, primitiveIndexOffset);
		}



		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context,
			&accel_options,
			build_inputs.data(),
			nInputs, // Number of build inputs
			&gas_buffer_sizes
		));
		CUdeviceptr d_temp_buffer_gas;
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_temp_buffer_gas),
			gas_buffer_sizes.tempSizeInBytes
		));
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_gas_output_buffer),
			gas_buffer_sizes.outputSizeInBytes
		));

		OPTIX_CHECK(optixAccelBuild(
			context,
			0,                  // CUDA stream
			&accel_options,
			build_inputs.data(),
			nInputs,    // num build inputs
			d_temp_buffer_gas,
			gas_buffer_sizes.tempSizeInBytes,
			d_gas_output_buffer,
			gas_buffer_sizes.outputSizeInBytes,
			&gas_handle,
			nullptr,            // emitted property list
			0                   // num emitted properties
		));

		params.handle = gas_handle;

		// We can now free the scratch space buffer used during build and the vertex
		// inputs, since they are not needed by our trivial shading method
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));

		for (int i = 0; i < scene.sphereInputs; i++)
		{
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabb_buffers[i])));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sbt_index[i])));
		}




		// TODO: maybe compact later???
		/*
		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
		CUdeviceptr d_temp_buffer_gas;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

		// non-compacted output
		CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
		size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
			compactedSizeOffset + 8
		));

		OptixAccelEmitDesc emitProperty = {};
		emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

		OPTIX_CHECK(optixAccelBuild(context,
			0,                  // CUDA stream
			&accel_options,
			&aabb_input,
			1,                  // num build inputs
			d_temp_buffer_gas,
			gas_buffer_sizes.tempSizeInBytes,
			d_buffer_temp_output_gas_and_compacted_size,
			gas_buffer_sizes.outputSizeInBytes,
			&gas_handle,
			&emitProperty,      // emitted property list
			1                   // num emitted properties
		));

		CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
		CUDA_CHECK(cudaFree((void*)d_aabb_buffer));

		size_t compacted_gas_size;
		CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

		if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
		{
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

			// use handle as input and output
			OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

			CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
		}
		else
		{
			d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
		}
		*/
	}
}


void OptiXTracer::Trace(const Scene & scene)
{
	params.image_width = width;
	params.image_height = height;
	params.depth = scene.maxDepth;
	SetupCamera(scene);
	SetupLights(scene);

	BuildPrimativeGAS(scene);

	InitSBT(scene);

	progress = 1;
	
}

void OptiXTracer::Fill(COLORREF* arr) 
{ 
	if (progress < 1)
		return;


	uchar4* device_pixels = nullptr;
	std::vector<uchar4>  host_pixels;
	
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&device_pixels), width*height * sizeof(uchar4)
	));

	host_pixels.resize(width*height);

	// Create stream
	CUstream stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	// Copy launch params
	params.image = device_pixels;

	CUdeviceptr d_param;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_param),
		&params, sizeof(params),
		cudaMemcpyHostToDevice
	));

	// Launch
	OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
	CUDA_SYNC_CHECK();

	// Copy output from device memory to host memory
	CUDA_CHECK(cudaMemcpy(
		static_cast<void*>(host_pixels.data()),
		device_pixels,
		width*height * sizeof(uchar4),
		cudaMemcpyDeviceToHost
	));
	

	// Copy pixels to buffer to be sent to device context
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned int offset = x + y * width;
			// AA RR GG BB
			COLORREF c;
			
			c = host_pixels[offset].z;
			c |= (host_pixels[offset].y << 8);
			c |= (host_pixels[offset].x << 16);
			c |= (host_pixels[offset].w << 24);

			arr[offset] = c;
		}
	}

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(device_pixels)));

}


void OptiXTracer::Cleanup() 
{ 
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));

	OPTIX_CHECK(optixPipelineDestroy(pipeline));

	OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_primative));
	OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_occlusion));
	OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));

	OPTIX_CHECK(optixModuleDestroy(module));
	OPTIX_CHECK(optixDeviceContextDestroy(context));
}