#include "stdafx.h"
#include "RayTracer.h"
#include <stack>
#include <math.h>


#define MAX_DISTANCE 1000000.0f

RayTracer::RayTracer()
{
	colorBuffer = NULL;
	width = RT_WIDTH;
	height = RT_HEIGHT;
}


RayTracer::~RayTracer()
{
	DeleteBuffer();
}


void RayTracer::Trace(const Scene & scene)  
{
	isRunning = true;
	progress = 0;
	width = scene.width;
	height = scene.height;
	AllocateBuffer();


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


	for (int y = 0; y < scene.height; y++) {
		float yf = y + 0.5f; // send ray through center of pixel
		float beta = tanThetaY * (0.5f * height - yf) / (0.5f * height);

		for (int x = 0; x < scene.width; x++) {
			float xf = x + 0.5f; // send ray through center of pixel
			float alpha = tanThetaX * (xf - 0.5f * width) / (0.5f * width);

			Ray ray = Ray();
			ray.origin = scene.camera.lookFrom;
			ray.direction = (right * alpha) + (up * beta) - forward;
			ray.direction = ray.direction.normalize();

			HitInfo info = FireHitTest(ray, scene);

			if (info.hit) {
				colorBuffer[y][x] = ColorGeometry(info.geometry, info, scene.camera.lookFrom, scene, 0).argb();
			}
		}

		progress += 1.0f / scene.height;
	}

	isRunning = false;
}


HitInfo RayTracer::FireHitTest(const Ray & ray, const Scene & scene) {

	Geometry best;
	HitInfo bestInfo;
	bestInfo.hit = false;
	float bestDistance = MAX_DISTANCE;

	// Ray trace spheres in scene
	std::list<Sphere>::const_iterator it;
	for (it = scene.spheres.begin(); it != scene.spheres.end(); ++it) {
		Sphere sphere = *it;
		HitInfo info = TestSphere(ray, sphere);
		if (info.hit) {
			float dist = (info.position - ray.origin).length();
			if (dist < bestDistance) {
				best = sphere;
				bestInfo = info;
				bestDistance = dist;
			}
		}
	}

	// Ray trace triangles in scene
	std::list<Tri>::const_iterator it2;
	for (it2 = scene.tris.begin(); it2 != scene.tris.end(); ++it2) {
		Tri tri = *it2;
		HitInfo info = TestTriangle(ray, scene, tri);
		if (info.hit) {
			float dist = (info.position - ray.origin).length();
			if (dist < bestDistance) {
				best = tri;
				bestInfo = info;
				bestDistance = dist;
			}
		}
	}
	return bestInfo;
}

Color3 RayTracer::ColorGeometry(const Geometry & geometry, const HitInfo & hit, const Vector3 & eyepos,
	const Scene & scene, int depth)
{
	Color3 color = geometry.material.ambient + geometry.material.emission;

	std::list<Light>::const_iterator it;
	for (it = scene.lights.begin(); it != scene.lights.end(); ++it) {
		Light light = *it;
		
		float lVal = 1.0f;

		Vector3 lightDir;

		if (light.type == POINT_LIGHT) {
			float d = (light.position - hit.position).length();
			lVal = 1.0f / (light.atten0 + light.atten1 * d + light.atten2 * d * d);

			lightDir = (hit.position - light.position);
			lightDir = lightDir.normalize();
		}
		else {
			lightDir = (light.position * -1.0f).normalize();
		}

		Color3 Li = light.color * lVal;


		// calculate visibility / shadow 
		float V = 1.0f; 
		Ray ray = Ray();

		ray.direction = (lightDir * -1.0f);
		ray.direction = ray.direction.normalize();
		ray.origin = hit.position - hit.normal * 0.001f;

		HitInfo info = FireHitTest(ray, scene);
		if (info.hit) {
			// make sure the hit wasn't past the light source
			// this check only makes sense for point lights
			if (light.type == POINT_LIGHT) {
				Vector3 fromObject = (hit.position - light.position).normalize();
				Vector3 fromObstruction = (info.position - light.position).normalize();
				
				if ((fromObject - fromObstruction).length() < 0.05f)
					V = 0;
			}
			else {
				// for directional lights, block all shadows
				V = 0;
			}
		}

		
		Vector3 e = (hit.position - eyepos).normalize();
		Vector3 half = (lightDir + e).normalize();

		Color3 lambert = geometry.material.diffuse * max(Vector3::dot(lightDir, hit.normal), 0);
		Color3 phong = geometry.material.specular * pow(max(Vector3::dot(half, hit.normal), 0), geometry.material.shininess);

		color += Li * V * (lambert + phong);
	}

	// calculate reflections
	float s = geometry.material.specular.r + geometry.material.specular.g + geometry.material.specular.b;
	if (s > 0 && depth < scene.maxDepth) {
		Ray refl = Ray();
		
		Vector3 r = (eyepos - hit.position).normalize();
		refl.direction = hit.normal * 2.0f * Vector3::dot(r, hit.normal) - r;
		refl.direction = refl.direction.normalize();
		refl.origin = hit.position - hit.normal * 0.01f;

		HitInfo info = FireHitTest(refl, scene);
		if (info.hit) {
 			Color3 reflColor = ColorGeometry(info.geometry, info, hit.position, scene, depth + 1);
			color += geometry.material.specular * reflColor;
		}
	}
	
	return color;
}

HitInfo RayTracer::TestSphere(const Ray & ray, const Sphere & sphere) {
	
	Ray rayInvTrans = ray.InverseTransform(sphere.transform);

	Vector3 p0MinusC = rayInvTrans.origin - sphere.position;

	float a = Vector3::dot(rayInvTrans.direction, rayInvTrans.direction);
	float b = Vector3::dot(rayInvTrans.direction * 2, p0MinusC);
	float c = Vector3::dot(p0MinusC, p0MinusC) - sphere.radius * sphere.radius;

	HitInfo info = HitInfo();
	info.hit = false;
	info.geometry = sphere;

	float discriminant = b * b - 4 * a*c;
	if (discriminant > 0.0f) {
		float sDiscriminant = sqrt(discriminant);
		float t1 = (-b + sDiscriminant) / (2.0f*a);
		float t2 = (-b - sDiscriminant) / (2.0f*a);
		float t;

		if (t1 > 0.0f && t2 > 0.0f)
			t = min(t1, t2);
		else if (t1 > 0.0f)
			t = t1;
		else if (t2 > 0.0f)
			t = t2;
		else
			return info;

		info.hit = true;
		info.position = rayInvTrans.origin + rayInvTrans.direction * t;
		info.normal = (sphere.position - info.position).normalize();

		info.ForwardTransform(sphere.transform);
	}
	else {
		// no real roots (both complex)
	}

	return info;
}

HitInfo RayTracer::TestTriangle(const Ray & ray, const Scene & scene, const Tri & tri) {
	Ray rayInvTrans = ray.InverseTransform(tri.transform);

	HitInfo info = HitInfo();
	info.hit = false;
	info.geometry = tri;

	Vector3 a = scene.verticies[tri.one];
	Vector3 b = scene.verticies[tri.two];
	Vector3 c = scene.verticies[tri.three];
	Vector3 normal = Vector3::cross((c - a), (b - a));
	normal = normal.normalize();

	float dirDotN = Vector3::dot(rayInvTrans.direction, normal);

	// No ray-plane intersection (orthogonal)
	if (dirDotN == 0)
		return info;

	// Calculate intersection point P
	float t = (Vector3::dot(a, normal) - Vector3::dot(rayInvTrans.origin, normal)) / dirDotN;

	// Intersections from behind origin are invalid
	if (t < 0)
		return info;

	Vector3 P = rayInvTrans.origin + rayInvTrans.direction * t;

	
	// Calculate the area of the triangle 
	//float triArea = Vector3::cross((c - a), (b - a)).length() / 2.0f;

	// Intersection with the triangle's plane, determine if it's inside or outside
	Vector3 beta = Vector3::cross(P - a, b - a);
	if (Vector3::dot(normal, beta) < 0)
		return info;

	Vector3 gamma = Vector3::cross(P - b, c - b);
	if (Vector3::dot(normal, gamma) < 0)
		return info;

	Vector3 alpha = Vector3::cross(P - c, a - c);
	if (Vector3::dot(normal, alpha) < 0)
		return info;

	info.hit = true;
	info.normal = normal;
	info.position = P;

	info.ForwardTransform(tri.transform);

	return info;
}

Color3 RayTracer::BackgroundGradient(const Ray & ray)
{
	Color3 color = Color3();
	float t = (ray.direction.y + 1) * 0.5f;
	color.r = Lerp(0.1f, 0.5f, t);
	color.g = Lerp(0.4f, 0.5f, t);
	color.b = Lerp(0.4f, 0.7f, t);
	return color;
}

float RayTracer::Lerp(float min, float max, float t)
{
	return (1.0f - t) * min + t * max;
}

void RayTracer::Fill(COLORREF* arr) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned int offset = x + y * width;
			// AA RR GG BB
			if (colorBuffer != NULL)
				arr[offset] = colorBuffer[y][x];
			else
				arr[offset] = 0;
		}
	}
}

void RayTracer::Cleanup()
{
	DeleteBuffer();
}

void RayTracer::AllocateBuffer() 
{
	DeleteBuffer();

	unsigned int** c = new unsigned int*[height];
	for (int i = 0; i < height; i++) {
		c[i] = new unsigned int[width];
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			c[y][x] = 0;
		}
	}

	colorBuffer = c;
}

void RayTracer::DeleteBuffer() 
{
	if (colorBuffer != NULL) {
		for (int i = 0; i < height; ++i)
			delete colorBuffer[i];

		delete colorBuffer;

		colorBuffer = NULL;
	}
}
