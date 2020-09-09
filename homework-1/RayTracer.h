#pragma once
#include "Scene.h"

class Ray {
public:
	Vector3 origin;
	Vector3 direction;

	Ray() { }
	Ray(Vector3 orig, Vector3 dir) {
		origin = orig;
		direction = dir;
	}

	Ray InverseTransform(const Transform & t) const {
		Transform inv = t.Invert();
		Vector3 originTrans = origin.ApplyTransformation(inv);

		Transform dt = Transform(t);
		dt.x4 = 0;
		dt.y4 = 0;
		dt.z4 = 0;
		Vector3 directionTrans = direction.ApplyTransformation(dt.Invert());
		return Ray(originTrans, directionTrans);
	}
};

class HitInfo {
public:
	bool hit;
	Vector3 position;
	Vector3 normal;
	Geometry geometry;

	void ForwardTransform(const Transform & t) {
		position = position.ApplyTransformation(t);
		normal = normal.ApplyTransformation(t.Invert().Transpose()).normalize();
	}
};

class RayTracer
{
public:
	RayTracer();
	virtual ~RayTracer();

	void Trace(const Scene & scene);

	void Fill(COLORREF* arr);
	float Lerp(float min, float max, float t);
	void Cleanup();

	float progress;
	bool isRunning;

private:
	Color3 ColorGeometry(const Geometry & geometry, const HitInfo & hit, const Vector3 & eyepos, const Scene & scene, int depth);
	Color3 BackgroundGradient(const Ray & ray);
	HitInfo FireHitTest(const Ray & ray, const Scene & scene);
	HitInfo TestSphere(const Ray & ray, const Sphere & sphere);
	HitInfo TestTriangle(const Ray & ray, const Scene & scene, const Tri & tri);

	void DeleteBuffer();
	void AllocateBuffer();

	unsigned int** colorBuffer;
	int width, height;
};

