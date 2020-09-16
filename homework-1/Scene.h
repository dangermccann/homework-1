#pragma once
#include <list>
#include <vector>
#include "Vector3.h"
#include "Transform.h"
#include "Color.h"
#include <sstream>
#include <string>



class Material {
public:
	Color3 diffuse, specular, emission;
	Color3 ambient;
	float shininess;

	Material() {
		shininess = 0;
	}
};

class Geometry {
public:
	Transform transform;
	Material material;
};

class Sphere : public Geometry {
public:
	Vector3 position;
	float radius;
};

class Tri : public Geometry {
public:
	unsigned int one, two, three;

	Tri(unsigned int _one, unsigned int _two, unsigned int _three) {
		one = _one;
		two = _two;
		three = _three;
	}
};

class Camera {
public:
	Vector3 lookFrom, lookAt, up;
	float fieldOfView;
};

#define POINT_LIGHT 0
#define DIRECTIONAL_LIGHT 1

class Light {
public:
	int type;
	Vector3 position;
	Color3 color;
	float atten0, atten1, atten2;
};

class QuadLight : public Geometry {
public:
	Vector3 a, ab, ac;
	Color3 intensity;

	void Verticies(Vector3 &v1, Vector3 &v2, Vector3 &v3, Vector3 &v4)
	{
		v1 = a;
		v2 = v1 + ab;
		v3 = v1 + ab + ac;
		v4 = v1 + ac;
	}
};

#define ERR_INVALID_FILE 1
#define ERR_FILE_NOT_FOUND 2

class Scene
{
public:
	int width, height, maxDepth, lightSamples, spp;
	int lightStratify;
	std::string outputFileName;
	std::string integrator;
	Scene();
	virtual ~Scene();

	int Parse(LPCWSTR path);

	std::list<Sphere> spheres;
	std::vector<Vector3> verticies;
	std::list<Tri> tris;
	std::list<Light> lights;
	std::list<QuadLight> quadLights;
	Camera camera;

	unsigned int triInputs;
	unsigned int sphereInputs;

private:
	bool ReadVals(std::stringstream &s, const int numvals, float* values);
};

