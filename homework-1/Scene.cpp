#include "stdafx.h"
#include "Scene.h"
#include <iostream>
#include <fstream>
#include <stack>
using namespace std;

Scene::Scene()
{
	width = RT_WIDTH;
	height = RT_HEIGHT;
	maxDepth = 5;
	integrator = "raytracer";
	lightSamples = 1;
	lightStratify = 0;
	nextEventEstimation = 0;
	russianRoulette = 0;
	spp = 1;
	importanceSampling = 0;
	gamma = 1;
}


Scene::~Scene()
{
}

int Scene::Parse(LPCWSTR path)
{
	ifstream in;
	string str, cmd;
	float values[16];

	stack<Transform> transforms;
	transforms.push(Transform());

	Material material;
	material.ambient = Color3(0.2f, 0.2f, 0.2f);	// default value

	Vector3 atten = Vector3(1, 0, 0);


	in.open(path);
	if (in.is_open()) {
		getline(in, str);
		while (in) {
			// ignore comments
			if ((str.find_first_not_of(" \t\r\n") != string::npos) && (str[0] != '#')) {
				std::stringstream s(str);
				s >> cmd;
				
				if (cmd == "size") {
					if (ReadVals(s, 2, values)) {
						width = (int) values[0];
						height = (int)values[1];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "integrator") {
					s >> integrator;

					if (integrator != "raytracer") {
						// Original integrator called for default value of ambient light.
						material.ambient = Color3(0, 0, 0);
					}
				}
				else if (cmd == "maxdepth") {
					if (ReadVals(s, 1, values)) {
						maxDepth = (int) values[0];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "lightsamples") {
					if (ReadVals(s, 1, values)) {
						lightSamples = (int)values[0];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "spp") {
					if (ReadVals(s, 1, values)) {
						spp = (int)values[0];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "lightstratify") {
					std::string onOff;
					s >> onOff;

					if (onOff == "on")
						lightStratify = 1;
				}
				else if (cmd == "nexteventestimation") {
					std::string onOff;
					s >> onOff;

					if (onOff == "on")
						nextEventEstimation = 1;
				}
				else if (cmd == "russianroulette") {
					std::string onOff;
					s >> onOff;

					if (onOff == "on")
						russianRoulette = 1;
				}
				else if (cmd == "importancesampling") {
					std::string is;
					s >> is;

					if (is == "hemisphere")
						importanceSampling = 0;
					else if (is == "cosine")
						importanceSampling = 1;
					else if (is == "brdf")
						importanceSampling = 2;
				}
				else if (cmd == "brdf") {
					std::string brdfVal;
					s >> brdfVal;

					if (brdfVal == "ggx")
						material.brdfAlgorithm = 1;
				}
				else if (cmd == "gamma") {
					if (ReadVals(s, 1, values)) {
						gamma = values[0];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "output") {
					s >> outputFileName;
				}
				else if (cmd == "camera") {
					if (ReadVals(s, 10, values)) {
						camera.lookFrom = Vector3(values[0], values[1], values[2]);
						camera.lookAt = Vector3(values[3], values[4], values[5]);
						camera.up = Vector3(values[6], values[7], values[8]);
						camera.fieldOfView = values[9];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "sphere") {
					if (ReadVals(s, 4, values)) {
						Sphere s;
						s.position = Vector3(values[0], values[1], values[2]);
						s.radius = values[3];
						s.material = material;
						s.transform = transforms.top();

						// increment build input count if necessary
						if (spheres.size() > 0) {
							if (!spheres.back().transform.Equals(s.transform)) {
								sphereInputs++;
							}
						}
						else {
							sphereInputs++;
						}

						spheres.push_back(s);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "maxverts") { /* ignore*/ }
				else if (cmd == "maxvertnorms ") { /* ignore */ }
				else if (cmd == "vertex") {
					if (ReadVals(s, 3, values)) {
						Vector3 v = Vector3(values[0], values[1], values[2]);
						verticies.push_back(v);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "vertexnormal ") {
					// TODO: implement
				}
				else if (cmd == "tri") {
					if (ReadVals(s, 3, values)) {
						Tri t = Tri((int)values[0], (int)values[1], (int)values[2]);
						t.material = material;
						t.transform = transforms.top();

						// increment build input count if necessary
						if (tris.size() > 0) {
							if (!tris.back().transform.Equals(t.transform)) {
								triInputs++;
							}
						}
						else {
							triInputs++;
						}

						tris.push_back(t);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "trinormal") {
					// TODO: implement
				}
				else if (cmd == "translate") {
					if (ReadVals(s, 3, values)) {
						Transform t = Transform::Translate(values[0], values[1], values[2]);
						Transform &t2 = transforms.top();
						t2 = t2 * t;
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "rotate") {
					if (ReadVals(s, 4, values)) {
						Transform t = Transform::Rotate(values[3], Vector3(values[0], values[1], values[2]));
						Transform &t2 = transforms.top();
						t2 = t2 * t;
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "scale") {
					if (ReadVals(s, 3, values)) {
						Transform t = Transform::Scale(values[0], values[1], values[2]);
						Transform &t2 = transforms.top();
						t2 = t2 * t;
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "pushTransform") {
					transforms.push(Transform(transforms.top()));
				}
				else if (cmd == "popTransform") {
					transforms.pop();

					if (transforms.empty()) {
						cout << "Empty transform stack";
						return ERR_INVALID_FILE;
					}
				}
				else if (cmd == "directional") {
					if (ReadVals(s, 6, values)) {
						Light light = Light();
						light.position = Vector3(values[0], values[1], values[2]);
						light.color = Color3(values[3], values[4], values[5]);
						light.atten0 = atten.x;
						light.atten1 = atten.y;
						light.atten2 = atten.z;
						light.type = DIRECTIONAL_LIGHT;
						lights.push_back(light);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "point") {
					if (ReadVals(s, 6, values)) {
						Light light = Light();
						light.position = Vector3(values[0], values[1], values[2]);
						light.color = Color3(values[3], values[4], values[5]);
						light.atten0 = atten.x;
						light.atten1 = atten.y;
						light.atten2 = atten.z;
						light.type = POINT_LIGHT;
						lights.push_back(light);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "attenuation") {
					if (ReadVals(s, 3, values)) {
						atten.x = values[0];
						atten.y = values[1];
						atten.z = values[2];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "quadLight") {
					if (ReadVals(s, 12, values)) {
						QuadLight ql = QuadLight();
						ql.a = Vector3(values[0], values[1], values[2]);
						ql.ab = Vector3(values[3], values[4], values[5]);
						ql.ac = Vector3(values[6], values[7], values[8]);
						ql.intensity = Color3(values[9], values[10], values[11]);

						ql.material.emission = ql.intensity;


						quadLights.push_back(ql);
					}
				}
				else if (cmd == "ambient") {
					if (ReadVals(s, 3, values)) {
						material.ambient = Color3(values[0], values[1], values[2]);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "diffuse") {
					if (ReadVals(s, 3, values)) {
						material.diffuse = Color3(values[0], values[1], values[2]);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "specular") {
					if (ReadVals(s, 3, values)) {
						material.specular = Color3(values[0], values[1], values[2]);
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "shininess") {
					if (ReadVals(s, 1, values)) {
						material.shininess = values[0];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "roughness") {
					if (ReadVals(s, 1, values)) {
						material.roughness = values[0];
					}
					else { return ERR_INVALID_FILE; }
				}
				else if (cmd == "emission") {
				if (ReadVals(s, 3, values)) {
					material.emission = Color3(values[0], values[1], values[2]);
				}
				else { return ERR_INVALID_FILE; }
				}
				else {
					cerr << "Unknown Command: " << cmd << " Skipping \n";
				}
			}

			getline(in, str);
		}
	}
	else {
		cerr << "Unable to Open Input Data File " << path << "\n";
		return ERR_FILE_NOT_FOUND;
	}

	return 0;
}

bool Scene::ReadVals(stringstream &s, const int numvals, float* values)
{
	for (int i = 0; i < numvals; i++) {
		s >> values[i];
		if (s.fail()) {
			cout << "Failed reading value " << i << " will skip\n";
			return false;
		}
	}
	return true;
}
