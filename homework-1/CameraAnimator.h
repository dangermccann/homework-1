#pragma once
#include "Scene.h"
class CameraAnimator
{
private:
	float speed;
	Camera* camera;

public:
	CameraAnimator(Camera* _camera, float _speed);
	void Animate(float deltaTime);

};

