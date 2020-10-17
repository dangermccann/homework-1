#include "stdafx.h"
#include "CameraAnimator.h"


CameraAnimator::CameraAnimator(Camera* _camera, float _speed) 
{
	camera = _camera;
	speed = _speed;
}

void CameraAnimator::Animate(float deltaTime)
{
	Vector3 P = camera->lookAt - camera->lookFrom;
	

	Transform rotate = Transform::Rotate(90, Vector3(1, 0, 0));
	Transform reverse = rotate.Invert();

	P = P.ApplyTransformation(rotate);

	float r = sqrtf(P.x * P.x + P.y * P.y + P.z * P.z);
	//float theta = atanf(P.z / P.x);
	//float phi = atanf(sqrtf(P.x * P.x + P.z * P.z) / P.y);
	float phi = atanf(P.y / P.x);
	float theta = acosf(P.z / r);

	//theta += speed * deltaTime;
	phi += speed * deltaTime;
	if (phi > PI)
	{
	//	phi *= -1;
	}

	P.x = r * sinf(theta) * cosf(phi);
	P.y = r * sinf(phi) * sinf(theta);
	P.z = r * cosf(theta);

	P = P.ApplyTransformation(reverse);

	camera->lookFrom = camera->lookAt - P;
}