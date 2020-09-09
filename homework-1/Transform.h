#pragma once
#include "Vector3.h"

bool invertMatrix(const float m[16], float * invOut);

class Transform
{
public:
	float x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, w1, w2, w3, w4;

	Transform();
	Transform(const float & _x1, const float & _y1, const float _z1, const float & _w1, 
			  const float & _x2, const float & _y2, const float _z2, const float & _w2,
			  const float & _x3, const float & _y3, const float _z3, const float & _w3,
			  const float & _x4, const float & _y4, const float _z4, const float & _w4);
	~Transform();

	Transform Invert() const;
	Transform Transpose() const;
	bool Equals(const Transform & t);
	bool IsClose(const Transform & t, const float tol);

	static Transform Rotate(const float degrees, const Vector3& axis);
	static Transform Scale(const float &sx, const float &sy, const float &sz);
	static Transform Translate(const float &tx, const float &ty, const float &tz);

	void ToArray(float (&ary)[12]) const;
	void ToArray16(float(&ary)[16]) const;

	Transform operator*(const Transform a);
	Transform operator+(const Transform a);
	Transform operator*(const float & a);
};

