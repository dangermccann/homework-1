#include "stdafx.h"
#include "Vector3.h"
#include "Transform.h"
#include <math.h>

Vector3::Vector3() { }
Vector3::Vector3(const float _x, const float _y, const float _z)
{
	x = _x;
	y = _y;
	z = _z;
}

Vector3::Vector3(const Vector3 &v)
{
	x = v.x;
	y = v.y;
	z = v.z;
}

Vector3::~Vector3() { }


float Vector3::dot(const Vector3& v1, const Vector3& v2) 
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vector3 Vector3::cross(const Vector3& v1, const Vector3& v2) 
{
	Vector3 ret;
	ret.x = v1.y*v2.z - v1.z*v2.y;
	ret.y = v1.z*v2.x - v1.x*v2.z;
	ret.z = v1.x*v2.y - v1.y*v2.x;
	return ret;
}

float Vector3::length() const
{
	return sqrtf(x*x + y*y + z*z);
}

Vector3 Vector3::normalize() const
{
	return Vector3(x, y, z) / length();
}


Vector3 Vector3::ApplyTransformation(const Transform & t) const {
	return Vector3(t.x1*x + t.x2*y + t.x3*z + t.x4, t.y1*x + t.y2*y + t.y3*z + t.y4, t.z1*x + t.z2*y + t.z3*z + t.z4);
}

Vector3 Vector3::operator+(const Vector3 & other) const
{
	return Vector3(x + other.x, y + other.y, z + other.z);
}

Vector3 Vector3::operator+=(const Vector3 & a)
{
	x += a.x;
	y += a.y;
	z += a.z;
	return *this;
}

Vector3 Vector3::operator-(const Vector3 & other) const
{
	return Vector3(x - other.x, y - other.y, z - other.z);
}

Vector3 Vector3::operator-=(const Vector3 & a)
{
	x -= a.x;
	y -= a.y;
	z -= a.z;
	return *this;
}

Vector3 Vector3::operator*(const float a) const
{
	return Vector3(x*a, y*a, z*a);
}

Vector3 Vector3::operator*=(const float a)
{
	x *= a;
	y *= a;
	z *= a;
	return *this;
}


Vector3 Vector3::operator/(const float a) const
{
	return Vector3(x/a, y/a, z/a);
}

bool Vector3::operator==(const Vector3 & v1) const
{
	return x == v1.x && y == v1.y && z == v1.z;
}
