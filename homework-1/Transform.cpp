#include "stdafx.h"
#include "Transform.h"
#include <ctgmath>


/*
	x1 x2 x3 x4
	y1 y2 y3 y4
	z1 z2 z3 x4
	w1 w2 w3 w4
*/


Transform::Transform() 
{ 
	x1 = y2 = z3 = w4 = 1;
	x2 = x3 = x4 = 0;
	y1 = y3 = y4 = 0;
	z1 = z2 = z4 = 0;
	w1 = w2 = w3 = 0;
}

Transform::Transform(const float & _x1, const float & _y1, const float _z1, const float & _w1,
					 const float & _x2, const float & _y2, const float _z2, const float & _w2,
					 const float & _x3, const float & _y3, const float _z3, const float & _w3,
					 const float & _x4, const float & _y4, const float _z4, const float & _w4)
{
	x1 = _x1;
	x2 = _x2;
	x3 = _x3;
	x4 = _x4;
	y1 = _y1;
	y2 = _y2;
	y3 = _y3;
	y4 = _y4;
	z1 = _z1;
	z2 = _z2;
	z3 = _z3;
	z4 = _z4;
	w1 = _w1;
	w2 = _w2;
	w3 = _w3;
	w4 = _w4;
}

Transform::~Transform() { }

bool invertMatrix(const float m[16], float * invOut)
{
	float inv[16], det;
	int i;

	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return false;

	det = 1.0f / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}

Transform Transform::Invert() const
{
	// https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
	// http://graphics.stanford.edu/courses/cs248-98-fall/Final/q4.html
	float m[16], out[16];

	m[0] = x1;
	m[1] = y1;
	m[2] = z1;
	m[3] = w1;
	m[4] = x2;
	m[5] = y2;
	m[6] = z2;
	m[7] = w2;
	m[8] = x3;
	m[9] = y3;
	m[10] = z3;
	m[11] = w3;
	m[12] = x4;
	m[13] = y4;
	m[14] = z4;
	m[15] = w4;

	invertMatrix(m, out);

	return Transform(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10],
		out[11], out[12], out[13], out[14], out[15]);
}

Transform Transform::Transpose() const
{
	return Transform(x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, w1, w2, w3, w4);
}

bool Transform::Equals(const Transform & t) {
	return (t.x1 == x1 && t.x2 == x2 && t.x3 == x3 && t.x4 == x4 &&
			t.y1 == y1 && t.y2 == y2 && t.y3 == y3 && t.y4 == y4 &&	
			t.z1 == z1 && t.z2 == z2 && t.z3 == z3 && t.z4 == z4 &&
			t.w1 == w1 && t.w2 == w2 && t.w3 == w3 && t.w4 == w4 );
}

bool Transform::IsClose(const Transform & t, const float tol) {
	return (
		abs(x1-t.x1)< tol && abs(x2 - t.x2) < tol && abs(x3 - t.x3) < tol && abs(x4 - t.x4) < tol &&
		abs(y1 - t.y1) < tol && abs(y2 - t.y2) < tol && abs(y3 - t.y3) < tol && abs(y4 - t.y4) < tol &&
		abs(z1 - t.z1) < tol && abs(z2 - t.z2) < tol && abs(z3 - t.z3) < tol && abs(z4 - t.z4) < tol &&
		abs(w1 - t.w1) < tol && abs(w2 - t.w2) < tol && abs(w3 - t.w3) < tol && abs(w4 - t.w4) < tol );
}

Transform Transform::Rotate(const float degrees, const Vector3 & axis)
{
	float rads = degrees * PI / 180.0f;

	float  dx = cos(rads);
	Transform rx = Transform(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
	rx = rx * dx;

	float dy = 1 - cos(rads);
	Transform ry = Transform(axis.x*axis.x, axis.x*axis.y, axis.x*axis.z, 0, axis.x*axis.y, axis.y*axis.y, axis.y*axis.z, 0, 
							 axis.x*axis.z, axis.y*axis.z, axis.z*axis.z, 0, 0, 0, 0, 0);
	ry = ry * dy;

	float dz = sin(rads);
	Transform rz = Transform(0, axis.z, -axis.y, 0, -axis.z, 0, axis.x, 0, axis.y, -axis.x, 0, 0, 0, 0, 0, 0);
	rz = rz * dz;

	Transform result = Transform(rx + ry + rz);

	// Rotate here operates on a 3x3 matrix.  To convert to homogenous coordinates, we would 
	// multiply by a 4x4 identity matrix, which essentially is what this does:
	result.w4 = 1;

	return result;
}

Transform Transform::Scale(const float & sx, const float & sy, const float & sz)
{
	Transform scale = Transform(sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1);
	return scale;
}

Transform Transform::Translate(const float & tx, const float & ty, const float & tz)
{
	Transform trans = Transform(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1);
	return trans;
}

void Transform::ToArray16(float(&ary)[16]) const
{
	ary[0] = x1;
	ary[1] = x2;
	ary[2] = x3;
	ary[3] = x4;
	ary[4] = y1;
	ary[5] = y2;
	ary[6] = y3;
	ary[7] = y4;
	ary[8] = z1;
	ary[9] = z2;
	ary[10] = z3;
	ary[11] = z4;
	ary[12] = w1;
	ary[13] = w2;
	ary[14] = w3;
	ary[15] = w4;
}
 
void Transform::ToArray(float(&ary)[12]) const
{
	ary[0] = x1;
	ary[1] = x2;
	ary[2] = x3;
	ary[3] = x4;
	ary[4] = y1;
	ary[5] = y2;
	ary[6] = y3;
	ary[7] = y4;
	ary[8] = z1;
	ary[9] = z2;
	ary[10] = z3;
	ary[11] = z4;	
}

Transform Transform::operator*(const Transform a)
{
	Transform t = Transform();
	t.x1 = x1 * a.x1 + x2 * a.y1 + x3 * a.z1 + x4 * a.w1;
	t.x2 = x1 * a.x2 + x2 * a.y2 + x3 * a.z2 + x4 * a.w2;
	t.x3 = x1 * a.x3 + x2 * a.y3 + x3 * a.z3 + x4 * a.w3;
	t.x4 = x1 * a.x4 + x2 * a.y4 + x3 * a.z4 + x4 * a.w4;

	t.y1 = y1 * a.x1 + y2 * a.y1 + y3 * a.z1 + y4 * a.w1;
	t.y2 = y1 * a.x2 + y2 * a.y2 + y3 * a.z2 + y4 * a.w2;
	t.y3 = y1 * a.x3 + y2 * a.y3 + y3 * a.z3 + y4 * a.w3;
	t.y4 = y1 * a.x4 + y2 * a.y4 + y3 * a.z4 + y4 * a.w4;

	t.z1 = z1 * a.x1 + z2 * a.y1 + z3 * a.z1 + z4 * a.w1;
	t.z2 = z1 * a.x2 + z2 * a.y2 + z3 * a.z2 + z4 * a.w2;
	t.z3 = z1 * a.x3 + z2 * a.y3 + z3 * a.z3 + z4 * a.w3;
	t.z4 = z1 * a.x4 + z2 * a.y4 + z3 * a.z4 + z4 * a.w4;

	t.w1 = w1 * a.x1 + w2 * a.y1 + w3 * a.z1 + w4 * a.w1;
	t.w2 = w1 * a.x2 + w2 * a.y2 + w3 * a.z2 + w4 * a.w2;
	t.w3 = w1 * a.x3 + w2 * a.y3 + w3 * a.z3 + w4 * a.w3;
	t.w4 = w1 * a.x4 + w2 * a.y4 + w3 * a.z4 + w4 * a.w4;

	
	return t;
}
Transform Transform::operator*(const float & a)
{
	Transform t = Transform(x1*a, y1*a, z1*a, w1*a,
							x2*a, y2*a, z2*a, w2*a,
							x3*a, y3*a, z3*a, w3*a,
							x4*a, y4*a, z4*a, w4*a);
	return t;
}

Transform Transform::operator+(const Transform a)
{
	Transform t = Transform(x1+a.x1, y1+a.y1, z1+a.z1, w1+a.w1,
							x2+a.x2, y2+a.y2, z2+a.z2, w2+a.w2,
							x3+a.x3, y3+a.y3, z3+a.z3, w3+a.w3,
							x4+a.x4, y4+a.y4, z4+a.z4, w4+a.w4);
	return t;
}