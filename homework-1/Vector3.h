#pragma once

typedef class Transform;

class Vector3
{
public:
	float x;
	float y;
	float z;

	Vector3();
	Vector3(const float _x, const float _y, const float _z);
	Vector3(const Vector3 &v);
	~Vector3();

	static Vector3 cross(const Vector3& v1, const Vector3& v2);
	static float dot(const Vector3& v1, const Vector3& v2);
	

	Vector3 normalize() const;
	float length() const;
	Vector3 ApplyTransformation(const Transform & t) const;


	// operators
	Vector3 operator+(const Vector3& other) const;
	Vector3 operator+=(const Vector3& other);
	Vector3 operator-(const Vector3& other) const;
	Vector3 operator-=(const Vector3& other);
	Vector3 operator*(const float a) const;
	Vector3 operator*=(const float a);
	Vector3 operator/(const float a) const;
	bool operator==(const Vector3& v1) const;
};

