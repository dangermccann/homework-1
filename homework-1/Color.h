#pragma once

class Color3 {
public:
	float r;
	float g;
	float b;

	Color3() {
		r = g = b = 0;
	}
	Color3(float _r, float _g, float _b) {
		r = _r;
		g = _g;
		b = _b;
	}

	unsigned int argb() {
		unsigned int c = 0;
		unsigned int ri = (int)floor(max(min(r, 1.0f), 0) * 255.0f);
		unsigned int gi = (int)floor(max(min(g, 1.0f), 0) * 255.0f);
		unsigned int bi = (int)floor(max(min(b, 1.0f), 0) * 255.0f);
		c = (ri << 16) | (gi << 8) | bi;
		return c;
	}

	inline static int add(int first, int second) {
		unsigned int result = ((first & 0x000000ff) + (second & 0x000000ff) & 0x000000ff);
		result |= ((first & 0x0000ff00) + (second & 0x0000ff00) & 0x0000ff00);
		result |= ((first & 0x00ff0000) + (second & 0x00ff0000) & 0x00ff0000);
		return result;
	}

	inline Color3 operator+(const Color3& other) const {
		return Color3(r + other.r, g + other.g, b + other.b);
	}
	inline Color3 operator+=(const Color3& other) {
		r += other.r;
		g += other.g;
		b += other.b;
		return *this;
	}

	inline Color3 operator*(const Color3& other) const {
		return Color3(r * other.r, g * other.g, b * other.b);
	}
	inline Color3 operator*=(const Color3& other) {
		r *= other.r;
		g *= other.g;
		b *= other.b;
		return *this;
	}

	inline Color3 operator*(const float val) const {
		return Color3(r * val, g * val, b * val);
	}
	inline Color3 operator*=(const float val) {
		r *= val;
		g *= val;
		b *= val;
		return *this;
	}
};