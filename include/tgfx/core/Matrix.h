/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Tencent is pleased to support the open source community by making tgfx available.
//
//  Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
//  Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//  in compliance with the License. You may obtain a copy of the License at
//
//      https://opensource.org/licenses/BSD-3-Clause
//
//  unless required by applicable law or agreed to in writing, software distributed under the
//  license is distributed on an "as is" basis, without warranties or conditions of any kind,
//  either express or implied. see the license for the specific language governing permissions
//  and limitations under the license.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstring>
#include "tgfx/core/Rect.h"

namespace tgfx {
const uint32_t MaxUInt32 = 0xFFFFFFFF;
const float MaxFloat     = 3.402823466e+38F;
const float MinPosFloat  = 1.175494351e-38F;
    
const float Pi     = 3.141592654f;
const float TwoPi  = 6.283185307f;
const float PiHalf = 1.570796327f;

const float Epsilon = 0.000001f;
const float ZeroEpsilon = 32.0f * MinPosFloat;  // Very small epsilon for checking against 0.0f

static inline float degToRad(float f)
{
    return f * 0.017453293f;
}

static inline float radToDeg(float f)
{
    return f * 57.29577951f;
}

inline bool IsFloatEqual(float x, float y, float tol = 0.0001f)
{
    bool bRet = false;
    if (std::abs(x - y) < tol)
    {
        bRet = true;
    }
    return bRet;
}
inline bool IsFloatZero(float val, float tol = 0.0001f)
{
    return IsFloatEqual(val, 0.0f, tol);
}
template<class T>
inline T Clamp(const T& t, const T& tLow, const T& tHigh)
{
    if (t < tHigh)
    {
        return ((t > tLow) ? t : tLow);
    }
    return tHigh;
}
template<class T>
inline T Mix(const T& t1, const T& t2, const float& ratio)
{
    return t1 * (1.0 - ratio) + t2 * ratio;
}
class Vec3f
{
public:
    float x, y, z;
    
    // ------------
    // Constructors
    // ------------
    Vec3f() : x( 0.0f ), y( 0.0f ), z( 0.0f )
    { 
    }
    
    Vec3f( const float x, const float y, const float z ) : x( x ), y( y ), z( z ) 
    {
    }

    Vec3f( const Vec3f &v ) : x( v.x ), y( v.y ), z( v.z )
    {
    }
    Vec3f& operator=(const Vec3f& other) {
      if (this != &other) { // 避免自赋值
          x = other.x;
          y = other.y;
          z = other.z;
      }
      return *this;
    }
    void set(const float x, const float y, const float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    // ------
    // Access
    // ------
    float &operator[]( uint32_t index )
    {
        return *(&x + index);
    }

    const float &operator[](uint32_t index) const
    {
        return *(&x + index);
    }
    
    // -----------
    // Comparisons
    // -----------
    bool operator==( const Vec3f &v ) const
    {
        return IsFloatEqual(x, v.x) && IsFloatEqual(y, v.y) && IsFloatEqual(z, v.z);
    }

    bool operator!=( const Vec3f &v ) const
    {
        return !(*this == v);
    }
    
    // ---------------------
    // Arithmetic operations
    // ---------------------
    Vec3f operator-() const
    {
        return Vec3f( -x, -y, -z );
    }

    Vec3f operator+( const Vec3f &v ) const
    {
        return Vec3f( x + v.x, y + v.y, z + v.z );
    }

    Vec3f &operator+=( const Vec3f &v )
    {
        return *this = *this + v;
    }

    Vec3f operator-( const Vec3f &v ) const 
    {
        return Vec3f( x - v.x, y - v.y, z - v.z );
    }

    Vec3f &operator-=( const Vec3f &v )
    {
        return *this = *this - v;
    }

    Vec3f operator*( const float f ) const
    {
        return Vec3f( x * f, y * f, z * f );
    }

    Vec3f operator*( const Vec3f &v ) const
    {
        return Vec3f( x * v.x, y * v.y, z * v.z );
    }
    Vec3f &operator*=( const Vec3f &v )
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    Vec3f &operator*=( const float f )
    {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    Vec3f operator/( const Vec3f &v ) const
    {
        return Vec3f( x / v.x, y / v.y, z / v.z );
    }
    Vec3f &operator/=( const Vec3f &v )
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }
    Vec3f operator/( const float f ) const
    {
        return Vec3f( x / f, y / f, z / f );
    }

    Vec3f &operator/=( const float f )
    {
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }

    // ----------------
    // Special products
    // ----------------
    float dot( const Vec3f &v ) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    Vec3f cross( const Vec3f &v ) const
    {
        return Vec3f( y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x );
    }

    // ----------------
    // Other operations
    // ----------------
    float length() const 
    {
        return sqrtf( x * x + y * y + z * z );
    }

    float sqrLength() const
    {
        return x * x + y * y + z * z;
    }

    Vec3f normalized() const
    {
        float l = length();
        if (l < 0.000001f)
        {
            return Vec3f(0.0f, 0.0f, 0.0f);
        }
        float invLen = 1.0f / l;
        return Vec3f( x * invLen, y * invLen, z * invLen );
    }

    void normalize()
    {
        float l = length();
        if (l < 0.000001f)
        {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
            return;
        }

        float invLen = 1.0f / l;
        x *= invLen;
        y *= invLen;
        z *= invLen;
    }

    static Vec3f Normalize(const Vec3f& v)
    {
        return v.normalized();
    }

    /*void fromRotation( float angleX, float angleY )
    {
        x = cosf( angleX ) * sinf( angleY ); 
        y = -sinf( angleX );
        z = cosf( angleX ) * cosf( angleY );
    }*/

    Vec3f toRotation() const
    {
        // Assumes that the unrotated view vector is (0, 0, -1)
        Vec3f v;
        
        if( y != 0 ) v.x = atan2f( y, sqrtf( x*x + z*z ) );
        if( x != 0 || z != 0 ) v.y = atan2f( -x, -z );

        return v;
    }

    Vec3f lerp( const Vec3f &v, float f ) const
    {
        return Vec3f( x + (v.x - x) * f, y + (v.y - y) * f, z + (v.z - z) * f ); 
    }

    static Vec3f Lerp(const Vec3f& a, const Vec3f& b, float f)
    {
        return a.lerp(b, f);
    }

    static float Angle(const Vec3f& from, const Vec3f& to)
    {
        float mod = from.sqrLength() * to.sqrLength();
        float dot = from.dot(to) / sqrt(mod);
        dot = Clamp(dot, -1.0f, 1.0f);
        
        float deg = radToDeg(acos(dot));

        return deg;
    }
};
class Vec4f
{
public:
    float x, y, z, w;

    Vec4f() : x( 0.0f ), y( 0.0f ), z( 0.0f ), w( 0.0f )
    {
    }

    explicit Vec4f( const float x, const float y, const float z, const float w ) :
        x( x ), y( y ), z( z ), w( w )
    {
    }

    explicit Vec4f( Vec3f v ) : x( v.x ), y( v.y ), z( v.z ), w( 1.0f )
    {
    }

    Vec4f operator+( const Vec4f &v ) const
    {
        return Vec4f( x + v.x, y + v.y, z + v.z, w + v.w );
    }

    Vec4f operator-(const Vec4f &v) const
    {
        return Vec4f(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    Vec4f operator-() const
    {
        return Vec4f( -x, -y, -z, -w );
    }
    
    Vec4f operator*( const float f ) const
    {
        return Vec4f( x * f, y * f, z * f, w * f );
    }

    Vec4f operator/(const float f) const
    {
        return Vec4f(x / f, y / f, z / f, w / f);
    }

    bool operator==(const Vec4f& right) const
    {
        return 
            IsFloatEqual(x, right.x) &&
            IsFloatEqual(y, right.y) && 
            IsFloatEqual(z, right.z) && 
            IsFloatEqual(w, right.w);
    }

    bool operator!=(const Vec4f& right) const
    {
        return !(*this == right);
    }

    void set(const float x, const float y, const float z, const float w)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
};
class Quaternion
{
public:
    float x, y, z, w;

    static const Quaternion& identity()
    {
        static Quaternion value(0.0f, 0.0f, 0.0f, 1.0f);
        return value;
    }

    // ------------
    // Constructors
    // ------------
    Quaternion() : x( 0.0f ), y( 0.0f ), z( 0.0f ), w( 0.0f ) 
    { 
    }
    
    explicit Quaternion( const float x, const float y, const float z, const float w ) :
        x( x ), y( y ), z( z ), w( w )
    {
    }
    
    Quaternion( const float eulerX, const float eulerY, const float eulerZ )
    {
      float roll = eulerX, pitch = eulerY, yaw = eulerZ;
      float cy = static_cast<float>(cos(yaw * 0.5));
      float sy = static_cast<float>(sin(yaw * 0.5));
      float cp = static_cast<float>(cos(pitch * 0.5));
      float sp = static_cast<float>(sin(pitch * 0.5));
      float cr = static_cast<float>(cos(roll * 0.5));
      float sr = static_cast<float>(sin(roll * 0.5));

      w = cr * cp * cy + sr * sp * sy;
      x = sr * cp * cy + cr * sp * sy;
      y = cr * sp * cy - sr * cp * sy;
      z = cr * cp * sy - sr * sp * cy;// Order: y * x * z, z axis as main axis
    }

    // ---------------------
    // Arithmetic operations
    // ---------------------
    Quaternion operator*( const Quaternion &q ) const
    {
        return Quaternion(
            y * q.z - z * q.y + q.x * w + x * q.w,
            z * q.x - x * q.z + q.y * w + y * q.w,
            x * q.y - y * q.x + q.z * w + z * q.w,
            w * q.w - (x * q.x + y * q.y + z * q.z)
        );
    }

    Quaternion &operator*=( const Quaternion &q )
    {
        return *this = *this * q;
    }

    bool operator ==(const Quaternion &q)
    {
        return IsFloatEqual(x, q.x) && IsFloatEqual(y, q.y) && IsFloatEqual(z, q.z) && IsFloatEqual(w, q.w);
    }

    bool operator !=(const Quaternion &q)
    {
        return !(*this == q);
    }

    Quaternion slerp( const Quaternion &q, const float t ) const
    {
        // Spherical linear interpolation between two quaternions
        // Note: SLERP is not commutative
        
        Quaternion q1( q );

        // Calculate cosine
        float cosTheta = x * q.x + y * q.y + z * q.z + w * q.w;

        // Use the shortest path
        if( cosTheta < 0 )
        {
            cosTheta = -cosTheta; 
            q1.x = -q.x; q1.y = -q.y;
            q1.z = -q.z; q1.w = -q.w;
        }

        // Initialize with linear interpolation
        float scale0 = 1 - t, scale1 = t;
        
        // Use spherical interpolation only if the quaternions are not very close
        if( (1 - cosTheta) > 0.001f )
        {
            // SLERP
            float theta = acosf( cosTheta );
            float sinTheta = sinf( theta );
            scale0 = sinf( (1 - t) * theta ) / sinTheta;
            scale1 = sinf( t * theta ) / sinTheta;
        } 
        
        // Calculate final quaternion
        return Quaternion(
            x * scale0 + q1.x * scale1, y * scale0 + q1.y * scale1,
            z * scale0 + q1.z * scale1, w * scale0 + q1.w * scale1
        );
    }

    Quaternion nlerp( const Quaternion &q, const float t ) const
    {
        // Normalized linear quaternion interpolation
        // Note: NLERP is faster than SLERP and commutative but does not yield constant velocity

        Quaternion qt;
        float cosTheta = x * q.x + y * q.y + z * q.z + w * q.w;
        
        // Use the shortest path and interpolate linearly
        if (cosTheta < 0)
        {
            qt = Quaternion(
                x + (-q.x - x) * t, y + (-q.y - y) * t,
                z + (-q.z - z) * t, w + (-q.w - w) * t
            );
        }
        else
        {
            qt = Quaternion(
                x + (q.x - x) * t, y + (q.y - y) * t,
                z + (q.z - z) * t, w + (q.w - w) * t
            );
        }

        // Return normalized quaternion
        float invLen = 1.0f / sqrtf( qt.x * qt.x + qt.y * qt.y + qt.z * qt.z + qt.w * qt.w );
        return Quaternion( qt.x * invLen, qt.y * invLen, qt.z * invLen, qt.w * invLen );
    }

    Quaternion inverted() const
    {
        float len = x * x + y * y + z * z + w * w;
        if( len > 0 )
        {
            float invLen = 1.0f / len;
            return Quaternion( -x * invLen, -y * invLen, -z * invLen, w * invLen );
        }
        else return Quaternion();
    }

    Vec3f operator *(const Vec3f& p) const
    {
        Quaternion p_ = *this * Quaternion(p.x, p.y, p.z, 0) * this->inverted();

        return Vec3f(p_.x, p_.y, p_.z);
    }

    float dot(const Quaternion& v) const
    {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }

    float length() const
    {
        return sqrtf(x * x + y * y + z * z + w * w);
    }

    Quaternion normalized() const
    {
        float l = length();
        if (l < Epsilon)
        {
            return identity();
        }
        
        return Quaternion(x / l, y / l, z / l, w / l);
    }

    void normalize()
    {
        //*this = this->normalized();

        float l = length();
        if (l < Epsilon)
        {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
            w = 1.0f;
        }
        x /= l;
        y /= l;
        z /= l;
        w /= l;
    }

    static Quaternion AngleAxis(float angle, const Vec3f& axis)
    {
        Vec3f v = Vec3f::Normalize(axis);
        float cosv, sinv;
        
        cosv = cos(degToRad(angle * 0.5f));
        sinv = sin(degToRad(angle * 0.5f));

        float x = v.x * sinv;
        float y = v.y * sinv;
        float z = v.z * sinv;
        float w = cosv;

        return Quaternion(x, y, z, w);
    }

    static Quaternion FromToRotation(const Vec3f& from_direction, const Vec3f& to_direction)
    {
        Vec3f origin = Vec3f::Normalize(from_direction);
        Vec3f fn = Vec3f::Normalize(to_direction);

        if (fn != origin)
        {
            if (!IsFloatZero(fn.sqrLength()) && !IsFloatZero(origin.sqrLength()))
            {
                float deg = Vec3f::Angle(origin, fn);
                Vec3f axis = origin.cross(fn);
                
                if (axis == Vec3f(0, 0, 0))
                {
                    if (!IsFloatZero(origin.x))
                    {
                        float x = -origin.y / origin.x;
                        float y = 1;
                        float z = 0;

                        axis = Vec3f(x, y, z);
                    }
                    else if (!IsFloatZero(origin.y))
                    {
                        float y = -origin.z / origin.y;
                        float x = 0;
                        float z = 1;

                        axis = Vec3f(x, y, z);
                    }
                    else
                    {
                        float z = -origin.x / origin.z;
                        float y = 0;
                        float x = 1;

                        axis = Vec3f(x, y, z);
                    }

                    return AngleAxis(deg, axis);
                }
                else
                {
                    return AngleAxis(deg, axis);
                }
            }
        }

        return identity();
    }

    static Quaternion LookRotation(const Vec3f& forward, const Vec3f& up)
    {
        Vec3f un = up.normalized();
        Vec3f fn = forward.normalized();

        Quaternion rot0 = FromToRotation(Vec3f(0, 1, 0), un);
        Vec3f f = rot0 * Vec3f(0, 0, 1);
        float deg = Vec3f::Angle(f, fn);
        Quaternion rot1;
        Vec3f axis = f.cross(fn);
        float d = axis.dot(un);
        if (d > 0)
        {
            rot1 = AngleAxis(deg, up);
        }
        else
        {
            rot1 = AngleAxis(-deg, up);
        }
        return rot1 * rot0;
    }

    static Quaternion Euler(const Vec3f& euler)
    {
        return Quaternion(euler.x, euler.y, euler.z);
    }

    static Quaternion EulerRadians(float x, float y, float z)
    {
        return Quaternion(x, y, z);
    }

    static Quaternion EulerDegree(float x, float y, float z)
    {
        return Quaternion(degToRad(x), degToRad(y), degToRad(z));
    }

	Vec3f toEulerAngles() const
	{
		float rx = asin(2 * (w * x - y * z));
		float ry = atan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y));
		float rz = atan2(2 * (w * z + x * y), 1 - 2 * (z * z + x * x));

		return Vec3f(radToDeg(rx), radToDeg(ry), radToDeg(rz));
	}
};
class Matrix4f
{
public:
    union
    {
        float c[4][4];    // Column major order for OpenGL: c[column][row]
        float x[16];
    };

    static Matrix4f TransMat( float x, float y, float z )
    {
        Matrix4f m;

        m.c[3][0] = x;
        m.c[3][1] = y;
        m.c[3][2] = z;

        return m;
    }

    static Matrix4f ScaleMat( float x, float y, float z )
    {
        Matrix4f m;
        
        m.c[0][0] = x;
        m.c[1][1] = y;
        m.c[2][2] = z;

        return m;
    }

    static Matrix4f RotMat(const Quaternion& q)
    {
        return Matrix4f(q);
    }

    static Matrix4f RotMat( float x, float y, float z ) // x, y, z in radians
    {
        // Rotation order: rotY * rotX * rotZ [* Vector], 以z轴作为第一主轴
        return Matrix4f( Quaternion( x, y, z ) );
    }

    static Matrix4f RotMat( Vec3f axis, float angle )
    {
        axis.normalize();
        axis *= sinf( angle * 0.5f );
        return Matrix4f( Quaternion( axis.x, axis.y, axis.z, cosf( angle * 0.5f ) ) );
    }

    static Matrix4f TRS(const Vec3f& translate, const Quaternion& rotate, const Vec3f& scale)
    {
        Matrix4f mt = TransMat(translate.x, translate.y, translate.z);
        Matrix4f mr = Matrix4f(rotate);
        Matrix4f ms = ScaleMat(scale.x, scale.y, scale.z);

        return mt * mr * ms;
    }

    static void LookAtRH(Matrix4f& result,
        const Vec3f& eye,
        const Vec3f& target,
        const Vec3f& up)
    {
      Vec3f f = eye - target; f.normalize();
      Vec3f s = up.cross(f); s.normalize();
      Vec3f u = f.cross(s); u.normalize();

      result.c[0][0] = s.x;
      result.c[1][0] = s.y;
      result.c[2][0] = s.z;
      result.c[0][1] = u.x;
      result.c[1][1] = u.y;
      result.c[2][1] = u.z;
      result.c[0][2] = f.x;
      result.c[1][2] = f.y;
      result.c[2][2] = f.z;
      result.c[3][0] = -s.dot(eye);
      result.c[3][1] = -u.dot(eye);
      result.c[3][2] = -f.dot(eye);
      result.c[3][3] = 1.0;
    }

    static Matrix4f LookAtMatRH(
        const Vec3f& eye,
        const Vec3f& target,
        const Vec3f& up)
    {
        Matrix4f m;
        Matrix4f::LookAtRH(m, eye, target, up);
        return m;
    }

	static Matrix4f LookAtMatLH(
		const Vec3f& eye,
		const Vec3f& target,
		const Vec3f& up)
	{
		Matrix4f m;
		Vec3f f = target - eye; f.normalize();
		Vec3f s = up.cross(f); s.normalize();
		Vec3f u = f.cross(s); u.normalize();

		m.c[0][0] = s.x;
		m.c[1][0] = s.y;
		m.c[2][0] = s.z;
		m.c[0][1] = u.x;
		m.c[1][1] = u.y;
		m.c[2][1] = u.z;
		m.c[0][2] = f.x;
		m.c[1][2] = f.y;
		m.c[2][2] = f.z;
		m.c[3][0] = -s.dot(eye);
		m.c[3][1] = -u.dot(eye);
		m.c[3][2] = -f.dot(eye);

		return m;
	}

  static Matrix4f PerspectiveMatRH(float fov, float aspect, float n, float f)
  {
      float frustumH = tanf(fov / 360.0f * Pi) * n;
      float frustumW = frustumH * aspect;
      return Matrix4f::PerspectiveMatRH(-frustumW, frustumW, -frustumH, frustumH, n, f);
  }

  static Matrix4f PerspectiveMatRH( float l, float r, float b, float t, float n, float f )
  {
      Matrix4f m;

      m.x[0] = 2 * n / (r - l);
      m.x[5] = 2 * n / (t - b);
      m.x[8] = (r + l) / (r - l);
      m.x[9] = (t + b) / (t - b);
      m.x[10] = -(f + n) / (f - n);
      m.x[11] = -1;
      m.x[14] = -2 * f * n / (f - n);
      m.x[15] = 0;

      return m;
  }

	static Matrix4f PerspectiveMatLH(float fov, float aspect, float n, float f)
	{
		float frustumH = tanf(fov / 360.0f * Pi) * n;
		float frustumW = frustumH * aspect;

		auto cal = [=](float l, float r, float b, float t, float n, float f)->Matrix4f
		{
			Matrix4f m;

			m.x[0] = 2 * n / (r - l);
			m.x[5] = 2 * n / (t - b);
			m.x[8] = -(r + l) / (r - l);
			m.x[9] = -(t + b) / (t - b);
			m.x[10] = (f + n) / (f - n);
			m.x[11] = 1;
			m.x[14] = -2 * f * n / (f - n);
			m.x[15] = 0;

			return m;
		};

		return cal(-frustumW, frustumW, -frustumH, frustumH, n, f);
	}

  static Matrix4f OrthoMatRH( float l, float r, float b, float t, float n, float f )
  {
      Matrix4f m;

      m.x[0] = 2 / (r - l);
      m.x[5] = 2 / (t - b);
      m.x[10] = -2 / (f - n);
      m.x[12] = -(r + l) / (r - l);
      m.x[13] = -(t + b) / (t - b);
      m.x[14] = -(f + n) / (f - n);

      return m;
  }

	static Matrix4f OrthoMatLH(float l, float r, float b, float t, float n, float f)
	{
		Matrix4f m;
		m.x[0] = 2 / (r - l);
		m.x[5] = 2 / (t - b);
		m.x[10] = 2 / (f - n);
		m.x[12] = -(r + l) / (r - l);
		m.x[13] = -(t + b) / (t - b);
		m.x[14] = -(f + n) / (f - n);

		return m;
	}

  static void fastMult43( Matrix4f &dst, const Matrix4f &m1, const Matrix4f &m2 )
  {
      // Note: dst may not be the same as m1 or m2

      float *dstx = dst.x;
      const float *m1x = m1.x;
      const float *m2x = m2.x;
      
      dstx[0] = m1x[0] * m2x[0] + m1x[4] * m2x[1] + m1x[8] * m2x[2];
      dstx[1] = m1x[1] * m2x[0] + m1x[5] * m2x[1] + m1x[9] * m2x[2];
      dstx[2] = m1x[2] * m2x[0] + m1x[6] * m2x[1] + m1x[10] * m2x[2];
      dstx[3] = 0.0f;

      dstx[4] = m1x[0] * m2x[4] + m1x[4] * m2x[5] + m1x[8] * m2x[6];
      dstx[5] = m1x[1] * m2x[4] + m1x[5] * m2x[5] + m1x[9] * m2x[6];
      dstx[6] = m1x[2] * m2x[4] + m1x[6] * m2x[5] + m1x[10] * m2x[6];
      dstx[7] = 0.0f;

      dstx[8] = m1x[0] * m2x[8] + m1x[4] * m2x[9] + m1x[8] * m2x[10];
      dstx[9] = m1x[1] * m2x[8] + m1x[5] * m2x[9] + m1x[9] * m2x[10];
      dstx[10] = m1x[2] * m2x[8] + m1x[6] * m2x[9] + m1x[10] * m2x[10];
      dstx[11] = 0.0f;

      dstx[12] = m1x[0] * m2x[12] + m1x[4] * m2x[13] + m1x[8] * m2x[14] + m1x[12] * m2x[15];
      dstx[13] = m1x[1] * m2x[12] + m1x[5] * m2x[13] + m1x[9] * m2x[14] + m1x[13] * m2x[15];
      dstx[14] = m1x[2] * m2x[12] + m1x[6] * m2x[13] + m1x[10] * m2x[14] + m1x[14] * m2x[15];
      dstx[15] = 1.0f;
  }

    // ------------
    // Constructors
    // ------------
    Matrix4f()
    {
        c[0][0] = 1; c[1][0] = 0; c[2][0] = 0; c[3][0] = 0;
        c[0][1] = 0; c[1][1] = 1; c[2][1] = 0; c[3][1] = 0;
        c[0][2] = 0; c[1][2] = 0; c[2][2] = 1; c[3][2] = 0;
        c[0][3] = 0; c[1][3] = 0; c[2][3] = 0; c[3][3] = 1;
    }

    Matrix4f(float m11, float m12, float m13, float m14, float m21, float m22, float m23, float m24,
             float m31, float m32, float m33, float m34, float m41, float m42, float m43, float m44)
    {
        x[ 0 ]  = m11; x[ 1 ]  = m21; x[ 2 ]  = m31; x[ 3 ]  = m41;
        x[ 4 ]  = m12; x[ 5 ]  = m22; x[ 6 ]  = m32; x[ 7 ]  = m42;
        x[ 8 ]  = m13; x[ 9 ]  = m23; x[ 10 ] = m33; x[ 11 ] = m43;
        x[ 12 ] = m14; x[ 13 ] = m24; x[ 14 ] = m34; x[ 15 ] = m44;
    }

    Matrix4f( const float *floatArray16 )
    {
        for(uint32_t i = 0; i < 4; ++i)
        {
            for(uint32_t j = 0; j < 4; ++j)
            {
                c[i][j] = floatArray16[i * 4 + j];
            }
        }
    }

    Matrix4f( const Quaternion &q )
    {
        // Calculate coefficients
        float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
        float xx = q.x * x2,  xy = q.x * y2,  xz = q.x * z2;
        float yy = q.y * y2,  yz = q.y * z2,  zz = q.z * z2;
        float wx = q.w * x2,  wy = q.w * y2,  wz = q.w * z2;

        c[0][0] = 1 - (yy + zz);  c[1][0] = xy - wz;    
        c[2][0] = xz + wy;        c[3][0] = 0;
        c[0][1] = xy + wz;        c[1][1] = 1 - (xx + zz);
        c[2][1] = yz - wx;        c[3][1] = 0;
        c[0][2] = xz - wy;        c[1][2] = yz + wx;
        c[2][2] = 1 - (xx + yy);  c[3][2] = 0;
        c[0][3] = 0;              c[1][3] = 0;
        c[2][3] = 0;              c[3][3] = 1;
    }

    Matrix4f(const Matrix4f &m) {
      if (this != &m) {
        memcpy(x, m.x, sizeof(Matrix4f));
      }
    }
     // ----------
    // Matrix sum
    // ----------
    Matrix4f operator+( const Matrix4f &m ) const 
    {
        Matrix4f mf;
        
        mf.x[0]  = x[0]  + m.x[0];
        mf.x[1]  = x[1]  + m.x[1];
        mf.x[2]  = x[2]  + m.x[2];
        mf.x[3]  = x[3]  + m.x[3];
        mf.x[4]  = x[4]  + m.x[4];
        mf.x[5]  = x[5]  + m.x[5];
        mf.x[6]  = x[6]  + m.x[6];
        mf.x[7]  = x[7]  + m.x[7];
        mf.x[8]  = x[8]  + m.x[8];
        mf.x[9]  = x[9]  + m.x[9];
        mf.x[10] = x[10] + m.x[10];
        mf.x[11] = x[11] + m.x[11];
        mf.x[12] = x[12] + m.x[12];
        mf.x[13] = x[13] + m.x[13];
        mf.x[14] = x[14] + m.x[14];
        mf.x[15] = x[15] + m.x[15];

        return mf;
    }

    Matrix4f &operator+=( const Matrix4f &m )
    {
        return *this = *this + m;
    }

    Matrix4f &operator=(const Matrix4f &m)
    {
        memcpy(x, m.x, sizeof(Matrix4f));
        return *this;
    }

    Matrix4f operator*( const Matrix4f &m ) const
    {
        Matrix4f mf;
        
        mf.x[0] = x[0] * m.x[0] + x[4] * m.x[1] + x[8] * m.x[2] + x[12] * m.x[3];
        mf.x[1] = x[1] * m.x[0] + x[5] * m.x[1] + x[9] * m.x[2] + x[13] * m.x[3];
        mf.x[2] = x[2] * m.x[0] + x[6] * m.x[1] + x[10] * m.x[2] + x[14] * m.x[3];
        mf.x[3] = x[3] * m.x[0] + x[7] * m.x[1] + x[11] * m.x[2] + x[15] * m.x[3];

        mf.x[4] = x[0] * m.x[4] + x[4] * m.x[5] + x[8] * m.x[6] + x[12] * m.x[7];
        mf.x[5] = x[1] * m.x[4] + x[5] * m.x[5] + x[9] * m.x[6] + x[13] * m.x[7];
        mf.x[6] = x[2] * m.x[4] + x[6] * m.x[5] + x[10] * m.x[6] + x[14] * m.x[7];
        mf.x[7] = x[3] * m.x[4] + x[7] * m.x[5] + x[11] * m.x[6] + x[15] * m.x[7];

        mf.x[8] = x[0] * m.x[8] + x[4] * m.x[9] + x[8] * m.x[10] + x[12] * m.x[11];
        mf.x[9] = x[1] * m.x[8] + x[5] * m.x[9] + x[9] * m.x[10] + x[13] * m.x[11];
        mf.x[10] = x[2] * m.x[8] + x[6] * m.x[9] + x[10] * m.x[10] + x[14] * m.x[11];
        mf.x[11] = x[3] * m.x[8] + x[7] * m.x[9] + x[11] * m.x[10] + x[15] * m.x[11];

        mf.x[12] = x[0] * m.x[12] + x[4] * m.x[13] + x[8] * m.x[14] + x[12] * m.x[15];
        mf.x[13] = x[1] * m.x[12] + x[5] * m.x[13] + x[9] * m.x[14] + x[13] * m.x[15];
        mf.x[14] = x[2] * m.x[12] + x[6] * m.x[13] + x[10] * m.x[14] + x[14] * m.x[15];
        mf.x[15] = x[3] * m.x[12] + x[7] * m.x[13] + x[11] * m.x[14] + x[15] * m.x[15];

        return mf;
    }

    Matrix4f operator*( float f ) const
    {
        Matrix4f m( *this );
        
        m.x[0]  *= f; m.x[1]  *= f; m.x[2]  *= f; m.x[3]  *= f;
        m.x[4]  *= f; m.x[5]  *= f; m.x[6]  *= f; m.x[7]  *= f;
        m.x[8]  *= f; m.x[9]  *= f; m.x[10] *= f; m.x[11] *= f;
        m.x[12] *= f; m.x[13] *= f; m.x[14] *= f; m.x[15] *= f;

        return m;
    }

    // ----------------------------
    // Vector-Matrix multiplication
    // ----------------------------
    Vec3f operator*( const Vec3f &v ) const
    {
        return Vec3f(
            v.x * c[0][0] + v.y * c[1][0] + v.z * c[2][0] + c[3][0],
            v.x * c[0][1] + v.y * c[1][1] + v.z * c[2][1] + c[3][1],
            v.x * c[0][2] + v.y * c[1][2] + v.z * c[2][2] + c[3][2]
        );
    }

    Vec4f operator*( const Vec4f &v ) const
    {
        return Vec4f(
            v.x * c[0][0] + v.y * c[1][0] + v.z * c[2][0] + v.w * c[3][0],
            v.x * c[0][1] + v.y * c[1][1] + v.z * c[2][1] + v.w * c[3][1],
            v.x * c[0][2] + v.y * c[1][2] + v.z * c[2][2] + v.w * c[3][2],
            v.x * c[0][3] + v.y * c[1][3] + v.z * c[2][3] + v.w * c[3][3]
        );
    }

    Vec3f multiplyPoint3x4(const Vec3f &v) const
    {
        return (*this) * v;
    }

    Vec3f multiplyDirection(const Vec3f& v) const
    {
        float vx, vy, vz;

        vx = v.x * c[0][0] + v.y * c[1][0] + v.z * c[2][0];
        vy = v.x * c[0][1] + v.y * c[1][1] + v.z * c[2][1];
        vz = v.x * c[0][2] + v.y * c[1][2] + v.z * c[2][2];

        return Vec3f(vx, vy, vz);
    }

    Vec3f mult33Vec( const Vec3f &v ) const
    {
        return Vec3f(
            v.x * c[0][0] + v.y * c[1][0] + v.z * c[2][0] + c[3][0],
            v.x * c[0][1] + v.y * c[1][1] + v.z * c[2][1] + c[3][1],
            v.x * c[0][2] + v.y * c[1][2] + v.z * c[2][2] + c[3][2]
        );
    }

    // ---------------
    // Transformations
    // ---------------
    void translate( const float x, const float y, const float z )
    {
        *this = TransMat( x, y, z ) * *this;
    }

    void scale( const float x, const float y, const float z )
    {
        *this = ScaleMat( x, y, z ) * *this;
    }

    void scaleMix( const float x, const float y, const float z )
    {
        c[0][0] *= x;  c[1][0] *= y;  c[2][0] *= z;
        c[0][1] *= x;  c[1][1] *= y;  c[2][1] *= z;
        c[0][2] *= x;  c[1][2] *= y;  c[2][2] *= z;
        c[0][3] *= x;  c[1][3] *= y;  c[2][3] *= z;
    }

    void rotate( const float x, const float y, const float z )
    {
        *this = RotMat( x, y, z ) * *this;
    }

    // ---------------
    // Other
    // ---------------
    Matrix4f& transpose()
    {
        for (uint32_t y = 0; y < 4; ++y)
        {
            for (uint32_t x = y + 1; x < 4; ++x)
            {
                float tmp = c[x][y];
                c[x][y] = c[y][x];
                c[y][x] = tmp;
            }
        }

        return *this;
    }

    Matrix4f transposed() const
    {
        Matrix4f m( *this );
        for(uint32_t y = 0; y < 4; ++y)
        {
            for(uint32_t x = y + 1; x < 4; ++x) 
            {
                float tmp = m.c[x][y];
                m.c[x][y] = m.c[y][x];
                m.c[y][x] = tmp;
            }
        }

        return m;
    }

    float determinant() const
    {
        return 
            c[0][3]*c[1][2]*c[2][1]*c[3][0] - c[0][2]*c[1][3]*c[2][1]*c[3][0] - c[0][3]*c[1][1]*c[2][2]*c[3][0] + c[0][1]*c[1][3]*c[2][2]*c[3][0] +
            c[0][2]*c[1][1]*c[2][3]*c[3][0] - c[0][1]*c[1][2]*c[2][3]*c[3][0] - c[0][3]*c[1][2]*c[2][0]*c[3][1] + c[0][2]*c[1][3]*c[2][0]*c[3][1] +
            c[0][3]*c[1][0]*c[2][2]*c[3][1] - c[0][0]*c[1][3]*c[2][2]*c[3][1] - c[0][2]*c[1][0]*c[2][3]*c[3][1] + c[0][0]*c[1][2]*c[2][3]*c[3][1] +
            c[0][3]*c[1][1]*c[2][0]*c[3][2] - c[0][1]*c[1][3]*c[2][0]*c[3][2] - c[0][3]*c[1][0]*c[2][1]*c[3][2] + c[0][0]*c[1][3]*c[2][1]*c[3][2] +
            c[0][1]*c[1][0]*c[2][3]*c[3][2] - c[0][0]*c[1][1]*c[2][3]*c[3][2] - c[0][2]*c[1][1]*c[2][0]*c[3][3] + c[0][1]*c[1][2]*c[2][0]*c[3][3] +
            c[0][2]*c[1][0]*c[2][1]*c[3][3] - c[0][0]*c[1][2]*c[2][1]*c[3][3] - c[0][1]*c[1][0]*c[2][2]*c[3][3] + c[0][0]*c[1][1]*c[2][2]*c[3][3];
    }

    Matrix4f inverted() const
    {
        Matrix4f m;

        float d = determinant();
        if( d == 0 ) return m;
        d = 1.0f / d;

        m.c[0][0] = d * (c[1][2]*c[2][3]*c[3][1] - c[1][3]*c[2][2]*c[3][1] + c[1][3]*c[2][1]*c[3][2] - c[1][1]*c[2][3]*c[3][2] - c[1][2]*c[2][1]*c[3][3] + c[1][1]*c[2][2]*c[3][3]);
        m.c[0][1] = d * (c[0][3]*c[2][2]*c[3][1] - c[0][2]*c[2][3]*c[3][1] - c[0][3]*c[2][1]*c[3][2] + c[0][1]*c[2][3]*c[3][2] + c[0][2]*c[2][1]*c[3][3] - c[0][1]*c[2][2]*c[3][3]);
        m.c[0][2] = d * (c[0][2]*c[1][3]*c[3][1] - c[0][3]*c[1][2]*c[3][1] + c[0][3]*c[1][1]*c[3][2] - c[0][1]*c[1][3]*c[3][2] - c[0][2]*c[1][1]*c[3][3] + c[0][1]*c[1][2]*c[3][3]);
        m.c[0][3] = d * (c[0][3]*c[1][2]*c[2][1] - c[0][2]*c[1][3]*c[2][1] - c[0][3]*c[1][1]*c[2][2] + c[0][1]*c[1][3]*c[2][2] + c[0][2]*c[1][1]*c[2][3] - c[0][1]*c[1][2]*c[2][3]);
        m.c[1][0] = d * (c[1][3]*c[2][2]*c[3][0] - c[1][2]*c[2][3]*c[3][0] - c[1][3]*c[2][0]*c[3][2] + c[1][0]*c[2][3]*c[3][2] + c[1][2]*c[2][0]*c[3][3] - c[1][0]*c[2][2]*c[3][3]);
        m.c[1][1] = d * (c[0][2]*c[2][3]*c[3][0] - c[0][3]*c[2][2]*c[3][0] + c[0][3]*c[2][0]*c[3][2] - c[0][0]*c[2][3]*c[3][2] - c[0][2]*c[2][0]*c[3][3] + c[0][0]*c[2][2]*c[3][3]);
        m.c[1][2] = d * (c[0][3]*c[1][2]*c[3][0] - c[0][2]*c[1][3]*c[3][0] - c[0][3]*c[1][0]*c[3][2] + c[0][0]*c[1][3]*c[3][2] + c[0][2]*c[1][0]*c[3][3] - c[0][0]*c[1][2]*c[3][3]);
        m.c[1][3] = d * (c[0][2]*c[1][3]*c[2][0] - c[0][3]*c[1][2]*c[2][0] + c[0][3]*c[1][0]*c[2][2] - c[0][0]*c[1][3]*c[2][2] - c[0][2]*c[1][0]*c[2][3] + c[0][0]*c[1][2]*c[2][3]);
        m.c[2][0] = d * (c[1][1]*c[2][3]*c[3][0] - c[1][3]*c[2][1]*c[3][0] + c[1][3]*c[2][0]*c[3][1] - c[1][0]*c[2][3]*c[3][1] - c[1][1]*c[2][0]*c[3][3] + c[1][0]*c[2][1]*c[3][3]);
        m.c[2][1] = d * (c[0][3]*c[2][1]*c[3][0] - c[0][1]*c[2][3]*c[3][0] - c[0][3]*c[2][0]*c[3][1] + c[0][0]*c[2][3]*c[3][1] + c[0][1]*c[2][0]*c[3][3] - c[0][0]*c[2][1]*c[3][3]);
        m.c[2][2] = d * (c[0][1]*c[1][3]*c[3][0] - c[0][3]*c[1][1]*c[3][0] + c[0][3]*c[1][0]*c[3][1] - c[0][0]*c[1][3]*c[3][1] - c[0][1]*c[1][0]*c[3][3] + c[0][0]*c[1][1]*c[3][3]);
        m.c[2][3] = d * (c[0][3]*c[1][1]*c[2][0] - c[0][1]*c[1][3]*c[2][0] - c[0][3]*c[1][0]*c[2][1] + c[0][0]*c[1][3]*c[2][1] + c[0][1]*c[1][0]*c[2][3] - c[0][0]*c[1][1]*c[2][3]);
        m.c[3][0] = d * (c[1][2]*c[2][1]*c[3][0] - c[1][1]*c[2][2]*c[3][0] - c[1][2]*c[2][0]*c[3][1] + c[1][0]*c[2][2]*c[3][1] + c[1][1]*c[2][0]*c[3][2] - c[1][0]*c[2][1]*c[3][2]);
        m.c[3][1] = d * (c[0][1]*c[2][2]*c[3][0] - c[0][2]*c[2][1]*c[3][0] + c[0][2]*c[2][0]*c[3][1] - c[0][0]*c[2][2]*c[3][1] - c[0][1]*c[2][0]*c[3][2] + c[0][0]*c[2][1]*c[3][2]);
        m.c[3][2] = d * (c[0][2]*c[1][1]*c[3][0] - c[0][1]*c[1][2]*c[3][0] - c[0][2]*c[1][0]*c[3][1] + c[0][0]*c[1][2]*c[3][1] + c[0][1]*c[1][0]*c[3][2] - c[0][0]*c[1][1]*c[3][2]);
        m.c[3][3] = d * (c[0][1]*c[1][2]*c[2][0] - c[0][2]*c[1][1]*c[2][0] + c[0][2]*c[1][0]*c[2][1] - c[0][0]*c[1][2]*c[2][1] - c[0][1]*c[1][0]*c[2][2] + c[0][0]*c[1][1]*c[2][2]);

        return m;
    }

    void decompose( Vec3f &trans, Vec3f &rot, Vec3f &scale ) const
    {
        // Getting translation is trivial
        trans = Vec3f( c[3][0], c[3][1], c[3][2] );

        // Scale is length of columns
        scale.x = sqrtf( c[0][0] * c[0][0] + c[0][1] * c[0][1] + c[0][2] * c[0][2] );
        scale.y = sqrtf( c[1][0] * c[1][0] + c[1][1] * c[1][1] + c[1][2] * c[1][2] );
        scale.z = sqrtf( c[2][0] * c[2][0] + c[2][1] * c[2][1] + c[2][2] * c[2][2] );

        if( scale.x == 0 || scale.y == 0 || scale.z == 0 ) return;

        // Detect negative scale with determinant and flip one arbitrary axis
        if( determinant() < 0 ) scale.x = -scale.x;

        // Combined rotation matrix YXZ
        //
        // Cos[y]*Cos[z]+Sin[x]*Sin[y]*Sin[z]   Cos[z]*Sin[x]*Sin[y]-Cos[y]*Sin[z]  Cos[x]*Sin[y]    
        // Cos[x]*Sin[z]                        Cos[x]*Cos[z]                       -Sin[x]
        // -Cos[z]*Sin[y]+Cos[y]*Sin[x]*Sin[z]  Cos[y]*Cos[z]*Sin[x]+Sin[y]*Sin[z]  Cos[x]*Cos[y]

        rot.x = asinf( -c[2][1] / scale.z );
        
        // Special case: Cos[x] == 0 (when Sin[x] is +/-1)
        float f = fabsf( c[2][1] / scale.z );
        if( f > 0.999f && f < 1.001f )
        {
            // Pin arbitrarily one of y or z to zero
            // Mathematical equivalent of gimbal lock
            rot.y = 0;
            
            // Now: Cos[x] = 0, Sin[x] = +/-1, Cos[y] = 1, Sin[y] = 0
            // => m[0][0] = Cos[z] and m[1][0] = Sin[z]
            rot.z = atan2f( -c[1][0] / scale.y, c[0][0] / scale.x );
        }
        // Standard case
        else
        {
            rot.y = atan2f( c[2][0] / scale.z, c[2][2] / scale.z );
            rot.z = atan2f( c[0][1] / scale.x, c[1][1] / scale.y );
        }
    }

    void extractMatrix3x3f(float matrix33[ 3 ][ 3 ])
    {
        for (int32_t i = 0; i < 3; ++i)
        {
            for (int32_t j = 0; j < 3; ++j)
            {
                matrix33[ i ][ j ] = c[ i ][ j ];
            }
        }
    }

    void set(const float *floatArray16)
    {
        for (uint32_t i = 0; i < 4; ++i)
        {
            for (uint32_t j = 0; j < 4; ++j)
            {
                c[i][j] = floatArray16[i * 4 + j];
            }
        }
    }

    void setCol( uint32_t col, const Vec4f& v ) 
    {
        x[col * 4 + 0] = v.x;
        x[col * 4 + 1] = v.y;
        x[col * 4 + 2] = v.z;
        x[col * 4 + 3] = v.w;
    }

    Vec4f getCol( uint32_t col ) const
    {
        return Vec4f( x[col * 4 + 0], x[col * 4 + 1], x[col * 4 + 2], x[col * 4 + 3] );
    }

    Vec4f getRow( uint32_t row ) const
    {
        return Vec4f( x[row + 0], x[row + 4], x[row + 8], x[row + 12] );
    }

    Vec3f getTrans() const
    {
        return Vec3f( c[3][0], c[3][1], c[3][2] );
    }
    
    // Note: Scale length is length of columns
    Vec3f getScaleLen() const
    {
        Vec3f scale;
        scale.x = sqrtf( c[0][0] * c[0][0] + c[0][1] * c[0][1] + c[0][2] * c[0][2] );
        scale.y = sqrtf( c[1][0] * c[1][0] + c[1][1] * c[1][1] + c[1][2] * c[1][2] );
        scale.z = sqrtf( c[2][0] * c[2][0] + c[2][1] * c[2][1] + c[2][2] * c[2][2] );
        return scale;
    }

    Vec3f getScale() const
    {
        return Vec3f(c[0][0], c[1][1], c[2][2]);
    }

    Vec3f getRot() const
    {
        Vec3f t, r, s;
        decompose(t, r, s);
        return r;
    }

    Quaternion getQuat() const
    {
      Vec3f t, r, s;
      decompose(t, r, s);
      Quaternion q(r.x, r.y, r.z);
      return q;
    }
};

/***
 * Matrix holds a 3x2 matrix for transforming coordinates. This allows mapping Point and vectors
 * with translation, scaling, skewing, and rotation. Together these types of transformations are
 * known as affine transformations. Affine transformations preserve the straightness of lines while
 * transforming, so that parallel lines stay parallel. Matrix elements are in row major order.
 * Matrix does not have a constructor, so it must be explicitly initialized.
 */
class Matrix {
 public:
  static Matrix MakeAll(const Matrix4f& mat4) {
    // tgfx::PrintError("Matrix MakeAll mat = ");
    // for (auto i  = 0 ; i < 16; i++) {
    //   tgfx::PrintError("x[%d] = %f ", i, mat4.x[i]);
    // }
    return Matrix::MakeAll(mat4.c[0][0], mat4.c[1][0], mat4.c[3][0], 
                            mat4.c[0][1], mat4.c[1][1], mat4.c[3][1]);
  }

  /**
   * Sets Matrix to scale by (sx, sy). Returned matrix is:
   *
   *       | sx  0  0 |
   *       |  0 sy  0 |
   *       |  0  0  1 |
   *
   *  @param sx  horizontal scale factor
   *  @param sy  vertical scale factor
   *  @return    Matrix with scale factors.
   */
  static Matrix MakeScale(float sx, float sy) {
    return {sx, 0, 0, 0, sy, 0};
  }

  /**
   * Sets Matrix to scale by (scale, scale). Returned matrix is:
   *
   *      | scale   0   0 |
   *      |   0   scale 0 |
   *      |   0     0   1 |
   *
   * @param scale  horizontal and vertical scale factor
   * @return       Matrix with scale factors.
   */
  static Matrix MakeScale(float scale) {
    return {scale, 0, 0, 0, scale, 0};
  }

  /**
   * Sets Matrix to translate by (tx, ty). Returned matrix is:
   *
   *       | 1 0 tx |
   *       | 0 1 ty |
   *       | 0 0  1 |
   *
   * @param tx  horizontal translation
   * @param ty  vertical translation
   * @return    Matrix with translation
   */
  static Matrix MakeTrans(float tx, float ty) {
    return {1, 0, tx, 0, 1, ty};
  }

  /**
   * Sets Matrix to skew by (kx, ky) about pivot point (0, 0).
   * @param kx  horizontal skew factor
   * @param ky  vertical skew factor
   * @return    Matrix with skew
   */
  static Matrix MakeSkew(float kx, float ky) {
    return {1, kx, 0, ky, 1, 0};
  }

  /**
   * Sets Matrix to rotate by |degrees| about a pivot point at (0, 0).
   * @param degrees  rotation angle in degrees (positive rotates clockwise)
   * @return     Matrix with rotation
   */
  static Matrix MakeRotate(float degrees) {
    Matrix m;
    m.setRotate(degrees);
    return m;
  }

  /**
   * Sets Matrix to rotate by |degrees| about a pivot point at (px, py).
   * @param degrees  rotation angle in degrees (positive rotates clockwise)
   * @param px       pivot on x-axis
   * @param py       pivot on y-axis
   * @return         Matrix with rotation
   */
  static Matrix MakeRotate(float degrees, float px, float py) {
    Matrix m;
    m.setRotate(degrees, px, py);
    return m;
  }

  /**
   * Sets Matrix to:
   *
   *      | scaleX  skewX transX |
   *      | skewY  scaleY transY |
   *      |   0      0      1    |
   *
   * @param scaleX  horizontal scale factor
   * @param skewX   horizontal skew factor
   * @param transX  horizontal translation
   * @param skewY   vertical skew factor
   * @param scaleY  vertical scale factor
   * @param transY  vertical translation
   * @return        Matrix constructed from parameters
   */
  static Matrix MakeAll(float scaleX, float skewX, float transX, float skewY, float scaleY,
                        float transY) {
    return {scaleX, skewX, transX, skewY, scaleY, transY};
  }

  /**
   * Returns reference to const identity Matrix. Returned Matrix is set to:
   *
   *       | 1 0 0 |
   *       | 0 1 0 |
   *       | 0 0 1 |
   *
   *   @return  const identity Matrix
   */
  static const Matrix& I();

  /**
   * Creates an identity Matrix:
   *    | 1 0 0 |
   *    | 0 1 0 |
   *    | 0 0 1 |
   */
  constexpr Matrix() : Matrix(1, 0, 0, 0, 1, 0) {
  }

  /**
   * Returns true if Matrix is identity. The identity matrix is:
   *
   *       | 1 0 0 |
   *       | 0 1 0 |
   *       | 0 0 1 |
   *
   * @return  Returns true if the Matrix has no effect.
   */
  bool isIdentity() const {
    return values[0] == 1 && values[1] == 0 && values[2] == 0 && values[3] == 0 && values[4] == 1 &&
           values[5] == 0;
  }

  /**
   * Returns one matrix value.
   */
  float operator[](int index) const {
    return values[index];
  }

  /**
   * Returns writable Matrix value.
   */
  float& operator[](int index) {
    return values[index];
  }

  /**
   * Returns one matrix value.
   */
  float get(int index) const {
    return values[index];
  }

  /**
   * Sets Matrix value.
   */
  void set(int index, float value) {
    values[index] = value;
  }

  /**
   * Copies six scalar values contained by Matrix into buffer, in member value ascending order:
   * ScaleX, SkewX, TransX, SkewY, ScaleY, TransY.
   * @param buffer  storage for six scalar values.
   */
  void get6(float buffer[6]) const {
    memcpy(buffer, values, 6 * sizeof(float));
  }

  /**
   * Sets Matrix to six scalar values in buffer, in member value ascending order:
   * ScaleX, SkewX, TransX, SkewY, ScaleY, TransY.
   * Sets matrix to:
   *
   *     | buffer[0] buffer[1] buffer[2] |
   *     | buffer[3] buffer[4] buffer[5] |
   *
   * @param buffer storage for six scalar values.
   */
  void set6(const float buffer[6]) {
    memcpy(values, buffer, 6 * sizeof(float));
  }

  /**
   * Copies nine scalar values contained by Matrix into buffer, in member value ascending order:
   * ScaleX, SkewX, TransX, SkewY, ScaleY, TransY, 0, 0, 1.
   * @param buffer  storage for nine scalar values
   */
  void get9(float buffer[9]) const;

  /**
   * Returns the horizontal scale factor.
   */
  float getScaleX() const {
    return values[SCALE_X];
  }

  /**
   * Returns the vertical scale factor.
   */
  float getScaleY() const {
    return values[SCALE_Y];
  }

  /**
   * Returns the vertical skew factor.
   */
  float getSkewY() const {
    return values[SKEW_Y];
  }

  /**
   * Returns the horizontal scale factor.
   */
  float getSkewX() const {
    return values[SKEW_X];
  }

  /**
   * Returns the horizontal translation factor.
   */
  float getTranslateX() const {
    return values[TRANS_X];
  }

  /**
   * Returns the vertical translation factor.
   */
  float getTranslateY() const {
    return values[TRANS_Y];
  }

  /**
   * Sets the horizontal scale factor.
   */
  void setScaleX(float v) {
    values[SCALE_X] = v;
  }

  /**
   * Sets the vertical scale factor.
   */
  void setScaleY(float v) {
    values[SCALE_Y] = v;
  }

  /**
   * Sets the vertical skew factor.
   */
  void setSkewY(float v) {
    values[SKEW_Y] = v;
  }

  /**
   * Sets the horizontal skew factor.
   */
  void setSkewX(float v) {
    values[SKEW_X] = v;
  }

  /**
   * Sets the horizontal translation.
   */
  void setTranslateX(float v) {
    values[TRANS_X] = v;
  }

  /**
   * Sets the vertical translation.
   */
  void setTranslateY(float v) {
    values[TRANS_Y] = v;
  }

  /**
   * Sets all values from parameters. Sets matrix to:
   *
   *      | scaleX  skewX transX |
   *      | skewY  scaleY transY |
   *      |   0      0      1    |
   *
   * @param scaleX  horizontal scale factor to store
   * @param skewX   horizontal skew factor to store
   * @param transX  horizontal translation to store
   * @param skewY   vertical skew factor to store
   * @param scaleY  vertical scale factor to store
   * @param transY  vertical translation to store
   */
  void setAll(float scaleX, float skewX, float transX, float skewY, float scaleY, float transY);

  /**
   * Sets Matrix to identity; which has no effect on mapped Point. Sets Matrix to:
   *
   *       | 1 0 0 |
   *       | 0 1 0 |
   *       | 0 0 1 |
   *
   * Also called setIdentity(); use the one that provides better inline documentation.
   */
  void reset();

  /**
   * Sets Matrix to identity; which has no effect on mapped Point. Sets Matrix to:
   *
   *       | 1 0 0 |
   *       | 0 1 0 |
   *       | 0 0 1 |
   *
   *  Also called reset(); use the one that provides better inline documentation.
   */
  void setIdentity() {
    this->reset();
  }

  /**
   * Sets Matrix to translate by (tx, ty).
   * @param tx  horizontal translation
   * @param ty  vertical translation
   */
  void setTranslate(float tx, float ty);

  /**
   * Sets Matrix to scale by sx and sy, about a pivot point at (px, py). The pivot point is
   * unchanged when mapped with Matrix.
   * @param sx  horizontal scale factor
   * @param sy  vertical scale factor
   * @param px  pivot on x-axis
   * @param py  pivot on y-axis
   */
  void setScale(float sx, float sy, float px, float py);

  /**
   * Sets Matrix to scale by sx and sy about at pivot point at (0, 0).
   * @param sx  horizontal scale factor
   * @param sy  vertical scale factor
   */
  void setScale(float sx, float sy);

  /**
   * Sets Matrix to rotate by degrees about a pivot point at (px, py). The pivot point is
   * unchanged when mapped with Matrix. Positive degrees rotates clockwise.
   *  @param degrees  angle of axes relative to upright axes
   *  @param px       pivot on x-axis
   *  @param py       pivot on y-axis
   */
  void setRotate(float degrees, float px, float py);

  /**
   * Sets Matrix to rotate by degrees about a pivot point at (0, 0). Positive degrees rotates
   * clockwise.
   * @param degrees  angle of axes relative to upright axes
   */
  void setRotate(float degrees);

  /**
   * Sets Matrix to rotate by sinValue and cosValue, about a pivot point at (px, py).
   * The pivot point is unchanged when mapped with Matrix.
   * Vector (sinValue, cosValue) describes the angle of rotation relative to (0, 1).
   * Vector length specifies the scale factor.
   */
  void setSinCos(float sinV, float cosV, float px, float py);

  /**
   * Sets Matrix to rotate by sinValue and cosValue, about a pivot point at (0, 0).
   * Vector (sinValue, cosValue) describes the angle of rotation relative to (0, 1).
   * Vector length specifies the scale factor.
   */
  void setSinCos(float sinV, float cosV);

  /**
   * Sets Matrix to skew by kx and ky, about a pivot point at (px, py). The pivot point is
   * unchanged when mapped with Matrix.
   * @param kx  horizontal skew factor
   * @param ky  vertical skew factor
   * @param px  pivot on x-axis
   * @param py  pivot on y-axis
   */
  void setSkew(float kx, float ky, float px, float py);

  /**
   * Sets Matrix to skew by kx and ky, about a pivot point at (0, 0).
   * @param kx  horizontal skew factor
   * @param ky  vertical skew factor
   */
  void setSkew(float kx, float ky);

  /**
   * Sets Matrix to Matrix a multiplied by Matrix b. Either a or b may be this.
   *
   * Given:
   *
   *          | A B C |      | J K L |
   *      a = | D E F |, b = | M N O |
   *          | G H I |      | P Q R |
   *
   * sets Matrix to:
   *
   *              | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
   *      a * b = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
   *              | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |
   *
   * @param a  Matrix on the left side of multiply expression
   * @param b  Matrix on the right side of multiply expression
   */
  void setConcat(const Matrix& a, const Matrix& b);

  /**
   * Preconcats the matrix with the specified scale. M' = M * S(sx, sy)
   */
  void preTranslate(float tx, float ty);

  /**
   * Postconcats the matrix with the specified scale. M' = S(sx, sy, px, py) * M
   */
  void preScale(float sx, float sy, float px, float py);

  /**
   * Preconcats the matrix with the specified scale. M' = M * S(sx, sy)
   */
  void preScale(float sx, float sy);

  /**
   * Preconcats the matrix with the specified rotation. M' = M * R(degrees, px, py)
   */
  void preRotate(float degrees, float px, float py);

  /**
   * Preconcats the matrix with the specified rotation. M' = M * R(degrees)
   */
  void preRotate(float degrees);

  /**
   * Preconcats the matrix with the specified skew. M' = M * K(kx, ky, px, py)
   */
  void preSkew(float kx, float ky, float px, float py);

  /**
   * Preconcats the matrix with the specified skew. M' = M * K(kx, ky)
   */
  void preSkew(float kx, float ky);

  /**
   * Preconcats the matrix with the specified matrix. M' = M * other
   */
  void preConcat(const Matrix& other);

  /**
   * Postconcats the matrix with the specified translation. M' = T(tx, ty) * M
   */
  void postTranslate(float tx, float ty);

  /**
   * Postconcats the matrix with the specified scale. M' = S(sx, sy, px, py) * M
   */
  void postScale(float sx, float sy, float px, float py);

  /**
   * Postconcats the matrix with the specified scale. M' = S(sx, sy) * M
   */
  void postScale(float sx, float sy);

  /**
   * Postconcats the matrix with the specified rotation. M' = R(degrees, px, py) * M
   */
  void postRotate(float degrees, float px, float py);

  /**
   * Postconcats the matrix with the specified rotation. M' = R(degrees) * M
   */
  void postRotate(float degrees);

  /**
   * Postconcats the matrix with the specified skew. M' = K(kx, ky, px, py) * M
   */
  void postSkew(float kx, float ky, float px, float py);

  /**
   * Postconcats the matrix with the specified skew. M' = K(kx, ky) * M
   */
  void postSkew(float kx, float ky);

  /**
   * Postconcats the matrix with the specified matrix. M' = other * M
   */
  void postConcat(const Matrix& other);

  /**
   * If this matrix can be inverted, return true and if the inverse is not null, set inverse to be
   * the inverse of this matrix. If this matrix cannot be inverted, ignore the inverse and return
   * false.
   */
  bool invert(Matrix* inverse) const {
    if (this->isIdentity()) {
      if (inverse) {
        inverse->reset();
      }
      return true;
    }
    return this->invertNonIdentity(inverse);
  }

  /**
   * Returns ture if the Matrix is invertible.
   */
  bool invertible() const;

  /**
   * Maps src Point array of length count to dst Point array of equal or greater length. Points are
   * mapped by multiplying each Point by Matrix. Given:
   *
   *                | A B C |        | x |
   *       Matrix = | D E F |,  pt = | y |
   *                | G H I |        | 1 |
   *
   * where
   *
   *       for (i = 0; i < count; ++i) {
   *           x = src[i].fX
   *           y = src[i].fY
   *       }
   *
   * each dst Point is computed as:
   *
   *                     |A B C| |x|                               Ax+By+C   Dx+Ey+F
   *       Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
   *                     |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
   *
   * src and dst may point to the same storage.
   *
   * @param dst    storage for mapped Point
   * @param src    Point to transform
   * @param count  number of Points to transform
   */
  void mapPoints(Point dst[], const Point src[], int count) const;

  /**
   * Maps pts Point array of length count in place. Points are mapped by multiplying each Point by
   * Matrix. Given:
   *
   *                 | A B C |        | x |
   *        Matrix = | D E F |,  pt = | y |
   *                 | G H I |        | 1 |
   *
   * where
   *
   *        for (i = 0; i < count; ++i) {
   *            x = pts[i].fX
   *            y = pts[i].fY
   *        }
   *
   * each resulting pts Point is computed as:
   *
   *                      |A B C| |x|                               Ax+By+C   Dx+Ey+F
   *        Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
   *                      |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
   *
   * @param pts    storage for mapped Point
   * @param count  number of Points to transform
   */
  void mapPoints(Point pts[], int count) const {
    this->mapPoints(pts, pts, count);
  }

  /**
   * Maps Point (x, y) to result. Point is mapped by multiplying by Matrix. Given:
   *
   *                | A B C |        | x |
   *       Matrix = | D E F |,  pt = | y |
   *                | G H I |        | 1 |
   *
   * the result is computed as:
   *
   *                     |A B C| |x|                               Ax+By+C   Dx+Ey+F
   *       Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
   *                     |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
   *
   * @param x       x-axis value of Point to map
   * @param y       y-axis value of Point to map
   * @param result  storage for mapped Point
   */
  void mapXY(float x, float y, Point* result) const;

  /**
   * Returns Point (x, y) multiplied by Matrix. Given:
   *
   *                | A B C |        | x |
   *       Matrix = | D E F |,  pt = | y |
   *                | G H I |        | 1 |
   *
   * the result is computed as:
   *
   *                     |A B C| |x|                               Ax+By+C   Dx+Ey+F
   *       Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
   *                     |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
   *
   * @param x  x-axis value of Point to map
   * @param y  y-axis value of Point to map
   * @return   mapped Point
   */
  Point mapXY(float x, float y) const {
    Point result = {};
    this->mapXY(x, y, &result);
    return result;
  }

  /**
   * Returns true if Matrix maps Rect to another Rect. If true, the Matrix is identity, or scales,
   * or rotates a multiple of 90 degrees, or mirrors on axes. In all cases, Matrix may also have
   * translation. Matrix form is either:
   *
   *         | scale-x    0    translate-x |
   *         |    0    scale-y translate-y |
   *         |    0       0         1      |
   *
   *     or
   *
   *        |    0     rotate-x translate-x |
   *        | rotate-y    0     translate-y |
   *        |    0        0          1      |
   *
   *    for non-zero values of scale-x, scale-y, rotate-x, and rotate-y.
   */
  bool rectStaysRect() const;

  /**
   * Sets dst to bounds of src corners mapped by Matrix.
   */
  void mapRect(Rect* dst, const Rect& src) const;

  /**
   * Sets rect to bounds of rect corners mapped by Matrix.
   */
  void mapRect(Rect* rect) const {
    mapRect(rect, *rect);
  }

  /**
   * Returns bounds of src corners mapped by Matrix.
   */
  Rect mapRect(const Rect& src) const {
    Rect dst = {};
    mapRect(&dst, src);
    return dst;
  }

  /** Compares a and b; returns true if a and b are numerically equal. Returns true even if sign
   * of zero values are different. Returns false if either Matrix contains NaN, even if the other
   * Matrix also contains NaN.
   */
  friend bool operator==(const Matrix& a, const Matrix& b);

  /**
   * Compares a and b; returns true if a and b are not numerically equal. Returns false even if
   * sign of zero values are different. Returns true if either Matrix contains NaN, even if the
   * other Matrix also contains NaN.
   */
  friend bool operator!=(const Matrix& a, const Matrix& b) {
    return !(a == b);
  }

  /**
   * Returns Matrix A multiplied by Matrix B.
   */
  friend Matrix operator*(const Matrix& a, const Matrix& b);

  /**
   * Returns the minimum scale factor of the Matrix by decomposing the scaling and skewing elements.
   * The scale factor is an absolute value and may not align with the x/y axes. Returns -1 if the
   * scale factor overflows.
   */
  float getMinScale() const;

  /**
   * Returns the maximum scale factor of the Matrix by decomposing the scaling and skewing elements.
   * The scale factor is an absolute value and may not align with the x/y axes. Returns -1 if the
   * scale factor overflows.
   */
  float getMaxScale() const;

  /**
   * Returns the scale components of the Matrix along the x and y axes. Both components are
   * absolute values.
   */
  Point getAxisScales() const;

  /**
   * Returns true if the Matrix contains a non-identity scale component.
   */
  bool hasNonIdentityScale() const;

  /**
   * Returns true if the Matrix is identity or contains only translation.
   */
  bool isTranslate() const;

  /**
   * Returns true if all elements of the matrix are finite. Returns false if any element is
   * infinity, or NaN.
   */
  bool isFinite() const;

 private:
  float values[6];
  /**
   * Matrix organizes its values in row order. These members correspond to each value in Matrix.
   */
  static constexpr int SCALE_X = 0;  //!< horizontal scale factor
  static constexpr int SKEW_X = 1;   //!< horizontal skew factor
  static constexpr int TRANS_X = 2;  //!< horizontal translation
  static constexpr int SKEW_Y = 3;   //!< vertical skew factor
  static constexpr int SCALE_Y = 4;  //!< vertical scale factor
  static constexpr int TRANS_Y = 5;  //!< vertical translation

  constexpr Matrix(float scaleX, float skewX, float transX, float skewY, float scaleY, float transY)
      : values{scaleX, skewX, transX, skewY, scaleY, transY} {
  }

  bool invertNonIdentity(Matrix* inverse) const;
  bool getMinMaxScaleFactors(float results[2]) const;
};
}  // namespace tgfx
