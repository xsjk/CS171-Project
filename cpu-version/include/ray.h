#pragma once

#include <core.h>

struct Ray{

    /// @brief origin point of ray
    Vec3f _origin;

    /// @brief normalized direction of the ray
    Vec3f _dir;

    /// @brief min distance of the ray (default = 1e-5)
    float _t_min;

    /// @brief max distance of the ray (default = INF_F)
    float _t_max;

    Ray(const Vec3f &o, const Vec3f&dir, float t_min, float t_max);

    Vec3f operator()(float t) const;
};