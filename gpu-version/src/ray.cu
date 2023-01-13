#include <ray.h>

// __device__ Ray::Ray(const Vec3f &o, const Vec3f&dir, float t_min = RAY_DEFAULT_MIN, float t_max = RAY_DEFAULT_MAX) : _origin(o), _dir(dir), _t_min(t_min), _t_max(t_max) {}

__device__ Vec3f Ray::operator()(float t) const {
    return _origin + t * _dir;
}