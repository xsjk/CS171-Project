#pragma once

#include "point.h"
#include "sampler.h"

class Reservoir{
public:
    Reservoir() = default;
    __device__ inline void update(const Point& newLightPoint, float w_new, const Sampler& sampler) {
        if (w_sum != 0) {
            w_sum += w_new;
            if (sampler.get1D() < w_new / w_sum)
                lightPoint = newLightPoint;
        }
        else {
            w_sum = w_new;
            lightPoint = newLightPoint;
        }
        M++;
    }
    bool isLight{false};
    Point visiblePoint;
    Point lightPoint;
    float w_sum{0.0f};
    float W{0.0f};
    unsigned int M {0};
    __device__ inline float p_hat() {
        return glm::length(visiblePoint.color * lightPoint.color * glm::dot(visiblePoint.normal, glm::normalize(lightPoint.position - visiblePoint.position)) / glm::dot(lightPoint.position - visiblePoint.position, lightPoint.position - visiblePoint.position) * INV_PI);
    }
    __device__ inline Vec3f get_color() {
        return (W * lightPoint.color * visiblePoint.color * INV_PI * glm::dot(visiblePoint.normal, glm::normalize(lightPoint.position - visiblePoint.position)) * glm::dot(lightPoint.position - visiblePoint.position, lightPoint.position - visiblePoint.position) * glm::dot(lightPoint.normal, glm::normalize(visiblePoint.position - lightPoint.position)) * 0.05f);
    }
    __device__ inline Vec3f get_multipleTerm() {
        return (W * visiblePoint.color * INV_PI * glm::dot(visiblePoint.normal, glm::normalize(lightPoint.position - visiblePoint.position)) * glm::dot(lightPoint.position - visiblePoint.position, lightPoint.position - visiblePoint.position) * glm::dot(lightPoint.normal, glm::normalize(visiblePoint.position - lightPoint.position))*0.7f);
    }
};