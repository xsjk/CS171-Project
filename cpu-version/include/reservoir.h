#ifndef _RESERVOIR_H_
#define _RESERVOIR_H_

#include "point.h"
#include "sampler.h"

class Reservoir{
    public:
        Reservoir() = default;
        void update(const Point& newLightPoint, float w_new, Sampler& sampler);
        bool isLight{false};
        Point visiblePoint;
        Point lightPoint;
        float w_sum{0.0f};
        float W{0.0f};
        unsigned int M {0};
};

#endif