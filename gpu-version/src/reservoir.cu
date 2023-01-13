#include "reservoir.h"
#include "sampler.h"

// __device__ void Reservoir::update(const Point& newLightPoint, float w_new, Sampler& sampler) {
//     if(w_sum != 0) {
//         w_sum += w_new;
//         if(sampler.get1D() < w_new / w_sum)
//             lightPoint = newLightPoint;
//     } else {
//         w_sum = w_new;
//         lightPoint = newLightPoint;
//     }
//     M++;
// }
