#pragma once

#include <random>
#include <core.h>

class Sampler {
public:
    Sampler() = default;
    __device__ Sampler(curandState& state) : state(&state) {}
    __device__ inline float get1D() const { return curand_uniform(state); }
    __device__ inline unsigned get1U(unsigned int start, unsigned int width) const {return curand(state) % width + start;}
    __device__ inline void setSeed(unsigned seed, unsigned index) { curand_init(seed, index, 0, state); }

private:
    curandState* state;
};
