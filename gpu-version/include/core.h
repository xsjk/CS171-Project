#pragma once

#include <glm/glm.hpp>
#include <random>
#include <limits>

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"


using Vec3f = glm::vec3;
using Vec3i = glm::ivec3;
using Vec2f = glm::vec2;
using Vec2i = glm::ivec2;

constexpr unsigned SEED = 1;

constexpr float POS_INF_F = 1e10;
constexpr float NEG_INF_F = -1e10;
constexpr float EPS = 1e-5;
constexpr float RAY_DEFAULT_MIN = 1e-5;
constexpr float RAY_DEFAULT_MAX = 1e7;
constexpr int LIGHT_CANDIDATE_NUM = 100;
constexpr int NEIGHBOR_COUNT = 30;
constexpr int NEIGHBOR_RANGE = 10;
constexpr float PI = 3.1415926;
constexpr float INV_PI = 0.31830989;

const Vec3f DEFAULT_BACKGROUND_COLOR {0,0,0};

//some const variables for BVH
constexpr unsigned int BVH_LEAF_MAX_SIZE = 4;
constexpr float t_traveral = 1.0f;
constexpr float t_intersect = 2.0f;

//some const variables for global illumination
constexpr float PROB = 0.6f;
constexpr int LIGHTCOUNT = 20;