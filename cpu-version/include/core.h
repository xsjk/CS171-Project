#pragma once

#include <glm/glm.hpp>
#include <random>
#include <limits>

using Vec3f = glm::vec3;
using Vec3i = glm::ivec3;
using Vec2f = glm::vec2;
using Vec2i = glm::ivec2;

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
constexpr Vec3f DEFAULT_BACKGROUND_COLOR {0,0,0};

//some const variables for BVH
constexpr unsigned int BVH_LEAF_MAX_SIZE = 4;
constexpr float t_traveral = 1.0f;
constexpr float t_intersect = 2.0f;

//some const variables for global illumination
constexpr float PROB = 0.6f;
constexpr Vec3f WORLD_UP = {0.0f,0.0f,1.0f};
constexpr int LIGHTCOUNT = 20;

template<typename T>
T max(const T& t){
  return t;
}

template<typename T, typename... Args>
T max(const T& t, Args ...args) {
  T other_max = max(args...);
  return t > other_max ? t : other_max;
}

template<typename T>
T min(const T& t) {
  return t;
}

template<typename T, typename... Args>
T min(const T& t, Args ...args) {
  T other_min = min(args...);
  return t < other_min ? t : other_min;
}

inline Vec3f cwiseMin(Vec3f v) {
  return v;
}

template<typename... Args>
Vec3f cwiseMin(Vec3f v, Args ...args) {
  Vec3f other_cwise_min = cwiseMin(args...);
  if(other_cwise_min.x < v.x) v.x = other_cwise_min.x;
  if(other_cwise_min.y < v.y) v.y = other_cwise_min.y;
  if(other_cwise_min.z < v.z) v.z = other_cwise_min.z;
  return v;
}

inline Vec3f cwiseMax(Vec3f v) {
  return v;
}

template<typename... Args>
Vec3f cwiseMax(Vec3f v, Args ...args) {
  Vec3f other_cwise_max = cwiseMax(args...);
  if(other_cwise_max.x > v.x) v.x = other_cwise_max.x;
  if(other_cwise_max.y > v.y) v.y = other_cwise_max.y;
  if(other_cwise_max.z > v.z) v.z = other_cwise_max.z;
  return v;
}