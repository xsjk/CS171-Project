#pragma once

namespace utils {
  __device__ __host__ 
  inline float clamp01(float v) {
    if (v > 1) v = 1;
    else if (v < 0) v = 0;
    return v;
  }

  __device__ __host__ 
  inline Vec3f clamp01(Vec3f v) {
    return {clamp01(v.x), clamp01(v.y), clamp01(v.z)};
  }

  inline uint8_t gammaCorrection(float radiance) {
    return static_cast<uint8_t>(255.f * clamp01(powf(radiance, 1.f / 2.2f)));
  }

  
  template<typename T>
  __device__ __host__ 
  T max(const T& t){
    return t;
  }

  template<typename T, typename... Args>
  __device__ __host__ 
  T max(const T& t, Args ...args) {
    T other_max = max(args...);
    return t > other_max ? t : other_max;
  }

  template<typename T>
  __device__ __host__ 
  T min(const T& t) {
    return t;
  }

  template<typename T, typename... Args>
  __device__ __host__ 
  T min(const T& t, Args ...args) {
    T other_min = min(args...);
    return t < other_min ? t : other_min;
  }

  __device__ __host__ 
  inline Vec3f cwiseMin(Vec3f v) {
    return v;
  }

  template<typename... Args>
  __device__ __host__ 
  Vec3f cwiseMin(Vec3f v, Args ...args) {
    Vec3f other_cwise_min = cwiseMin(args...);
    if(other_cwise_min.x < v.x) v.x = other_cwise_min.x;
    if(other_cwise_min.y < v.y) v.y = other_cwise_min.y;
    if(other_cwise_min.z < v.z) v.z = other_cwise_min.z;
    return v;
  }

  __device__ __host__ 
  inline Vec3f cwiseMax(Vec3f v) {
    return v;
  }

  template<typename... Args>
  __device__ __host__ 
  Vec3f cwiseMax(Vec3f v, Args ...args) {
    Vec3f other_cwise_max = cwiseMax(args...);
    if(other_cwise_max.x > v.x) v.x = other_cwise_max.x;
    if(other_cwise_max.y > v.y) v.y = other_cwise_max.y;
    if(other_cwise_max.z > v.z) v.z = other_cwise_max.z;
    return v;
  }

  template<class T>
  __device__ __host__ 
  inline T interpolate(float u, float v, T _0, T _1, T _2) {
    return (1 - u - v) * _0 + u * _1 + v * _2;
  }

  template<typename T>
  inline void CPU2GPU(T *&ptr, size_t n) {
    T *tmp;
    cudaMalloc(&tmp, n * sizeof(T));
    cudaMemcpy(tmp, ptr, n * sizeof(T), cudaMemcpyHostToDevice);
    delete[] ptr;
    ptr = tmp;
  }

  template<typename T>
  inline void GPU2CPU(T *&ptr, size_t n) {
    T *tmp = new T[n];
    cudaMemcpy(tmp, ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(ptr);
    ptr = tmp;
  }
}
