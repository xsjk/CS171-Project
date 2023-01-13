#ifndef _UTILS_H_
#define _UTILS_H_


namespace utils {
  static inline float clamp01(float v) {
    if (v > 1) v = 1;
    else if (v < 0) v = 0;
    return v;
  }

  static inline uint8_t gammaCorrection(float radiance) {
    return static_cast<uint8_t>(255.f * clamp01(powf(radiance, 1.f / 2.2f)));
  }
}

#endif