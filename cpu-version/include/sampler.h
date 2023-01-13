#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include <random>

class Sampler {
 public:
  Sampler() = default;
  inline float get1D() { return r_dis(engine); }
  inline unsigned int get1U(unsigned int start, unsigned int width) {return u_dis(engine)%width + start;}
  inline void setSeed(int i) { engine.seed(i); }
 private:
  std::default_random_engine engine;
  std::uniform_real_distribution<float> r_dis;
  std::uniform_int_distribution<unsigned> u_dis;
};

#endif