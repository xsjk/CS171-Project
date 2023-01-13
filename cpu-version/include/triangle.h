#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "core.h"

struct Triangle {
  bool emissive;
  Vec3i v_idx;
  Vec3i n_idx;
  Vec3i c_idx;
};

#endif