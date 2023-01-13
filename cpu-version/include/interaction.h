#ifndef _INTERACTION_H_
#define _INTERACTION_H_

#include "core.h"

struct Interaction {
  enum Type { LIGHT, GEOMETRY, NONE };
  int triangle_idx;
  Vec2f uv;  //v0v1v2中的v1v2的系数
  float dist{RAY_DEFAULT_MAX};
  Type type{Type::NONE};
};

#endif