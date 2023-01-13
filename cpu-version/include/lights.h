#ifndef _LIGHT_H_
#define _LIGHT_H_

#include "core.h"
#include "sampler.h"
#include "triangle.h"
#include <memory>

class Lights{
  public:
    Lights() = default;
    Sampler* set_sampler(Sampler*);
    std::shared_ptr<std::vector<Vec3f>> set_positions(std::shared_ptr<std::vector<Vec3f>>);
    std::shared_ptr<std::vector<Vec3f>> set_normals(std::shared_ptr<std::vector<Vec3f>>);
    std::shared_ptr<std::vector<Vec3f>> set_colors(std::shared_ptr<std::vector<Vec3f>>);
    std::shared_ptr<std::vector<Triangle>> set_triangles(std::shared_ptr<std::vector<Triangle>>);

    void sample(Vec3f& samplePos, Vec3f& sampleNormal, Vec3f& sampleColor, float& pdf) const;
  private:
    std::shared_ptr<std::vector<Triangle>> _emissive_triangles;
    std::shared_ptr<std::vector<Vec3f>> _pos;
    std::shared_ptr<std::vector<Vec3f>> _norms;
    std::shared_ptr<std::vector<Vec3f>> _colors;
    Sampler* _sampler;
};

#endif