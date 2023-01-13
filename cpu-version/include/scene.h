#ifndef _SCENE_H_
#define _SCENE_H_

#include <memory>
#include "core.h"
#include "triangle.h"
#include "camera.h"
#include "reservoir.h"
#include "sampler.h"
#include "lights.h"
#include "accel.h"

class Interaction;

class Scene {
  public:
    Scene();
    void add_positions(std::shared_ptr<std::vector<Vec3f>>);
    void add_normals(std::shared_ptr<std::vector<Vec3f>>);
    void add_colors(std::shared_ptr<std::vector<Vec3f>>);
    void add_triangles(std::shared_ptr<std::vector<Triangle>>);
    void build_bvh();
    Sampler* set_sampler(Sampler*);
    Lights& get_lights();
    Sampler* get_sampler();
  
    bool isShadowed(const Ray&) const;
    void intersect(const Ray&, Interaction& interaction) const;

    std::shared_ptr<std::vector<Vec3f>> _pos;
    std::shared_ptr<std::vector<Vec3f>> _norms;
    std::shared_ptr<std::vector<Vec3f>> _colors;
    std::shared_ptr<std::vector<Triangle>> _triangles;
    
  private:
    BVH _bvh;
    Sampler* _sampler;
    Lights _lights;
  friend class BVH;
};

#endif