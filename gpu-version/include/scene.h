#pragma once

#include <memory>
#include "core.h"
#include "triangle.h"
#include "camera.h"
#include "reservoir.h"
#include "accel.h"
#include "sampler.h"
#include "utils.h"

class Interaction;

class Scene {
  public:
    Scene();
    void set_positions(const std::vector<Vec3f>&);
    void set_normals(const std::vector<Vec3f>&);
    void set_colors(const std::vector<Vec3f>&);
    void set_triangles(const std::vector<Triangle>&);
    void build_bvh();

    void CPU2GPU();
    void GPU2CPU();

    void free_device();
    
    __device__ void sample_lights(Vec3f& samplePos, Vec3f& sampleNormal, Vec3f& sampleColor, float& pdf, const Sampler& sampler) const;

    __device__ inline bool isShadowed(const Ray& ray) const {
        return !_bvh.visiality_test(ray, 0);
    }

    __device__ inline void intersect(const Ray& ray, Interaction& interaction) const {
        _bvh.bvhHit(interaction, ray, 0);
    }


    Vec3f* _pos;
    Vec3f* _norms;
    Vec3f* _colors;
    Triangle* _triangles;
    Triangle* _emissive_triangles;

    unsigned n_pos;
    unsigned n_norms;
    unsigned n_colors;
    unsigned n_triangles;
    unsigned n_emissive_triangles;

    BVH& get_bvh();

  private:
    BVH _bvh;
  friend class BVH;
};

