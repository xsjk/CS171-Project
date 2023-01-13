#pragma once

#include "core.h"
#include "ray.h"
#include "interaction.h"
#include "triangle.h"
#include "utils.h"

#include <vector>
#include <memory>

class Scene;

struct AABB{
    Vec3f _lower_bnd;
    Vec3f _upper_bnd;
    AABB();
    AABB(Vec3f lower, Vec3f upper);
    AABB(Vec3f v1, Vec3f v2, Vec3f pos3);
    float cal_prob(const AABB& other) const;
    AABB& Merge(const AABB& other);

    __device__ inline bool intersect(const Ray& ray, float& t_in, float& t_out) const {
        float dir_frac_x = ray._dir.x == 0.0f ? 1.0e32 : 1.0f / ray._dir.x;
        float dir_frac_y = ray._dir.y == 0.0f ? 1.0e32 : 1.0f / ray._dir.y;
        float dir_frac_z = ray._dir.z == 0.0f ? 1.0e32 : 1.0f / ray._dir.z;

        float tx1 = (_lower_bnd.x - ray._origin.x) * dir_frac_x;
        float tx2 = (_upper_bnd.x - ray._origin.x) * dir_frac_x;
        float ty1 = (_lower_bnd.y - ray._origin.y) * dir_frac_y;
        float ty2 = (_upper_bnd.y - ray._origin.y) * dir_frac_y;
        float tz1 = (_lower_bnd.z - ray._origin.z) * dir_frac_z;
        float tz2 = (_upper_bnd.z - ray._origin.z) * dir_frac_z;

        t_in = utils::max(utils::min(tx1, tx2), utils::min(ty1, ty2), utils::min(tz1, tz2));
        t_out = utils::min(utils::max(tx1, tx2), utils::max(ty1, ty2), utils::max(tz1, tz2));

        t_in = utils::max(t_in, ray._t_min);
        t_out = utils::min(t_out, ray._t_max);

        return t_out < 0 ? false : t_out >= t_in;
    }

    Vec3f getCenter() const;
    int getCenter(int dim) const;
    float getDist(unsigned int dim) const;
    bool isOverlap(const AABB& other) const;
};

constexpr int INTERAL_NODE = -1;

class BVH{
public:
    /// @brief just convenient for building the bvh
    struct Triangle_Info_AABB{
        Triangle triangle;
        AABB aabb;
    };

    struct Bvh_Node{
        union LEFT{
            int type;
            int start;
        } left;

        union RIGHT{
            int right_node_idx;
            int end;
        } right;

        AABB aabb;
    };

private:
    /*
    some other type information of the triangles
    */
    Bvh_Node root, *_bvh_nodes, *_bvh_nodes_end;
    unsigned n_bvh_nodes;

public:
    void build_BVH();
    Bvh_Node* get_nodes();
    void CPU2GPU();
    void GPU2CPU();
    __device__ void bvhHit(Interaction & interaction, const Ray& ray, int node_idx) const;
    __device__ bool visiality_test(const Ray& ray, int node_idx) const;
    void set_scene(Scene* scene);

private:
    void build_bvh_recursively(int l, int r, int& right, std::vector<Triangle_Info_AABB>& triangles_aabb, Triangle*& ptr);
    thrust::device_ptr<Scene> _scene;
};

