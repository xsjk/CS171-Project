#ifndef _ACCEL_H_
#define _ACCEL_H_

#include "core.h"
#include "ray.h"
#include "interaction.h"
#include "triangle.h"

#include <vector>
#include <memory>

class Scene;

struct AABB{
    Vec3f _lower_bnd;
    Vec3f _upper_bnd;
    AABB();
    AABB(Vec3f lower, Vec3f upper);
    AABB(Vec3f pos1, Vec3f pos2, Vec3f pos3);
    float cal_prob(const AABB& other) const;
    AABB& Merge(const AABB& other);
    bool intersect(const Ray& ray, float& t_in, float& t_out) const;
    Vec3f getCenter() const;
    int getCenter(int dim) const;
    float getDist(unsigned int dim) const;
    bool isOverlap(const AABB& other) const;
};

constexpr int INTERAL_NODE = -1;

class BVH{
    private:
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
        } root;

        std::shared_ptr<std::vector<Vec3f>> _positions;
        std::shared_ptr<std::vector<Vec3f>> _normals;
        std::shared_ptr<std::vector<Vec3f>> _colors;
        /*
        some other type information of the triangles
        */
        std::shared_ptr<std::vector<Triangle>> _triangles;
        std::vector<Bvh_Node> _bvh_nodes;

    public:
        BVH() = default;
        std::shared_ptr<std::vector<Vec3f>> set_positions(std::shared_ptr<std::vector<Vec3f>> positions);
        std::shared_ptr<std::vector<Vec3f>> set_normals(std::shared_ptr<std::vector<Vec3f>> normals);
        std::shared_ptr<std::vector<Vec3f>> set_colors(std::shared_ptr<std::vector<Vec3f>> colors);
        std::shared_ptr<std::vector<Triangle>> set_triangles(std::shared_ptr<std::vector<Triangle>> triangles);
        void build_BVH();
        void bvhHit(Interaction & interaction, const Ray& ray, int node_idx) const;
        bool visiality_test(const Ray& ray, int node_idx) const;
        Scene* set_scene(Scene* scene);
    
    private:
        void build_bvh_recursively(int l, int r, int& right, std::vector<Triangle_Info_AABB>& triangles_aabb);
        Scene* _scene;
};

#endif

