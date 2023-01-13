#include "accel.h"
#include "core.h"
#include "scene.h"
#include <assert.h>
#include <ctime>
#include <iostream>
#include <algorithm>


AABB::AABB() : _lower_bnd(POS_INF_F, POS_INF_F, POS_INF_F), 
               _upper_bnd(NEG_INF_F, NEG_INF_F, NEG_INF_F) {}

AABB::AABB(Vec3f lower, Vec3f upper) : _lower_bnd(lower), 
                                       _upper_bnd(upper) {}

AABB::AABB(Vec3f pos1, Vec3f pos2, Vec3f pos3) {
    _lower_bnd = cwiseMin(pos1, pos2, pos3);
    _upper_bnd = cwiseMax(pos1, pos2, pos3);
}

float AABB::cal_prob(const AABB& bigger) const {
    return (getDist(0) * (getDist(1)+getDist(2)) + getDist(1) * getDist(2)) / 
           (bigger.getDist(0) * (bigger.getDist(1) + bigger.getDist(2)) + bigger.getDist(1) * bigger.getDist(2));
}

AABB& AABB::Merge(const AABB& other) {
    _lower_bnd = min(_lower_bnd, other._lower_bnd);
    _upper_bnd = max(_upper_bnd, other._upper_bnd);
    return *this;
}

bool AABB::intersect(const Ray& ray, float& t_in, float& t_out) const {
    float dir_frac_x = ray._dir.x == 0.0f ? 1.0e32 : 1.0f / ray._dir.x;
    float dir_frac_y = ray._dir.y == 0.0f ? 1.0e32 : 1.0f / ray._dir.y;
    float dir_frac_z = ray._dir.z == 0.0f ? 1.0e32 : 1.0f / ray._dir.z;

    float tx1 = (_lower_bnd.x - ray._origin.x) * dir_frac_x;
    float tx2 = (_upper_bnd.x - ray._origin.x) * dir_frac_x;
    float ty1 = (_lower_bnd.y - ray._origin.y) * dir_frac_y;
    float ty2 = (_upper_bnd.y - ray._origin.y) * dir_frac_y;
    float tz1 = (_lower_bnd.z - ray._origin.z) * dir_frac_z;
    float tz2 = (_upper_bnd.z - ray._origin.z) * dir_frac_z;

    t_in = max(min(tx1, tx2), min(ty1, ty2), min(tz1, tz2));
    t_out = min(max(tx1, tx2), max(ty1, ty2), max(tz1, tz2));

    t_in = max(t_in, ray._t_min);
    t_out = min(t_out, ray._t_max);

    return t_out < 0 ? false : t_out >= t_in;
}

inline Vec3f AABB::getCenter() const {
    return (_lower_bnd + _upper_bnd) / 2.0f;
}

inline int AABB::getCenter(int dim) const {
    return (_lower_bnd[dim] + _upper_bnd[dim]) / 2.0f;
}

inline float AABB::getDist(unsigned int dim) const {
    return _upper_bnd[dim] - _lower_bnd[dim];
}

inline bool AABB::isOverlap(const AABB &other) const {
    return (_lower_bnd.x >= other._lower_bnd.x && _lower_bnd.x <= other._upper_bnd.x || other._lower_bnd.x >= _lower_bnd.x && other._lower_bnd.x <= _upper_bnd.x) &&
           (_lower_bnd.y >= other._lower_bnd.y && _lower_bnd.y <= other._upper_bnd.y || other._lower_bnd.y >= _lower_bnd.y && other._lower_bnd.y <= _upper_bnd.y) &&
           (_lower_bnd.z >= other._lower_bnd.z && _lower_bnd.z <= other._upper_bnd.z || other._lower_bnd.z >= _lower_bnd.z && other._lower_bnd.z <= _upper_bnd.z);
}

std::shared_ptr<std::vector<Vec3f>> BVH::set_positions(std::shared_ptr<std::vector<Vec3f>> positions) {
    return _positions = positions;
}

std::shared_ptr<std::vector<Vec3f>> BVH::set_normals(std::shared_ptr<std::vector<Vec3f>> normals) {
    return _normals = normals;
}

std::shared_ptr<std::vector<Triangle>> BVH::set_triangles(std::shared_ptr<std::vector<Triangle>> triangles) {
    return _triangles = triangles;
}

std::shared_ptr<std::vector<Vec3f>> BVH::set_colors(std::shared_ptr<std::vector<Vec3f>> colors) {
    return _colors = colors;
}

Scene* BVH::set_scene(Scene* scene) {
    return _scene = scene;
}

void BVH::build_BVH() {
    time_t start, stop;
    time(&start);
    auto tmp_triangles = _triangles;
    _triangles = std::make_shared<std::vector<Triangle>>();
    std::vector<Triangle_Info_AABB> triangles_aabbs;
    triangles_aabbs.reserve(tmp_triangles->size());
    for(auto triangle : *tmp_triangles)
        triangles_aabbs.push_back(Triangle_Info_AABB{.triangle = triangle, 
                                                     .aabb = AABB((*_positions)[triangle.v_idx.x], 
                                                                  (*_positions)[triangle.v_idx.y],  
                                                                  (*_positions)[triangle.v_idx.z])});
    _bvh_nodes.reserve(triangles_aabbs.size());

    int right;
    build_bvh_recursively(0,triangles_aabbs.size(), right, triangles_aabbs);

    _scene->_triangles = _triangles;

    time(&stop);

    double diff = difftime(stop, start);
    int hrs = int(diff) / 3600;
    int mins = int(diff) / 60 - hrs * 60;
    int secs = int(diff) - hrs * 3600 - mins * 60;
    std::cout << "------BVH Generation Complete------" << std::endl;
    std::cout << "Time Taken: " << hrs << " hours, " << mins << " minites, " << secs << " seconds" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}

void BVH::build_bvh_recursively(int l, int r, int& right_child, std::vector<Triangle_Info_AABB>& triangles_aabb) {
    AABB total_aabb;
    for(int i = l ; i < r ; ++i)
        total_aabb.Merge(triangles_aabb[i].aabb);
    int dim;
    float max_delta = NEG_INF_F;
    if(total_aabb.getDist(0) > max_delta) max_delta = total_aabb.getDist(0), dim = 0;
    if(total_aabb.getDist(1) > max_delta) max_delta = total_aabb.getDist(1), dim = 1;
    if(total_aabb.getDist(2) > max_delta) max_delta = total_aabb.getDist(2), dim = 2;
    if(max_delta == 0.0f || r - l <= BVH_LEAF_MAX_SIZE) {
        for(int i = l ; i < r ; ++i)
            _triangles->push_back(triangles_aabb[i].triangle);
        right_child = _bvh_nodes.size();
        _bvh_nodes.push_back(Bvh_Node{.left = l, .right = r, .aabb = total_aabb});
        return;
    }
    std::sort(triangles_aabb.begin() + l, triangles_aabb.begin() + r, 
              [&](const Triangle_Info_AABB& l, const Triangle_Info_AABB& r)->bool {
                return l.aabb.getCenter(dim) < r.aabb.getCenter(dim);
              });

    std::vector<AABB> right(r-l);
    right[r-l-1] = triangles_aabb[r-1].aabb;
    for(int i = r-2 ; i >= l ; --i)
        right[i-l] = AABB(triangles_aabb[i].aabb).Merge(right[i+1-l]);
    assert(right.size() == r-l);
    float min_cost = POS_INF_F;
    int mid = -1;
    AABB left;
    
    for(int divide_idx = l+1 ; divide_idx < r ; ++divide_idx) {
        left.Merge(triangles_aabb[divide_idx-1].aabb);
        float local_cost = t_traveral + left.cal_prob(total_aabb) * t_intersect * (divide_idx - l) + right[divide_idx-l].cal_prob(total_aabb) * t_intersect * (r - divide_idx);
        if(local_cost < min_cost) min_cost = local_cost, mid = divide_idx;
    }
    right.clear();
    assert(mid != -1);
    int this_idx = _bvh_nodes.size();
    _bvh_nodes.push_back(Bvh_Node{.left = INTERAL_NODE, .right = 0, .aabb = total_aabb});
    build_bvh_recursively(l, mid, right_child, triangles_aabb);
    build_bvh_recursively(mid, r, right_child, triangles_aabb);

    _bvh_nodes[this_idx].right.right_node_idx = right_child;
    right_child = this_idx;
}

void BVH::bvhHit(Interaction& interaction, const Ray& ray, int node_idx = 0) const {
    if(_bvh_nodes[node_idx].left.type != INTERAL_NODE) {
        for(int i = _bvh_nodes[node_idx].left.start ; i != _bvh_nodes[node_idx].right.end ; ++i) {
            //intersect with a triangle
            Vec3f v0 = (*_positions)[(*_triangles)[i].v_idx.x];
            Vec3f v1 = (*_positions)[(*_triangles)[i].v_idx.y];
            Vec3f v2 = (*_positions)[(*_triangles)[i].v_idx.z];
            Vec3f v0v1 = v1 - v0;
            Vec3f v0v2 = v2 - v0;
            Vec3f pvec = glm::cross(ray._dir, v0v2);
            float det  = glm::dot(v0v1, pvec);
            float invDet = 1.0f / det;
            Vec3f tvec = ray._origin - v0;
            float u = glm::dot(tvec, pvec) * invDet;
            if(u < 0.f || u > 1.f) continue;
            Vec3f qvec = glm::cross(tvec, v0v1);
            float v = glm::dot(ray._dir, qvec) * invDet;
            if(v < 0.f || u + v > 1.f) continue;
            float t = glm::dot(v0v2, qvec) * invDet;
            if(t < ray._t_min || t > ray._t_max || t > interaction.dist) continue;

            interaction.dist = t;
            interaction.type = (*_triangles)[i].emissive ? Interaction::Type::LIGHT : Interaction::Type::GEOMETRY;
            interaction.triangle_idx = i;
            interaction.uv.x = u;
            interaction.uv.y = v;
        }
    } else {

        float t_in, t_out;
        int leftIndex = node_idx+1;
        int rightIndex = _bvh_nodes[node_idx].right.right_node_idx;
        if(_bvh_nodes[leftIndex].aabb.intersect(ray, t_in, t_out)) bvhHit(interaction, ray, leftIndex);
        if(_bvh_nodes[rightIndex].aabb.intersect(ray, t_in, t_out)) bvhHit(interaction, ray, rightIndex);


        // float t_in_l, t_out_l, t_in_r, t_out_r;
        // bool hit_left = _bvh_nodes[node_idx+1].aabb.intersect(ray, t_in_l, t_out_l);
        // bool hit_right = _bvh_nodes[_bvh_nodes[node_idx].right.right_node_idx].aabb.intersect(ray, t_in_r, t_out_r);
        // if(hit_left && hit_right) {
        //     if(t_in_l < t_in_r) {
        //         bvhHit(interaction, ray, node_idx+1);
        //         if(interaction.type != Interaction::Type::NONE && interaction.dist < t_in_r) return;
        //         bvhHit(interaction, ray, _bvh_nodes[node_idx].right.right_node_idx);
        //     } else {
        //         bvhHit(interaction, ray, _bvh_nodes[node_idx].right.right_node_idx);
        //         if(interaction.type != Interaction::Type::NONE && interaction.dist < t_in_l) return;
        //         bvhHit(interaction, ray, node_idx+1);
        //     }
        // } else if(hit_left) {
        //     bvhHit(interaction, ray, node_idx+1);
        // } else if(hit_right) {
        //     bvhHit(interaction, ray, _bvh_nodes[node_idx].right.right_node_idx);
        // }
    }
}


/// @brief 
/// @param ray 
/// @param node_idx 
/// @return false represented shadowed
bool BVH::visiality_test(const Ray& ray, int node_idx = 0) const {
    if(_bvh_nodes[node_idx].left.type != INTERAL_NODE) {
        for(int i = _bvh_nodes[node_idx].left.start ; i != _bvh_nodes[node_idx].right.end ; ++i) {
            Vec3f v0 = (*_positions)[(*_triangles)[i].v_idx.x];
            Vec3f v1 = (*_positions)[(*_triangles)[i].v_idx.y];
            Vec3f v2 = (*_positions)[(*_triangles)[i].v_idx.z];
            Vec3f v0v1 = v1 - v0;
            Vec3f v0v2 = v2 - v0;
            Vec3f pvec = glm::cross(ray._dir, v0v2);
            float det  = glm::dot(v0v1, pvec);
            float invDet = 1.0f / det;
            Vec3f tvec = ray._origin - v0;
            float u = glm::dot(tvec, pvec) * invDet;
            if(u < 0.f || u > 1.f) continue;
            Vec3f qvec = glm::cross(tvec, v0v1);
            float v = glm::dot(ray._dir, qvec) * invDet;
            if(v < 0.f || u + v > 1.f) continue;
            float t = glm::dot(v0v2, qvec) * invDet;
            if(t > ray._t_min && t < ray._t_max) return false;
        }
        return true;
    } else {
        float t_in_l, t_out_l, t_in_r, t_out_r;
        bool hit_left_aabb = _bvh_nodes[node_idx+1].aabb.intersect(ray, t_in_l, t_out_l);
        bool hit_right_aabb = _bvh_nodes[_bvh_nodes[node_idx].right.right_node_idx].aabb.intersect(ray, t_in_r, t_out_r);
        if(t_in_l < ray._t_max && hit_left_aabb)
            if(!visiality_test(ray, node_idx+1))
                return false;
        if(t_in_r < ray._t_max && hit_right_aabb)
            if(!visiality_test(ray, _bvh_nodes[node_idx].right.right_node_idx))
                return false;
        return true;
    }
}