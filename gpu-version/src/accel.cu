#include "accel.h"
#include "core.h"
#include "scene.h"
#include "utils.h"
#include <assert.h>
#include <ctime>
#include <iostream>
#include <algorithm>


AABB::AABB() : _lower_bnd(POS_INF_F, POS_INF_F, POS_INF_F), 
               _upper_bnd(NEG_INF_F, NEG_INF_F, NEG_INF_F) {}

AABB::AABB(Vec3f lower, Vec3f upper) : _lower_bnd(lower), 
                                       _upper_bnd(upper) {}

AABB::AABB(Vec3f v1, Vec3f v2, Vec3f pos3) {
    _lower_bnd = utils::cwiseMin(v1, v2, pos3);
    _upper_bnd = utils::cwiseMax(v1, v2, pos3);
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

#include <iostream>

void BVH::set_scene(Scene* scene) {
    _scene = thrust::device_pointer_cast(scene);
}

void BVH::build_BVH() {
    time_t start, stop;
    time(&start);
    std::vector<Triangle_Info_AABB> triangles_aabbs;
    triangles_aabbs.reserve(_scene.get()->n_triangles);
    for(auto i = 0; i < _scene.get()->n_triangles; ++i) {
        const auto& t = _scene.get()->_triangles[i];
        triangles_aabbs.push_back({ t, AABB(   _scene.get()->_pos[t.v_idx.x], 
                                                    _scene.get()->_pos[t.v_idx.y],  
                                                    _scene.get()->_pos[t.v_idx.z])});
    }
    _bvh_nodes_end = _bvh_nodes = new Bvh_Node[_scene.get()->n_triangles];

    int right;
    auto ptr = _scene.get()->_triangles;
    build_bvh_recursively(0, _scene.get()->n_triangles, right, triangles_aabbs, ptr);

    n_bvh_nodes = _bvh_nodes_end - _bvh_nodes;

    time(&stop);

    double diff = difftime(stop, start);
    int hrs = int(diff) / 3600;
    int mins = int(diff) / 60 - hrs * 60;
    int secs = int(diff) - hrs * 3600 - mins * 60;
    std::cout << "------BVH Generation Complete------" << std::endl;
    std::cout << "Time Taken: " << hrs << " hours, " << mins << " minites, " << secs << " seconds" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}

void BVH::build_bvh_recursively(int l, int r, int& right_child, std::vector<Triangle_Info_AABB>& triangles_aabb, Triangle*& tri) {
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
            *tri++ = triangles_aabb[i].triangle;
        right_child = _bvh_nodes_end - _bvh_nodes;
        *_bvh_nodes_end++ = Bvh_Node{
            l, r, total_aabb
        };
        //*_bvh_nodes_end++ = Bvh_Node{ .left = l, .right = r, .aabb = total_aabb };
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
    int this_idx = _bvh_nodes_end - _bvh_nodes;
    *_bvh_nodes_end++ = Bvh_Node{INTERAL_NODE, 0, total_aabb};
    build_bvh_recursively(l, mid, right_child, triangles_aabb, tri);
    build_bvh_recursively(mid, r, right_child, triangles_aabb, tri);

    _bvh_nodes[this_idx].right.right_node_idx = right_child;
    right_child = this_idx;
}

BVH::Bvh_Node* BVH::get_nodes() {
    return _bvh_nodes;
}


void BVH::CPU2GPU() {
    utils::CPU2GPU(_bvh_nodes, n_bvh_nodes);
    _bvh_nodes_end = _bvh_nodes + n_bvh_nodes;
}

void BVH::GPU2CPU() {
    utils::GPU2CPU(_bvh_nodes, n_bvh_nodes);
    _bvh_nodes_end = _bvh_nodes + n_bvh_nodes;
}
