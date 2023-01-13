#include "scene.h"
#include "accel.h"
#include "utils.h"

Scene::Scene() { 
    _bvh.set_scene(this); 
}


void Scene::set_positions(const std::vector<Vec3f>& positions) { 
    auto ptr = _pos = new Vec3f[n_pos = positions.size()];
    for (const auto& p: positions)
        *ptr++ = p;
    
}
void Scene::set_normals(const std::vector<Vec3f>& normals) {
    auto ptr = _norms = new Vec3f[n_norms = normals.size()];
    for (const auto& n: normals)
        *ptr++ = n;
    
}
void Scene::set_colors(const std::vector<Vec3f>& colors) { 
    auto ptr = _colors = new Vec3f[n_colors = colors.size()];
    for (const auto& c: colors) 
        *ptr++ = c;
}

void Scene::set_triangles(const std::vector<Triangle>& triangles) {
    auto ptr = _triangles = new Triangle[n_triangles = triangles.size()];
    std::vector<Triangle> emissive_triangle;
    emissive_triangle.reserve(n_triangles);
    for(const auto& t : triangles) {
        *ptr++ = t;
        if(t.emissive)
            emissive_triangle.push_back(t);
    }
    ptr = _emissive_triangles = new Triangle[n_emissive_triangles = emissive_triangle.size()];
    for(const auto &t : emissive_triangle)
        *ptr++ = t;
}


void Scene::build_bvh() {
    _bvh.build_BVH();
}

BVH& Scene::get_bvh() {
    return _bvh;
}



void Scene::CPU2GPU() {

    utils::CPU2GPU(_pos, n_pos);
    utils::CPU2GPU(_norms, n_norms);
    utils::CPU2GPU(_colors, n_colors);
    utils::CPU2GPU(_triangles, n_triangles);
    utils::CPU2GPU(_emissive_triangles, n_emissive_triangles);

    _bvh.CPU2GPU();

}

void Scene::GPU2CPU() {

    utils::GPU2CPU(_pos, n_pos);
    utils::GPU2CPU(_norms, n_norms);
    utils::GPU2CPU(_colors, n_colors);
    utils::GPU2CPU(_triangles, n_triangles);
    utils::GPU2CPU(_emissive_triangles, n_emissive_triangles);

    _bvh.GPU2CPU();

}

void Scene::free_device() {

    cudaFree(_pos);
    cudaFree(_norms);
    cudaFree(_colors);
    cudaFree(_triangles);
    cudaFree(_emissive_triangles);
    
}


