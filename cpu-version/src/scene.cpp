#include "scene.h"
#include "accel.h"

Scene::Scene() {
    _bvh.set_scene(this);
}


void Scene::add_positions(std::shared_ptr<std::vector<Vec3f>> positions) {
    _pos = positions;
    _bvh.set_positions(positions);
    _lights.set_positions(positions);
}

void Scene::add_normals(std::shared_ptr<std::vector<Vec3f>> normals) {
    _norms = normals;
    _bvh.set_normals(normals);
    _lights.set_normals(normals);
}

void Scene::add_colors(std::shared_ptr<std::vector<Vec3f>> colors) {
    _colors = colors;
    _bvh.set_colors(colors);
    _lights.set_colors(colors);
}



void Scene::add_triangles(std::shared_ptr<std::vector<Triangle>> triangles) {
    _triangles = triangles;
    _bvh.set_triangles(triangles);
    std::shared_ptr<std::vector<Triangle>> emissive_triangles = std::make_shared<std::vector<Triangle>>();
    for(auto triangle : *triangles)
        if(triangle.emissive)
            emissive_triangles->push_back(triangle);
    _lights.set_triangles(emissive_triangles);
}

void Scene::build_bvh() {
    _bvh.build_BVH();
}

Sampler* Scene::set_sampler(Sampler* sampler) {
    return _sampler = _lights.set_sampler(sampler);
}

bool Scene::isShadowed(const Ray& ray) const {
    return !_bvh.visiality_test(ray, 0);
}

void Scene::intersect(const Ray& ray, Interaction& interaction) const {
    return _bvh.bvhHit(interaction, ray, 0);
}

Lights& Scene::get_lights() {
    return _lights;
}

Sampler* Scene::get_sampler() {
    return _sampler;
}