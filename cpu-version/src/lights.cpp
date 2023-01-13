#include "lights.h"

Sampler* Lights::set_sampler(Sampler* sampler) {
    return _sampler = sampler;
}

std::shared_ptr<std::vector<Vec3f>> Lights::set_positions(std::shared_ptr<std::vector<Vec3f>> positions) {
    return _pos = positions;
}

std::shared_ptr<std::vector<Vec3f>> Lights::set_normals(std::shared_ptr<std::vector<Vec3f>> normals) {
    return _norms = normals; 
}

std::shared_ptr<std::vector<Triangle>> Lights::set_triangles(std::shared_ptr<std::vector<Triangle>> emissive_triangles) {
    return _emissive_triangles = emissive_triangles;
}

std::shared_ptr<std::vector<Vec3f>> Lights::set_colors(std::shared_ptr<std::vector<Vec3f>> colors) {
    return _colors = colors;
}

void Lights::sample(Vec3f& samplePos, Vec3f& sampleNormal, Vec3f& sampleColor, float& pdf) const {
    unsigned int idx = _sampler->get1U(0,_emissive_triangles->size());
    float u = _sampler->get1D();
    float v = _sampler->get1D();
    if(u + v > 1.0f) u = 1.0f - u, v = 1.0f - v;
    samplePos = u * (*_pos)[(*_emissive_triangles)[idx].v_idx.y] + v * (*_pos)[(*_emissive_triangles)[idx].v_idx.z] + (1.0f - u - v) * (*_pos)[(*_emissive_triangles)[idx].v_idx.x];
    sampleNormal = glm::normalize(u * (*_norms)[(*_emissive_triangles)[idx].n_idx.y] + v * (*_norms)[(*_emissive_triangles)[idx].n_idx.z] + (1.0f - u - v) * (*_norms)[(*_emissive_triangles)[idx].n_idx.x]);
    sampleColor = u * (*_colors)[(*_emissive_triangles)[idx].c_idx.y] + v * (*_colors)[(*_emissive_triangles)[idx].c_idx.z] + (1.0f - u - v) * (*_colors)[(*_emissive_triangles)[idx].c_idx.x];
    Vec3f v1v0 = (*_pos)[(*_emissive_triangles)[idx].v_idx.y] - (*_pos)[(*_emissive_triangles)[idx].v_idx.x];
    Vec3f v2v0 = (*_pos)[(*_emissive_triangles)[idx].v_idx.z] - (*_pos)[(*_emissive_triangles)[idx].v_idx.x];
    float s = glm::length(glm::cross(v2v0, v1v0)) / 2.0f;
    pdf = 1.0f / s / float(_emissive_triangles->size());
}