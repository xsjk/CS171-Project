#include "obj_loder.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <iostream>
#include <cmath>

bool ObjLoader::loadObj(const std::string& path, Vec3f color, bool emissive, Vec3f translation, float scale) {
  std::cout << "-- Loading model " << path << std::endl;
  std::vector<Vec3f> positions; //save the positions of vectices of an object
  std::vector<Vec3f> normals;   //save the normals of vectices of an object
  std::vector<int> v_index;     //three consective integer represents the positions of three vectices of a triangle
  std::vector<int> n_index;     //three consective integer represents the normals of three vertices of a triangle
  tinyobj::ObjReaderConfig readerConfig;
  tinyobj::ObjReader reader;
  if (!reader.ParseFromFile(path, readerConfig)) {
    if (!reader.Error().empty()) {
      std::cerr << "TinyObjReader: " << reader.Error();
    }
    exit(1);
  }
  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }
  auto &attrib = reader.GetAttrib();
  auto &shapes = reader.GetShapes();
  auto &materials = reader.GetMaterials();

  //load the positions of an object
  for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
    positions.emplace_back(attrib.vertices[i], attrib.vertices[i + 1],
                          attrib.vertices[i + 2]);
  }
  //load positions end

  //load the normals an of an object
  for (size_t i = 0; i < attrib.normals.size(); i += 3) {
    //assert the normals is normalized
    // assert(glm::abs(1.0f - glm::length(Vec3f{attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2]})) < 0.001f);
    normals.emplace_back(attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2]);
  }
  //load normals end

  //load the infomation of triangles
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      auto fv = size_t(shapes[s].mesh.num_face_vertices[f]);

      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        v_index.push_back(idx.vertex_index);
        // tIndex.push_back(idx.texcoord_index);
        n_index.push_back(idx.normal_index);
      }
      index_offset += fv;
    }
  }
  std::cout << "  # vertices: " << attrib.vertices.size() / 3 << std::endl;
  std::cout << "  # faces: " << v_index.size() / 3 << std::endl;
  //load infomation of triangles end

  //change the positions to world space
  for(auto& position : positions)
    position = position * scale + translation;
  //change positions end

  //merge the new information with previous information
    //save new index
    int newPositionsStartIndex = _positions->size();
    int newNormalsStartIndex   = _normals->size();
    int newColorStartIndex     = _colors->size();
    int newTrianglesStartIndex = _triangles->size();
    //save new index end
    
    //save new information
      //new positions
      _positions->insert(_positions->end(), positions.begin(), positions.end());
      //new normals
      _normals->insert(_normals->end(), normals.begin(), normals.end());
      //new color
      _colors->push_back(color);
      //new triangles
      assert(v_index.size() == n_index.size());
      assert(v_index.size() % 3 == 0);
      for(int i = 0 ; i < v_index.size() / 3 ; ++i) 
        _triangles->push_back(Triangle{.emissive = emissive, 
                                       .v_idx = newPositionsStartIndex + Vec3i{v_index[3*i], v_index[3*i+1], v_index[3*i+2]},
                                       .n_idx = newNormalsStartIndex   + Vec3i{n_index[3*i], n_index[3*i+1], n_index[3*i+2]},
                                       .c_idx = newColorStartIndex     + Vec3i{0,0,0}});
    //save new information end
    return true;
}

std::shared_ptr<std::vector<Vec3f>> ObjLoader::getPositions() {
  return _positions;
}

std::shared_ptr<std::vector<Vec3f>> ObjLoader::getNormals() {
  return _normals; 
}

std::shared_ptr<std::vector<Vec3f>> ObjLoader::getColors() {
  return _colors;
}

std::shared_ptr<std::vector<Triangle>> ObjLoader::getTriangles() {
  return _triangles;
}