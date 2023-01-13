#ifndef _OBJ_LODER_H_
#define _OBJ_LODER_H_

#include "core.h"
#include "triangle.h"

#include <string>
#include <memory>
#include <vector>


static bool loadObj(const std::string &path, std::vector<Vec3f> &vertices,
                    std::vector<Vec3f> &normals, std::vector<int> &v_index, std::vector<int> &n_index);


class ObjLoader{
  public:
    ObjLoader() = default;
    bool loadObj(const std::string &path, Vec3f color, bool emissive, Vec3f translation, float scale);
    std::shared_ptr<std::vector<Vec3f>> getPositions();
    std::shared_ptr<std::vector<Vec3f>> getNormals();
    std::shared_ptr<std::vector<Vec3f>> getColors();
    std::shared_ptr<std::vector<Triangle>> getTriangles();
  private:
    std::shared_ptr<std::vector<Vec3f>>     _positions = std::make_shared<std::vector<Vec3f>>();
    std::shared_ptr<std::vector<Vec3f>>     _normals   = std::make_shared<std::vector<Vec3f>>();
    std::shared_ptr<std::vector<Vec3f>>     _colors    = std::make_shared<std::vector<Vec3f>>();
    std::shared_ptr<std::vector<Triangle>>  _triangles = std::make_shared<std::vector<Triangle>>();
};

#endif // _OBJ_LODER_H_