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
    const std::vector<Vec3f>& getPositions();
    const std::vector<Vec3f>& getNormals();
    const std::vector<Vec3f>& getColors();
    const std::vector<Triangle>& getTriangles();
  private:
    std::vector<Vec3f>     _positions;
    std::vector<Vec3f>     _normals  ;
    std::vector<Vec3f>     _colors   ;
    std::vector<Triangle>  _triangles;
};

#endif // _OBJ_LODER_H_