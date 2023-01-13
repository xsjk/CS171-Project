#pragma once

#include "core.h"

class ImageRGB {
public:
    ImageRGB() = delete;
    ImageRGB(int width, int height);
    [[nodiscard]] float getAspectRatio() const;
    __device__ __host__ [[nodiscard]] inline Vec2i getResolution() const { return _resolution; }
    void setPixel(int x, int y, const Vec3f &value);
    void fromGPU(Vec3f* ptr);
    void fromCPU(Vec3f* ptr);
    void writeImgToFile(const std::string &file_name);
private:
    std::vector<Vec3f> data;
    Vec2i _resolution;
};