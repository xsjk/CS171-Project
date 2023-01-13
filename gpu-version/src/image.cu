#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include "image.h"
#include "utils.h"

ImageRGB::ImageRGB(int width, int height)
        : _resolution(width, height) {
    data.resize(width * height);
}

float ImageRGB::getAspectRatio() const {
    return static_cast<float>(_resolution.x) / static_cast<float>(_resolution.y);
}

void ImageRGB::setPixel(int x, int y, const Vec3f &value) {
    data[x + _resolution.x * y] = value;
}

void ImageRGB::fromGPU(Vec3f* ptr) {
    cudaMemcpy(data.data(), ptr, _resolution.x * _resolution.y * sizeof(Vec3f), cudaMemcpyDeviceToHost);
}

void ImageRGB::fromCPU(Vec3f* ptr) {
    memcpy(data.data(), ptr, _resolution.x * _resolution.y * sizeof(Vec3f));
}

void ImageRGB::writeImgToFile(const std::string &file_name) {
    std::cerr << "write to file: " << file_name << std::endl;
    std::vector<uint8_t> rgb_data;
    rgb_data.reserve(_resolution.x * _resolution.y * 3);
    for (int x = 0; x < _resolution.x; ++x) {
        for (int y = 0; y < _resolution.y; ++y) {
            int i = _resolution.x * y + x;
            rgb_data.push_back(utils::gammaCorrection(data[i].x));
            rgb_data.push_back(utils::gammaCorrection(data[i].y));
            rgb_data.push_back(utils::gammaCorrection(data[i].z));
        }
    }

    stbi_flip_vertically_on_write(true);
    stbi_write_png(file_name.c_str(), _resolution.x, _resolution.y, 3, rgb_data.data(), 0);
}

