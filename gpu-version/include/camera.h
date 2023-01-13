#pragma once

#include "ray.h"
#include "image.h"
#include <memory>

class Camera {
public:
    Camera();

    void lookAt(const Vec3f &look_at, const Vec3f &ref_up = {0, 0, 1});
    void setPosition(const Vec3f &pos);
    [[nodiscard]] Vec3f getPosition() const;
    void setFov(float new_fov);
    [[nodiscard]] float getFov() const;
    void setImage(std::shared_ptr<ImageRGB>&);
    void setImage(ImageRGB*);
    [[nodiscard]] ImageRGB* getImage();
    [[nodiscard]] const Vec2i& getResolution();

    __device__ inline Ray generateRay(float dx, float dy) {
        dx = dx / float(resolution.x) * 2.0f - 1.0f;
        dy = dy / float(resolution.y) * 2.0f - 1.0f;
        return { position, glm::normalize(dx * right + dy * up + forward), RAY_DEFAULT_MIN, RAY_DEFAULT_MAX };
    }

private:
    Vec3f position;
    Vec3f forward;
    Vec3f up;
    Vec3f right;
    float focal_len;
    float fov;
    Vec2i resolution;
    ImageRGB* image; // this is a host_pointer
};
