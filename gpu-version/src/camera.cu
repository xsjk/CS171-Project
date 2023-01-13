#include "camera.h"
#include "core.h"
#include "ray.h"

Camera::Camera()
    : position(0, 0, 4), fov(45), focal_len(1) {
}


void Camera::lookAt(const Vec3f &look_at, const Vec3f &ref_up) {
    forward = glm::normalize(look_at - position);
    right = glm::normalize(glm::cross(forward, ref_up));
    up = glm::normalize(glm::cross(right, forward));
    float half_fov = glm::radians(fov / 2);
    float y_len = tanf(half_fov) * focal_len;
    up *= y_len;
    float xLen = y_len * image->getAspectRatio();
    right *= xLen;
}

void Camera::setPosition(const Vec3f &pos) {
    position = pos;
}

Vec3f Camera::getPosition() const {
    return position;
}

void Camera::setFov(float new_fov) {
    fov = new_fov;
}

float Camera::getFov() const {
    return fov;
}

void Camera::setImage(std::shared_ptr<ImageRGB> &img) {
    setImage(img.get());
}

void Camera::setImage(ImageRGB* img) {
    image = img;
    resolution = img->getResolution();
}

const Vec2i& Camera::getResolution() {
    return resolution;
}

ImageRGB* Camera::getImage() {
    return image;
}
