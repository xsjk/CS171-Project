#pragma once

#include "core.h"
#include "camera.h"
#include "scene.h"
#include <memory>

class Integrator;

__global__ void set_seed(thrust::device_ptr<Integrator>, unsigned);
__global__ void render_kernel(thrust::device_ptr<Integrator>);
__global__ void spatial_reuse(thrust::device_ptr<Integrator>);
__global__ void set_notFirstFrame(thrust::device_ptr<Integrator>);

class Integrator {
public:
    Integrator() {};
    Integrator(const std::shared_ptr<Camera>&, const std::shared_ptr<Scene>&);
    ~Integrator();
    // void render();
    void render_cuda();
    __device__ void render_kernel(int x, int y);
    __device__ void spatial_reuse(int x, int y);
    __device__ void test(int x, int y);
    bool isFirstFrame;

    curandState* _samplers;
    Vec2i _resolution;
// private:

    thrust::device_ptr<Integrator> this_device;
    Camera* _camera_host;
    thrust::device_ptr<Camera> _camera;
    thrust::device_ptr<Scene> _scene;
    thrust::device_ptr<Reservoir> _prevReservoirs;
    thrust::device_ptr<Reservoir> _curReservoirs;
    thrust::device_ptr<Vec3f> _output;


};