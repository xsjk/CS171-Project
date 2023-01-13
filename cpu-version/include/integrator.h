#ifndef _INTEGRATOR_H_
#define _INTEGRATOR_H_

#include "core.h"
#include "camera.h"
#include "scene.h"
#include <memory>

class Integrator {
  public:
    Integrator(std::shared_ptr<Camera>, std::shared_ptr<Scene>);
    void render();
    bool isFirstFrame;
  private:
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<Scene> _scene;
    std::vector<Reservoir> _prevReservoirs;
};

#endif