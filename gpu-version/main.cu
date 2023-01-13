#include "integrator.h"
#include "obj_loader.h"
#include <iostream>
#include <string>

const Vec3f grey{ 0.725f, 0.71f , 0.68f };
const Vec3f green{ 0.14f , 0.45f , 0.091f };
const Vec3f red{ 0.63f , 0.065f, 0.05f };
const Vec3f blue{ 0.14f , 0.091f, 0.78f };
const Vec3f white{ 1.0f  , 1.0f  , 1.0f };

int main() {
    ObjLoader objLoader;


    // add some geometry
    // left
    objLoader.loadObj("../../assets/left.obj", grey, false, { 0.0f,0.0f,0.0f }, 1.0f);
    // right
    objLoader.loadObj("../../assets/right.obj", grey, false, { 0.0f,0.0f,0.0f }, 1.0f);
    // floor
    objLoader.loadObj("../../assets/floor.obj", grey, false, { 0.0f,0.0f,0.0f }, 1.0f);
    // ceiling
    objLoader.loadObj("../../assets/ceiling.obj", grey, false, { 0.0f,0.0f,0.0f }, 1.0f);
    // back
    objLoader.loadObj("../../assets/back.obj", grey, false, { 0.0f,0.0f,0.0f }, 1.0f);
    // // front
    // objLoader.loadObj("../../assets/front.obj", grey, false, {0.0f,0.0f,0.0f},1.0f);
    // bunny
    objLoader.loadObj("../../assets/stanford_bunny.obj", grey, false, { 0.1,0,0 }, 6.0f);
    // objLoader.loadObj("../../assets/stanford_bunny.obj", grey, false, {-0.4,-0.1,0.2}, 4.0f);
    // // bunny
    // objLoader.loadObj("../../assets/stanford_bunny.obj", grey, false, {0.3,0.4,-0.2}, 4.0f);
    // //dragon
    // objLoader.loadObj("../../assets/stanford_dragon.obj", grey, false, {0.3,0.4,-0.2}, 4.0f);
    // short box
    // objLoader.loadObj("../../assets/short_box.obj", grey, false, {0,0,0}, 1);

    //add geometry end

    // add lights
    // left red light
    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,1.0f,0.0f }, 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,1.25f,0.0f }, 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,0.75f,0.0f }, 0.25f);

    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,1.0f,0.5f }, 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,1.25f,0.5f }, 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,0.75f,0.5f }, 0.25f);

    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,1.0f,-0.5f }, 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,1.25f,-0.5f }, 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj", red, true, { -0.99f,0.75f,-0.5f }, 0.25f);
    // objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.0f,0.0f} , 0.25f);
    // right green light
    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,1.0f,0.0f }, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,1.25f,0.0f }, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,0.75f,0.0f }, 0.25f);

    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,1.0f,0.5f }, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,1.25f,0.5f }, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,0.75f,0.5f }, 0.25f);

    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,1.0f,-0.5f }, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,1.25f,-0.5f }, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj", green, true, { 0.99f,0.75f,-0.5f }, 0.25f);
    // objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.0f,0.0f}, 0.25f);
    // top white light
    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { 0.0f,1.99f,0.25f }, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { 0.0f,1.99f,0.0f }, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { 0.0f,1.99f,-0.25f }, 0.25f);

    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { 0.5f,1.99f,0.25f }, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { 0.5f,1.99f,0.0f }, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { 0.5f,1.99f,-0.25f }, 0.25f);

    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { -0.5f,1.99f,0.25f }, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { -0.5f,1.99f,0.0f }, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white, true, { -0.5f,1.99f,-0.25f }, 0.25f);
    // objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.0f,1.99f,0.0f}, 0.25f);
    // // back blue light
    objLoader.loadObj("../../assets/backLight.obj", blue, true, { 0.0f,1.0f,-0.99f }, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue, true, { 0.0f,1.25f,-0.99f }, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue, true, { 0.0f,0.75f,-0.99f }, 0.25f);

    objLoader.loadObj("../../assets/backLight.obj", blue, true, { 0.5f,1.0f,-0.99f }, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue, true, { 0.5f,1.25f,-0.99f }, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue, true, { 0.5f,0.75f,-0.99f }, 0.25f);


    objLoader.loadObj("../../assets/backLight.obj", blue, true, { -0.5f,1.0f,-0.99f }, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue, true, { -0.5f,1.25f,-0.99f }, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue, true, { -0.5f,0.75f,-0.99f }, 0.25f);

    //add lights end

    auto scene = std::make_shared<Scene>();
    auto image = std::make_shared<ImageRGB>(400, 400);
    auto camera = std::make_shared<Camera>();

    Sampler s;

    camera->setImage(image);
    camera->setPosition({ 0.0f,1.0f, 4.0f });
    camera->lookAt({ 0.0f,1.0f,0.0f }, { 0.0f,1.0f,0.0f });

    scene->set_colors(objLoader.getColors());
    scene->set_positions(objLoader.getPositions());
    scene->set_normals(objLoader.getNormals());
    scene->set_triangles(objLoader.getTriangles());
    scene->build_bvh();

    Integrator integrator(camera, scene);
    integrator.render_cuda();
    image->writeImgToFile("../image/result1.png");
    integrator.render_cuda();
    image->writeImgToFile("../image/result2.png");
    integrator.render_cuda();
    image->writeImgToFile("../image/result3.png");
    integrator.render_cuda();
    image->writeImgToFile("../image/result4.png");
    integrator.render_cuda();
    image->writeImgToFile("../image/result5.png");
    return 0;
}