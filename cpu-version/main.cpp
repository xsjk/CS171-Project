#include "integrator.h"
#include "obj_loder.h"
#include <iostream>
#include <string>

constexpr Vec3f grey  {0.725f, 0.71f , 0.68f};
constexpr Vec3f green {0.14f , 0.45f , 0.091f};
constexpr Vec3f red   {0.63f , 0.065f, 0.05f};
constexpr Vec3f blue  {0.14f , 0.091f, 0.78f};
constexpr Vec3f white {1.0f  , 1.0f  , 1.0f};
constexpr Vec3f yellow{0.72f , 0.7f  , 0.1f};

int main() {
    ObjLoader objLoader;

    // add some geometry
    // left
    objLoader.loadObj("../../assets/left.obj", grey, false, {0.0f,0.0f,0.0f}, 1.0f);
    // right
    objLoader.loadObj("../../assets/right.obj", grey, false, {0.0f,0.0f,0.0f}, 1.0f);
    // floor
    objLoader.loadObj("../../assets/floor.obj", grey, false, {0.0f,0.0f,0.0f},1.0f);
    // ceiling
    objLoader.loadObj("../../assets/ceiling.obj", grey, false, {0.0f,0.0f,0.0f}, 1.0f);
    // back
    objLoader.loadObj("../../assets/back.obj", grey, false, {0.0f,0.0f,0.0f},1.0f);
    // // front
    // objLoader.loadObj("../../assets/front.obj", grey, false, {0.0f,0.0f,0.0f},1.0f);
    // bunny
    objLoader.loadObj("../../assets/stanford_bunny.obj", grey, false, {0.1,0,0}, 6.0f);
    // objLoader.loadObj("../../assets/stanford_bunny.obj", grey, false, {-0.4,-0.1,0.2}, 4.0f);
    // // bunny
    // objLoader.loadObj("../../assets/stanford_bunny.obj", grey, false, {0.3,0.4,-0.2}, 4.0f);
    // // dragon
    // objLoader.loadObj("../../assets/stanford_dragon.obj", grey, false, {0.3,0.4,-0.2}, 4.0f);
    // short box
    // objLoader.loadObj("../../assets/short_box.obj", grey, false, {0,0,0}, 1);
    
    //add geometry end

    // add lights
    // left red light
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.0f,0.0f} , 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.25f,0.0f} , 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,0.75f,0.0f} , 0.25f);
    
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.0f,0.5f} , 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.25f,0.5f} , 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,0.75f,0.5f} , 0.25f);
    
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.0f,-0.5f} , 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.25f,-0.5f} , 0.25f);
    objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,0.75f,-0.5f} , 0.25f);
    // objLoader.loadObj("../../assets/leftLight.obj",    red,   true, {-0.99f,1.0f,0.0f} , 0.25f);
    // right green light
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.0f,0.0f}, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.25f,0.0f}, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,0.75f,0.0f}, 0.25f);
    
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.0f,0.5f}, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.25f,0.5f}, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,0.75f,0.5f}, 0.25f);
    
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.0f,-0.5f}, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.25f,-0.5f}, 0.25f);
    objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,0.75f,-0.5f}, 0.25f);
    // objLoader.loadObj("../../assets/rightLight.obj",   green,   true, {0.99f,1.0f,0.0f}, 0.25f);
    // top white light
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.0f,1.99f,0.25f}, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.0f,1.99f,0.0f}, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.0f,1.99f,-0.25f}, 0.25f);
    
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.5f,1.99f,0.25f}, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.5f,1.99f,0.0f}, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.5f,1.99f,-0.25f}, 0.25f);
    
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {-0.5f,1.99f,0.25f}, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {-0.5f,1.99f,0.0f}, 0.25f);
    objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {-0.5f,1.99f,-0.25f}, 0.25f);
    // objLoader.loadObj("../../assets/ceilingLight.obj", white,   true, {0.0f,1.99f,0.0f}, 0.25f);
    // // back blue light
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {0.0f,1.0f,-0.99f}, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {0.0f,1.25f,-0.99f}, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {0.0f,0.75f,-0.99f}, 0.25f);
    
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {0.5f,1.0f,-0.99f}, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {0.5f,1.25f,-0.99f}, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {0.5f,0.75f,-0.99f}, 0.25f);

    
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {-0.5f,1.0f,-0.99f}, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {-0.5f,1.25f,-0.99f}, 0.25f);
    objLoader.loadObj("../../assets/backLight.obj", blue,   true, {-0.5f,0.75f,-0.99f}, 0.25f);

    // // floor yellow  light
    // objLoader.loadObj("../../assets/ceilingLight.obj", yellow,   true, {0.0f,0.01f,0.25f}, 0.25f);
    // objLoader.loadObj("../../assets/ceilingLight.obj", yellow,   true, {0.0f,0.01f,0.0f}, 0.25f);
    // objLoader.loadObj("../../assets/ceilingLight.obj", yellow,   true, {0.0f,0.01f,-0.25f}, 0.25f);
    
    // objLoader.loadObj("../../assets/ceilingLight.obj", yellow,   true, {0.5f,0.01f,0.25f}, 0.25f);
    // objLoader.loadObj("../../assets/ceilingLight.obj", yellow,   true, {0.5f,0.01f,0.0f}, 0.25f);
    // objLoader.loadObj("../../assets/ceilingLight.obj", yellow,   true, {0.5f,0.01f,-0.25f}, 0.25f);
    
    // objLoader.loadObj("../../assets/backLight.obj", blue,   true, {0.0f,1.0f,-0.99f}, 0.25f);
    // // front blue light
    // objLoader.loadObj("../../assets/frontLight.obj", blue,   true, {0.0f,1.0f,0.99f}, 0.3f);
    // // front blue light
    // objLoader.loadObj("../../assets/frontLight.obj", blue,   true, {0.0f,1.0f,4.1f}, 1);


    //add lights end

    auto scene = std::make_shared<Scene>();
    auto image = std::make_shared<ImageRGB> (1000,1000);
    auto camera = std::make_shared<Camera>();

    Sampler s;

    camera->setImage(image);
    camera->setPosition({0.0f,1.0f, 4.0f});
    camera->lookAt({0.0f,1.0f,0.0f}, {0.0f,1.0f,0.0f});

    scene->add_colors(objLoader.getColors());
    scene->add_positions(objLoader.getPositions());
    scene->add_normals(objLoader.getNormals());
    scene->add_triangles(objLoader.getTriangles());
    scene->set_sampler(&s);
    scene->build_bvh();
    
    Integrator integrator(camera, scene);
    integrator.render();
    image->writeImgToFile("../image/onlyBunny/result1.png");   
    integrator.render();
    image->writeImgToFile("../image/onlyBunny/result2.png");
    integrator.render();
    image->writeImgToFile("../image/onlyBunny/result3.png");
    integrator.render();
    image->writeImgToFile("../image/onlyBunny/result4.png");
    integrator.render();
    image->writeImgToFile("../image/onlyBunny/result5.png");                                                            
    return 0;
}