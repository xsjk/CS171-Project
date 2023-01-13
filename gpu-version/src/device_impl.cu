#include "integrator.h"
#include "core.h"
#include "utils.h"
#include "accel.h"
#include <iostream>


Integrator::Integrator(const std::shared_ptr<Camera>& camera, const std::shared_ptr<Scene>& scene) : isFirstFrame(true) {
    

    const Triangle* t = scene->_triangles;
    const Vec3f* _pos = scene->_pos;
    /*{
        constexpr auto i = 7373;
        Vec3f v0 = _pos[t[i].v_idx.x];
        Vec3f v1 = _pos[t[i].v_idx.y];
        Vec3f v2 = _pos[t[i].v_idx.z];

        printf("t[%d].v0 = pos[%d] = {%f, %f, %f}\n", i, t[i].v_idx.x, v0.x, v0.y, v0.z);
        printf("t[%d].v1 = pos[%d] = {%f, %f, %f}\n", i, t[i].v_idx.y, v1.x, v1.y, v1.z);
        printf("t[%d].v2 = pos[%d] = {%f, %f, %f}\n", i, t[i].v_idx.z, v2.x, v2.y, v2.z);
    }*/

    scene->CPU2GPU();
    _camera_host = camera.get();
    _camera = thrust::device_malloc<Camera>(1);
    _scene = thrust::device_malloc<Scene>(1);
    // printf("scene_ptr: %p\n", _scene.get());
    // printf("triangle_ptr: %p\n", scene->_triangles);
     printf("pos_ptr: %p n_pos: %d\n", scene->_pos, scene->n_pos);
     
    // printf("norm_ptr: %p\n", scene->_norms);
    // printf("color_ptr: %p\n", scene->_colors);
    
    // BVH::Bvh_Node b = thrust::device_pointer_cast(scene->get_bvh().get_nodes())[0];
    // printf("bvh_nodes.left %d\n", b.left);
    // printf("bvh_nodes.right %d\n", b.right);
    // printf("bvh_nodes.aabb.min %f %f %f\n", b.aabb._lower_bnd.x, b.aabb._lower_bnd.y, b.aabb._lower_bnd.z);
    // printf("bvh_nodes.aabb.max %f %f %f\n", b.aabb._upper_bnd.x, b.aabb._upper_bnd.y, b.aabb._upper_bnd.z);

    // Vec3f color = thrust::device_pointer_cast(scene->_colors)[1];
    // printf("color : (%f, %f, %f)\n", color.r, color.g, color.b);
    scene->get_bvh().set_scene(_scene.get());
    
    _resolution = camera->getResolution();
    _prevReservoirs = thrust::device_malloc<Reservoir>(_resolution.x * _resolution.y);
    _curReservoirs = thrust::device_malloc<Reservoir>(_resolution.x * _resolution.y);
    _output = thrust::device_malloc<Vec3f>(_resolution.x * _resolution.y);
    _samplers = thrust::device_malloc<curandState>(_resolution.x * _resolution.y).get();

    *_camera = *camera;
    *_scene = *scene;
}

Integrator::~Integrator() {
    // thrust::device_free(_camera);
    // thrust::device_free(_scene);
    // thrust::device_free(_prevReservoirs);
    // thrust::device_free(_curReservoirs);
    // thrust::device_free(_output);
}

__device__ void Integrator::render_kernel(int x, int y) {
    int idx = x * _resolution.y + y;

    const Sampler& sampler = _samplers[idx];
    const Scene& scene = *_scene;

    Vec3f visiblePointPosition;
    Vec3f visiblePointNormal;
    Vec3f visiblePointColor;
    Vec3f lightPointPosition;
    Vec3f lightPointNormal;
    Vec3f lightPointColor;
    Vec3f visibleToLight;
    float pdf;
    float p_hat;
    float LdotN;
    Reservoir newReservoir;
    Reservoir& curReservoir = _curReservoirs.get()[idx];
    Reservoir& prevReservoir = _prevReservoirs.get()[idx];
    Point newLightPoint;



    //start produce visible point
    Ray rayToVisiblePoint = _camera.get()->generateRay(x, y);

    /*printf("generated ray:\tdir={%f, %f, %f}\torigin={%f, %f, %f}\t t_min=%f\tt_max=%f\n",
        rayToVisiblePoint._dir.x,
        rayToVisiblePoint._dir.y,
        rayToVisiblePoint._dir.z,
        rayToVisiblePoint._origin.x,
        rayToVisiblePoint._origin.y,
        rayToVisiblePoint._origin.z,
        rayToVisiblePoint._t_min,
        rayToVisiblePoint._t_max
    );*/

    Interaction interaction;
    scene.intersect(rayToVisiblePoint, interaction);
     //if the ray doesn't hit any geometry or light in the scene, then M = 0, W = 0;
     //if the ray hits any geometry, including light
    if(interaction.type != Interaction::NONE) {

        float u = interaction.uv.x;
        float v = interaction.uv.y;
        Triangle tri = scene._triangles[interaction.triangle_idx];
        Vec3f v0 = scene._pos[tri.v_idx.x];
        Vec3f v1 = scene._pos[tri.v_idx.y];
        Vec3f v2 = scene._pos[tri.v_idx.z];
        Vec3f n0 = scene._norms[tri.n_idx.x];
        Vec3f n1 = scene._norms[tri.n_idx.y];
        Vec3f n2 = scene._norms[tri.n_idx.z];
        Vec3f c0 = scene._colors[tri.c_idx.x];
        Vec3f c1 = scene._colors[tri.c_idx.y];
        Vec3f c2 = scene._colors[tri.c_idx.z];



        visiblePointPosition = utils::interpolate(u, v, v0, v1, v2);
        visiblePointNormal = glm::normalize(utils::interpolate(u, v, n0, n1, n2));
        visiblePointColor = utils::interpolate(u, v, c0, c1, c2);

        /*printf("v0 = (%f, %f, %f)   v1 = (%f, %f, %f)   v2 = ги%f, %f, %f)   (u, v) = (%f, %f)\n",
            v0.x, v0.y, v0.z,
            v1.x, v1.y, v1.z,
            v2.x, v2.y, v2.z,
            u, v
        );*/
        newReservoir.visiblePoint = {visiblePointPosition, visiblePointNormal, visiblePointColor};
        newReservoir.isLight = (interaction.type == Interaction::LIGHT); 
    //end produce visible point


    //start generate light candidates by resample importance sampling(RIS) according to algorithm 3
        for(int j = 0 ; j < LIGHT_CANDIDATE_NUM ; ++j) {
            scene.sample_lights(lightPointPosition, lightPointNormal, lightPointColor, pdf, sampler);
            float dist = glm::length(visiblePointPosition - lightPointPosition);
            visibleToLight = glm::normalize(lightPointPosition - visiblePointPosition);
            Ray shadowRay (visiblePointPosition, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
            if(!scene.isShadowed(shadowRay)) {
                newLightPoint = Point{lightPointPosition, lightPointNormal, lightPointColor};
                LdotN = utils::max(glm::dot(visiblePointNormal, visibleToLight), 0.0f);
                p_hat = (LdotN == 0.0f) ? 0.0f : glm::length(visiblePointColor * lightPointColor * INV_PI * LdotN / dist / dist);
                newReservoir.update(newLightPoint, p_hat / pdf, sampler);
            } else {
                ++newReservoir.M;
            }

        }
      newReservoir.W = newReservoir.w_sum / utils::max(newReservoir.p_hat(), 0.0001f) / utils::max(float(newReservoir.M),0.0001f);
      curReservoir = newReservoir;
      //end RIS
    }
    if(!isFirstFrame) {

        //printf("!first Frame\n");

        // this is not tested
        
        //ignore the situation where visible point is not valid or previous reservoir is invalid
        if(curReservoir.M > 0 && prevReservoir.M > 0) {
            //start temporal reuse
            Reservoir temporalReservoir = curReservoir;
            visibleToLight = glm::normalize(prevReservoir.lightPoint.position - temporalReservoir.visiblePoint.position);
            float dist = glm::length(prevReservoir.lightPoint.position - temporalReservoir.visiblePoint.position);
            Ray shadowRay (temporalReservoir.visiblePoint.position, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
            LdotN = utils::max(0.0f, glm::dot(temporalReservoir.visiblePoint.normal, visibleToLight));
            p_hat = (scene.isShadowed(shadowRay) || LdotN == 0.0f) ? 0.0f : glm::length(temporalReservoir.visiblePoint.color * prevReservoir.lightPoint.color * INV_PI * LdotN / dist / dist);
            prevReservoir.M = int(utils::min(20.f * float(curReservoir.M), float(prevReservoir.M)));
            temporalReservoir.update(prevReservoir.lightPoint, p_hat * prevReservoir.W * float(prevReservoir.M), sampler);
            //set M value
            temporalReservoir.M = curReservoir.M + prevReservoir.M;
            //set W value
            temporalReservoir.W = temporalReservoir.w_sum / utils::max(p_hat, 0.0001f) / utils::max(float(temporalReservoir.M), 0.0001f);
            curReservoir = temporalReservoir;
            //end temporal reuse

            //printf("curReservoir.W = %f\n", curReservoir.W);
            //check the correctness of temporal reuse
            // _prevReservoirs[x * resolution.y + y] = temporalReservoir;
            // Vec3f color = temporalReservoir.W * temporalReservoir.lightPoint.color * temporalReservoir.visiblePoint.color * INV_PI * glm::dot(temporalReservoir.visiblePoint.normal, glm::normalize(temporalReservoir.lightPoint.position - temporalReservoir.visiblePoint.position)) * glm::dot(temporalReservoir.lightPoint.position - temporalReservoir.visiblePoint.position, temporalReservoir.lightPoint.position - temporalReservoir.visiblePoint.position) * glm::dot(temporalReservoir.lightPoint.normal, glm::normalize(temporalReservoir.visiblePoint.position - temporalReservoir.lightPoint.position));
            // _camera->getImage()->setPixel(x,y,color);
            //check end
        }
    }


}

__device__ void Scene::sample_lights(Vec3f& samplePos, Vec3f& sampleNormal, Vec3f& sampleColor, float& pdf, const Sampler& sampler) const {
    unsigned int idx = sampler.get1U(0, n_emissive_triangles);
    float u = sampler.get1D();
    float v = sampler.get1D();
    if (u + v > 1) u = 1 - u, v = 1 - v;
    //printf("(u, v) = (%f, %f)\n", u, v);
    
    Triangle tri = _emissive_triangles[idx];
    Vec3f v0 = _pos[tri.v_idx.x];
    Vec3f v1 = _pos[tri.v_idx.y];
    Vec3f v2 = _pos[tri.v_idx.z];
    Vec3f n0 = _norms[tri.n_idx.x];
    Vec3f n1 = _norms[tri.n_idx.y];
    Vec3f n2 = _norms[tri.n_idx.z];
    Vec3f c0 = _colors[tri.c_idx.x];
    Vec3f c1 = _colors[tri.c_idx.y];
    Vec3f c2 = _colors[tri.c_idx.z];
    Vec3f v1v0 = v1 - v0;
    Vec3f v2v0 = v2 - v0;
    samplePos = utils::interpolate(u, v, v0, v1, v2);
    sampleNormal = glm::normalize(utils::interpolate(u, v, n0, n1, n2));
    sampleColor = utils::interpolate(u, v, c0, c1, c2);
    float s = glm::length(glm::cross(v2v0, v1v0)) / 2;
    pdf = 1 / s / float(n_emissive_triangles);
}

__device__ void Integrator::spatial_reuse(int x, int y) {

    int idx = x * _resolution.y + y;
    const Scene& scene = *_scene;
    const Sampler& sampler = _samplers[idx];
    Reservoir& curReservoir = _curReservoirs.get()[idx];
    Reservoir& prevrReservoir = _prevReservoirs.get()[idx];
    const Vec3f WORLD_UP = {0.0f,0.0f,1.0f};


    //printf("_curReservoirs[%d].W = %f\t .M=%d\n", idx, _curReservoirs.get()[idx].W, _curReservoirs.get()[idx].M);


    //
    if(curReservoir.M > 0) {
        
        //printf("spatial reuse  curReservoir.M>0\n");

        Reservoir spatialReservoir = curReservoir;
        int lightSampleCnt = spatialReservoir.M;
        for(int i = 0 ; i < NEIGHBOR_COUNT ; ++i) {
            int neighborXIndex = x + (sampler.get1D() * 2.0f - 1.0f) * NEIGHBOR_RANGE;
            int neighborYIndex = y + (sampler.get1D() * 2.0f - 1.0f) * NEIGHBOR_RANGE;
            neighborXIndex = utils::max(0, utils::min(neighborXIndex, _resolution.x - 1));
            neighborYIndex = utils::max(0, utils::min(neighborYIndex, _resolution.y - 1));
            Reservoir neighbor = _curReservoirs[neighborXIndex * _resolution.y + neighborYIndex];
            //judge wether the spatial neighbor is valid
            if(neighbor.M == 0) continue;
            float dist = glm::length(neighbor.lightPoint.position - curReservoir.visiblePoint.position);
            Vec3f visibleToLight = glm::normalize(neighbor.lightPoint.position - curReservoir.visiblePoint.position);
            Ray shadowRay (curReservoir.visiblePoint.position, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
            if(!scene.isShadowed(shadowRay)) {
                float LdotN = utils::max(0.0f, glm::dot(curReservoir.visiblePoint.normal, visibleToLight));
                float p_hat = (LdotN == 0.0f) ? 0.0f : glm::length(curReservoir.visiblePoint.color * neighbor.lightPoint.color * INV_PI * LdotN / glm::dot(neighbor.lightPoint.position - curReservoir.visiblePoint.position, neighbor.lightPoint.position - curReservoir.visiblePoint.position));
                spatialReservoir.update(neighbor.lightPoint, p_hat * neighbor.W * float(neighbor.M), sampler);
            }
            lightSampleCnt += neighbor.M; 
        }
        spatialReservoir.M = lightSampleCnt;
        spatialReservoir.W = spatialReservoir.w_sum / utils::max((spatialReservoir.p_hat()), 0.0001f) / utils::max(float(spatialReservoir.M), 0.0001f);
        prevrReservoir = spatialReservoir;

        if(spatialReservoir.isLight) {
            _output[idx] = spatialReservoir.visiblePoint.color;
            return;
        }
        //end spatial reuse
        //if the intersect point is not in the light, start global illumiantion
        Vec3f plusTerm = spatialReservoir.get_color();
        Vec3f multipleTerm = spatialReservoir.get_multipleTerm();
        Vec3f thisPosition = spatialReservoir.visiblePoint.position;
        Vec3f thisNormal = spatialReservoir.visiblePoint.normal;
        //a random number in [0,1]
        float randNum = sampler.get1D();

        while(randNum < PROB) {
            
            randNum = sampler.get1D();
            float X1 = sampler.get1D();
            float X2 = sampler.get1D();
            //local space
            Vec3f toNext = {sqrtf(X1)*cosf(2.0f * PI * X2), sqrtf(X2)*sinf(2.0f * PI * X2), sqrtf(1.0f - X1)};
            multipleTerm /= glm::dot(toNext, WORLD_UP) / PI / PROB;
            //change to world space
            if(thisNormal != WORLD_UP) {
                Vec3f newXAxis = glm::normalize(glm::cross(WORLD_UP, thisNormal));
                Vec3f newYAxis = glm::normalize(glm::cross(thisNormal, newXAxis));
                toNext = newXAxis * toNext.x;
                
                //toNext = newXAxis * toNext.x + newYAxis * toNext.y + thisNormal * toNext.z;
                //assert(abs(1.0f - glm::length(toNext)) < 0.001f);
            }
            Ray nextRay (thisPosition, toNext, RAY_DEFAULT_MIN, RAY_DEFAULT_MAX);
            Interaction nextInteraction;
            scene.intersect(nextRay, nextInteraction);
            if(nextInteraction.type == Interaction::GEOMETRY) {
                //calculate direct light by RIS
                float pdf;
                Vec3f lightPointPosition;
                Vec3f lightPointNormal;
                Vec3f lightPointColor;
                Vec3f visiblePointPosition;
                Vec3f visiblePointNormal;
                Vec3f visiblePointColor;
                Vec3f visibleToLight;
                Reservoir tmpReservoir;
                Point newLightPoint;
                float LdotN;
                float p_hat;

                float u = nextInteraction.uv.x;
                float v = nextInteraction.uv.y;
                Triangle tri = scene._triangles[nextInteraction.triangle_idx];
                Vec3f v0 = scene._pos[tri.v_idx.x];
                Vec3f v1 = scene._pos[tri.v_idx.y];
                Vec3f v2 = scene._pos[tri.v_idx.z];
                Vec3f n0 = scene._norms[tri.n_idx.x];
                Vec3f n1 = scene._norms[tri.n_idx.y];
                Vec3f n2 = scene._norms[tri.n_idx.z];
                Vec3f c0 = scene._colors[tri.c_idx.z];
                Vec3f c1 = scene._colors[tri.c_idx.z];
                Vec3f c2 = scene._colors[tri.c_idx.z];

                tmpReservoir.visiblePoint.position = visiblePointPosition = thisPosition = utils::interpolate(u, v, v0, v1, v2);
                tmpReservoir.visiblePoint.normal   = visiblePointNormal = thisNormal = utils::interpolate(u, v, n0, n1, n2);
                tmpReservoir.visiblePoint.color    = visiblePointColor = utils::interpolate(u, v, c0, c1, c2);
                for(int j = 0 ; j < LIGHTCOUNT ; ++j) {
                    scene.sample_lights(lightPointPosition, lightPointNormal, lightPointColor, pdf, sampler);
                float dist = glm::length(visiblePointPosition - lightPointPosition);
                visibleToLight = glm::normalize(lightPointPosition - visiblePointPosition);
                Ray shadowRay (visiblePointPosition, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
                if(!scene.isShadowed(shadowRay)) {
                    newLightPoint = Point{lightPointPosition, lightPointNormal, lightPointColor};
                    LdotN = utils::max(glm::dot(visiblePointNormal, visibleToLight), 0.0f);
                    p_hat = (LdotN == 0.0f) ? 0.0f : glm::length(visiblePointColor * lightPointColor * INV_PI * LdotN / dist / dist);
                    tmpReservoir.update(newLightPoint, p_hat / pdf, sampler);
                } else {
                    tmpReservoir.M++;
                }
                }
                tmpReservoir.W = tmpReservoir.w_sum / utils::max((tmpReservoir.p_hat()), 0.0001f) / utils::max(float(tmpReservoir.M),0.0001f);
                plusTerm += tmpReservoir.get_color() * multipleTerm;
                multipleTerm *= tmpReservoir.get_multipleTerm();
            } else {
                break;
            }
        }
        //end global illumination
        _output[idx] = plusTerm;
    }

    // _output[idx] = Vec3f(0,1,1);

}


void Integrator::render_cuda() {

    if (isFirstFrame) {
        this_device = thrust::device_malloc<Integrator>(1);
        *this_device = *this;
    }

    dim3 block(8, 8, 1);   
    dim3 grid(_resolution.x / block.x, _resolution.y / block.y, 1);

    if (isFirstFrame)
        ::set_seed <<< grid, block >>> (this_device, SEED);

    ::render_kernel <<< grid, block >>> (this_device);  

    if (isFirstFrame) {
        isFirstFrame = false;
        ::set_notFirstFrame <<< 1, 1 >>> (this_device);
    }

    ::spatial_reuse <<< grid, block >>> (this_device);  
    
    _camera_host->getImage()->fromGPU(_output.get());
}

__global__ void set_seed(thrust::device_ptr<Integrator> i, unsigned seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= i.get()->_resolution.x || y >= i.get()->_resolution.y) return;
    int idx = x * i.get()->_resolution.y + y;
    Sampler(i.get()->_samplers[idx]).setSeed(seed, idx);
}

__global__ void render_kernel(thrust::device_ptr<Integrator> i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= i.get()->_resolution.x || y >= i.get()->_resolution.y) return;
    i.get()->render_kernel(x, y);
}

__global__ void spatial_reuse(thrust::device_ptr<Integrator> i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= i.get()->_resolution.x || y >= i.get()->_resolution.y) return;
    i.get()->spatial_reuse(x, y);

}

__global__ void set_notFirstFrame(thrust::device_ptr<Integrator> i) {
    i.get()->isFirstFrame = false;
}


__device__ void BVH::bvhHit(Interaction& interaction, const Ray& ray, int node_idx = 0) const {

    const Scene& scene = *_scene;
    //return;
    // printf("BVH::bvhhit triangles %p\n", scene._colors);
    // Vec3f color = scene._colors[1];
    // printf("color %f %f %f\n", color.x, color.y, color.z);
    // Bvh_Node b = _bvh_nodes[0];
    // printf("bvh_nodes.left %d\n", b.left);
    // printf("bvh_nodes.right %d\n", b.right);
    // printf("bvh_nodes.aabb.min %f %f %f\n", b.aabb._lower_bnd.x, b.aabb._lower_bnd.y, b.aabb._lower_bnd.z);
    // printf("bvh_nodes.aabb.max %f %f %f\n", b.aabb._upper_bnd.x, b.aabb._upper_bnd.y, b.aabb._upper_bnd.z);
    // printf("BVH::bvhHit %d\n", _bvh_nodes[node_idx].left.type);
    // bool flag = 
    if(_bvh_nodes[node_idx].left.type != INTERAL_NODE) {
        for (int i = _bvh_nodes[node_idx].left.start ; i != _bvh_nodes[node_idx].right.end ; ++i) {


            //intersect with a triangle
            Triangle tri = scene._triangles[i];
            Vec3f* _pos = scene._pos;
            Vec3f v0 = _pos[tri.v_idx.x];
            Vec3f v1 = _pos[tri.v_idx.y];
            Vec3f v2 = _pos[tri.v_idx.z];
            //printf("t[%d].v0  = p[%d] = {%f, %f, %f} \n", i, tri.v_idx.x, v0.x, v0.y, v0.z);
            //printf("t[%d].v1  = p[%d] = {%f, %f, %f} \n", i, tri.v_idx.y, v1.x, v1.y, v1.z);
            //printf("t[%d].v2  = p[%d] = {%f, %f, %f} \n",i, tri.v_idx.z,  v2.x, v2.y, v2.z);

            Vec3f v0v1 = v1 - v0;
            Vec3f v0v2 = v2 - v0;
            Vec3f pvec = glm::cross(ray._dir, v0v2);
            float det  = glm::dot(v0v1, pvec);
            float invDet = 1.0f / det;
            Vec3f tvec = ray._origin - v0;
            float u = glm::dot(tvec, pvec) * invDet;
            // printf("\n");

            if(u < 0 || u > 1) continue;

            Vec3f qvec = glm::cross(tvec, v0v1);
            float v = glm::dot(ray._dir, qvec) * invDet;
            if(v < 0 || u + v > 1) continue;


            float t = glm::dot(v0v2, qvec) * invDet;
            if(t < ray._t_min || t > ray._t_max || t > interaction.dist) continue;

            //printf("(u, v, t) = (%f\t%f\t%f)\n", u, v, t);

            interaction.dist = t;
            interaction.type = tri.emissive ? Interaction::Type::LIGHT : Interaction::Type::GEOMETRY;
            interaction.triangle_idx = i;
            interaction.uv.x = u;
            interaction.uv.y = v;
        }
    } else {
        float t_in_l, t_out_l, t_in_r, t_out_r;
        bool hit_left = _bvh_nodes[node_idx+1].aabb.intersect(ray, t_in_l, t_out_l);
        bool hit_right = _bvh_nodes[_bvh_nodes[node_idx].right.right_node_idx].aabb.intersect(ray, t_in_r, t_out_r);
        //printf("%d %d", hit_left, hit_right);
        if(hit_left && hit_right) {
            //printf("hit both t_in_l=%f t_in_r v=%f \n", t_in_l, t_in_r);
            if(t_in_l < t_in_r) {
                bvhHit(interaction, ray, node_idx+1);
                if(interaction.type != Interaction::Type::NONE && interaction.dist < t_in_r) return;
                bvhHit(interaction, ray, _bvh_nodes[node_idx].right.right_node_idx);
            } else {
                bvhHit(interaction, ray, _bvh_nodes[node_idx].right.right_node_idx);
                if(interaction.type != Interaction::Type::NONE && interaction.dist < t_in_l) return;
                bvhHit(interaction, ray, node_idx+1);
            }
        } else if(hit_left) {
            //printf("hit left\n");
            bvhHit(interaction, ray, node_idx+1);
        } else if(hit_right) {
            //printf("hit_right\n");
            bvhHit(interaction, ray, _bvh_nodes[node_idx].right.right_node_idx);
        }
    }

}


/// @brief 
/// @param ray 
/// @param node_idx 
/// @return false represented shadowed
__device__ bool BVH::visiality_test(const Ray& ray, int node_idx = 0) const {
    const Scene& scene = *_scene;

    if(_bvh_nodes[node_idx].left.type != INTERAL_NODE) {
        for(int i = _bvh_nodes[node_idx].left.start ; i != _bvh_nodes[node_idx].right.end ; ++i) {
            Triangle tri = scene._triangles[i];
            Vec3f v0 = scene._pos[tri.v_idx.x];
            Vec3f v1 = scene._pos[tri.v_idx.y];
            Vec3f v2 = scene._pos[tri.v_idx.z];
            Vec3f v0v1 = v1 - v0;
            Vec3f v0v2 = v2 - v0;
            Vec3f pvec = glm::cross(ray._dir, v0v2);
            float det  = glm::dot(v0v1, pvec);
            float invDet = 1.0f / det;
            Vec3f tvec = ray._origin - v0;
            float u = glm::dot(tvec, pvec) * invDet;
            if(u < 0 || u > 1) continue;
            Vec3f qvec = glm::cross(tvec, v0v1);
            float v = glm::dot(ray._dir, qvec) * invDet;
            if(v < 0 || u + v > 1) continue;
            float t = glm::dot(v0v2, qvec) * invDet;
            if(t > ray._t_min && t < ray._t_max) return false;
        }
        return true;
    } else {
        float t_in_l, t_out_l, t_in_r, t_out_r;
        bool hit_left_aabb = _bvh_nodes[node_idx+1].aabb.intersect(ray, t_in_l, t_out_l);
        bool hit_right_aabb = _bvh_nodes[_bvh_nodes[node_idx].right.right_node_idx].aabb.intersect(ray, t_in_r, t_out_r);
        if(t_in_l < ray._t_max && hit_left_aabb)
            if(!visiality_test(ray, node_idx+1))
                return false;
        if(t_in_r < ray._t_max && hit_right_aabb)
            if(!visiality_test(ray, _bvh_nodes[node_idx].right.right_node_idx))
                return false;
        return true;
    }
}


