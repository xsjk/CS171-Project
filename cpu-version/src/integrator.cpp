#include "integrator.h"
#include "core.h"
#include <iostream>
#include <omp.h>


#define interpolate(u, v, v0, v1, v2) ((1.0f - u - v) * v0 + u * v1 + v * v2)
#define p_hat(r) glm::length(r.visiblePoint.color * r.lightPoint.color * max(0.0f, glm::dot(r.visiblePoint.normal, glm::normalize(r.lightPoint.position - r.visiblePoint.position))) / glm::dot(r.lightPoint.position - r.visiblePoint.position, r.lightPoint.position - r.visiblePoint.position) * INV_PI)
#define get_color(r) (r.W * r.lightPoint.color * r.visiblePoint.color * INV_PI * glm::dot(r.visiblePoint.normal, glm::normalize(r.lightPoint.position - r.visiblePoint.position)) * glm::dot(r.lightPoint.position - r.visiblePoint.position, r.lightPoint.position - r.visiblePoint.position) * glm::dot(r.lightPoint.normal, glm::normalize(r.visiblePoint.position - r.lightPoint.position))*0.7f)
#define get_mulipleTerm(r) (r.W * r.visiblePoint.color * INV_PI * glm::dot(r.visiblePoint.normal, glm::normalize(r.lightPoint.position - r.visiblePoint.position)) * glm::dot(r.lightPoint.position - r.visiblePoint.position, r.lightPoint.position - r.visiblePoint.position) * glm::dot(r.lightPoint.normal, glm::normalize(r.visiblePoint.position - r.lightPoint.position))*0.7f)

Integrator::Integrator(std::shared_ptr<Camera> camera, std::shared_ptr<Scene> scene) : isFirstFrame(true) {
  _camera = camera;
  _scene = scene;
  Vec2i resolution = _camera->getImage()->getResolution();
  _prevReservoirs.resize(resolution.x * resolution.y);
}

void Integrator::render() {
  Vec2i resolution = _camera->getImage()->getResolution();
  int cnt = 0;
  std::vector<Reservoir> currentReservoirs(resolution.x * resolution.y);
#pragma omp parallel for schedule(dynamic), shared(cnt)
  for(int x = 0 ; x < resolution.x ; ++x) {
#pragma omp atomic
    ++cnt;
    printf("\r%.02f%%", cnt * 50.0 / resolution.x);
    for(int y = 0 ; y < resolution.y ; ++y) {
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
      Point newLightPoint;

      //start produce visible point
      Ray rayToVisiblePoint = _camera->generateRay(float(x), float(y));
      Interaction interaction;
      _scene->intersect(rayToVisiblePoint, interaction);
      //if the ray doesn't hit any geometry or light in the scene, then M = 0;
      //if the ray hits any geometry, including light
      if(interaction.type != Interaction::NONE) {
        visiblePointPosition = interpolate(interaction.uv.x, interaction.uv.y, (*_scene->_pos)[(*_scene->_triangles)[interaction.triangle_idx].v_idx.x], (*_scene->_pos)[(*_scene->_triangles)[interaction.triangle_idx].v_idx.y], (*_scene->_pos)[(*_scene->_triangles)[interaction.triangle_idx].v_idx.z]);
        visiblePointNormal = glm::normalize(interpolate(interaction.uv.x, interaction.uv.y, (*_scene->_norms)[(*_scene->_triangles)[interaction.triangle_idx].n_idx.x], (*_scene->_norms)[(*_scene->_triangles)[interaction.triangle_idx].n_idx.y], (*_scene->_norms)[(*_scene->_triangles)[interaction.triangle_idx].n_idx.z]));
        visiblePointColor = interpolate(interaction.uv.x, interaction.uv.y, (*_scene->_colors)[(*_scene->_triangles)[interaction.triangle_idx].c_idx.x], (*_scene->_colors)[(*_scene->_triangles)[interaction.triangle_idx].c_idx.y], (*_scene->_colors)[(*_scene->_triangles)[interaction.triangle_idx].c_idx.z]);
        newReservoir.visiblePoint = {.position = visiblePointPosition, .normal = visiblePointNormal, .color = visiblePointColor};
        newReservoir.isLight = (interaction.type == Interaction::LIGHT); 
      //end produce visible point

      //start generate light candidates by resample importance sampling(RIS) according to algorithm 3
        for(int j = 0 ; j < LIGHT_CANDIDATE_NUM ; ++j) {
          _scene->get_lights().sample(lightPointPosition, lightPointNormal, lightPointColor, pdf);
          float dist = glm::length(visiblePointPosition - lightPointPosition);
          visibleToLight = glm::normalize(lightPointPosition - visiblePointPosition);
          Ray shadowRay (visiblePointPosition, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
          if(!_scene->isShadowed(shadowRay)) {
            newLightPoint = Point{.position = lightPointPosition, .normal = lightPointNormal, .color = lightPointColor};
            LdotN = max(glm::dot(visiblePointNormal, visibleToLight), 0.0f);
            p_hat = (LdotN == 0.0f) ? 0.0f : glm::length(visiblePointColor * lightPointColor * INV_PI * LdotN / dist / dist);
            newReservoir.update(newLightPoint, p_hat / pdf, *_scene->get_sampler());
          } else {
            newReservoir.M++;
          }
        }
        newReservoir.W = newReservoir.w_sum / max(p_hat(newReservoir), 0.0001f) / max(float(newReservoir.M),0.0001f);
        currentReservoirs[x * resolution.y + y] = newReservoir;
        // Vec3f color = get_color(newReservoir);
        // _camera->getImage()->setPixel(x,y,color);
        // if(isFirstFrame) _prevReservoirs[x * resolution.y + y] = newReservoir;
      //end RIS
      }
      if(!isFirstFrame) {
        //ignore the situation where visible point is not valid or previous reservoir is invalid
        if(currentReservoirs[x * resolution.y + y].M > 0 && _prevReservoirs[x * resolution.y + y].M > 0) {
          //start temporal reuse
          Reservoir temporalReservoir = currentReservoirs[x * resolution.y + y];
          visibleToLight = glm::normalize(_prevReservoirs[x * resolution.y + y].lightPoint.position - temporalReservoir.visiblePoint.position);
          float dist = glm::length(_prevReservoirs[x * resolution.y + y].lightPoint.position - temporalReservoir.visiblePoint.position);
          Ray shadowRay (temporalReservoir.visiblePoint.position, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
          LdotN = max(0.0f, glm::dot(temporalReservoir.visiblePoint.normal, visibleToLight));
          p_hat = (_scene->isShadowed(shadowRay) || LdotN == 0.0f) ? 0.0f : glm::length(temporalReservoir.visiblePoint.color * _prevReservoirs[x * resolution.y + y].lightPoint.color * INV_PI * LdotN / dist / dist);
          _prevReservoirs[x * resolution.y + y].M = int(min(20.f * float(currentReservoirs[x * resolution.y + y].M), float(_prevReservoirs[x * resolution.y + y].M)));
          temporalReservoir.update(_prevReservoirs[x * resolution.y + y].lightPoint, p_hat * _prevReservoirs[x * resolution.y + y].W * float(_prevReservoirs[x * resolution.y + y].M), *_scene->get_sampler());
          //set M value
          temporalReservoir.M = currentReservoirs[x * resolution.y + y].M + _prevReservoirs[x * resolution.y + y].M;
          //set W value
          temporalReservoir.W = temporalReservoir.w_sum / max(p_hat, 0.0001f) / max(float(temporalReservoir.M), 0.0001f);
          currentReservoirs[x * resolution.y + y] = temporalReservoir;
          //end temporal reuse

          //check the correctness of temporal reuse
          // _prevReservoirs[x * resolution.y + y] = temporalReservoir;
          // Vec3f color = temporalReservoir.W * temporalReservoir.lightPoint.color * temporalReservoir.visiblePoint.color * INV_PI * glm::dot(temporalReservoir.visiblePoint.normal, glm::normalize(temporalReservoir.lightPoint.position - temporalReservoir.visiblePoint.position)) * glm::dot(temporalReservoir.lightPoint.position - temporalReservoir.visiblePoint.position, temporalReservoir.lightPoint.position - temporalReservoir.visiblePoint.position) * glm::dot(temporalReservoir.lightPoint.normal, glm::normalize(temporalReservoir.visiblePoint.position - temporalReservoir.lightPoint.position));
          // _camera->getImage()->setPixel(x,y,color);
          //check end
        }
      }
    }
  }
  isFirstFrame = false;
  //start spatial reuse


  cnt = 0;
#pragma omp parallel for schedule(dynamic), shared(cnt)
  for(int x = 0 ; x < resolution.x ; ++x) {
#pragma omp atomic
    ++cnt;
    printf("\r%.02f%%", cnt * 50.0 / resolution.x + 50);
    for(int y = 0 ; y < resolution.y ; ++y) {
      //ignore invalid visible point
      if(currentReservoirs[x * resolution.y + y].M > 0) {
        Reservoir spatialReservoir = currentReservoirs[x * resolution.y + y];
        int lightSampleCnt = spatialReservoir.M;
        for(int i = 0 ; i < NEIGHBOR_COUNT ; ++i) {
          int neighborXIndex = x + (_scene->get_sampler()->get1D() * 2.0f - 1.0f) * NEIGHBOR_RANGE;
          int neighborYIndex = y + (_scene->get_sampler()->get1D() * 2.0f - 1.0f) * NEIGHBOR_RANGE;
          neighborXIndex = max(0, min(neighborXIndex, resolution.x - 1));
          neighborYIndex = max(0, min(neighborYIndex, resolution.y - 1));
          Reservoir neighbor = currentReservoirs[neighborXIndex * resolution.y + neighborYIndex];
          //judge wether the spatial neighbor is valid
          if(neighbor.M == 0) continue;
          float dist = glm::length(neighbor.lightPoint.position - currentReservoirs[x * resolution.y + y].visiblePoint.position);
          Vec3f visibleToLight = glm::normalize(neighbor.lightPoint.position - currentReservoirs[x * resolution.y + y].visiblePoint.position);
          Ray shadowRay (currentReservoirs[x * resolution.y + y].visiblePoint.position, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
          if(!_scene->isShadowed(shadowRay)) {
            float LdotN = max(0.0f, glm::dot(currentReservoirs[x * resolution.y + y].visiblePoint.normal, visibleToLight));
            float p_hat = (LdotN == 0.0f) ? 0.0f : glm::length(currentReservoirs[x * resolution.y + y].visiblePoint.color * neighbor.lightPoint.color * INV_PI * LdotN / glm::dot(neighbor.lightPoint.position - currentReservoirs[x * resolution.y + y].visiblePoint.position, neighbor.lightPoint.position - currentReservoirs[x * resolution.y + y].visiblePoint.position));
            spatialReservoir.update(neighbor.lightPoint, p_hat * neighbor.W * float(neighbor.M), *_scene->get_sampler());
          }
          lightSampleCnt += neighbor.M; 
        }
        spatialReservoir.M = lightSampleCnt;
        spatialReservoir.W = spatialReservoir.w_sum / max(p_hat(spatialReservoir), 0.0001f) / max(float(spatialReservoir.M), 0.0001f);
        _prevReservoirs[x * resolution.y + y] = spatialReservoir;
        // Vec3f color =(!spatialReservoir.isLight) ? get_color(spatialReservoir) : spatialReservoir.visiblePoint.color;
        // _camera->getImage()->setPixel(x, y, color);
        if(spatialReservoir.isLight) {
          _camera->getImage()->setPixel(x, y, spatialReservoir.visiblePoint.color);
          continue;
        }
        //end spatial reuse
        //if the intersect point is not in the light, start global illumiantion
        Vec3f plusTerm = get_color(spatialReservoir);
        Vec3f multipleTerm = get_mulipleTerm(spatialReservoir);
        Vec3f thisPosition = spatialReservoir.visiblePoint.position;
        Vec3f thisNormal = spatialReservoir.visiblePoint.normal;
        //a random number in [0,1]
        float randNum = _scene->get_sampler()->get1D();
        while(randNum < PROB) {
          randNum = _scene->get_sampler()->get1D();
          float X1 = _scene->get_sampler()->get1D();
          float X2 = _scene->get_sampler()->get1D();
          //local space
          Vec3f toNext = {sqrtf(X1)*cosf(2.0f * PI * X2), sqrtf(X2)*sinf(2.0f * PI * X2), sqrtf(1.0f - X1)};
          multipleTerm /= glm::dot(toNext, WORLD_UP) / PI / PROB;
          //change to world space
          if(thisNormal != WORLD_UP) {
            Vec3f newXAxis = glm::normalize(glm::cross(WORLD_UP, thisNormal));
            //newZAxis = thisNormal
            Vec3f newYAxis = glm::normalize(glm::cross(thisNormal, newXAxis));
            toNext = newXAxis * toNext.x + newYAxis * toNext.y + thisNormal * toNext.z;
            assert(abs(1.0f - glm::length(toNext)) < 0.001f);
          }
          Ray nextRay (thisPosition, toNext, RAY_DEFAULT_MIN, RAY_DEFAULT_MAX);
          Interaction nextInteraction;
          _scene->intersect(nextRay, nextInteraction);
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
            tmpReservoir.visiblePoint.position = visiblePointPosition = thisPosition = interpolate(nextInteraction.uv.x, nextInteraction.uv.y, (*_scene->_pos)[(*_scene->_triangles)[nextInteraction.triangle_idx].v_idx.x], (*_scene->_pos)[(*_scene->_triangles)[nextInteraction.triangle_idx].v_idx.y], (*_scene->_pos)[(*_scene->_triangles)[nextInteraction.triangle_idx].v_idx.z]);
            tmpReservoir.visiblePoint.normal   = visiblePointNormal = thisNormal = interpolate(nextInteraction.uv.x, nextInteraction.uv.y, (*_scene->_norms)[(*_scene->_triangles)[nextInteraction.triangle_idx].n_idx.x], (*_scene->_norms)[(*_scene->_triangles)[nextInteraction.triangle_idx].n_idx.y], (*_scene->_norms)[(*_scene->_triangles)[nextInteraction.triangle_idx].n_idx.z]);
            tmpReservoir.visiblePoint.color    = visiblePointColor = interpolate(nextInteraction.uv.x, nextInteraction.uv.y, (*_scene->_colors)[(*_scene->_triangles)[nextInteraction.triangle_idx].c_idx.x], (*_scene->_pos)[(*_scene->_triangles)[nextInteraction.triangle_idx].c_idx.y], (*_scene->_pos)[(*_scene->_triangles)[nextInteraction.triangle_idx].c_idx.z]);
            for(int j = 0 ; j < LIGHTCOUNT ; ++j) {
              _scene->get_lights().sample(lightPointPosition, lightPointNormal, lightPointColor, pdf);
              float dist = glm::length(visiblePointPosition - lightPointPosition);
              visibleToLight = glm::normalize(lightPointPosition - visiblePointPosition);
              Ray shadowRay (visiblePointPosition, visibleToLight, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
              if(!_scene->isShadowed(shadowRay)) {
                newLightPoint = Point{.position = lightPointPosition, .normal = lightPointNormal, .color = lightPointColor};
                LdotN = max(glm::dot(visiblePointNormal, visibleToLight), 0.0f);
                p_hat = (LdotN == 0.0f) ? 0.0f : glm::length(visiblePointColor * lightPointColor * INV_PI * LdotN / dist / dist);
                tmpReservoir.update(newLightPoint, p_hat / pdf, *_scene->get_sampler());
              } else {
                tmpReservoir.M++;
              }
            }
            tmpReservoir.W = tmpReservoir.w_sum / max(p_hat(tmpReservoir), 0.0001f) / max(float(tmpReservoir.M),0.0001f);
            plusTerm += get_color(tmpReservoir) * multipleTerm;
            multipleTerm *= get_mulipleTerm(tmpReservoir);
          } else {
            break;
          }
        }
        //end global illumination
        _camera->getImage()->setPixel(x, y, plusTerm);
      }
    }
  }
}