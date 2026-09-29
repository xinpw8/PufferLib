#pragma once

#include "../../../vendor/cJSON.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rek_eval {
struct FrameRequest {
    std::array<float,72> qpos{};
    bool has_tick=false,has_generation=false;
    double tick=0,generation=0;
    int follow_side=-1;
};

inline FrameRequest frame_request(const cJSON* command) {
    const auto* op=cJSON_GetObjectItemCaseSensitive(command,"op");
    if(!cJSON_IsString(op)||std::string(op->valuestring)!="frame")
        throw std::runtime_error("Renderer accepts frame requests only");
    const auto* q=cJSON_GetObjectItemCaseSensitive(command,"qpos");
    if(!cJSON_IsArray(q)||cJSON_GetArraySize(q)!=72)
        throw std::runtime_error("Renderer requires exactly72 qpos values");
    FrameRequest request;
    for(int i=0;i<72;i++) {
        const auto* item=cJSON_GetArrayItem(q,i);
        if(!cJSON_IsNumber(item)||!std::isfinite(item->valuedouble))
            throw std::runtime_error("Renderer qpos must be finite numbers");
        const float value=static_cast<float>(item->valuedouble);
        if(!std::isfinite(value))throw std::runtime_error("Renderer qpos exceeds float32 range");
        request.qpos[i]=value;
    }
    auto identity=[&](const char* name,bool& present,double& value){
        const auto* item=cJSON_GetObjectItemCaseSensitive(command,name);
        if(!item)return;
        if(!cJSON_IsNumber(item)||!std::isfinite(item->valuedouble)||item->valuedouble<0
                ||item->valuedouble>9007199254740991.0||std::floor(item->valuedouble)!=item->valuedouble)
            throw std::runtime_error(std::string("Invalid renderer ")+name);
        present=true;value=item->valuedouble;
    };
    identity("snapshotTick",request.has_tick,request.tick);
    identity("generation",request.has_generation,request.generation);
    const auto* side=cJSON_GetObjectItemCaseSensitive(command,"followSide");
    if(side) {
        if(!cJSON_IsNumber(side)||(side->valuedouble!=0&&side->valuedouble!=1))
            throw std::runtime_error("Invalid renderer followSide");
        request.follow_side=int(side->valuedouble);
    }
    return request;
}
}
