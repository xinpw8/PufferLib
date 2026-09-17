#pragma once
#include "../../../vendor/cJSON.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rek5_modes {
struct Identity { std::string opponent,observation,geometry; int contact_substeps; };
inline std::string select(const cJSON* fast,const char* field,const char* fallback,const char* alternative){
    const cJSON* value=fast?cJSON_GetObjectItemCaseSensitive(fast,field):nullptr;
    if(!value)return fallback;
    if(!cJSON_IsString(value)||!value->valuestring)throw std::runtime_error(std::string("Invalid ")+field+" type");
    std::string result=value->valuestring;
    if(result!=fallback&&result!=alternative)throw std::runtime_error(std::string("Invalid ")+field);
    return result;
}
inline Identity configure(const cJSON* fast){
    if(fast&&!cJSON_IsObject(fast))throw std::runtime_error("fast must be an object");
    Identity out{select(fast,"opponent_controller","v4_scripted","recovered_bot1_v1"),
        select(fast,"observation_mode","v4_logical","rendered_pose_v1"),
        select(fast,"geometry_mode","bounding_spheres","primitive_samples_v1"),4};
    const cJSON* samples=fast?cJSON_GetObjectItemCaseSensitive(fast,"contact_substeps"):nullptr;
    if(samples){
        if(!cJSON_IsNumber(samples)||!std::isfinite(samples->valuedouble)||
            samples->valuedouble<1||samples->valuedouble>16||samples->valuedouble!=std::floor(samples->valuedouble))
            throw std::runtime_error("contact_substeps must be an integer in [1,16]");
        out.contact_substeps=int(samples->valuedouble);
    }
    // Explicit config, including omitted legacy fields, overrides ambient flags.
    setenv("REK_FAST_OPPONENT",out.opponent.c_str(),1);
    setenv("REK_FAST_OBSERVATION",out.observation.c_str(),1);
    setenv("REK_FAST_GEOMETRY",out.geometry.c_str(),1);
    setenv("REK_FAST_CONTACT_SUBSTEPS",std::to_string(out.contact_substeps).c_str(),1);
    if(out.geometry=="bounding_spheres")out.contact_substeps=0;
    std::printf("{\"event\":\"compact_mode_identity\",\"opponent_controller\":\"%s\",\"observation_mode\":\"%s\",\"geometry_mode\":\"%s\",\"contact_substeps\":%d,\"source\":\"runtime_config\"}\n",out.opponent.c_str(),out.observation.c_str(),out.geometry.c_str(),out.contact_substeps);
    return out;
}
inline std::string json_fields(const Identity& identity,const std::string& scoring){
    // Only call after the caller has validated scoring and configure has
    // validated the fixed-enumeration mode names and substep count.
    return ",\"opponent_controller\":\""+identity.opponent+"\",\"observation_mode\":\""+
        identity.observation+"\",\"scoring_mode\":\""+scoring+"\",\"geometry_mode\":\""+
        identity.geometry+"\",\"contact_substeps\":"+std::to_string(identity.contact_substeps);
}
}
