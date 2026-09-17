#pragma once
#include "../../../vendor/cJSON.h"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace rek5_modes {
struct Identity { std::string opponent,observation; };
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
        select(fast,"observation_mode","v4_logical","rendered_pose_v1")};
    // Explicit config, including omitted legacy fields, overrides ambient flags.
    setenv("REK_FAST_OPPONENT",out.opponent.c_str(),1);
    setenv("REK_FAST_OBSERVATION",out.observation.c_str(),1);
    std::printf("{\"event\":\"compact_mode_identity\",\"opponent_controller\":\"%s\",\"observation_mode\":\"%s\",\"source\":\"runtime_config\"}\n",out.opponent.c_str(),out.observation.c_str());
    return out;
}
inline std::string json_fields(const Identity& identity,const std::string& scoring){
    // Only call after the caller has validated scoring and configure has
    // validated the two fixed-enumeration mode names.
    return ",\"opponent_controller\":\""+identity.opponent+"\",\"observation_mode\":\""+
        identity.observation+"\",\"scoring_mode\":\""+scoring+"\"";
}
}
