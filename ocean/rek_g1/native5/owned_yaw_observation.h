#ifndef REK_NATIVE5_OWNED_YAW_OBSERVATION_H
#define REK_NATIVE5_OWNED_YAW_OBSERVATION_H
#include <cstring>
#include <stdexcept>

// Versioned owned-command intent, never an angular velocity or playback claim.
// Column 187 is identically zero in the legacy 223-feature interface.
namespace rek_owned_yaw {
constexpr int kColumn=187;
constexpr const char* kLegacySchema="rek.native5.scaled_polar_xy.v1";
constexpr const char* kSchema="rek.native5.scaled_polar_xy.owned_yaw_v2";
#ifdef __CUDACC__
#define REK_OWNED_YAW_HD __host__ __device__
#else
#define REK_OWNED_YAW_HD
#endif
REK_OWNED_YAW_HD inline bool valid_desired(int category){return category>=1&&category<=15;}
REK_OWNED_YAW_HD inline float desired_yaw(int category){
    switch(category){
        case 6:case 8:case 10:case 12:case 14:return 1.f;
        case 7:case 9:case 11:case 13:case 15:return -1.f;
        default:return 0.f;
    }
}
REK_OWNED_YAW_HD inline float pending_value(bool enabled,bool busy,int desired){
    return enabled&&busy?desired_yaw(desired):0.f;
}
#undef REK_OWNED_YAW_HD
inline bool enabled(const char* schema){
    if(!schema||!std::strcmp(schema,kLegacySchema))return false;
    if(!std::strcmp(schema,kSchema))return true;
    throw std::invalid_argument("unsupported_owned_yaw_observation_schema");
}
inline const char* schema(bool value){return value?kSchema:kLegacySchema;}
inline bool training_compatible(bool owned,const char* backend,const char* frozen){
    return !owned||(backend&&!std::strcmp(backend,"semantic_cuda")&&
        (!frozen||!frozen[0]||!std::strcmp(frozen,"None")));
}
}
#endif
