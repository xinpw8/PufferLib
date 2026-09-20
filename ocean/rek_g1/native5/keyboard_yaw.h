#ifndef REK_NATIVE5_KEYBOARD_YAW_H
#define REK_NATIVE5_KEYBOARD_YAW_H
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rek_keyboard_yaw {
enum class Mode { LegacyVelocitySlew=0, KeyboardReset=1 };
// Recovered schedule contract's expected command parameter, not a measured
// physical actuator time constant or a claim about a current live config.
constexpr float kRampSeconds=.5f;
inline Mode parse(const char* value){
    if(!value||!std::strcmp(value,"legacy_velocity_slew_v1"))return Mode::LegacyVelocitySlew;
    if(!std::strcmp(value,"keyboard_reset_v1"))return Mode::KeyboardReset;
    throw std::invalid_argument("REK_FAST_YAW_COMMAND_must_be_legacy_velocity_slew_v1_or_keyboard_reset_v1");
}
inline const char* name(Mode mode){
    return mode==Mode::KeyboardReset?"keyboard_reset_v1":"legacy_velocity_slew_v1";
}
struct State {float ramp,sign;};
#ifdef __CUDACC__
#define REK_KEYBOARD_HD __host__ __device__
#else
#define REK_KEYBOARD_HD
#endif
// G1HeldInputScheduleContract.AdvanceKeyboardYaw, for finite inputs and dt>=0.
// This is command state, not a model of actuator angular momentum. The caller
// supplies rawYaw=0 while busy. Positive ramp ignores raw magnitude, as in C#.
REK_KEYBOARD_HD inline float advance(State& state,float rawYaw,float dt,
                                     float rampSeconds,float yawSpeed){
    if(rampSeconds<=0.f||rawYaw==0.f){state={0.f,0.f};return rawYaw*yawSpeed;}
    const float sign=rawYaw>0.f?1.f:-1.f;
    float ramp=state.sign==sign?state.ramp:0.f;
    ramp=fminf(1.f,ramp+dt/rampSeconds);
    state={ramp,sign};
    return ramp*sign*yawSpeed;
}
#undef REK_KEYBOARD_HD
}
#endif
