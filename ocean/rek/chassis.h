// The two robots REK fields, as of v0.0.119.
//
//   L100 — formerly the Unitree G1
//   H100 — formerly the EngineAI T-800
//
// Geometry and mass below are measured off the manufacturers' own URDFs, not
// from spec-sheet marketing numbers:
//
//   L100: unitreerobotics/unitree_ros, robots/g1_description/g1_29dof.urdf
//   H100: engineai-robotics/engineai_robotics_native_sdk,
//         assets/resource/robot/t800/urdf/serial_t800.urdf
//
// Limb lengths are the summed joint-origin offsets down one arm (shoulder roll
// to hand) and one leg (hip roll to foot). Measured:
//
//                                   L100      H100     ratio
//   mass (kg)                      35.115    84.988    2.42x
//   arm: shoulder -> hand (m)       0.410     0.567    1.38x
//   leg: hip -> foot (m)            0.698     1.028    1.47x
//   shoulder height above pelvis    0.238     0.383    1.61x
//   shoulder half-width (m)         0.100     0.163    1.63x
//
// Note the H100 URDF masses sum to 85.0 kg while EngineAI publish 75 kg for the
// T-800. The URDF is the number used here because it is the same kind of
// measurement taken the same way for both robots; the published figure may
// exclude the battery or a different hand assembly.
//
// The four combat multipliers at the bottom of each entry are NOT measurements.
// They are derived from the mass ratio by an exponent that is a config kwarg,
// because how much a 2.4x mass advantage should translate into knockdown
// resistance is a game-balance decision REK made and we cannot read off a URDF.
// Extraction against the shipped game is what should eventually replace them.

#pragma once

typedef enum {
    LIMB_ARM = 0,
    LIMB_LEG = 1,
} Limb;

typedef struct {
    const char* name;
    float mass;            // kg, summed from the URDF's link inertials
    float arm_len;         // shoulder -> hand, metres
    float leg_len;         // hip -> foot, metres
    float shoulder_half_w; // half the shoulder span, metres
    float body_radius;     // collision/footprint radius, metres
} ChassisDef;

// body_radius is shoulder half-width plus a fixed 0.18 m margin for the torso
// and arms, which reproduces the 0.28 m the single-chassis version used for the
// L100 and scales the H100 from the same rule rather than a second guess.
static const ChassisDef REK_CHASSIS[] = {
    // name    mass     arm     leg    sh_half_w  body_r
    {"L100",  35.115f, 0.410f, 0.698f,  0.100f,   0.280f},
    {"H100",  84.988f, 0.567f, 1.028f,  0.163f,   0.343f},
};

#define NUM_CHASSIS ((int)(sizeof(REK_CHASSIS) / sizeof(REK_CHASSIS[0])))
#define CHASSIS_L100 0
#define CHASSIS_H100 1

// Mass used as the "neutral" point, so a chassis at this mass gets multiplier
// 1.0 and the config defaults stay meaningful. The lighter robot is the
// reference, making every H100 multiplier a readable number above 1.
#define REK_REF_MASS (REK_CHASSIS[CHASSIS_L100].mass)

static inline const ChassisDef* rek_chassis(int idx) {
    if (idx < 0 || idx >= NUM_CHASSIS) idx = 0;
    return &REK_CHASSIS[idx];
}

// Limb a move swings, used to scale its reach onto whichever chassis threw it.
static inline float rek_limb_len(const ChassisDef* c, Limb limb) {
    return (limb == LIMB_LEG) ? c->leg_len : c->arm_len;
}
