#pragma once

#include "primitive_contacts.cuh"

// Current G1 importData body-zone mapping. Native IsScoring accepts zones
// Head=1, Torso=2, Pelvis=3, LeftHip=12 and RightHip=13. This model has no
// Head-zone body: its head-shaped capsule belongs to the Torso-zone body.
// Names are suffixes; prepend player__ or opponent__ for model lookup.
namespace rek5_native_contact {
inline constexpr int LegacyTargetCount=3;
inline constexpr int TargetCount=9;
inline constexpr const char* TargetContract="g1_scoring_bodyzones_1_2_3_12_13_v1";
inline constexpr const char* TargetNames[TargetCount]={
    "mjgeom_3021","mjgeom_3285","mjgeom_3064",
    "mjgeom_3337","mjgeom_3141","mjgeom_3399",
    "mjgeom_3024","mjgeom_3062","mjgeom_3406"
};
inline constexpr const char* TargetBodies[TargetCount]={
    "pelvis_3266","torso_link_3347","torso_link_3347",
    "left_hip_pitch_link_3457","left_hip_roll_link_3425","left_hip_yaw_link_2943",
    "right_hip_pitch_link_3469","right_hip_roll_link_3345","right_hip_yaw_link_3191"
};
inline constexpr int TargetKinds[TargetCount]={
    rek5_primitive::Box,rek5_primitive::Box,rek5_primitive::Capsule,
    rek5_primitive::Box,rek5_primitive::Box,rek5_primitive::Capsule,
    rek5_primitive::Box,rek5_primitive::Box,rek5_primitive::Capsule
};
inline constexpr int TargetZones[TargetCount]={3,2,2,12,12,12,13,13,13};
}
