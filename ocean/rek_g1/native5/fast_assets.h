#pragma once

#include "runtime_api.h"
#include "primitive_contacts.cuh"
#include "../g1_hit_detector.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Offline kinematic bake. No integrated dynamics or measured root translation
// is implied by these pose samples. Coordinates are metres, relative to the
// fighter's world XY origin, with configured clip yaw removal and root height
// included. Root XY samples in the actual bundle are effectively zero.
struct FastFrame {
    float q[29];
    float root_wxyz[4];
    float strike_xyz[6][3]; // feet L/R, hands L/R, knees L/R
    float strike_radius[6];
    float target_xyz[3][3]; // pelvis, torso, head
    float target_radius[3];
    float root_z;
    float clip_yaw; // normalized source yaw before configured yaw removal
    rek5_primitive::Shape strike_shapes[12]; // four spheres per foot, hands, shins
    rek5_primitive::Shape target_shapes[3]; // pelvis, torso, head
};

struct FastRoute {
    int offset;
    int count;
    float fps; // sample rate after baking: 50 Hz
    int loop;
    int source_clip_id;
    int move; // runtime move index, or -1
    int start_frame;
    int end_frame;
    float playback_speed;
    float blend_in_seconds;
    float blend_out_seconds;
    float yaw_blend;
};

struct FastAssets {
    std::array<FastRoute,24> routes{};
    std::vector<FastFrame> frames;
    std::array<int,33> action_to_route{}; // continue= -1; held combinations resolve translation first
    std::array<std::uint32_t,17> move_duration_ticks{};
    int qindices[2][29]{};
    int vindices[2][29]{};
    float initial_qpos[72]{};
    float spawn_xy[2][2]{};
    float initial_heading[2]{};
    float floor_height=0;
    float arena_half_extent[2]{};
    float yaw_ramp_seconds=.5f;
    float settle_linear_speed=.03f;
    float settle_yaw_rate=.03f;
    float stop_brake_rate=2;
    int strike_limb[12]{0,0,0,0,1,1,1,1,2,3,4,5}; // feet L/R, hands L/R, shins L/R
    std::string model_sha256;
    std::string manifest_sha256;
    std::string features_sha256;
    std::string provenance_json;
    // Verified recovered metadata. Geometry and velocity remain kinematic proxies.
    std::array<RekG1ImpactEvent,29> impact_events{};
    int impact_offsets[24]{},impact_counts[24]{};
    RekG1HitDetectorConfig recovered_hit_config{};
    bool recovered_catalog_compatible=false;
};

// CPU work is restricted to loading and forward kinematics before training.
// Uses mj_kinematics, never mj_step, mj_forward, or controller inference.
FastAssets load_fast_assets(const RekNative5Config& config);
