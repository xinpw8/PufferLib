#pragma once

#include <cuda_runtime.h>
#include <mujoco/mujoco.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rek5 {

// GPU views have the exact layout and conventions of RpsDescriptor. Shapes:
// qpos[A,72], qvel[A,70], base[A,2,4], angular[A,2,3], time[A];
// body fields[A,B,*], geometry fields[A,G,*], contact fields[capacity,*].
// xquat/base use WXYZ; body_map/geom_map local quaternions use XYZW.
struct PhysicsDescriptor {
    int arenas = 0, bodies = 0, geoms = 0, capacity = 0;
    float *qpos = nullptr, *qvel = nullptr, *base = nullptr;
    float *angular = nullptr, *time = nullptr;
    float *xpos = nullptr, *xquat = nullptr, *xmat = nullptr;
    float *xipos = nullptr, *ximat = nullptr, *com = nullptr, *cvel = nullptr;
    float *geom_xpos = nullptr, *geom_xmat = nullptr;
    int *contact_geom = nullptr, *contact_world = nullptr, *nacon = nullptr;
    float *contact_dist = nullptr, *contact_pos = nullptr, *contact_frame = nullptr;
    int *counts = nullptr, *offsets = nullptr;
    float *body_map = nullptr, *geom_map = nullptr, *pre_centers = nullptr;
};

struct Physics {
    PhysicsDescriptor data;
    cudaStream_t stream = nullptr;
    void* native_handle = nullptr;
    mjModel* model = nullptr;
    float* ctrl = nullptr;                 // [arenas,58], owned device buffer
    int* stats = nullptr;                  // [arenas,4], native-owned device buffer
    int* finite_status = nullptr;          // reporting-only semantic field check
    std::array<int,58> joint_qpos{}, joint_qvel{}, actuator_ids{};
    std::array<int,2> root_bodies{};        // source MuJoCo body IDs
    std::array<float,8> initial_heading_wxyz{}; // source spawn root quaternions
    std::vector<int> geom_bodyid, geom_type, geom_contype, geom_conaffinity;
    std::vector<float> geom_size, initial_qpos, model_qpos0, joint_limits;
    std::vector<float> host_body_map, host_geom_map;
    std::vector<float> packed_bodies, packed_shapes, packed_joints, packed_roots;
    std::string model_sha256, export_sha256, assets_sha256;
    std::vector<void*> allocations;
};

// Host-only compilation and frame mapping. Calls mj_kinematics once; performs
// no CPU physics steps, CUDA initialization, or Python invocation.
Physics* physics_load_model(const char* xml_path, const char* export_json_path);

// Creates the existing independent-contact Puffysics solver (mode 0) and all
// semantic GPU buffers, completing startup work before returning. At most one
// live GPU physics handle is supported. After ordering any outstanding work,
// the caller may change Physics.stream; the native handle owns no stream.
Physics* physics_create(const char* xml_path, const char* export_json_path,
                        int arenas, cudaStream_t stream);
void physics_step(Physics* physics, const float* device_ctrl);
void physics_forward_selected(Physics* physics, const uint8_t* device_mask);
void physics_refresh(Physics* physics);

// Explicit reporting boundary, synchronizes this physics stream. Cumulative
// failure counters survive every masked forward/reset.
std::vector<int> physics_stats(Physics* physics);
void physics_check_status(Physics* physics);
void physics_close(Physics* physics) noexcept;

} // namespace rek5
