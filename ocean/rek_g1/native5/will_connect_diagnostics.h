#pragma once

#include "will_connect.h"
#include "../g1_fight_state.h"
#include <cmath>
#include <cstdint>

// Read-only host diagnostic. No observation, reward, action or physics writes.
// The attached model is an uncalibrated upright head/torso approximation.
namespace rek5_will_connect {
inline constexpr const char* Schema = "rek.will_connect.diagnostic.v1";
// Native moves 1/4 occupy registry positions 5/8 (human_eval_server.py's
// MOVE_REGISTRY_ORDER; fast_assets.cpp's move_order). Category = 16 + registry
// position. These identities are independent of keyboard bindings.
inline constexpr int Actions[2] = {21, 24};
inline constexpr const char* Moves[2] = {"left_jab_processed", "right_jab_processed"};
inline constexpr const char* Labels[2] = {"Left jab", "Right jab"};
// Applicability guard for the upright proxy, not an official fall threshold.
inline constexpr float MaxTiltRadians = 0.35f;

struct Attack {
    wc_result result{};
    bool available = false;
    const char* reason = nullptr;
};
struct Diagnostic {
    wc_body bodies[2]{};
    bool body_valid[2]{};
    Attack attacks[2][2]{};
};

inline wc_v3 rotate(const float* q, wc_v3 v) {
    const wc_v3 u = wc_v(q[1], q[2], q[3]);
    const wc_v3 t = wc__mul(wc__cross(u, v), 2.f);
    return wc__add(v, wc__add(wc__mul(t, q[0]), wc__cross(u, t)));
}

// MuJoCo free-joint translation is world-frame; angular qvel is root-local.
// Forward is root-local +X, verified for this G1's recovered forwardYawOffset.
inline const char* body(const float* p, const float* v, wc_body& out) {
    for (int k = 0; k < 7; ++k) if (!std::isfinite(p[k])) return "invalid_root_state";
    for (int k = 0; k < 6; ++k) if (!std::isfinite(v[k])) return "invalid_root_state";
    double norm2 = 0;
    for (int k = 3; k < 7; ++k) norm2 += double(p[k]) * p[k];
    if (norm2 < 1e-12 || !std::isfinite(norm2)) return "invalid_root_quaternion";
    float q[4];
    for (int k = 0; k < 4; ++k) q[k] = float(p[k + 3] / std::sqrt(norm2));
    const wc_v3 up = rotate(q, wc_v(0, 0, 1));
    if (up.z < std::cos(MaxTiltRadians)) return "outside_upright_proxy";
    const wc_v3 forward = rotate(q, wc_v(1, 0, 0));
    const float horizontal2 = forward.x * forward.x + forward.y * forward.y;
    if (horizontal2 < 1e-6f) return "degenerate_facing";
    const wc_v3 omega = rotate(q, wc_v(v[3], v[4], v[5]));
    const wc_v3 derivative = wc__cross(omega, forward);
    out.pos = wc_v(p[0], p[1], p[2]);
    out.vel = wc_v(v[0], v[1], v[2]);
    out.fwd = wc_v(forward.x, forward.y, 0);
    out.yaw_rate = (forward.x * derivative.y - forward.y * derivative.x) / horizontal2;
    return std::isfinite(out.yaw_rate) ? nullptr : "invalid_root_state";
}

inline wc_result invalid() {
    wc_result r{};
    r.outcome = WC_INVALID; r.light = WC_OFF; r.t_contact = -1.f;
    return r;
}

inline Diagnostic evaluate(const float* qpos, const float* qvel,
        const uint8_t* masks, int phase, bool terminal, uint32_t failure_bits,
        bool physical_velocity_contract = true) {
    Diagnostic out;
    const char* reason = nullptr;
    if (!physical_velocity_contract) reason = "unsupported_backend_velocity";
    else if (!qpos || !qvel || !masks) reason = "missing_snapshot";
    else if (failure_bits) reason = "runtime_failure";
    else if (terminal || phase != REK_G1_FIGHT_ROUND_ACTIVE) reason = "round_inactive";
    if (!reason) for (int k = 0; k < 66; ++k)
        if (masks[k] > 1) { reason = "invalid_action_mask"; break; }
    if (!reason) {
        for (int side = 0; side < 2; ++side) {
            const char* error = body(qpos + side * 36, qvel + side * 35, out.bodies[side]);
            out.body_valid[side] = error == nullptr;
            if (error && !reason) reason = error;
        }
    }
    if (!reason) {
        const wc_hurtbox& h = WC_HURTBOX_G1;
        wc_v3 a = out.bodies[0].pos, b = out.bodies[1].pos;
        a.z += h.torso_lo; b.z += h.torso_lo;
        wc_v3 a_top = a, b_top = b;
        a_top.z += h.torso_hi - h.torso_lo;
        b_top.z += h.torso_hi - h.torso_lo;
        float t;
        if (wc__seg_seg(a, a_top, b, b_top, &t) <= 2.f * h.torso_r)
            reason = "proxy_bodies_overlap";
    }
    for (int side = 0; side < 2; ++side) for (int move = 0; move < 2; ++move) {
        Attack& a = out.attacks[side][move];
        a.result = invalid();
        a.reason = reason;
        if (!a.reason && !masks[side * 33 + Actions[move]]) a.reason = "attack_unavailable";
        if (a.reason) continue;
        const wc_attack& spec = move == 0 ? WC_ATK_U : WC_ATK_I;
        a.result = wc_eval(&spec, &out.bodies[side], &out.bodies[1 - side],
                          &WC_HURTBOX_G1, nullptr, 0, &WC_CFG_DEFAULT);
        a.available = a.result.outcome != WC_INVALID;
        if (!a.available) a.reason = "invalid_prediction";
    }
    return out;
}
}
