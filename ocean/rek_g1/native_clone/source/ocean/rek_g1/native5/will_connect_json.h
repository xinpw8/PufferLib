#pragma once

#include "will_connect_diagnostics.h"
#include "../../../vendor/cJSON.h"

namespace rek5_will_connect {
inline const char* outcome(wc_outcome value) {
    switch (value) {
        case WC_MISS: return "miss";
        case WC_HIT: return "hit";
        case WC_WEAK: return "weak";
        case WC_BLOCKED: return "blocked";
        default: return "invalid";
    }
}
inline const char* light(wc_light value) {
    switch (value) {
        case WC_RED: return "red";
        case WC_AMBER: return "amber";
        case WC_GREEN: return "green";
        default: return "off";
    }
}
inline void optional_number(cJSON* o, const char* key, float value, bool present) {
    if (present && std::isfinite(value)) cJSON_AddNumberToObject(o, key, value);
    else cJSON_AddNullToObject(o, key);
}
inline void vector(cJSON* o, const char* key, wc_v3 value) {
    const float values[] = {value.x, value.y, value.z};
    cJSON_AddItemToObject(o, key, cJSON_CreateFloatArray(values, 3));
}
// Caller owns the returned JSON tree. All values derive from this snapshot.
inline cJSON* json(const Diagnostic& diagnostic) {
    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "schema", Schema);
    cJSON_AddStringToObject(root, "calibration", "placeholder");
    cJSON_AddStringToObject(root, "guardMode", "unavailable");
    cJSON_AddStringToObject(root, "targetModel", "upright_head_sphere_torso_capsule");
    cJSON_AddStringToObject(root, "frame", "mujoco_z_up_root_local_x_forward");
    cJSON_AddBoolToObject(root, "officialScoringValidated", false);
    cJSON* fighters = cJSON_AddArrayToObject(root, "fighters");
    for (int side = 0; side < 2; ++side) {
        cJSON* fighter = cJSON_CreateObject();
        cJSON_AddItemToArray(fighters, fighter);
        cJSON_AddNumberToObject(fighter, "side", side);
        if (diagnostic.body_valid[side]) {
            cJSON* input = cJSON_AddObjectToObject(fighter, "input");
            vector(input, "position", diagnostic.bodies[side].pos);
            vector(input, "velocity", diagnostic.bodies[side].vel);
            vector(input, "forward", diagnostic.bodies[side].fwd);
            optional_number(input, "yawRate", diagnostic.bodies[side].yaw_rate, true);
        } else cJSON_AddNullToObject(fighter, "input");
        cJSON* attacks = cJSON_AddArrayToObject(fighter, "attacks");
        for (int move = 0; move < 2; ++move) {
            const Attack& a = diagnostic.attacks[side][move];
            const wc_result& r = a.result;
            cJSON* attack = cJSON_CreateObject();
            cJSON_AddItemToArray(attacks, attack);
            cJSON_AddStringToObject(attack, "move", Moves[move]);
            cJSON_AddStringToObject(attack, "label", Labels[move]);
            cJSON_AddNumberToObject(attack, "action", Actions[move]);
            cJSON_AddBoolToObject(attack, "available", a.available);
            cJSON_AddStringToObject(attack, "outcome", outcome(r.outcome));
            cJSON_AddStringToObject(attack, "light", light(r.light));
            if (a.reason) cJSON_AddStringToObject(attack, "reason", a.reason);
            else cJSON_AddNullToObject(attack, "reason");
            optional_number(attack, "margin", r.margin, a.available);
            optional_number(attack, "contactTime", r.t_contact, a.available && r.t_contact >= 0);
            optional_number(attack, "relativeSpeed", r.rel_speed, a.available && r.t_contact >= 0);
            if (a.available && r.t_contact >= 0) vector(attack, "contactPoint", r.p_contact);
            else cJSON_AddNullToObject(attack, "contactPoint");
            cJSON_AddStringToObject(attack, "target", r.target == WC_TGT_HEAD ? "proxy_head" :
                                   r.target == WC_TGT_TORSO ? "proxy_torso" : "none");
        }
    }
    return root;
}
}
