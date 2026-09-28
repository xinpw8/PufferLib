#include "will_connect_json.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>

int main() {
    using namespace rek5_will_connect;
    float p[72]{}, v[70]{}; uint8_t masks[66];
    std::fill(masks, masks + 66, 1);
    p[2] = p[38] = .65f; p[3] = 1; p[36] = .5f; p[37] = .16f; p[42] = 1;
    const auto baseline = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(baseline.attacks[0][0].available && baseline.attacks[0][0].result.outcome == WC_HIT);
    assert(Actions[0] == 21 && Actions[1] == 24);
    // Read-only adapter must not change the exact snapshot or masks.
    float original_p[72], original_v[70]; uint8_t original_masks[66];
    std::copy(p, p + 72, original_p); std::copy(v, v + 70, original_v);
    std::copy(masks, masks + 66, original_masks);
    (void)evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(std::memcmp(original_p, p, sizeof p) == 0);
    assert(std::memcmp(original_v, v, sizeof v) == 0);
    assert(std::memcmp(original_masks, masks, sizeof masks) == 0);
    masks[21] = 0;
    auto r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(!r.attacks[0][0].available && r.attacks[0][0].result.light == WC_OFF);
    assert(std::strcmp(r.attacks[0][0].reason, "attack_unavailable") == 0);
    assert(r.attacks[0][1].available && r.attacks[1][0].available);
    masks[21] = 1; masks[33 + 24] = 0;
    r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(r.attacks[0][1].available && !r.attacks[1][1].available);
    masks[33 + 24] = 1;
    r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0, false);
    assert(!r.attacks[0][0].available && std::strcmp(r.attacks[0][0].reason, "unsupported_backend_velocity") == 0);
    masks[21] = 2;
    r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(!r.attacks[0][0].available && std::strcmp(r.attacks[0][0].reason, "invalid_action_mask") == 0);
    masks[21] = 1;
    for (int mode = 0; mode < 3; ++mode) {
        r = evaluate(p, v, masks, mode == 0 ? REK_G1_FIGHT_ROUND_COUNTDOWN : REK_G1_FIGHT_ROUND_ACTIVE,
                     mode == 1, mode == 2 ? 8 : 0);
        for (const auto& side : r.attacks) for (const auto& attack : side)
            assert(!attack.available && attack.result.light == WC_OFF && attack.reason);
    }
    // Tilted free-joint rotation: local angular Z maps to heading rate w/cos(pitch).
    float root[7] = {0, 0, .65f, std::cos(.1f), 0, std::sin(.1f), 0};
    float velocity[6] = {1, 2, 3, 0, 0, 2}; wc_body b{};
    assert(body(root, velocity, b) == nullptr);
    assert(std::fabs(b.yaw_rate - 2.f / std::cos(.2f)) < 1e-5f);
    assert(b.vel.x == 1 && b.vel.y == 2 && b.vel.z == 3);
    for (int k = 3; k < 7; ++k) root[k] *= -3.f;
    assert(body(root, velocity, b) == nullptr && std::fabs(b.yaw_rate - 2.f / std::cos(.2f)) < 1e-5f);
    p[3] = std::cos(.3f); p[5] = std::sin(.3f);
    r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(std::strcmp(r.attacks[0][0].reason, "outside_upright_proxy") == 0);
    p[3] = 1; p[5] = 0; p[36] = .1f; p[37] = 0;
    r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(std::strcmp(r.attacks[0][0].reason, "proxy_bodies_overlap") == 0);
    p[36] = .5f; p[37] = .16f;
    // Translation and common-velocity invariance must survive the snapshot adapter.
    for (int s = 0; s < 2; ++s) { p[36*s] += 10; p[36*s+1] -= 5; v[35*s+1] = 1; }
    r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(r.attacks[0][0].result.outcome == baseline.attacks[0][0].result.outcome);
    assert(std::fabs(r.attacks[0][0].result.t_contact - baseline.attacks[0][0].result.t_contact) < 1e-5f);
    p[36] = std::numeric_limits<float>::quiet_NaN();
    r = evaluate(p, v, masks, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    cJSON* j = json(r);
    cJSON* attack = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(
        cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(j, "fighters"), 0), "attacks"), 0);
    assert(cJSON_IsNull(cJSON_GetObjectItemCaseSensitive(attack, "margin")));
    assert(cJSON_IsFalse(cJSON_GetObjectItemCaseSensitive(attack, "available")));
    assert(cJSON_IsFalse(cJSON_GetObjectItemCaseSensitive(j, "officialScoringValidated")));
    cJSON_Delete(j);
    r = evaluate(nullptr, nullptr, nullptr, REK_G1_FIGHT_ROUND_ACTIVE, false, 0);
    assert(std::strcmp(r.attacks[0][0].reason, "missing_snapshot") == 0);
    std::cout << "Snapshot conversion, state gating, semantic mapping, invariance and JSON checks passed\n";
}
