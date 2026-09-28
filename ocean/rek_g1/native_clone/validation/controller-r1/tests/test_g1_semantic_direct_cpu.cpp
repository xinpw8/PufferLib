/* Runs the actual scheduler kernel bodies serially with real native scalar and
 * composer implementations. No CUDA runtime, policy or physics is initialized.
 * Synthetic clips and callbacks test dispatch contracts, not motion parity. */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <initializer_list>
#define REK_G1_SEMANTIC_CPU_TEST 1
#define __device__
#define __global__
struct CpuLaunchDimension { size_t x; };
static CpuLaunchDimension blockIdx{0}, blockDim{1}, threadIdx{0}, gridDim{1};
#include "g1_semantic_scheduler_cuda.cu"
#if defined(REK_G1_SEMANTIC_BASELINE_SOURCE)
namespace source_baseline {
#include REK_G1_SEMANTIC_BASELINE_SOURCE
}
#endif

namespace {
int checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) { std::fprintf(stderr, "FAILED: %s\n", message); std::exit(1); }
}
int slerp(void*, const float* a, const float* b, float t, float* out) {
    for (int i = 0; i < 4; ++i) out[i] = a[i] + t * (b[i] - a[i]);
    return 1;
}
int atan_backend(void*, float y, float x, float* out) { *out = atan2f(y, x); return 1; }
int sincos_backend(void*, float x, float* s, float* c) { *s = sinf(x); *c = cosf(x); return 1; }
int matcher(void*, const SonicMotionComposerNativeLayer* target,
        const SonicMotionComposerNativeLayer*, float* out) {
    *out = float(target->start_frame); return 1;
}
struct Assets {
    float positions[24][4 * 29]{};
    float roots[24][4 * 4]{};
    RekG1CudaComposerCommand routes[24]{};
    int32_t kinds[24]{}, moves[17]{};
    uint32_t durations[17]{}, mirror_indices[29]{};
    uint8_t mirror_negate[29]{};
    RekG1SemanticActionTableStorage table{};
    RekG1CudaSemanticConfig config{{0.02f, 0.5f}, {1, 1, 1, 50}, {0.03f, 0.03f, 2, 1}};
    SonicMotionComposerNativeReferenceTiming timing{};
    SonicMotionComposerNativeMirrorTable mirror{mirror_indices, mirror_negate, 29, 29};
    Assets() {
        for (int route = 0; route < 24; ++route) {
            for (int frame = 0; frame < 4; ++frame) {
                roots[route][frame * 4] = 1;
                for (int joint = 0; joint < 29; ++joint) positions[route][frame * 29 + joint] = 0.001f * float(route + frame);
            }
            routes[route] = {REK_G1_CUDA_COMPOSER_PLAY,
                {positions[route], roots[route], 4 * 29, 4 * 4, 4, 50},
                {0, route < 7, 1, 0, 3, 0.04f, 0.04f, 0}, 0};
            kinds[route] = route == 0 ? 0 : route < 5 ? 1 : route < 7 ? 2 : 3;
        }
        for (int move = 0; move < 17; ++move) { moves[move] = 23 - move; durations[move] = 3; }
        for (int joint = 0; joint < 29; ++joint) mirror_indices[joint] = uint32_t(joint);
        for (int frame = 0; frame < 10; ++frame) { timing.current_offsets[frame] = frame * 5; timing.next_offsets[frame] = frame * 5 + 1; }
        int status = -1;
        table_kernel(&table, 1, durations, &status);
        require(status == 0, "semantic table setup");
    }
};
struct Fixture {
    static constexpr size_t count = 2;
    RekG1CudaSemanticRow rows[count]{};
    SonicMotionComposerNative composers[count]{};
    float heading[count * 4]{}, observation[count * 12]{};
    uint8_t masks[count * 33]{}, flags[count]{1, 1}, zeros[count]{};
    float positions[count * 290]{}, next_positions[count * 290]{}, rotations[count * 40]{};
    SonicMotionComposerNativeReferenceOutput references[count]{};
    RekG1CudaSemanticBuffers buffers{};
    float actions[count]{}, velocity[count * 6]{};
    int32_t phase[count]{};
    uint8_t suspended[count]{}, enabled[count]{1, 1};
    RekG1CudaDirectCommand commands[count]{};
    RekG1CudaDirectResult results[count]{};
    bool use_baseline = false;
    explicit Fixture(Assets& assets, bool baseline = false) : use_baseline(baseline) {
        const SonicMotionComposerNativeBackends backend{slerp, atan_backend, sincos_backend, matcher, nullptr};
        for (size_t i = 0; i < count; ++i) {
            require(sonic_motion_composer_native_init(&composers[i], 50, &backend) == 0, "composer setup");
            references[i] = {positions + i * 290, next_positions + i * 290, rotations + i * 40, 290, 290, 40};
            heading[i * 4] = 1;
            commands[i].move_index = -1;
        }
        buffers = {rows, composers, &assets.table, &assets.config, assets.routes, assets.kinds,
            assets.moves, &assets.timing, &assets.mirror, references, heading, observation, masks};
        reset();
        require(rows[0].status == 0 && rows[1].status == 0, "scheduler reset");
    }
    void reset() {
#if defined(REK_G1_SEMANTIC_BASELINE_SOURCE)
        if (use_baseline) { source_baseline::reset_kernel(&buffers, flags, heading, count); return; }
#endif
        reset_kernel(&buffers, flags, heading, count);
    }
    void pre() { pre_kernel(&buffers, actions, velocity, suspended, count, commands, enabled, results); }
    void legacy_pre() {
#if defined(REK_G1_SEMANTIC_BASELINE_SOURCE)
        source_baseline::pre_kernel(&buffers, actions, velocity, suspended, count);
#else
        pre_kernel(&buffers, actions, velocity, suspended, count);
#endif
    }
    void post(const uint8_t* input_reset = nullptr, const uint8_t* reset_event = nullptr,
            const uint8_t* terminal = nullptr) {
#if defined(REK_G1_SEMANTIC_BASELINE_SOURCE)
        if (use_baseline) {
            source_baseline::post_kernel(&buffers, velocity, phase, suspended, input_reset ? input_reset : zeros,
                reset_event ? reset_event : zeros, terminal ? terminal : zeros, count);
            return;
        }
#endif
        post_kernel(&buffers, velocity, phase, suspended, input_reset ? input_reset : zeros,
            reset_event ? reset_event : zeros, terminal ? terminal : zeros, count);
    }
};
void identical(const Fixture& a, const Fixture& b) {
    require(std::memcmp(a.rows, b.rows, sizeof(a.rows)) == 0, "disabled rows are byte-identical");
    require(std::memcmp(a.composers, b.composers, sizeof(a.composers)) == 0, "disabled composers are byte-identical");
    require(std::memcmp(a.heading, b.heading, sizeof(a.heading)) == 0, "disabled headings are byte-identical");
    require(std::memcmp(a.observation, b.observation, sizeof(a.observation)) == 0, "disabled observations are byte-identical");
    require(std::memcmp(a.masks, b.masks, sizeof(a.masks)) == 0, "disabled masks are byte-identical");
    require(std::memcmp(a.positions, b.positions, sizeof(a.positions)) == 0, "disabled reference positions are byte-identical");
    require(std::memcmp(a.next_positions, b.next_positions, sizeof(a.next_positions)) == 0, "disabled next positions are byte-identical");
    require(std::memcmp(a.rotations, b.rotations, sizeof(a.rotations)) == 0, "disabled rotations are byte-identical");
}
void test_continuous(Assets& assets) {
    Fixture f(assets);
    f.commands[0].velocity = {0.8f, 0.434f, -1.23f};
    f.rows[0].recovery_active = 1;
    f.pre();
    require(f.results[0].status == 0, "continuous status");
    require(std::memcmp(&f.rows[0].effective_velocity, &f.commands[0].velocity, sizeof(RekG1NativeVelocityCommand)) == 0,
        "all continuous components retained despite unrelated fall/recovery mask state");
    require(f.composers[0].current_layer.per_tick > 1, "native playback preserves above-unit magnitude");
    f.post();
    require(f.rows[0].status == 0 && f.heading[3] < 0, "continuous native heading integration");
}
void test_moves(Assets& assets) {
    for (int move = 0; move < 17; ++move) {
        Fixture f(assets);
        f.commands[0].velocity = {0.8f, 0, 0}; f.pre(); f.post();
        require(!f.rows[0].translation_settled, "fixture has active translation");
        f.commands[0].velocity = {}; f.commands[0].move_index = move;
        f.pre();
        require(f.results[0].move_attempted && f.results[0].move_accepted && !f.results[0].move_rejected, "assigned move accepted without learner translation gate");
        require(f.results[0].applied_route == assets.moves[move], "assigned move uses move_routes mapping");
        require(f.composers[0].action_playing && !f.rows[0].locomotion.locomotion_active, "accepted move clears native locomotion flag");
        f.post();
        f.commands[0].move_index = -1;
        for (int tick = 0; tick < 5; ++tick) { f.pre(); f.post(); }
        require(f.rows[0].status == 0 && !f.composers[0].action_playing, "direct move uses actual composer completion");
    }
}
void test_rejection_and_cancel(Assets& assets) {
    for (const int invalid : {-2, 17, 100}) {
        Fixture f(assets);
        f.commands[0].move_index = invalid; f.commands[0].rejection_velocity = {0.321f, 0.111f, -0.27f};
        f.pre();
        require(f.results[0].move_rejected && f.results[0].rejection_reason == REK_G1_DIRECT_INVALID_MOVE && !f.results[0].status, "invalid assigned index is ordinary rejection");
        require(std::memcmp(&f.rows[0].effective_velocity, &f.commands[0].rejection_velocity, sizeof(RekG1NativeVelocityCommand)) == 0, "rejection velocity applies in same tick");
    }
    Fixture f(assets);
    f.commands[0].move_index = 16; f.pre(); f.post();
    const int move_id = f.composers[0].action_move_id;
    f.commands[0].move_index = 0; f.commands[0].rejection_velocity = {0.21f, 0.13f, 0.7f};
    f.pre();
    require(f.results[0].move_rejected && f.results[0].rejection_reason == REK_G1_DIRECT_PUNCHING && !f.results[0].status, "playing move rejects interruption");
    require(f.composers[0].action_move_id == move_id, "rejected move did not replace composer action");
    require(f.rows[0].effective_velocity.forward == 0.21f && f.rows[0].effective_velocity.strafe == 0.13f && f.rows[0].effective_velocity.yaw == 0,
        "rejection velocity reaches native restrict-yaw rule");
    f.commands[0].move_index = -1; f.commands[0].cancel_action = 1; f.commands[0].input_recovering = 1;
    const auto recovering_composer = f.composers[0];
    f.pre();
    require(!f.results[0].cancelled && f.composers[0].action_playing, "CancelPunch ignored while input recovering");
    require(std::memcmp(&recovering_composer, &f.composers[0], sizeof(recovering_composer)) == 0, "ignored cancel preserves composer");
    f.commands[0].input_recovering = 0; f.pre();
    require(f.results[0].cancelled && !f.composers[0].action_playing && f.composers[0].current_layer.config.loop,
        "native CancelAction followed by idle");
    require(f.results[0].applied_route == 0 && f.composers[0].action_move_id == move_id, "cancel does not start an attack");
    const auto idle_composer = f.composers[0]; f.pre();
    require(!f.results[0].cancelled && std::memcmp(&idle_composer, &f.composers[0], sizeof(idle_composer)) == 0, "cancel when not punching is no-op");
    Fixture recovery(assets);
    recovery.commands[0].move_index = 2; recovery.commands[0].input_recovering = 1;
    recovery.commands[0].rejection_velocity = {0.33f, 0, 0}; recovery.pre();
    require(recovery.results[0].rejection_reason == REK_G1_DIRECT_RECOVERING && !recovery.results[0].status, "native recovery rejects move");
    require(recovery.rows[0].effective_velocity.forward == 0.33f, "input recovery does not invent FixedUpdate zeroing");
}
void test_lifecycle_and_handoff(Assets& assets) {
    Fixture f(assets);
    f.commands[0].move_index = 3; f.pre();
    const auto playing = f.composers[0];
    const auto adapter = f.rows[0].adapter;
    f.enabled[0] = 0; f.pre();
    require(f.results[0].status == REK_G1_DIRECT_HANDOFF_UNSUPPORTED, "active direct-to-categorical handoff explicit unsupported");
    require(std::memcmp(&playing, &f.composers[0], sizeof(playing)) == 0 && std::memcmp(&adapter, &f.rows[0].adapter, sizeof(adapter)) == 0,
        "unsupported handoff preserves action and adapter");
    reset_kernel(&f.buffers, f.flags, f.heading, Fixture::count);
    require(!f.rows[0].direct_native_active && !f.rows[0].status && !f.composers[0].action_playing, "existing lifecycle reset clears mode/status");
    f.commands[0].move_index = -1; f.enabled[0] = 1; f.pre(); f.post();
    f.enabled[0] = 0; f.actions[0] = 1; f.pre(); f.post();
    require(!f.rows[0].status && !f.rows[0].direct_native_active, "quiescent handoff works");
    Fixture suspended(assets);
    suspended.suspended[0] = 1; suspended.commands[0].move_index = 4;
    float old_positions[290], old_next[290], old_rotations[40], old_heading[4];
    std::memcpy(old_positions, suspended.positions, sizeof(old_positions));
    std::memcpy(old_next, suspended.next_positions, sizeof(old_next));
    std::memcpy(old_rotations, suspended.rotations, sizeof(old_rotations));
    std::memcpy(old_heading, suspended.heading, sizeof(old_heading));
    suspended.pre();
    require(suspended.results[0].suspended && suspended.results[0].move_accepted && !suspended.results[0].move_rejected
        && suspended.results[0].rejection_reason == REK_G1_DIRECT_NOT_REJECTED, "motor suspension does not reject native move dispatch");
    const auto suspended_composer = suspended.composers[0]; suspended.post();
    require(suspended.composers[0].action_playing && std::memcmp(&suspended_composer, &suspended.composers[0], sizeof(suspended_composer)) == 0,
        "suspended accepted action installs but does not advance");
    require(std::memcmp(old_heading, suspended.heading, sizeof(old_heading)) == 0, "suspension retains heading");
    require(std::memcmp(old_positions, suspended.positions, sizeof(old_positions)) == 0
        && std::memcmp(old_next, suspended.next_positions, sizeof(old_next)) == 0
        && std::memcmp(old_rotations, suspended.rotations, sizeof(old_rotations)) == 0, "suspension retains reference rows");
    suspended.commands[0].move_index = -1; suspended.commands[0].cancel_action = 1; suspended.pre();
    require(suspended.results[0].cancelled && !suspended.composers[0].action_playing && suspended.results[0].applied_route == 0,
        "native cancellation remains active during motor suspension");
    suspended.commands[0].cancel_action = 0; suspended.commands[0].velocity = {0.321f, 0.123f, -0.21f}; suspended.pre();
    require(std::memcmp(&suspended.rows[0].effective_velocity, &suspended.commands[0].velocity, sizeof(RekG1NativeVelocityCommand)) == 0,
        "continuous native input writes remain active during motor suspension");
    suspended.suspended[0] = 0; suspended.commands[0].move_index = 4; suspended.commands[0].velocity = {}; suspended.pre(); suspended.post();
    require(!suspended.results[0].suspended && suspended.results[0].move_accepted && suspended.composers[0].current_layer.cursor > 0,
        "unsuspended action resumes native advancement");
    suspended.post(suspended.flags, suspended.flags, suspended.flags);
    require(!suspended.rows[0].adapter.scheduler.active && !suspended.rows[0].action_busy, "terminal/input reset lifecycle retained");
    Fixture invalid(assets);
    invalid.commands[0].velocity.forward = std::numeric_limits<float>::quiet_NaN(); invalid.pre();
    require(invalid.results[0].status == 202, "nonfinite direct velocity fails");
    Fixture measured(assets);
    measured.suspended[0] = 1; measured.velocity[5] = std::numeric_limits<float>::infinity(); measured.pre();
    require(measured.results[0].status == 202, "suspended measured velocity still validated");
}
void test_disabled_equivalence(Assets& assets) {
    Fixture legacy(assets, true), direct(assets);
    direct.enabled[0] = direct.enabled[1] = 0;
    direct.commands[0].velocity.forward = direct.commands[1].velocity.forward = std::numeric_limits<float>::quiet_NaN();
    direct.commands[0].move_index = direct.commands[1].move_index = 100;
    for (int tick = 0; tick < 1200; ++tick) {
        if (tick % 31 == 0) {
            legacy.reset(); direct.reset();
        }
        for (size_t i = 0; i < Fixture::count; ++i) {
            const int candidate = (tick * 13 + int(i) * 7) % 33;
            int selected = candidate;
            if (!legacy.masks[i * 33 + selected]) {
                selected = 0;
                while (selected < 33 && !legacy.masks[i * 33 + selected]) ++selected;
                require(selected < 33, "legacy mask has a legal action");
            }
            const float action = float(selected);
            legacy.actions[i] = direct.actions[i] = action;
        }
        legacy.legacy_pre(); direct.pre(); identical(legacy, direct);
        legacy.post(); direct.post(); identical(legacy, direct);
        if (legacy.rows[0].status || legacy.rows[1].status) {
            std::fprintf(stderr, "legacy fixture tick=%d status=%d,%d action=%g,%g remaining=%u,%u cursor=%g,%g playing=%d,%d\n",
                tick, legacy.rows[0].status, legacy.rows[1].status, legacy.actions[0], legacy.actions[1],
                legacy.rows[0].semantic.remaining_ticks, legacy.rows[1].semantic.remaining_ticks,
                legacy.composers[0].current_layer.cursor, legacy.composers[1].current_layer.cursor,
                legacy.composers[0].action_playing, legacy.composers[1].action_playing);
        }
        require(!legacy.rows[0].status && !legacy.rows[1].status, "legacy sequence stays valid");
    }
}
}
int main() {
    require(blocks_for(0) == 0 && blocks_for(129) == 2, "launch shape helper");
    Assets assets;
#if defined(REK_G1_SEMANTIC_BASELINE_SOURCE)
    require(source_baseline::blocks_for(129) == 2, "baseline launch shape helper");
    int baseline_table_status = -1;
    source_baseline::table_kernel(&assets.table, 1, assets.durations, &baseline_table_status);
    require(baseline_table_status == 0, "baseline semantic table");
#endif
    test_continuous(assets); test_moves(assets); test_rejection_and_cancel(assets);
    test_lifecycle_and_handoff(assets); test_disabled_equivalence(assets);
    std::printf("{\"event\":\"semantic_direct_cpu_tests\",\"checks\":%d,\"failures\":0,\"gpu_calls\":0,\"disabled_equivalence_ticks\":1200,\"pinned_source_baseline\":%s}\n", checks,
#if defined(REK_G1_SEMANTIC_BASELINE_SOURCE)
        "true"
#else
        "false"
#endif
    );
}
