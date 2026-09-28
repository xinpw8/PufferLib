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
    *out = float(target->start_frame) + 0.25f; return 1;
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


void test_deactivate(Assets& assets) {
    for(int route: {0,1,7})for(int mode: {0,1}) {
        Fixture f(assets); f.flags[1]=0;
        require(sonic_motion_composer_native_play_action(&f.composers[0],&assets.routes[route].clip,&assets.routes[route].config)==0,"set prior route");
        f.composers[0].from_layer.active=1;f.composers[0].pending_heading_delta=0.25f;
        auto& row=f.rows[0];row.direct_native_active=mode;
        row.effective_velocity={.25f,-.5f,.75f};row.locomotion.locomotion_active=1;
        row.locomotion.stop_braking=1;row.locomotion.has_momentum=1;
        row.locomotion.transition_settling=1;row.locomotion.transition_from_route_id=REK_G1_NATIVE_FORWARD;
        row.locomotion.current_route_id=REK_G1_NATIVE_FORWARD;row.locomotion.last_driven_command={1,0,0};
        row.semantic.kind=REK_G1_SEMANTIC_DISCRETE_MOVE;
        const auto adapter=row.adapter,otherrow=f.rows[1];const auto other=f.composers[1];
        auto expected=f.composers[0];
        require(sonic_motion_composer_native_cancel_action(&expected)==0,"unconditional original cancel");
        require(sonic_motion_composer_native_play_action(&expected,&assets.routes[0].clip,&assets.routes[0].config)==0,"original physical idle");
        deactivate_kernel(&f.buffers,f.flags,Fixture::count);
        require(!row.status,"deactivate status");require(!memcmp(&expected,&f.composers[0],sizeof(expected)),"original composer call sequence");
        require(!row.effective_velocity.forward&&!row.effective_velocity.strafe&&!row.effective_velocity.yaw,"zero effective velocity");
        require(!row.locomotion.locomotion_active&&!row.locomotion.stop_braking&&!row.locomotion.has_momentum,"three deactivation flags cleared");
        require(row.locomotion.transition_settling==1&&row.locomotion.current_route_id==REK_G1_NATIVE_FORWARD&&row.locomotion.last_driven_command.forward==1,"unwritten fields retained");
        require(row.direct_native_active==mode&&!memcmp(&adapter,&row.adapter,sizeof(adapter)),"mode and adapter ownership retained");
        require(row.semantic.kind==0,"expired tick-local semantic cleared");
        require(!memcmp(&other,&f.composers[1],sizeof(other))&&!memcmp(&otherrow,&f.rows[1],sizeof(otherrow)),"unselected fighter unchanged");
    }
}
void test_inactive(Assets& assets) {
    for(int direct: {0,1})for(int suspended: {0,1}){
        Fixture f(assets);f.enabled[0]=direct;f.rows[0].direct_native_active=1-direct;
        f.suspended[0]=suspended;f.commands[0].move_index=0;f.commands[0].cancel_action=1;f.commands[0].velocity={1,-1,.5f};f.actions[0]=16;
        uint8_t active[2]={0,1};const auto before=f.rows[0],composer=f.composers[0];
        for(float& x:f.positions)x=-777;
        pre_kernel(&f.buffers,f.actions,f.velocity,f.suspended,Fixture::count,f.commands,f.enabled,f.results,active);
        require(!f.rows[0].status,"inactive no protocol error");require(!memcmp(&before,&f.rows[0],sizeof(before)),"inactive scheduler fields unchanged");
        require(!memcmp(&composer,&f.composers[0],sizeof(composer)),"inactive composer not dispatched");
        require(!f.results[0].move_accepted&&!f.results[0].cancelled,"inactive no accepted attack/cancel");
        require(f.results[0].move_attempted==direct&&f.results[0].move_rejected==direct,"inactive direct attempt reports rejection");
        require(f.results[0].rejection_reason==(direct?REK_G1_DIRECT_INPUT_INACTIVE:REK_G1_DIRECT_NOT_REJECTED),"inactive reason explicit");
        require((f.positions[0]==-777)==bool(suspended),"reference sampled only unsuspended");
    }
    Fixture f(assets);uint8_t active[2]={0,0};f.commands[0].move_index=-1;
    pre_kernel(&f.buffers,f.actions,f.velocity,f.suspended,Fixture::count,f.commands,f.enabled,f.results,active);
    require(!f.results[0].move_attempted&&!f.results[0].move_rejected&&f.results[0].rejection_reason==REK_G1_DIRECT_INPUT_INACTIVE,"inactive neutral is not a rejected move");
}
void test_active_parity(Assets& assets) {
    Fixture a(assets),b(assets);uint8_t active[2]={1,1};
    a.commands[0].velocity=b.commands[0].velocity={.5f,-.25f,.75f};a.commands[1].move_index=b.commands[1].move_index=0;
    pre_kernel(&a.buffers,a.actions,a.velocity,a.suspended,Fixture::count,a.commands,a.enabled,a.results);
    pre_kernel(&b.buffers,b.actions,b.velocity,b.suspended,Fixture::count,b.commands,b.enabled,b.results,active);
    require(!memcmp(a.rows,b.rows,sizeof(a.rows)),"active row parity");require(!memcmp(a.composers,b.composers,sizeof(a.composers)),"active composer parity");
    require(!memcmp(a.results,b.results,sizeof(a.results)),"active feedback parity");require(!memcmp(a.positions,b.positions,sizeof(a.positions)),"active reference parity");
    Fixture invalid(assets);active[0]=2;const auto prior=invalid.composers[0];
    pre_kernel(&invalid.buffers,invalid.actions,invalid.velocity,invalid.suspended,Fixture::count,invalid.commands,invalid.enabled,invalid.results,active);
    require(invalid.rows[0].status==400&&!memcmp(&prior,&invalid.composers[0],sizeof(prior)),"invalid activity fails before composer mutation");
}
}
int main(){Assets assets;test_deactivate(assets);test_inactive(assets);test_active_parity(assets);printf("{\"checks\":%d,\"failures\":0,\"gpu_calls\":0}\n",checks);}
