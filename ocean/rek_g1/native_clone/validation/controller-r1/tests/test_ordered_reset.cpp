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

void test_ordered(Assets& assets) {
    for(int route: {0,1,2,7})for(auto order: {REK_G1_RESET_RUNNER_THEN_INPUT,REK_G1_RESET_INPUT_THEN_RUNNER}) {
        Fixture f(assets);
        require(sonic_motion_composer_native_play_action(&f.composers[0],&assets.routes[route].clip,&assets.routes[route].config)==0,"set prior route");
        f.composers[0].current_layer.cursor=2.75f;
        f.composers[0].current_layer.speed=0.625f;
        f.composers[0].pending_heading_delta=0.3f;
        f.rows[0].direct_native_active=1;
        f.rows[0].locomotion.current_route_id=static_cast<RekG1NativeRouteId>(1);
        f.rows[0].locomotion.last_driven_command={0.25f,-0.5f,0.75f};
        f.rows[0].locomotion.stop_brake_command={-0.25f,0.5f,-0.75f};
        f.rows[0].locomotion.locomotion_active=1;
        f.rows[0].locomotion.has_momentum=1;
        f.rows[0].effective_velocity={0.125f,0.25f,-0.5f};
        auto expected=f.composers[0];
        if(order==REK_G1_RESET_INPUT_THEN_RUNNER)
            require(sonic_motion_composer_native_play_action(&expected,&assets.routes[0].clip,&assets.routes[0].config)==0,"original Input PlayIdle");
        require(sonic_motion_composer_native_reset(&expected)==0,"original Runner Reset");
        float q[29],ref[4],head[4],base[8]={0.8f,0,0,0.6f,1,0,0,0};
        require(sonic_motion_composer_native_reference_frame(&expected,0,&assets.mirror,q,ref)==0,"reference at runner callback");
        require(rek_g1_initial_heading(&expected.backends,base,ref,head),"runner InitHeading");
        if(order==REK_G1_RESET_RUNNER_THEN_INPUT)
            require(sonic_motion_composer_native_play_action(&expected,&assets.routes[0].clip,&assets.routes[0].config)==0,"original Input PlayIdle after runner");
        f.flags[1]=0;const auto other=f.composers[1];const auto otherrow=f.rows[1];
        reset_ordered_kernel(&f.buffers,f.flags,base,order,1,Fixture::count);
        require(f.rows[0].status==0,"ordered reset status");
        require(!memcmp(&expected,&f.composers[0],sizeof(expected)),"ordered reset retains exact callback state");
        require(!memcmp(head,f.heading,sizeof(head)),"heading computed at runner callback");
        require(!memcmp(&other,&f.composers[1],sizeof(other)),"unselected composer untouched");
        require(!memcmp(&otherrow,&f.rows[1],sizeof(otherrow)),"unselected scheduler untouched");
        require(f.rows[0].direct_native_active==1,"body reset preserves dispatch mode");
        require(f.rows[0].locomotion.current_route_id==static_cast<RekG1NativeRouteId>(1),"body reset retains passive route id");
        require(f.rows[0].locomotion.last_driven_command.forward==0.25f && f.rows[0].locomotion.stop_brake_command.forward==-0.25f,"body reset retains last and brake commands");
        require(!f.rows[0].locomotion.locomotion_active && !f.rows[0].locomotion.transition_settling && !f.rows[0].locomotion.stop_braking && !f.rows[0].locomotion.has_momentum,"four native reset flags cleared");
        require(f.rows[0].effective_velocity.yaw==-0.5f,"runner VelocityCommand retained");
        if(order==REK_G1_RESET_RUNNER_THEN_INPUT){
            require(f.composers[0].from_layer.active,"ordinary idle crossfade retained");
            require(f.composers[0].from_layer.clip.dof_position_mujoco==assets.routes[route].clip.dof_position_mujoco,"prior clip retained");
            require(f.composers[0].from_layer.cursor==0,"prior clip reset cursor retained");
            require(f.composers[0].from_layer.speed==0.625f,"prior speed retained");
        }else require(!f.composers[0].from_layer.active,"runner last clears outgoing active only");
        reset_ordered_kernel(&f.buffers,f.flags,base,order,0,Fixture::count);
        require(f.rows[0].direct_native_active==0,"explicit mode-clear supported");
    }
    Fixture bad(assets);auto previous=bad.composers[0];bad.flags[1]=0;
    reset_ordered_kernel(&bad.buffers,bad.flags,bad.heading,static_cast<RekG1CudaResetOrder>(0),1,Fixture::count);
    require(bad.rows[0].status==400,"invalid order fails closed");
    require(!memcmp(&previous,&bad.composers[0],sizeof(previous)),"invalid order preserves composer");
    for(int i=0;i<33;++i)require(bad.masks[i]==0,"invalid order no action mask");
    Fixture nonfinite(assets);previous=nonfinite.composers[0];float base[8]={NAN,0,0,0,1,0,0,0};
    reset_ordered_kernel(&nonfinite.buffers,nonfinite.flags,base,REK_G1_RESET_RUNNER_THEN_INPUT,1,Fixture::count);
    require(nonfinite.rows[0].status==311,"nonfinite base rejected");
    require(!memcmp(&previous,&nonfinite.composers[0],sizeof(previous)),"failed heading preserves composer");
}
}
int main(){Assets assets;test_ordered(assets);printf("{\"checks\":%d,\"failures\":0,\"gpu_calls\":0}\n",checks);}
