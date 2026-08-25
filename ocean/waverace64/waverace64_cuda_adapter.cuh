#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "waverace64_recomp_device.cuh"

// Device translation of the state adapter in waverace64.h. The generated
// cartridge code owns the physics transition. This file owns the Puffer-facing
// action, observation, reward, terminal, curriculum, and log contract.

#define WR64_CUDA_OBS_SIZE 57
#define WR64_CUDA_NUM_ATNS 5
#define WR64_CUDA_MAX_STEPS 14400
#define WR64_CUDA_GAME_UPDATE_HZ 20u
#define WR64_CUDA_SPEED_SCALE 55.555557f
#define WR64_CUDA_MAX_COURSE_NODES 64
#define WR64_CUDA_MAX_WAVE_VARIANTS 128u

#define WR64_CUDA_RDRAM_SIZE 0x800000u
#define WR64_CUDA_RDRAM_MASK 0x1FFFFFFFu

#define WR64_CUDA_STATE_RACING 0x28u
#define WR64_CUDA_ADDR_GAMESTATE 0x800DAB24u
#define WR64_CUDA_ADDR_RACE_READY 0x801CE638u
#define WR64_CUDA_ADDR_MODE_STATE 0x801CE650u
#define WR64_CUDA_ADDR_COURSE_ID 0x800D8170u
#define WR64_CUDA_ADDR_GAME_MODE 0x801CE620u
#define WR64_CUDA_ADDR_PLAYERS 0x800DAB28u
#define WR64_CUDA_ADDR_RIDERS 0x801982F0u
#define WR64_CUDA_ADDR_ACTIVE_RIDER 0x800D48DCu
#define WR64_CUDA_COURSE_SUNNY_BEACH 1u
#define WR64_CUDA_MODE_TIME_TRIALS 0u

#define WR64_CUDA_PHYSICS_BASE 0x80192690u
#define WR64_CUDA_PHYSICS_POS 0x0044u
#define WR64_CUDA_PHYSICS_SPEED 0x0B90u
#define WR64_CUDA_PHYSICS_BASIS_0_X 0x006Cu
#define WR64_CUDA_PHYSICS_BASIS_0_Y 0x0070u
#define WR64_CUDA_PHYSICS_BASIS_0_Z 0x0074u
#define WR64_CUDA_PHYSICS_BASIS_1_X 0x0078u
#define WR64_CUDA_PHYSICS_BASIS_1_Y 0x007Cu
#define WR64_CUDA_PHYSICS_BASIS_1_Z 0x0080u
#define WR64_CUDA_PHYSICS_BASIS_2_X 0x0084u
#define WR64_CUDA_PHYSICS_BASIS_2_Y 0x0088u
#define WR64_CUDA_PHYSICS_BASIS_2_Z 0x008Cu
#define WR64_CUDA_PHYSICS_FORWARD_X 0x1434u
#define WR64_CUDA_PHYSICS_FORWARD_Z 0x1438u
#define WR64_CUDA_PHYSICS_RECOVERY 0x1608u
#define WR64_CUDA_PHYSICS_STATE 0x0C54u
#define WR64_CUDA_PHYSICS_STATE_FRAME 0x0C58u

#define WR64_CUDA_RIDERS_BASE 0x801C2938u
#define WR64_CUDA_RIDER_LAP 0x0000u
#define WR64_CUDA_RIDER_NODE 0x000Cu
#define WR64_CUDA_RIDER_POWER 0x012Cu
#define WR64_CUDA_RIDER_MISSES 0x0134u
#define WR64_CUDA_RIDER_DQ 0x013Cu
#define WR64_CUDA_RIDER_LAP_TIME 0x0168u
#define WR64_CUDA_RIDER_LAP_SPLIT_1 0x0178u
#define WR64_CUDA_RIDER_TOTAL_TIME 0x019Cu
#define WR64_CUDA_RIDER_ENDED 0x02ECu
#define WR64_CUDA_RIDER_FINISHED 0x02F4u

#define WR64_CUDA_CONFIG_LAPS_ADDR 0x801CE618u
#define WR64_CUDA_TARGET_LAPS_ADDR 0x801CE728u

#define WR64_CUDA_COURSE_PRIMARY 0x801AEE20u
#define WR64_CUDA_COURSE_NODE_STRIDE 0x0104u
#define WR64_CUDA_COURSE_NODE_X 0x0000u
#define WR64_CUDA_COURSE_NODE_Z 0x0008u
#define WR64_CUDA_COURSE_NODE_LATERAL_X 0x0078u
#define WR64_CUDA_COURSE_NODE_LATERAL_Z 0x007Cu
#define WR64_CUDA_COURSE_NODE_LENGTH 0x0088u
#define WR64_CUDA_COURSE_NODE_TYPE 0x0098u
#define WR64_CUDA_COURSE_NODE_NEXT 0x00A0u
#define WR64_CUDA_COURSE_NODE_COUNT 0x801BB120u
#define WR64_CUDA_COURSE_START_NODE 0x801BB12Cu

#define WR64_CUDA_WATER_GRID 0x80162420u
#define WR64_CUDA_WATER_LEVEL 0x80192458u
#define WR64_CUDA_WATER_ROWS 384
#define WR64_CUDA_WATER_COLS 128

#define WR64_CUDA_BTN_A 0x8000u
#define WR64_CUDA_BTN_B 0x4000u
#define WR64_CUDA_BTN_Z 0x2000u
#define WR64_CUDA_BTN_R 0x0010u

#define WR64_CUDA_WAVE_TICK_SALT UINT32_C(0xB9DCF3C0)
#define WR64_CUDA_ADAPTER_ERROR_RESET_CONTRACT 0x57410001
#define WR64_CUDA_ADAPTER_ERROR_ROUTE 0x57410002
#define WR64_CUDA_ADAPTER_ERROR_ENV_FAULT 0x57410003

typedef struct Log {
    float perf;
    float score;
    float distance;
    float checkpoints;
    float misses;
    float success_rate;
    float failure_rate;
    float disqualification_rate;
    float safety_timeout_rate;
    float env_fault_rate;
    float mean_speed;
    float episode_return;
    float episode_length;
    float target_laps;
    float three_lap_success_rate;
    float successful_race_time_ms;
    float successful_lap_1_ms;
    float successful_lap_2_ms;
    float successful_lap_3_ms;
    float n;
} Log;
typedef Log WR64CudaLog;

typedef struct WR64CudaState {
    int tick;
    float prev_a;
    float prev_y;
    float prev_b;
    float episode_return;
    float dist_total;
    float progress_total;
    float max_progress;
    float velocity_x;
    float velocity_y;
    float velocity_z;
    float prev_course_progress;
    int32_t prev_node;
    int32_t prev_lap;
    int32_t checkpoints;
    int32_t prev_misses;
    int32_t misses;
    int32_t recovery;
    int32_t success;
    int32_t failed;
    int32_t disqualified;
    int32_t safety_timeout;
    int32_t env_fault;
} WR64CudaState;

typedef struct WR64CudaAdapter {
    // Keep the log first so a GPU Env can embed this structure at byte zero and
    // retain PufferLib's flat parallel-log reduction contract.
    WR64CudaLog log;
    WR64CudaState state;

    int frameskip;
    unsigned int rng;
    int32_t randomize_waves;
    uint32_t wave_seed;
    uint32_t wave_rng_state;
    uint32_t wave_episode;
    uint32_t wave_boot_variant;
    uint32_t active_wave_variant;
    int32_t wave_variants;

    float reward_speed;
    float reward_progress;
    float reward_slip;
    float reward_checkpoint;
    float reward_miss;
    float reward_finish;
    float reward_fail;
    float discount;
    int32_t reward_mode;

    int32_t curriculum_start_laps;
    int32_t curriculum_max_laps;
    int32_t curriculum_successes_per_lap;
    int32_t curriculum_laps;
    int32_t curriculum_successes;

    float vertical_origin;
    float route_arc[WR64_CUDA_MAX_COURSE_NODES];
    int32_t route_pred[WR64_CUDA_MAX_COURSE_NODES];
    float route_total;
    int32_t route_nodes;
    int32_t route_valid;
} WR64CudaAdapter;

typedef struct WR64CudaStepPrelude {
    float potential_before;
    float max_progress_before;
} WR64CudaStepPrelude;

static_assert(WR64_CUDA_OBS_SIZE == 57, "Wave Race CUDA observation contract changed");
static_assert(WR64_CUDA_NUM_ATNS == 5, "Wave Race CUDA action contract changed");
static_assert(offsetof(WR64CudaAdapter, log) == 0,
    "Wave Race CUDA log must remain the first adapter field");
static_assert(sizeof(((WR64CudaAdapter*)0)->route_arc) / sizeof(float)
        == WR64_CUDA_MAX_COURSE_NODES,
    "Wave Race CUDA route cache size changed");

__device__ static inline int wr64_cuda_addr_ok(uint32_t va, uint32_t len) {
    return va >= UINT32_C(0x80000000)
        && ((va & WR64_CUDA_RDRAM_MASK) + len) <= WR64_CUDA_RDRAM_SIZE;
}

__device__ static inline uint32_t wr64_cuda_u(
        const uint8_t* rdram, uint32_t va) {
    if (!wr64_cuda_addr_ok(va, 4)) return 0;
    return *(const uint32_t*)(rdram + (va & WR64_CUDA_RDRAM_MASK));
}

__device__ static inline uint16_t wr64_cuda_h(
        const uint8_t* rdram, uint32_t va) {
    if (!wr64_cuda_addr_ok(va, 2)) return 0;
    return *(const uint16_t*)(rdram + ((va & WR64_CUDA_RDRAM_MASK) ^ 2u));
}

__device__ static inline void wr64_cuda_w(
        uint8_t* rdram, uint32_t va, uint32_t value) {
    if (!wr64_cuda_addr_ok(va, 4)) return;
    *(uint32_t*)(rdram + (va & WR64_CUDA_RDRAM_MASK)) = value;
}

__device__ static inline float wr64_cuda_f(
        const uint8_t* rdram, uint32_t va) {
    return __uint_as_float(wr64_cuda_u(rdram, va));
}

__device__ static inline int32_t wr64_cuda_i32_bits(uint32_t value) {
    union {
        uint32_t u;
        int32_t i;
    } bits;
    bits.u = value;
    return bits.i;
}

__device__ static inline float wr64_cuda_f32_bits(uint32_t value) {
    return __uint_as_float(value);
}

__device__ static inline int32_t wr64_cuda_add32(int32_t a, int32_t b) {
    return wr64_cuda_i32_bits((uint32_t)a + (uint32_t)b);
}

__device__ static inline int32_t wr64_cuda_sub32(int32_t a, int32_t b) {
    return wr64_cuda_i32_bits((uint32_t)a - (uint32_t)b);
}

__device__ static inline int32_t wr64_cuda_mul32(int32_t a, int32_t b) {
    return wr64_cuda_i32_bits((uint32_t)a * (uint32_t)b);
}

__device__ static inline int32_t wr64_cuda_shl32(int32_t value, unsigned shift) {
    return wr64_cuda_i32_bits((uint32_t)value << shift);
}

__device__ static inline int32_t wr64_cuda_asr6(int32_t value) {
    return value >= 0 ? value / 64 : -(((-value) + 63) / 64);
}

__device__ static inline int32_t wr64_cuda_asr8_s16(int16_t value) {
    int32_t wide = value;
    return wide >= 0 ? wide / 256 : -(((-wide) + 255) / 256);
}

__device__ static inline int32_t wr64_cuda_water_q(
        const uint8_t* rdram, int32_t row, int32_t col) {
    uint32_t index = (uint32_t)row * WR64_CUDA_WATER_COLS + (uint32_t)col;
    int16_t raw = (int16_t)wr64_cuda_h(
        rdram, WR64_CUDA_WATER_GRID + 4u * index);
    return wr64_cuda_asr8_s16(raw);
}

// Translation of func_8004D30C. Compile the including CUDA translation unit
// with --fmad=false --ftz=false --prec-div=true --prec-sqrt=true.
__device__ static inline float wr64_cuda_water_height(
        const uint8_t* rdram, float x, float z) {
    const float k0 = wr64_cuda_f32_bits(UINT32_C(0x3F93CD3A));
    const float k1 = wr64_cuda_f32_bits(UINT32_C(0x3F13CD3A));
    volatile float vzf = k0 * z;
    int32_t v = ((int32_t)vzf) % 24576;
    int32_t j = wr64_cuda_asr6(v);
    volatile float u0 = k1 * z;
    volatile float uf = u0 + x;
    int32_t u = ((int32_t)uf) % 24576;
    int32_t i = wr64_cuda_asr6(u);
    int32_t h0 = wr64_cuda_water_q(rdram,
        (j + (i & -128) + 1536) % WR64_CUDA_WATER_ROWS, i & 127);
    int32_t fv = wr64_cuda_sub32(wr64_cuda_shl32(j, 6), v);
    int32_t fu = wr64_cuda_sub32(wr64_cuda_shl32(i, 6), u);
    int32_t sx;
    int32_t sz;
    if (fv < fu) {
        int32_t h1 = wr64_cuda_water_q(rdram,
            (j + (i & -128) + 1537) % WR64_CUDA_WATER_ROWS, i & 127);
        int32_t ip = wr64_cuda_add32(i, 1);
        int32_t h2 = wr64_cuda_water_q(rdram,
            (j + (ip & -128) + 1537) % WR64_CUDA_WATER_ROWS, ip & 127);
        sx = wr64_cuda_sub32(h1, h2);
        sz = wr64_cuda_sub32(h0, h1);
    } else {
        int32_t ip = wr64_cuda_add32(i, 1);
        int32_t h1 = wr64_cuda_water_q(rdram,
            (j + (ip & -128) + 1536) % WR64_CUDA_WATER_ROWS, ip & 127);
        int32_t h2 = wr64_cuda_water_q(rdram,
            (j + (ip & -128) + 1537) % WR64_CUDA_WATER_ROWS, ip & 127);
        sx = wr64_cuda_sub32(h0, h1);
        sz = wr64_cuda_sub32(h1, h2);
    }
    int32_t denominator = wr64_cuda_add32(
        wr64_cuda_add32(wr64_cuda_mul32(sx, sx), wr64_cuda_mul32(sz, sz)),
        4096);
    denominator = wr64_cuda_shl32(denominator, 12);
    volatile float denominator_f = (float)denominator;
    volatile float root = sqrtf(denominator_f);
    int32_t level = (int32_t)wr64_cuda_u(rdram, WR64_CUDA_WATER_LEVEL);
    int32_t numerator = wr64_cuda_add32(wr64_cuda_mul32(sx, fu),
        wr64_cuda_shl32(wr64_cuda_add32(level, h0), 12));
    numerator = wr64_cuda_add32(numerator, wr64_cuda_mul32(sz, fv));
    volatile float numerator_f = (float)numerator;
    volatile float result = numerator_f / root;
    return result;
}

__device__ static inline int32_t wr64_cuda_active_rider(
        const uint8_t* rdram) {
    int32_t rider = (int32_t)wr64_cuda_u(rdram, WR64_CUDA_ADDR_ACTIVE_RIDER);
    int32_t riders = (int32_t)wr64_cuda_u(rdram, WR64_CUDA_ADDR_RIDERS);
    return riders == 1 && rider == 0 ? 0 : -1;
}

__device__ static inline uint32_t wr64_cuda_rider_addr(
        const uint8_t* rdram, uint32_t offset) {
    return wr64_cuda_active_rider(rdram) == 0
        ? WR64_CUDA_RIDERS_BASE + offset : 0;
}

__device__ static inline uint32_t wr64_cuda_physics_addr(
        const uint8_t* rdram, uint32_t offset) {
    return wr64_cuda_active_rider(rdram) == 0
        ? WR64_CUDA_PHYSICS_BASE + offset : 0;
}

__device__ static inline int32_t wr64_cuda_lap(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_LAP));
}

__device__ static inline int32_t wr64_cuda_node(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_NODE));
}

__device__ static inline int32_t wr64_cuda_misses(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_MISSES));
}

__device__ static inline int32_t wr64_cuda_power(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_POWER));
}

__device__ static inline int32_t wr64_cuda_lap_time_ms(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_LAP_TIME));
}

__device__ static inline int32_t wr64_cuda_lap_split_ms(
        const uint8_t* rdram, int lap) {
    if (lap < 0 || lap >= 3) return 0;
    return (int32_t)wr64_cuda_u(rdram, wr64_cuda_rider_addr(
        rdram, WR64_CUDA_RIDER_LAP_SPLIT_1 + 4u * (uint32_t)lap));
}

__device__ static inline int32_t wr64_cuda_race_time_ms(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_TOTAL_TIME));
}

__device__ static inline float wr64_cuda_physics_speed(const uint8_t* rdram) {
    return wr64_cuda_f(
        rdram, wr64_cuda_physics_addr(rdram, WR64_CUDA_PHYSICS_SPEED));
}

__device__ static inline int32_t wr64_cuda_disqualified(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_DQ));
}

__device__ static inline int32_t wr64_cuda_ended(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_ENDED));
}

__device__ static inline int32_t wr64_cuda_finished(const uint8_t* rdram) {
    return (int32_t)wr64_cuda_u(
        rdram, wr64_cuda_rider_addr(rdram, WR64_CUDA_RIDER_FINISHED));
}

__device__ static inline int32_t wr64_cuda_target_laps(const uint8_t* rdram) {
    int32_t laps = (int32_t)wr64_cuda_u(rdram, WR64_CUDA_TARGET_LAPS_ADDR);
    return laps >= 1 && laps <= 16 ? laps : 0;
}

__device__ static inline int32_t wr64_cuda_recovery(const uint8_t* rdram) {
    uint32_t physics = wr64_cuda_physics_addr(rdram, 0);
    int32_t state = (int32_t)wr64_cuda_u(
        rdram, physics + WR64_CUDA_PHYSICS_STATE);
    int32_t frame = (int32_t)wr64_cuda_u(
        rdram, physics + WR64_CUDA_PHYSICS_STATE_FRAME);
    if (wr64_cuda_h(rdram, physics + WR64_CUDA_PHYSICS_RECOVERY) != 0
            || state == 23 || (state == 7 && frame < 56)) return 2;
    return state == 24 ? 1 : 0;
}

__device__ static inline uint32_t wr64_cuda_course_count_addr(uint32_t base) {
    return base == WR64_CUDA_COURSE_PRIMARY
        ? WR64_CUDA_COURSE_NODE_COUNT : 0;
}

__device__ static inline uint32_t wr64_cuda_course_addr(
        uint32_t base, int32_t node, uint32_t offset) {
    return base + (uint32_t)node * WR64_CUDA_COURSE_NODE_STRIDE + offset;
}

__device__ static inline int32_t wr64_cuda_node_count(
        const uint8_t* rdram, uint32_t base) {
    uint32_t address = wr64_cuda_course_count_addr(base);
    if (address == 0) return 0;
    int32_t count = (int32_t)wr64_cuda_u(rdram, address);
    return count > 0 && count <= WR64_CUDA_MAX_COURSE_NODES ? count : 0;
}

__device__ static inline int32_t wr64_cuda_sanitize_node(
        const uint8_t* rdram, uint32_t base, int32_t node) {
    int32_t count = wr64_cuda_node_count(rdram, base);
    return node >= 0 && node < count ? node : -1;
}

__device__ static inline int wr64_cuda_race_identity_valid(
        const uint8_t* rdram) {
    return wr64_cuda_active_rider(rdram) == 0
        && wr64_cuda_sanitize_node(
            rdram, WR64_CUDA_COURSE_PRIMARY, wr64_cuda_node(rdram)) >= 0;
}

__device__ static inline int wr64_cuda_reset_contract_valid(
        const uint8_t* rdram, int32_t target_laps) {
    return wr64_cuda_race_identity_valid(rdram)
        && wr64_cuda_u(rdram, WR64_CUDA_ADDR_GAMESTATE)
            == WR64_CUDA_STATE_RACING
        && wr64_cuda_u(rdram, WR64_CUDA_ADDR_RACE_READY) == 1
        && wr64_cuda_u(rdram, WR64_CUDA_ADDR_MODE_STATE) == 2
        && wr64_cuda_u(rdram, WR64_CUDA_ADDR_COURSE_ID)
            == WR64_CUDA_COURSE_SUNNY_BEACH
        && wr64_cuda_u(rdram, WR64_CUDA_ADDR_GAME_MODE)
            == WR64_CUDA_MODE_TIME_TRIALS
        && wr64_cuda_u(rdram, WR64_CUDA_ADDR_PLAYERS) == 1
        && wr64_cuda_u(rdram, WR64_CUDA_ADDR_RIDERS) == 1
        && wr64_cuda_target_laps(rdram) == target_laps
        && wr64_cuda_lap_time_ms(rdram) == 0
        && wr64_cuda_race_time_ms(rdram) == 0;
}

__device__ static inline int32_t wr64_cuda_node_type(
        const uint8_t* rdram, int32_t node) {
    node = wr64_cuda_sanitize_node(rdram, WR64_CUDA_COURSE_PRIMARY, node);
    return node < 0 ? -1 : (int32_t)wr64_cuda_u(rdram,
        wr64_cuda_course_addr(
            WR64_CUDA_COURSE_PRIMARY, node, WR64_CUDA_COURSE_NODE_TYPE));
}

__device__ static inline float wr64_cuda_buoy_side(
        const uint8_t* rdram, int32_t node) {
    int32_t type = wr64_cuda_node_type(rdram, node);
    return type == 0 ? -1.f : (type == 1 ? 1.f : 0.f);
}

__device__ static inline int32_t wr64_cuda_next_node(
        const uint8_t* rdram, uint32_t base, int32_t node) {
    node = wr64_cuda_sanitize_node(rdram, base, node);
    if (node < 0) return -1;
    return wr64_cuda_sanitize_node(rdram, base,
        (int32_t)wr64_cuda_u(rdram, wr64_cuda_course_addr(
            base, node, WR64_CUDA_COURSE_NODE_NEXT)));
}

__device__ static inline void wr64_cuda_target(const uint8_t* rdram,
        float* x, float* z, int32_t* node_out, uint32_t* course_out) {
    uint32_t base = WR64_CUDA_COURSE_PRIMARY;
    int32_t node = wr64_cuda_sanitize_node(
        rdram, base, wr64_cuda_node(rdram));
    if (node < 0) {
        *x = wr64_cuda_f(rdram,
            wr64_cuda_physics_addr(rdram, WR64_CUDA_PHYSICS_POS));
        *z = wr64_cuda_f(rdram,
            wr64_cuda_physics_addr(rdram, WR64_CUDA_PHYSICS_POS + 8));
        *node_out = -1;
        *course_out = 0;
        return;
    }
    *x = wr64_cuda_f(rdram,
        wr64_cuda_course_addr(base, node, WR64_CUDA_COURSE_NODE_X));
    *z = wr64_cuda_f(rdram,
        wr64_cuda_course_addr(base, node, WR64_CUDA_COURSE_NODE_Z));
    *node_out = node;
    *course_out = base;
}

__device__ static inline int wr64_cuda_pass_point(const uint8_t* rdram,
        int32_t node, float* x, float* z) {
    node = wr64_cuda_sanitize_node(rdram, WR64_CUDA_COURSE_PRIMARY, node);
    if (node < 0) return 0;
    uint32_t address = wr64_cuda_course_addr(
        WR64_CUDA_COURSE_PRIMARY, node, 0);
    *x = wr64_cuda_f(rdram, address + WR64_CUDA_COURSE_NODE_X);
    *z = wr64_cuda_f(rdram, address + WR64_CUDA_COURSE_NODE_Z);
    int32_t type = wr64_cuda_node_type(rdram, node);
    if (type == 0 || type == 1) {
        float side = type == 0 ? -400.f : 400.f;
        *x += side * wr64_cuda_f(
            rdram, address + WR64_CUDA_COURSE_NODE_LATERAL_X);
        *z += side * wr64_cuda_f(
            rdram, address + WR64_CUDA_COURSE_NODE_LATERAL_Z);
    }
    return isfinite(*x) && isfinite(*z);
}

__device__ static inline void wr64_cuda_position(
        const uint8_t* rdram, float* x, float* y, float* z) {
    uint32_t pos = wr64_cuda_physics_addr(rdram, WR64_CUDA_PHYSICS_POS);
    *x = wr64_cuda_f(rdram, pos);
    *y = wr64_cuda_f(rdram, pos + 4);
    *z = wr64_cuda_f(rdram, pos + 8);
}

__device__ static inline void wr64_cuda_heading(const uint8_t* rdram,
        float vx, float vz, float* hx, float* hz) {
    *hx = wr64_cuda_f(rdram,
        wr64_cuda_physics_addr(rdram, WR64_CUDA_PHYSICS_FORWARD_X));
    *hz = wr64_cuda_f(rdram,
        wr64_cuda_physics_addr(rdram, WR64_CUDA_PHYSICS_FORWARD_Z));
    float norm = sqrtf(*hx * *hx + *hz * *hz);
    if (!isfinite(norm) || norm < 1e-4f) {
        norm = sqrtf(vx * vx + vz * vz);
        *hx = norm > 1e-4f ? vx / norm : 0.f;
        *hz = norm > 1e-4f ? vz / norm : 1.f;
        return;
    }
    *hx /= norm;
    *hz /= norm;
}

__device__ static inline int wr64_cuda_progress_advance(
        const WR64CudaAdapter* adapter, const uint8_t* rdram,
        int32_t from_lap, int32_t from_node, int32_t to_lap, int32_t to_node,
        float* distance_out) {
    (void)adapter;
    uint32_t base = WR64_CUDA_COURSE_PRIMARY;
    int32_t count = wr64_cuda_node_count(rdram, base);
    int32_t lap = from_lap;
    int32_t node = wr64_cuda_sanitize_node(rdram, base, from_node);
    to_node = wr64_cuda_sanitize_node(rdram, base, to_node);
    *distance_out = 0.f;
    if (count == 0 || node < 0 || to_node < 0 || to_lap < from_lap) return 0;
    float distance = 0.f;
    for (int32_t hops = 0; hops <= count; hops++) {
        if (lap == to_lap && node == to_node) {
            *distance_out = distance;
            return hops;
        }
        if (hops == count) break;
        float length = wr64_cuda_f(rdram, wr64_cuda_course_addr(
            base, node, WR64_CUDA_COURSE_NODE_LENGTH));
        if (!isfinite(length) || length < 0.f) return 0;
        distance += length;
        if ((int32_t)wr64_cuda_u(rdram, wr64_cuda_course_addr(
                base, node, WR64_CUDA_COURSE_NODE_TYPE)) == 3) {
            lap++;
        }
        node = wr64_cuda_next_node(rdram, base, node);
        if (node < 0) return 0;
    }
    return 0;
}

__device__ static inline int wr64_cuda_build_route(
        WR64CudaAdapter* adapter, const uint8_t* rdram) {
    uint32_t base = WR64_CUDA_COURSE_PRIMARY;
    int32_t count = wr64_cuda_node_count(rdram, base);
    int32_t start = wr64_cuda_sanitize_node(rdram, base,
        (int32_t)wr64_cuda_u(rdram, WR64_CUDA_COURSE_START_NODE));
    for (int32_t i = 0; i < WR64_CUDA_MAX_COURSE_NODES; i++) {
        adapter->route_arc[i] = NAN;
        adapter->route_pred[i] = -1;
    }
    adapter->route_total = 0.f;
    adapter->route_nodes = 0;
    adapter->route_valid = 0;
    if (count < 2 || start < 0) return 0;
    uint8_t seen[WR64_CUDA_MAX_COURSE_NODES] = {0};
    int32_t node = start;
    float total = 0.f;
    for (int32_t hop = 0; hop < count; hop++) {
        if (node < 0 || node >= count || seen[node]) return 0;
        seen[node] = 1;
        int32_t next = wr64_cuda_next_node(rdram, base, node);
        float length = wr64_cuda_f(rdram, wr64_cuda_course_addr(
            base, node, WR64_CUDA_COURSE_NODE_LENGTH));
        if (next < 0 || !isfinite(length) || length <= 0.f
                || adapter->route_pred[next] != -1) return 0;
        adapter->route_arc[node] = total;
        adapter->route_pred[next] = node;
        total += length;
        adapter->route_nodes++;
        node = next;
        if (node == start) break;
    }
    if (node != start || adapter->route_nodes < 2
            || !isfinite(total) || total <= 0.f) return 0;
    adapter->route_total = total;
    adapter->route_valid = 1;
    return 1;
}

__device__ static inline int wr64_cuda_course_progress(
        const WR64CudaAdapter* adapter, const uint8_t* rdram,
        float* absolute_out, float* fraction_out) {
    if (!adapter->route_valid || !isfinite(adapter->route_total)
            || adapter->route_total <= 0.f) return 0;
    uint32_t base = WR64_CUDA_COURSE_PRIMARY;
    int32_t target = wr64_cuda_sanitize_node(
        rdram, base, wr64_cuda_node(rdram));
    if (target < 0) return 0;
    int32_t pred = adapter->route_pred[target];
    if (pred < 0 || pred >= WR64_CUDA_MAX_COURSE_NODES
            || !isfinite(adapter->route_arc[pred])) return 0;
    float x, y, z;
    wr64_cuda_position(rdram, &x, &y, &z);
    (void)y;
    float ax = wr64_cuda_f(rdram,
        wr64_cuda_course_addr(base, pred, WR64_CUDA_COURSE_NODE_X));
    float az = wr64_cuda_f(rdram,
        wr64_cuda_course_addr(base, pred, WR64_CUDA_COURSE_NODE_Z));
    float bx = wr64_cuda_f(rdram,
        wr64_cuda_course_addr(base, target, WR64_CUDA_COURSE_NODE_X));
    float bz = wr64_cuda_f(rdram,
        wr64_cuda_course_addr(base, target, WR64_CUDA_COURSE_NODE_Z));
    float dx = bx - ax;
    float dz = bz - az;
    float norm2 = dx * dx + dz * dz;
    float length = wr64_cuda_f(rdram,
        wr64_cuda_course_addr(base, pred, WR64_CUDA_COURSE_NODE_LENGTH));
    if (!isfinite(x) || !isfinite(z) || !isfinite(norm2) || norm2 <= 1e-6f
            || !isfinite(length) || length <= 0.f) return 0;
    float u = ((x - ax) * dx + (z - az) * dz) / norm2;
    if (u < 0.f) u = 0.f;
    if (u > 1.f) u = 1.f;
    float arc = adapter->route_arc[pred] + u * length;
    int32_t lap = wr64_cuda_lap(rdram);
    if (lap < 0 || !isfinite(arc)) return 0;
    *absolute_out = (float)lap * adapter->route_total + arc;
    *fraction_out = arc / adapter->route_total;
    return isfinite(*absolute_out) && isfinite(*fraction_out);
}

__device__ static inline float wr64_cuda_route_fraction(
        const WR64CudaAdapter* adapter, const uint8_t* rdram) {
    float absolute = 0.f;
    float fraction = 0.f;
    return wr64_cuda_course_progress(
        adapter, rdram, &absolute, &fraction) ? fraction : 0.f;
}

__device__ static inline void wr64_cuda_local_target_features(
        float x, float z, float hx, float hz, float target_x, float target_z,
        float route_total, float* forward, float* lateral,
        float* distance_fraction) {
    float dx = target_x - x;
    float dz = target_z - z;
    float distance = hypotf(dx, dz);
    float inverse = isfinite(distance) && distance > 1e-4f
        ? 1.f / distance : 0.f;
    *forward = (hx * dx + hz * dz) * inverse;
    *lateral = (hz * dx - hx * dz) * inverse;
    *distance_fraction = route_total > 0.f ? distance / route_total : 0.f;
    if (*distance_fraction < 0.f) *distance_fraction = 0.f;
    if (*distance_fraction > 1.f) *distance_fraction = 1.f;
}

template<typename ObsT>
__device__ static inline void wr64_cuda_compute_observations(
        const WR64CudaAdapter* adapter, const WR64DeviceMachine* machine,
        const uint8_t* rdram, ObsT* observations) {
    float o[WR64_CUDA_OBS_SIZE];
    float x, y, z;
    wr64_cuda_position(rdram, &x, &y, &z);
    float vx = adapter->state.velocity_x;
    float vz = adapter->state.velocity_z;
    float sp = sqrtf(vx * vx + vz * vz);
    float inv = sp > 1e-4f ? 1.f / sp : 0.f;
    float hx, hz;
    wr64_cuda_heading(rdram, vx, vz, &hx, &hz);
    float target_x, target_z;
    int32_t target_node;
    uint32_t course;
    wr64_cuda_target(rdram, &target_x, &target_z, &target_node, &course);
    (void)course;
    float gdx = target_x - x;
    float gdz = target_z - z;
    float gd = sqrtf(gdx * gdx + gdz * gdz);
    float gate_inv = isfinite(gd) && gd > 1e-4f ? 1.f / gd : 0.f;
    float gate_fraction = adapter->route_total > 0.f
        ? gd / adapter->route_total : 0.f;
    if (gate_fraction < 0.f) gate_fraction = 0.f;
    if (gate_fraction > 1.f) gate_fraction = 1.f;
    int32_t target_laps = wr64_cuda_target_laps(rdram);
    float lap_fraction = target_laps > 0
        ? (float)wr64_cuda_lap(rdram) / (float)target_laps : 0.f;
    if (lap_fraction < 0.f) lap_fraction = 0.f;
    if (lap_fraction > 1.f) lap_fraction = 1.f;

    float pass_x = target_x;
    float pass_z = target_z;
    (void)wr64_cuda_pass_point(rdram, target_node, &pass_x, &pass_z);
    float pass_forward, pass_lateral, pass_distance;
    wr64_cuda_local_target_features(x, z, hx, hz, pass_x, pass_z,
        adapter->route_total, &pass_forward, &pass_lateral, &pass_distance);

    int32_t next_node = wr64_cuda_next_node(
        rdram, WR64_CUDA_COURSE_PRIMARY, target_node);
    float next_x = pass_x;
    float next_z = pass_z;
    (void)wr64_cuda_pass_point(rdram, next_node, &next_x, &next_z);
    float next_forward, next_lateral, next_distance;
    wr64_cuda_local_target_features(x, z, hx, hz, next_x, next_z,
        adapter->route_total, &next_forward, &next_lateral, &next_distance);
    int32_t node_type = wr64_cuda_node_type(rdram, target_node);

    o[0] = x * 1e-3f;
    o[1] = z * 1e-3f;
    o[2] = vx / WR64_CUDA_SPEED_SCALE;
    o[3] = vz / WR64_CUDA_SPEED_SCALE;
    o[4] = hx;
    o[5] = hz;
    o[6] = (y - adapter->vertical_origin) * 0.01f;
    o[7] = sp > 1e-4f ? (hx * vz - hz * vx) * inv : 0.f;
    o[8] = sp / WR64_CUDA_SPEED_SCALE;
    o[9] = (machine->pad_buttons & WR64_CUDA_BTN_A) ? 1.f : 0.f;
    o[10] = (machine->pad_buttons & WR64_CUDA_BTN_B) ? 1.f : 0.f;
    o[11] = (machine->pad_buttons & WR64_CUDA_BTN_Z) ? 1.f : 0.f;
    o[12] = (machine->pad_buttons & WR64_CUDA_BTN_R) ? 1.f : 0.f;
    o[13] = machine->pad_stick_x * (1.f / 80.f);
    o[14] = machine->pad_stick_y * (1.f / 80.f);
    o[15] = (float)adapter->state.tick / (float)WR64_CUDA_MAX_STEPS;
    o[16] = wr64_cuda_buoy_side(rdram, target_node);
    o[17] = (hx * gdx + hz * gdz) * gate_inv;
    o[18] = (hz * gdx - hx * gdz) * gate_inv;
    o[19] = gate_fraction;
    o[20] = wr64_cuda_route_fraction(adapter, rdram);
    o[21] = (float)wr64_cuda_misses(rdram) * 0.2f;
    o[22] = lap_fraction;
    o[23] = adapter->route_total > 0.f && target_laps > 0
        ? adapter->state.progress_total
            / (adapter->route_total * (float)target_laps)
        : 0.f;
    o[24] = pass_forward;
    o[25] = pass_lateral;
    o[26] = pass_distance;
    o[27] = next_forward;
    o[28] = next_lateral;
    o[29] = next_distance;
    o[30] = node_type == 4 ? 1.f : 0.f;
    o[31] = 0.5f * (float)wr64_cuda_recovery(rdram);
    o[32] = adapter->route_nodes > 0 && target_laps > 0
        ? (float)adapter->state.checkpoints
            / ((float)adapter->route_nodes * (float)target_laps)
        : 0.f;
    uint32_t physics = wr64_cuda_physics_addr(rdram, 0);
    o[33] = adapter->state.velocity_y / WR64_CUDA_SPEED_SCALE;
    o[34] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_0_X);
    o[35] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_0_Y);
    o[36] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_0_Z);
    o[37] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_1_X);
    o[38] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_1_Y);
    o[39] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_1_Z);
    o[40] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_2_X);
    o[41] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_2_Y);
    o[42] = wr64_cuda_f(rdram, physics + WR64_CUDA_PHYSICS_BASIS_2_Z);

    const float lateral_offsets[3] = {-96.f, 0.f, 96.f};
    const float forward_offsets[4] = {-64.f, 64.f, 192.f, 384.f};
    int water_index = 43;
    for (int forward_i = 0; forward_i < 4; forward_i++) {
        for (int lateral_i = 0; lateral_i < 3; lateral_i++) {
            float forward = forward_offsets[forward_i];
            float lateral = lateral_offsets[lateral_i];
            float sample_x = x + forward * hx + lateral * hz;
            float sample_z = z + forward * hz - lateral * hx;
            o[water_index++] =
                (wr64_cuda_water_height(rdram, sample_x, sample_z) - y) * 0.01f;
        }
    }
    o[55] = wr64_cuda_physics_speed(rdram) / WR64_CUDA_SPEED_SCALE;
    o[56] = (float)wr64_cuda_power(rdram) * 0.2f;
    for (int i = 0; i < WR64_CUDA_OBS_SIZE; i++) {
        if (!isfinite(o[i])) o[i] = 0.f;
        observations[i] = (ObsT)o[i];
    }
}

__device__ static inline float wr64_cuda_reward_potential(
        const WR64CudaAdapter* adapter) {
    float lap_progress = adapter->route_total > 0.f
        ? adapter->state.progress_total / adapter->route_total : 0.f;
    float potential = adapter->reward_progress * lap_progress
        + adapter->reward_checkpoint * (float)adapter->state.checkpoints;
    return isfinite(potential) ? potential : 0.f;
}

__host__ __device__ static inline uint32_t wr64_cuda_wave_mix32(uint32_t value) {
    value ^= value >> 16;
    value *= UINT32_C(0x7FEB352D);
    value ^= value >> 15;
    value *= UINT32_C(0x846CA68B);
    value ^= value >> 16;
    return value;
}

__host__ __device__ static inline uint32_t wr64_cuda_wave_stream_seed(
        uint32_t base_seed, uint32_t env_index) {
    return wr64_cuda_wave_mix32(base_seed
        ^ wr64_cuda_wave_mix32(env_index + UINT32_C(0xD1B54A35)));
}

__device__ static inline uint32_t wr64_cuda_wave_next_variant(
        WR64CudaAdapter* adapter) {
    uint32_t mask = (uint32_t)adapter->wave_variants - 1u;
    uint32_t offset = (wr64_cuda_wave_mix32(adapter->wave_seed)
        + (uint32_t)adapter->rng) & mask;
    uint32_t stride = ((adapter->wave_rng_state >> 16) | 1u) & mask;
    if (stride == 0) stride = 1;
    uint32_t index = (offset + adapter->wave_episode * stride) & mask;
    adapter->wave_episode++;
    return index;
}

__device__ static inline int8_t wr64_cuda_stick_x(int index) {
    switch (index) {
        case 0: return -80; case 1: return -68; case 2: return -56;
        case 3: return -44; case 4: return -32; case 5: return -20;
        case 6: return -10; case 7: return 0; case 8: return 10;
        case 9: return 20; case 10: return 32; case 11: return 44;
        case 12: return 56; case 13: return 68; default: return 80;
    }
}

__device__ static inline int8_t wr64_cuda_stick_y(int index) {
    switch (index) {
        case 0: return -80; case 1: return -56; case 2: return -32;
        case 3: return -12; case 4: return 0; case 5: return 12;
        case 6: return 32; case 7: return 56; default: return 80;
    }
}

__device__ static inline WR64CudaStepPrelude wr64_cuda_adapter_begin_step(
        WR64CudaAdapter* adapter, WR64DeviceMachine* machine,
        const float* actions) {
    WR64CudaStepPrelude prelude;
    prelude.potential_before = wr64_cuda_reward_potential(adapter);
    prelude.max_progress_before = adapter->state.max_progress;
    int ax = (int)actions[0];
    ax = ax < 0 ? 0 : (ax > 14 ? 14 : ax);
    int ay = (int)actions[1];
    ay = ay < 0 ? 0 : (ay > 8 ? 8 : ay);
    machine->pad_stick_x = wr64_cuda_stick_x(ax);
    machine->pad_stick_y = wr64_cuda_stick_y(ay);
    machine->pad_buttons =
        (((int)actions[2] & 1) ? WR64_CUDA_BTN_A : 0)
        | (((int)actions[3] & 1) ? WR64_CUDA_BTN_B : 0)
        | (((int)actions[4] & 1) ? WR64_CUDA_BTN_R : 0);
    return prelude;
}

__device__ static inline void wr64_cuda_record_curriculum_success(
        WR64CudaAdapter* adapter) {
    if (adapter->curriculum_laps >= adapter->curriculum_max_laps) return;
    adapter->curriculum_successes++;
    if (adapter->curriculum_successes
            >= adapter->curriculum_successes_per_lap) {
        adapter->curriculum_laps++;
        adapter->curriculum_successes = 0;
    }
}

// Mirrors puf_eval_reset bookkeeping. The backend restores the canonical race
// bytes or selected authentic wave variant, then calls reset_after_restore.
__device__ static inline void wr64_cuda_adapter_prepare_eval_reset(
        WR64CudaAdapter* adapter) {
    adapter->curriculum_laps = 3;
    adapter->curriculum_successes = 0;
    adapter->wave_episode = 0;
}

__device__ static inline void wr64_cuda_add_log(
        WR64CudaAdapter* adapter, const uint8_t* rdram) {
    WR64CudaState* state = &adapter->state;
    WR64CudaLog* log = &adapter->log;
    log->episode_length += state->tick;
    log->episode_return += state->episode_return;
    log->distance += state->dist_total;
    log->mean_speed += state->tick
        ? (float)WR64_CUDA_GAME_UPDATE_HZ * state->dist_total / state->tick
        : 0.f;
    log->score += state->progress_total;
    log->checkpoints += (float)state->checkpoints;
    log->misses += (float)state->misses;
    log->success_rate += state->success ? 1.f : 0.f;
    log->failure_rate += state->failed
        && !state->disqualified && !state->safety_timeout ? 1.f : 0.f;
    log->disqualification_rate += state->disqualified ? 1.f : 0.f;
    log->safety_timeout_rate += state->safety_timeout ? 1.f : 0.f;
    log->env_fault_rate += state->env_fault ? 1.f : 0.f;
    int32_t target_laps = wr64_cuda_target_laps(rdram);
    log->target_laps += (float)target_laps;
    log->three_lap_success_rate += state->success && target_laps == 3 ? 1.f : 0.f;
    if (state->success) {
        log->successful_race_time_ms += (float)wr64_cuda_race_time_ms(rdram);
        log->successful_lap_1_ms += (float)wr64_cuda_lap_split_ms(rdram, 0);
        log->successful_lap_2_ms += (float)wr64_cuda_lap_split_ms(rdram, 1);
        log->successful_lap_3_ms += (float)wr64_cuda_lap_split_ms(rdram, 2);
    }
    float perf = adapter->route_total > 0.f && target_laps > 0
        ? state->max_progress / (adapter->route_total * (float)target_laps)
        : 0.f;
    if (perf < 0.f) perf = 0.f;
    if (perf > 1.f) perf = 1.f;
    log->perf += perf;
    log->n += 1;
}

template<typename ObsT>
__device__ static inline int wr64_cuda_adapter_reset_after_restore(
        WR64CudaAdapter* adapter, WR64DeviceMachine* machine,
        uint8_t* rdram, ObsT* observations) {
    if (!wr64_cuda_reset_contract_valid(rdram, 3)) {
        machine->error = WR64_CUDA_ADAPTER_ERROR_RESET_CONTRACT;
        return 0;
    }
    wr64_cuda_w(rdram, WR64_CUDA_CONFIG_LAPS_ADDR,
        (uint32_t)adapter->curriculum_laps);
    wr64_cuda_w(rdram, WR64_CUDA_TARGET_LAPS_ADDR,
        (uint32_t)adapter->curriculum_laps);
    machine->pad_buttons = 0;
    machine->pad_stick_x = 0;
    machine->pad_stick_y = 0;
    uint8_t* state_bytes = (uint8_t*)&adapter->state;
    for (size_t i = 0; i < sizeof(adapter->state); i++) state_bytes[i] = 0;
    adapter->state.prev_lap = wr64_cuda_lap(rdram);
    adapter->state.prev_misses = wr64_cuda_misses(rdram);
    adapter->state.recovery = wr64_cuda_recovery(rdram);
    float x, y, z;
    wr64_cuda_position(rdram, &x, &y, &z);
    adapter->vertical_origin = y;
    adapter->state.prev_node = wr64_cuda_node(rdram);
    adapter->state.prev_a = x;
    adapter->state.prev_y = y;
    adapter->state.prev_b = z;
    float fraction = 0.f;
    if (!wr64_cuda_reset_contract_valid(rdram, adapter->curriculum_laps)
            || !wr64_cuda_course_progress(adapter, rdram,
                &adapter->state.prev_course_progress, &fraction)
            || adapter->state.recovery != 0
            || wr64_cuda_disqualified(rdram)
            || wr64_cuda_ended(rdram)
            || wr64_cuda_finished(rdram)) {
        machine->error = WR64_CUDA_ADAPTER_ERROR_RESET_CONTRACT;
        return 0;
    }
    wr64_cuda_compute_observations(adapter, machine, rdram, observations);
    return 1;
}

template<typename ObsT>
__device__ static inline int wr64_cuda_adapter_finish_step(
        WR64CudaAdapter* adapter, WR64DeviceMachine* machine,
        uint8_t* rdram, uint32_t game_state, int elapsed_frames,
        WR64CudaStepPrelude prelude, ObsT* observations,
        float* reward, float* terminal) {
    WR64CudaState* state = &adapter->state;
    state->tick += elapsed_frames;
    int race_identity_valid = wr64_cuda_race_identity_valid(rdram);

    float x, y, z;
    wr64_cuda_position(rdram, &x, &y, &z);
    float da = x - state->prev_a;
    float dy = y - state->prev_y;
    float db = z - state->prev_b;
    float raw_dist = sqrtf(da * da + dy * dy + db * db);
    int teleported = !isfinite(raw_dist)
        || raw_dist > 500.f * (float)elapsed_frames;
    int recovery = wr64_cuda_recovery(rdram);
    int stable_motion = race_identity_valid && !teleported
        && recovery == 0 && state->recovery == 0;
    if (!stable_motion) {
        da = 0.f;
        dy = 0.f;
        db = 0.f;
    }
    float step_dist = sqrtf(da * da + dy * dy + db * db);
    state->dist_total += step_dist;

    float frame_scale = 1.f / (float)elapsed_frames;
    float vx = da * frame_scale;
    float vy = dy * frame_scale;
    float vz = db * frame_scale;
    state->velocity_x = vx;
    state->velocity_y = vy;
    state->velocity_z = vz;
    float speed = sqrtf(vx * vx + vz * vz);
    float hx, hz;
    wr64_cuda_heading(rdram, vx, vz, &hx, &hz);
    float slip = speed > 1e-4f
        ? fabsf((hx * vz - hz * vx) / speed) : 0.f;

    int32_t lap = wr64_cuda_lap(rdram);
    int32_t node = wr64_cuda_node(rdram);
    float crossed_distance = 0.f;
    int gained = stable_motion ? wr64_cuda_progress_advance(
        adapter, rdram, state->prev_lap, state->prev_node,
        lap, node, &crossed_distance) : 0;
    (void)crossed_distance;
    int checkpoint_discontinuity =
        (lap != state->prev_lap || node != state->prev_node) && gained == 0;

    float absolute_progress = state->prev_course_progress;
    float route_fraction = 0.f;
    int progress_valid = wr64_cuda_course_progress(
        adapter, rdram, &absolute_progress, &route_fraction);
    (void)route_fraction;
    float progress = 0.f;
    if (stable_motion && progress_valid && !checkpoint_discontinuity) {
        float delta = absolute_progress - state->prev_course_progress;
        float max_delta = 500.f * (float)elapsed_frames;
        if (isfinite(delta) && fabsf(delta) <= max_delta) progress = delta;
    }
    state->progress_total += progress;
    if (state->progress_total > state->max_progress) {
        state->max_progress = state->progress_total;
    }
    float frontier_progress = state->max_progress - prelude.max_progress_before;
    if (!isfinite(frontier_progress) || frontier_progress < 0.f) {
        frontier_progress = 0.f;
    }

    int32_t misses = wr64_cuda_misses(rdram);
    int miss_events = misses - state->prev_misses;
    if (miss_events < 0 || miss_events > 8) miss_events = 0;
    state->misses += miss_events;
    int checkpoint_events = gained - miss_events;
    if (checkpoint_events < 0) checkpoint_events = 0;
    state->checkpoints += checkpoint_events;

    int disqualified = wr64_cuda_disqualified(rdram) != 0;
    int official_finish = wr64_cuda_finished(rdram) != 0;
    int generic_end = wr64_cuda_ended(rdram) != 0;
    int success = !disqualified && official_finish;
    int failed = !success
        && (disqualified || (!official_finish && generic_end));
    int env_fault = !success && !failed
        && !disqualified && !official_finish && !generic_end
        && (!race_identity_valid || game_state != WR64_CUDA_STATE_RACING);
    if (env_fault) machine->error = WR64_CUDA_ADAPTER_ERROR_ENV_FAULT;
    int safety_timeout = !success && !failed
        && state->tick >= WR64_CUDA_MAX_STEPS;
    if (safety_timeout) failed = 1;

    int episode_terminal = success || failed || env_fault;
    float potential_after = wr64_cuda_reward_potential(adapter);
    float shaping;
    if (adapter->reward_mode == 2) {
        shaping = adapter->route_total > 0.f
            ? adapter->reward_progress * frontier_progress / adapter->route_total
            : 0.f;
        shaping += adapter->reward_checkpoint * (float)checkpoint_events;
    } else {
        shaping = episode_terminal && adapter->reward_mode == 0
            ? -prelude.potential_before
            : adapter->discount * potential_after - prelude.potential_before;
    }
    if (stable_motion) {
        shaping += adapter->reward_speed * step_dist
            - adapter->reward_slip * slip * (float)elapsed_frames;
    }
    if (!isfinite(shaping)) shaping = 0.f;
    float r = shaping;
    r -= adapter->reward_miss * (float)miss_events;
    if (!episode_terminal) r -= adapter->reward_fail * (1.f - adapter->discount);
    if (success) r += adapter->reward_finish;
    if (failed) r -= adapter->reward_fail;
    if (!isfinite(r)) r = 0.f;
    *reward = r;
    state->episode_return += r;
    state->success = success;
    state->failed = failed;
    state->disqualified = disqualified;
    state->safety_timeout = safety_timeout;
    state->env_fault = env_fault;
    *terminal = episode_terminal ? 1.f : 0.f;

    if (episode_terminal) {
        wr64_cuda_add_log(adapter, rdram);
        if (success) wr64_cuda_record_curriculum_success(adapter);
        return 1;
    }

    state->prev_a = x;
    state->prev_y = y;
    state->prev_b = z;
    state->prev_node = node;
    state->prev_lap = lap;
    state->prev_misses = misses;
    state->recovery = recovery;
    if (progress_valid) state->prev_course_progress = absolute_progress;
    wr64_cuda_compute_observations(adapter, machine, rdram, observations);
    return 0;
}
