// Wave Race 64 -- PufferLib environment backed by statically recompiled cartridge
// CPU code. Hardware video and audio paths are replaced by headless runtime
// shims. Actions enter through the emulated controller port, observations are
// read from the game's RDRAM, and the optional evaluator renders those same
// authoritative state fields without executing the N64 graphics pipeline.
//
// Each instance owns a WRMachine, including its RDRAM and suspended game context.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __linux__
#include <sched.h>
#endif

typedef float obs_t;
#include "pufferenv.h"
#include "wr_env.h"

#define WR64_OBS_SIZE   57
#define OBS_SIZE        WR64_OBS_SIZE
#define NUM_ATNS        5
#define ACT_SIZES       {15, 9, 2, 2, 2}
#define PUF_STEPS_PER_SEC 10
#define PUF_STEPS_PER_SEC_ENV(env) \
    ((int)WR_GAME_UPDATE_HZ / ((env)->frameskip > 0 ? (env)->frameskip : 1))

// WRMachine and its saved ucontext retain pointers into the containing Env.
// The upstream default vec initializer may realloc after puf_init, so Wave Race
// must allocate the exact final array once and initialize every machine in place.
#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define PUFFER_ENV_DISCOUNT_FROM_TRAIN
#define PUFFER_ENV_INTERNAL_FRAMESKIP
#define PUFFER_ENV_UNCLIPPED_REWARDS
#define PUFFER_ENV_EVAL_RESET
#define PUFFER_ENV_EXACT_OUTCOMES

// Episode time is measured in WR_GAME_UPDATE_HZ guest updates, not policy
// decisions. The supported runtime currently defines that rate as 20 Hz.
// 14,400 updates is 720 s and leaves the native ~599.25 s Time Trial timeout
// in control during ordinary episodes.
#define WR64_MAX_STEPS 14400

// Use the runtime's public decomp-derived addresses directly. Keeping one
// source of truth prevents an adapter rebuild from silently reading stale
// fields after a runtime-header update.
#define WR64_ACTIVE_RIDER_ADDR   WR_ADDR_ACTIVE_RIDER
#define WR64_PHYSICS_BASE        WR_PHYSICS_BASE
#define WR64_PHYSICS_STRIDE      WR_PHYSICS_STRIDE
#define WR64_PHYSICS_POS         WR_PHYSICS_POS
#define WR64_PHYSICS_SPEED       WR_PHYSICS_SPEED
#define WR64_PHYSICS_FORWARD_X   WR_PHYSICS_FORWARD_X
#define WR64_PHYSICS_FORWARD_Z   WR_PHYSICS_FORWARD_Z
#define WR64_RIDER_COUNT_ADDR    WR_ADDR_RIDERS
#define WR64_CONFIG_LAPS_ADDR    0x801CE618u
#define WR64_TARGET_LAPS_ADDR    0x801CE728u
#define WR64_COURSE_PRIMARY      WR_COURSE_PRIMARY
#define WR64_COURSE_NODE_STRIDE  WR_COURSE_NODE_STRIDE
#define WR64_COURSE_NODE_X       WR_COURSE_NODE_X
#define WR64_COURSE_NODE_Z       WR_COURSE_NODE_Z
#define WR64_COURSE_NODE_LENGTH  WR_COURSE_NODE_LENGTH
#define WR64_COURSE_NODE_TYPE    WR_COURSE_NODE_TYPE
#define WR64_COURSE_NODE_NEXT    WR_COURSE_NODE_NEXT
#define WR64_COURSE_NODE_COUNT   WR_ADDR_COURSE_NODE_COUNT
#define WR64_COURSE_START_NODE   WR_ADDR_COURSE_START_NODE
#define WR64_MAX_RIDERS          WR_MAX_RIDERS
#define WR64_MAX_COURSE_NODES    WR_MAX_COURSE_NODES
#define WR64_RIDER_MISSES        WR_RIDER_MISSES
#define WR64_RIDER_DQ            WR_RIDER_DISQUALIFIED
#define WR64_RIDER_POWER         WR_RIDER_POWER
#define WR64_RIDER_LAP_TIME      WR_RIDER_LAP_TIME
#define WR64_RIDER_LAP_SPLIT_1   WR_RIDER_LAP_SPLIT_1
#define WR64_RIDER_TOTAL_TIME    WR_RIDER_TOTAL_TIME
#define WR64_RIDER_ENDED         WR_RIDER_ENDED
#define WR64_RIDER_FINISHED      WR_RIDER_FINISHED
#define WR64_PHYSICS_RECOVERY    WR_PHYSICS_RECOVERY
#define WR64_PHYSICS_STATE       WR_PHYSICS_STATE
#define WR64_PHYSICS_STATE_FRAME WR_PHYSICS_STATE_FRAME
#define WR64_SPEED_SCALE         55.555557f

// Dynamic-water state used by the game's own func_8004D30C surface query.
// The query operates on a 384 by 128 triangular height lattice. Each entry is
// two guest halfwords; only the signed high byte of the first halfword is used
// by this physics query.
#define WR64_WATER_GRID          0x80162420u
#define WR64_WATER_LEVEL         0x80192458u
#define WR64_WATER_SCALE_Z       0x800E92A0u
#define WR64_WATER_SCALE_XZ      0x800E92A4u
#define WR64_WATER_ROWS          384
#define WR64_WATER_COLS          128

// Additional fields in the decomp-derived primary course node. Live XYZ drives
// current-target effects; anchor XZ is where the original buoy model is placed.
#define WR64_COURSE_NODE_Y        0x0004u
#define WR64_COURSE_NODE_ANCHOR_X 0x0024u
#define WR64_COURSE_NODE_ANCHOR_Z 0x0028u
#define WR64_COURSE_NODE_TANGENT_X 0x0070u
#define WR64_COURSE_NODE_TANGENT_Z 0x0074u
#define WR64_COURSE_NODE_DISABLED 0x009Cu
#define WR64_COURSE_NODE_ENABLED  0x00C8u

#define WR64_RENDER_WATER_DIM     33
#define WR64_RENDER_WATER_SAMPLES \
    (WR64_RENDER_WATER_DIM * WR64_RENDER_WATER_DIM)

// Stick detents, as a hand actually holds them.
static const int8_t WR64_STICK_X[15] = {-80,-68,-56,-44,-32,-20,-10,0,10,20,32,44,56,68,80};
static const int8_t WR64_STICK_Y[9]  = {-80,-56,-32,-12,0,12,32,56,80};

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
    float n;
} Log;

typedef struct Client Client;

typedef struct WR64RenderNode {
    int32_t index;
    int32_t next;
    int32_t type;
    int32_t valid;
    float live_x;
    float live_y;
    float live_z;
    float anchor_x;
    float anchor_z;
    float tangent_x;
    float tangent_z;
    float lateral_x;
    float lateral_z;
    float length;
    float pass_x;
    float pass_z;
} WR64RenderNode;

// Immutable, pointer-free projection of the gameplay state used by human eval.
// The renderer consumes this copy and cannot mutate RDRAM or adapter state.
typedef struct WR64RenderState {
    uint32_t version;
    uint32_t game_state;
    uint32_t course_id;
    uint32_t game_mode;
    uint64_t machine_ticks;
    int32_t tick;
    int32_t lap;
    int32_t target_laps;
    int32_t race_time_ms;
    int32_t lap_time_ms;
    int32_t lap_splits_ms[3];
    int32_t speed_kmh;
    int32_t power;
    int32_t race_position;
    int32_t target_node;
    int32_t next_node;
    int32_t misses;
    int32_t checkpoints;
    int32_t recovery;
    int32_t physics_state;
    int32_t physics_state_frame;
    int32_t disqualified;
    int32_t ended;
    int32_t finished;
    int32_t success;
    int32_t failed;
    int32_t node_count;
    uint16_t pad_buttons;
    int8_t pad_stick_x;
    int8_t pad_stick_y;
    float position[3];
    float velocity[3];
    float heading[2];
    float basis[9];
    float speed_per_second;
    float route_fraction;
    float progress_fraction;
    float route_total;
    float water_level;
    float water_origin_x;
    float water_origin_z;
    float water_spacing;
    WR64RenderNode nodes[WR64_MAX_COURSE_NODES];
    float water[WR64_RENDER_WATER_SAMPLES];
} WR64RenderState;

// Episode bookkeeping stays separate from the 8 MiB RDRAM backing and the
// machine-specific suspended stack used for exact reset.
typedef struct State {
    int   tick;
    float prev_a, prev_y, prev_b;
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
} State;

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    Client* client;

    WRMachine  machine;
    WRSnapshot snap;
    int   booted;
    State state;

    int   num_agents;
    int   frameskip;
    unsigned int rng;
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
    char* rom_path;
    float vertical_origin;
    float route_arc[WR64_MAX_COURSE_NODES];
    int32_t route_pred[WR64_MAX_COURSE_NODES];
    float route_total;
    int32_t route_nodes;
    int32_t route_valid;
};
typedef Env WaveRace64;

#ifdef PUFFER_WAVERACE64_RENDER
static void wr64_render_close(WaveRace64* env);
static void wr64_render_human_controls(WaveRace64* env);
static void wr64_render_capture_terminal(WaveRace64* env);
static void wr64_render_reset_episode(WaveRace64* env);
static int wr64_render_is_paused(const WaveRace64* env);
static int wr64_render_terminal_ready(const WaveRace64* env);
static void wr64_render_draw(WaveRace64* env);
#define PUF_EVAL_RENDER_PAUSED(env) wr64_render_is_paused(env)
#define PUF_EVAL_RENDER_TERMINAL(env) wr64_render_terminal_ready(env)
#define PUF_EVAL_POLICY_DURING_HUMAN 1
#endif

static inline uint32_t wr64_u(WaveRace64* e, uint32_t va) {
    return wr_rd32(e->machine.rdram, va);
}

static inline float wr64_f(WaveRace64* e, uint32_t va) {
    uint32_t bits = wr_rd32(e->machine.rdram, va);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline int32_t wr64_i32_bits(uint32_t value) {
    int32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static inline float wr64_f32_bits(uint32_t value) {
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static inline int32_t wr64_add32(int32_t a, int32_t b) {
    return wr64_i32_bits((uint32_t)a + (uint32_t)b);
}

static inline int32_t wr64_sub32(int32_t a, int32_t b) {
    return wr64_i32_bits((uint32_t)a - (uint32_t)b);
}

static inline int32_t wr64_mul32(int32_t a, int32_t b) {
    return wr64_i32_bits((uint32_t)a * (uint32_t)b);
}

static inline int32_t wr64_shl32(int32_t value, unsigned shift) {
    return wr64_i32_bits((uint32_t)value << shift);
}

static inline int32_t wr64_asr6(int32_t value) {
    return value >= 0 ? value / 64 : -(((-value) + 63) / 64);
}

static inline int32_t wr64_asr8_s16(int16_t value) {
    int32_t wide = value;
    return wide >= 0 ? wide / 256 : -(((-wide) + 255) / 256);
}

static inline int32_t wr64_water_q(const WaveRace64* e,
        int32_t row, int32_t col) {
    uint32_t index = (uint32_t)row * WR64_WATER_COLS + (uint32_t)col;
    int16_t raw = (int16_t)wr_rd16(e->machine.rdram,
        WR64_WATER_GRID + 4u * index);
    return wr64_asr8_s16(raw);
}

// Bit-exact host translation of the game's func_8004D30C water-height query.
// This implementation was compared against the recompiled function at 262,144
// randomized points over 64 randomized complete height fields with zero bit
// mismatches. Keep strict float sequencing and compile with -ffp-contract=off.
static inline float wr64_water_height(const WaveRace64* e, float x, float z) {
    const float k0 = wr64_f32_bits(UINT32_C(0x3F93CD3A));
    const float k1 = wr64_f32_bits(UINT32_C(0x3F13CD3A));
    volatile float vzf = k0 * z;
    int32_t v = ((int32_t)vzf) % 24576;
    int32_t j = wr64_asr6(v);
    volatile float u0 = k1 * z;
    volatile float uf = u0 + x;
    int32_t u = ((int32_t)uf) % 24576;
    int32_t i = wr64_asr6(u);
    int32_t h0 = wr64_water_q(e,
        (j + (i & -128) + 1536) % WR64_WATER_ROWS, i & 127);
    int32_t fv = wr64_sub32(wr64_shl32(j, 6), v);
    int32_t fu = wr64_sub32(wr64_shl32(i, 6), u);
    int32_t sx;
    int32_t sz;
    if (fv < fu) {
        int32_t h1 = wr64_water_q(e,
            (j + (i & -128) + 1537) % WR64_WATER_ROWS, i & 127);
        int32_t ip = wr64_add32(i, 1);
        int32_t h2 = wr64_water_q(e,
            (j + (ip & -128) + 1537) % WR64_WATER_ROWS, ip & 127);
        sx = wr64_sub32(h1, h2);
        sz = wr64_sub32(h0, h1);
    } else {
        int32_t ip = wr64_add32(i, 1);
        int32_t h1 = wr64_water_q(e,
            (j + (ip & -128) + 1536) % WR64_WATER_ROWS, ip & 127);
        int32_t h2 = wr64_water_q(e,
            (j + (ip & -128) + 1537) % WR64_WATER_ROWS, ip & 127);
        sx = wr64_sub32(h0, h1);
        sz = wr64_sub32(h1, h2);
    }
    int32_t denominator = wr64_add32(
        wr64_add32(wr64_mul32(sx, sx), wr64_mul32(sz, sz)), 4096);
    denominator = wr64_shl32(denominator, 12);
    volatile float denominator_f = (float)denominator;
    volatile float root = sqrtf(denominator_f);
    int32_t level = (int32_t)wr_rd32(
        e->machine.rdram, WR64_WATER_LEVEL);
    int32_t numerator = wr64_add32(wr64_mul32(sx, fu),
        wr64_shl32(wr64_add32(level, h0), 12));
    numerator = wr64_add32(numerator, wr64_mul32(sz, fv));
    volatile float numerator_f = (float)numerator;
    volatile float result = numerator_f / root;
    return result;
}

static inline uint16_t wr64_h(WaveRace64* e, uint32_t va) {
    return wr_rd16(e->machine.rdram, va);
}

static inline int32_t wr64_active_rider(WaveRace64* e) {
    int32_t rider = (int32_t)wr64_u(e, WR64_ACTIVE_RIDER_ADDR);
    int32_t riders = (int32_t)wr64_u(e, WR64_RIDER_COUNT_ADDR);
    return riders == 1 && rider == 0 ? 0 : -1;
}

static inline uint32_t wr64_rider_addr(WaveRace64* e, uint32_t offset) {
    int32_t rider = wr64_active_rider(e);
    return rider == 0 ? WR_RIDERS_BASE + offset : 0;
}

static inline uint32_t wr64_physics_addr(WaveRace64* e, uint32_t offset) {
    int32_t rider = wr64_active_rider(e);
    return rider == 0 ? WR64_PHYSICS_BASE + offset : 0;
}

// The player's rider record, from the decomp's RiderStruct array.
static inline int32_t wr64_lap(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR_RIDER_LAP));
}
static inline int32_t wr64_node(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR_RIDER_NODE));
}

static inline int32_t wr64_misses(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR64_RIDER_MISSES));
}

static inline int32_t wr64_power(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR64_RIDER_POWER));
}

static inline int32_t wr64_lap_time_ms(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR64_RIDER_LAP_TIME));
}

static inline int32_t wr64_lap_split_ms(WaveRace64* e, int lap) {
    if (lap < 0 || lap >= 3) return 0;
    return (int32_t)wr64_u(e, wr64_rider_addr(e,
        WR64_RIDER_LAP_SPLIT_1 + 4u*(uint32_t)lap));
}

static inline int32_t wr64_race_time_ms(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR64_RIDER_TOTAL_TIME));
}

static inline float wr64_physics_speed(WaveRace64* e) {
    return wr64_f(e, wr64_physics_addr(e, WR64_PHYSICS_SPEED));
}

static inline int32_t wr64_speed_to_kmh(float raw) {
    if (!isfinite(raw)) return 0;
    int32_t whole = (int32_t)raw;
    int32_t kmh = (int32_t)((float)whole * 1.8f);
    return kmh >= 1000 ? 999 : kmh;
}

static inline int32_t wr64_speed_kmh(WaveRace64* e) {
    return wr64_speed_to_kmh(wr64_physics_speed(e));
}

static inline int32_t wr64_disqualified(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR64_RIDER_DQ));
}

static inline int32_t wr64_ended(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR64_RIDER_ENDED));
}

static inline int32_t wr64_finished(WaveRace64* e) {
    return (int32_t)wr64_u(e, wr64_rider_addr(e, WR64_RIDER_FINISHED));
}

static inline int32_t wr64_target_laps(WaveRace64* e) {
    int32_t laps = (int32_t)wr64_u(e, WR64_TARGET_LAPS_ADDR);
    return laps >= 1 && laps <= 16 ? laps : 0;
}

static inline int32_t wr64_recovery(WaveRace64* e) {
    uint32_t physics = wr64_physics_addr(e, 0);
    int32_t state = (int32_t)wr64_u(e, physics + WR64_PHYSICS_STATE);
    int32_t frame = (int32_t)wr64_u(e, physics + WR64_PHYSICS_STATE_FRAME);
    if (wr64_h(e, physics + WR64_PHYSICS_RECOVERY) != 0
            || state == 23 || (state == 7 && frame < 56)) return 2;
    return state == 24 ? 1 : 0;
}

static inline int wr64_environment_fault(WaveRace64* e, uint32_t game_state,
                                         int race_identity_valid) {
    return !wr64_disqualified(e) && !wr64_finished(e) && !wr64_ended(e)
        && (!race_identity_valid || game_state != WR_STATE_RACING);
}

static inline void wr64_abort_environment_fault(WaveRace64* e,
                                                 uint32_t game_state) {
    fprintf(stderr,
        "[waverace64] fatal env fault tick=%d state=%#x ready=%u mode_state=%u "
        "course=%u game_mode=%u players=%u riders=%u active_rider=%d "
        "target_laps=%d node=%d\n",
        e->state.tick, game_state, wr64_u(e, WR_ADDR_RACE_READY),
        wr64_u(e, WR_ADDR_MODE_STATE), wr64_u(e, WR_ADDR_COURSE_ID),
        wr64_u(e, WR_ADDR_GAME_MODE), wr64_u(e, WR_ADDR_PLAYERS),
        wr64_u(e, WR_ADDR_RIDERS),
        (int32_t)wr64_u(e, WR64_ACTIVE_RIDER_ADDR), wr64_target_laps(e),
        wr64_node(e));
    abort();
}

static inline uint32_t wr64_course_count_addr(uint32_t base) {
    return base == WR64_COURSE_PRIMARY ? WR64_COURSE_NODE_COUNT : 0;
}

static inline uint32_t wr64_course_addr(uint32_t base, int32_t node, uint32_t offset) {
    return base + (uint32_t)node * WR64_COURSE_NODE_STRIDE + offset;
}

static inline int32_t wr64_node_count(WaveRace64* e, uint32_t base) {
    uint32_t address = wr64_course_count_addr(base);
    if (address == 0) return 0;
    int32_t count = (int32_t)wr64_u(e, address);
    return count > 0 && count <= WR64_MAX_COURSE_NODES ? count : 0;
}

static inline int32_t wr64_sanitize_node(WaveRace64* e, uint32_t base, int32_t node) {
    int32_t count = wr64_node_count(e, base);
    return node >= 0 && node < count ? node : -1;
}

static inline int wr64_race_identity_valid(WaveRace64* e) {
    return wr64_active_rider(e) == 0
        && wr64_sanitize_node(e, WR64_COURSE_PRIMARY, wr64_node(e)) >= 0;
}

static inline int wr64_reset_contract_valid(WaveRace64* e, int32_t target_laps) {
    return wr64_race_identity_valid(e)
        && wr64_u(e, WR_ADDR_GAMESTATE) == WR_STATE_RACING
        && wr64_u(e, WR_ADDR_RACE_READY) == 1
        && wr64_u(e, WR_ADDR_MODE_STATE) == 2
        && wr64_u(e, WR_ADDR_COURSE_ID) == WR_COURSE_SUNNY_BEACH
        && wr64_u(e, WR_ADDR_GAME_MODE) == WR_MODE_TIME_TRIALS
        && wr64_u(e, WR_ADDR_PLAYERS) == 1
        && wr64_u(e, WR_ADDR_RIDERS) == 1
        && wr64_target_laps(e) == target_laps
        && wr64_lap_time_ms(e) == 0
        && wr64_race_time_ms(e) == 0;
}

static inline int32_t wr64_node_type(WaveRace64* e, int32_t node) {
    node = wr64_sanitize_node(e, WR64_COURSE_PRIMARY, node);
    return node < 0 ? -1 : (int32_t)wr64_u(e,
        wr64_course_addr(WR64_COURSE_PRIMARY, node, WR_COURSE_NODE_TYPE));
}

static inline float wr64_buoy_side(WaveRace64* e, int32_t node) {
    int32_t type = wr64_node_type(e, node);
    return type == 0 ? -1.f : (type == 1 ? 1.f : 0.f);
}

static inline int32_t wr64_next_node(WaveRace64* e, uint32_t base, int32_t node) {
    node = wr64_sanitize_node(e, base, node);
    if (node < 0) return -1;
    return wr64_sanitize_node(e, base,
        (int32_t)wr64_u(e, wr64_course_addr(base, node, WR64_COURSE_NODE_NEXT)));
}

static inline void wr64_target(WaveRace64* e, float* x, float* z,
                               int32_t* node_out, uint32_t* course_out) {
    // RiderStruct+0xC and primary node+0xA0 are the game's authoritative
    // checkpoint topology. The current node is the gate being approached.
    uint32_t base = WR64_COURSE_PRIMARY;
    int32_t node = wr64_sanitize_node(e, base, wr64_node(e));
    if (node < 0) {
        *x = wr64_f(e, wr64_physics_addr(e, WR64_PHYSICS_POS));
        *z = wr64_f(e, wr64_physics_addr(e, WR64_PHYSICS_POS + 8));
        *node_out = -1;
        *course_out = 0;
        return;
    }

    *x = wr64_f(e, wr64_course_addr(base, node, WR64_COURSE_NODE_X));
    *z = wr64_f(e, wr64_course_addr(base, node, WR64_COURSE_NODE_Z));
    *node_out = node;
    *course_out = base;
}

static inline int wr64_pass_point(WaveRace64* e, int32_t node,
                                  float* x, float* z) {
    node = wr64_sanitize_node(e, WR64_COURSE_PRIMARY, node);
    if (node < 0) return 0;
    uint32_t address = wr64_course_addr(WR64_COURSE_PRIMARY, node, 0);
    *x = wr64_f(e, address + WR_COURSE_NODE_X);
    *z = wr64_f(e, address + WR_COURSE_NODE_Z);
    int32_t type = wr64_node_type(e, node);
    if (type == 0 || type == 1) {
        float side = type == 0 ? -400.f : 400.f;
        *x += side * wr64_f(e, address + WR_COURSE_NODE_LATERAL_X);
        *z += side * wr64_f(e, address + WR_COURSE_NODE_LATERAL_Z);
    }
    return isfinite(*x) && isfinite(*z);
}

static inline void wr64_position(WaveRace64* e, float* x, float* y, float* z) {
    uint32_t pos = wr64_physics_addr(e, WR64_PHYSICS_POS);
    *x = wr64_f(e, pos);
    *y = wr64_f(e, pos + 4);
    *z = wr64_f(e, pos + 8);
}

static inline void wr64_heading(WaveRace64* e, float vx, float vz, float* hx, float* hz) {
    *hx = wr64_f(e, wr64_physics_addr(e, WR64_PHYSICS_FORWARD_X));
    *hz = wr64_f(e, wr64_physics_addr(e, WR64_PHYSICS_FORWARD_Z));
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

static inline int wr64_progress_advance(WaveRace64* e,
                                        int32_t from_lap, int32_t from_node,
                                        int32_t to_lap, int32_t to_node,
                                        float* distance_out) {
    uint32_t base = WR64_COURSE_PRIMARY;
    int32_t count = wr64_node_count(e, base);
    int32_t lap = from_lap;
    int32_t node = wr64_sanitize_node(e, base, from_node);
    to_node = wr64_sanitize_node(e, base, to_node);
    *distance_out = 0.f;
    if (count == 0 || node < 0 || to_node < 0 || to_lap < from_lap) return 0;

    float distance = 0.f;
    for (int32_t hops = 0; hops <= count; hops++) {
        if (lap == to_lap && node == to_node) {
            *distance_out = distance;
            return hops;
        }
        if (hops == count) break;
        float length = wr64_f(e,
            wr64_course_addr(base, node, WR64_COURSE_NODE_LENGTH));
        if (!isfinite(length) || length < 0.f) return 0;
        distance += length;
        if ((int32_t)wr64_u(e,
                wr64_course_addr(base, node, WR64_COURSE_NODE_TYPE)) == 3) {
            lap++;
        }
        node = wr64_next_node(e, base, node);
        if (node < 0) return 0;
    }
    return 0;
}

static inline int wr64_build_route(WaveRace64* e) {
    uint32_t base = WR64_COURSE_PRIMARY;
    int32_t count = wr64_node_count(e, base);
    int32_t start = wr64_sanitize_node(e, base,
        (int32_t)wr64_u(e, WR64_COURSE_START_NODE));
    for (int32_t i = 0; i < WR64_MAX_COURSE_NODES; i++) {
        e->route_arc[i] = NAN;
        e->route_pred[i] = -1;
    }
    e->route_total = 0.f;
    e->route_nodes = 0;
    e->route_valid = 0;
    if (count < 2 || start < 0) return 0;

    uint8_t seen[WR64_MAX_COURSE_NODES] = {0};
    int32_t node = start;
    float total = 0.f;
    for (int32_t hop = 0; hop < count; hop++) {
        if (node < 0 || node >= count || seen[node]) return 0;
        seen[node] = 1;
        int32_t next = wr64_next_node(e, base, node);
        float length = wr64_f(e,
            wr64_course_addr(base, node, WR64_COURSE_NODE_LENGTH));
        if (next < 0 || !isfinite(length) || length <= 0.f
                || e->route_pred[next] != -1) return 0;
        e->route_arc[node] = total;
        e->route_pred[next] = node;
        total += length;
        e->route_nodes++;
        node = next;
        if (node == start) break;
    }
    if (node != start || e->route_nodes < 2
            || !isfinite(total) || total <= 0.f) return 0;
    e->route_total = total;
    e->route_valid = 1;
    return 1;
}

static inline int wr64_course_progress(WaveRace64* e, float* absolute_out,
                                       float* fraction_out) {
    if (!e->route_valid || !isfinite(e->route_total)
            || e->route_total <= 0.f) return 0;
    uint32_t base = WR64_COURSE_PRIMARY;
    int32_t target = wr64_sanitize_node(e, base, wr64_node(e));
    if (target < 0) return 0;
    int32_t pred = e->route_pred[target];
    if (pred < 0 || pred >= WR64_MAX_COURSE_NODES
            || !isfinite(e->route_arc[pred])) return 0;

    float x, y, z;
    wr64_position(e, &x, &y, &z);
    (void)y;
    float ax = wr64_f(e, wr64_course_addr(base, pred, WR64_COURSE_NODE_X));
    float az = wr64_f(e, wr64_course_addr(base, pred, WR64_COURSE_NODE_Z));
    float bx = wr64_f(e, wr64_course_addr(base, target, WR64_COURSE_NODE_X));
    float bz = wr64_f(e, wr64_course_addr(base, target, WR64_COURSE_NODE_Z));
    float dx = bx - ax;
    float dz = bz - az;
    float norm2 = dx*dx + dz*dz;
    float length = wr64_f(e,
        wr64_course_addr(base, pred, WR64_COURSE_NODE_LENGTH));
    if (!isfinite(x) || !isfinite(z) || !isfinite(norm2) || norm2 <= 1e-6f
            || !isfinite(length) || length <= 0.f) return 0;
    float u = ((x - ax)*dx + (z - az)*dz) / norm2;
    if (u < 0.f) u = 0.f;
    if (u > 1.f) u = 1.f;
    float arc = e->route_arc[pred] + u*length;
    int32_t lap = wr64_lap(e);
    if (lap < 0 || !isfinite(arc)) return 0;
    *absolute_out = (float)lap*e->route_total + arc;
    *fraction_out = arc / e->route_total;
    return isfinite(*absolute_out) && isfinite(*fraction_out);
}

static inline float wr64_route_fraction(WaveRace64* e) {
    float absolute = 0.f;
    float fraction = 0.f;
    return wr64_course_progress(e, &absolute, &fraction) ? fraction : 0.f;
}

static inline float wr64_finite_or_zero(float value) {
    return isfinite(value) ? value : 0.f;
}

static inline void wr64_capture_render_state(
        WaveRace64* e, WR64RenderState* out) {
    memset(out, 0, sizeof(*out));
    out->version = 2;
    out->game_state = wr64_u(e, WR_ADDR_GAMESTATE);
    out->course_id = wr64_u(e, WR_ADDR_COURSE_ID);
    out->game_mode = wr64_u(e, WR_ADDR_GAME_MODE);
    out->machine_ticks = e->machine.ticks;
    out->tick = e->state.tick;
    out->lap = wr64_lap(e);
    out->target_laps = wr64_target_laps(e);
    out->race_time_ms = wr64_race_time_ms(e);
    out->lap_time_ms = wr64_lap_time_ms(e);
    for (int lap = 0; lap < 3; lap++) {
        out->lap_splits_ms[lap] = wr64_lap_split_ms(e, lap);
    }
    out->speed_kmh = wr64_speed_kmh(e);
    out->power = wr64_power(e);
    out->race_position = (int32_t)wr64_u(
        e, wr64_rider_addr(e, WR_RIDER_RACE_POSITION));
    out->target_node = wr64_sanitize_node(
        e, WR64_COURSE_PRIMARY, wr64_node(e));
    out->next_node = wr64_next_node(
        e, WR64_COURSE_PRIMARY, out->target_node);
    out->misses = wr64_misses(e);
    out->checkpoints = e->state.checkpoints;
    out->recovery = wr64_recovery(e);
    uint32_t physics = wr64_physics_addr(e, 0);
    out->physics_state = (int32_t)wr64_u(
        e, physics + WR64_PHYSICS_STATE);
    out->physics_state_frame = (int32_t)wr64_u(
        e, physics + WR64_PHYSICS_STATE_FRAME);
    out->disqualified = wr64_disqualified(e) != 0;
    out->ended = wr64_ended(e) != 0;
    out->finished = wr64_finished(e) != 0;
    out->success = e->state.success;
    out->failed = e->state.failed;
    out->pad_buttons = e->machine.pad_buttons;
    out->pad_stick_x = e->machine.pad_stick_x;
    out->pad_stick_y = e->machine.pad_stick_y;

    wr64_position(e, &out->position[0], &out->position[1], &out->position[2]);
    out->velocity[0] = e->state.velocity_x;
    out->velocity[1] = e->state.velocity_y;
    out->velocity[2] = e->state.velocity_z;
    wr64_heading(e, out->velocity[0], out->velocity[2],
        &out->heading[0], &out->heading[1]);
    static const uint32_t basis_offsets[9] = {
        WR_PHYSICS_BASIS_0_X, WR_PHYSICS_BASIS_0_Y,
        WR_PHYSICS_BASIS_0_Z, WR_PHYSICS_BASIS_1_X,
        WR_PHYSICS_BASIS_1_Y, WR_PHYSICS_BASIS_1_Z,
        WR_PHYSICS_BASIS_2_X, WR_PHYSICS_BASIS_2_Y,
        WR_PHYSICS_BASIS_2_Z,
    };
    for (int i = 0; i < 9; i++) {
        out->basis[i] = wr64_f(e, physics + basis_offsets[i]);
    }
    out->speed_per_second = hypotf(
        out->velocity[0], out->velocity[2]) * (float)WR_GAME_UPDATE_HZ;
    out->route_fraction = wr64_route_fraction(e);
    out->route_total = e->route_total;
    float denominator = e->route_total * (float)out->target_laps;
    out->progress_fraction = denominator > 0.f
        ? e->state.progress_total / denominator : 0.f;
    out->water_level = (float)(int32_t)wr64_u(e, WR64_WATER_LEVEL);

    out->node_count = wr64_node_count(e, WR64_COURSE_PRIMARY);
    for (int32_t i = 0; i < out->node_count; i++) {
        WR64RenderNode* node = &out->nodes[i];
        uint32_t address = wr64_course_addr(WR64_COURSE_PRIMARY, i, 0);
        node->index = i;
        node->next = wr64_next_node(e, WR64_COURSE_PRIMARY, i);
        node->type = (int32_t)wr64_u(e, address + WR64_COURSE_NODE_TYPE);
        node->valid = (int32_t)wr64_u(
            e, address + WR64_COURSE_NODE_DISABLED) == 0
            && (int32_t)wr64_u(e, address + WR64_COURSE_NODE_ENABLED) != 0;
        node->live_x = wr64_f(e, address + WR64_COURSE_NODE_X);
        node->live_y = wr64_f(e, address + WR64_COURSE_NODE_Y);
        node->live_z = wr64_f(e, address + WR64_COURSE_NODE_Z);
        node->anchor_x = wr64_f(e, address + WR64_COURSE_NODE_ANCHOR_X);
        node->anchor_z = wr64_f(e, address + WR64_COURSE_NODE_ANCHOR_Z);
        node->tangent_x = wr64_f(e, address + WR64_COURSE_NODE_TANGENT_X);
        node->tangent_z = wr64_f(e, address + WR64_COURSE_NODE_TANGENT_Z);
        node->lateral_x = wr64_f(e, address + WR_COURSE_NODE_LATERAL_X);
        node->lateral_z = wr64_f(e, address + WR_COURSE_NODE_LATERAL_Z);
        node->length = wr64_f(e, address + WR64_COURSE_NODE_LENGTH);
        if (!wr64_pass_point(e, i, &node->pass_x, &node->pass_z)) {
            node->pass_x = node->live_x;
            node->pass_z = node->live_z;
        }
        float* values = &node->live_x;
        const int value_count = 12;
        for (int value = 0; value < value_count; value++) {
            values[value] = wr64_finite_or_zero(values[value]);
        }
    }

    out->water_spacing = 128.f;
    float half = 0.5f * (float)(WR64_RENDER_WATER_DIM - 1);
    out->water_origin_x = out->position[0] - half * out->water_spacing;
    out->water_origin_z = out->position[2] - half * out->water_spacing;
    for (int row = 0; row < WR64_RENDER_WATER_DIM; row++) {
        float z = out->water_origin_z + (float)row * out->water_spacing;
        for (int col = 0; col < WR64_RENDER_WATER_DIM; col++) {
            float x = out->water_origin_x + (float)col * out->water_spacing;
            float height = wr64_water_height(e, x, z);
            out->water[row * WR64_RENDER_WATER_DIM + col]
                = wr64_finite_or_zero(height);
        }
    }

    float* scalar_values = out->position;
    const int scalar_count = 3 + 3 + 2 + 9 + 8;
    for (int i = 0; i < scalar_count; i++) {
        scalar_values[i] = wr64_finite_or_zero(scalar_values[i]);
    }
}

static inline uint64_t wr64_render_hash_bytes(uint64_t hash,
        const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

// Hash the logical projection, not its object representation. This keeps the
// render change detector independent of compiler-inserted structure padding.
static inline uint64_t wr64_render_state_hash(const WR64RenderState* state) {
    uint64_t hash = UINT64_C(14695981039346656037);
#define WR64_RENDER_HASH(field) \
    hash = wr64_render_hash_bytes(hash, &(field), sizeof(field))
    WR64_RENDER_HASH(state->version);
    WR64_RENDER_HASH(state->game_state);
    WR64_RENDER_HASH(state->course_id);
    WR64_RENDER_HASH(state->game_mode);
    WR64_RENDER_HASH(state->machine_ticks);
    WR64_RENDER_HASH(state->tick);
    WR64_RENDER_HASH(state->lap);
    WR64_RENDER_HASH(state->target_laps);
    WR64_RENDER_HASH(state->race_time_ms);
    WR64_RENDER_HASH(state->lap_time_ms);
    WR64_RENDER_HASH(state->lap_splits_ms);
    WR64_RENDER_HASH(state->speed_kmh);
    WR64_RENDER_HASH(state->power);
    WR64_RENDER_HASH(state->race_position);
    WR64_RENDER_HASH(state->target_node);
    WR64_RENDER_HASH(state->next_node);
    WR64_RENDER_HASH(state->misses);
    WR64_RENDER_HASH(state->checkpoints);
    WR64_RENDER_HASH(state->recovery);
    WR64_RENDER_HASH(state->physics_state);
    WR64_RENDER_HASH(state->physics_state_frame);
    WR64_RENDER_HASH(state->disqualified);
    WR64_RENDER_HASH(state->ended);
    WR64_RENDER_HASH(state->finished);
    WR64_RENDER_HASH(state->success);
    WR64_RENDER_HASH(state->failed);
    WR64_RENDER_HASH(state->node_count);
    WR64_RENDER_HASH(state->pad_buttons);
    WR64_RENDER_HASH(state->pad_stick_x);
    WR64_RENDER_HASH(state->pad_stick_y);
    WR64_RENDER_HASH(state->position);
    WR64_RENDER_HASH(state->velocity);
    WR64_RENDER_HASH(state->heading);
    WR64_RENDER_HASH(state->basis);
    WR64_RENDER_HASH(state->speed_per_second);
    WR64_RENDER_HASH(state->route_fraction);
    WR64_RENDER_HASH(state->progress_fraction);
    WR64_RENDER_HASH(state->route_total);
    WR64_RENDER_HASH(state->water_level);
    WR64_RENDER_HASH(state->water_origin_x);
    WR64_RENDER_HASH(state->water_origin_z);
    WR64_RENDER_HASH(state->water_spacing);
    int32_t node_count = state->node_count;
    if (node_count < 0) node_count = 0;
    if (node_count > WR64_MAX_COURSE_NODES) {
        node_count = WR64_MAX_COURSE_NODES;
    }
    for (int32_t i = 0; i < node_count; i++) {
        const WR64RenderNode* node = &state->nodes[i];
        WR64_RENDER_HASH(node->index);
        WR64_RENDER_HASH(node->next);
        WR64_RENDER_HASH(node->type);
        WR64_RENDER_HASH(node->valid);
        WR64_RENDER_HASH(node->live_x);
        WR64_RENDER_HASH(node->live_y);
        WR64_RENDER_HASH(node->live_z);
        WR64_RENDER_HASH(node->anchor_x);
        WR64_RENDER_HASH(node->anchor_z);
        WR64_RENDER_HASH(node->tangent_x);
        WR64_RENDER_HASH(node->tangent_z);
        WR64_RENDER_HASH(node->lateral_x);
        WR64_RENDER_HASH(node->lateral_z);
        WR64_RENDER_HASH(node->length);
        WR64_RENDER_HASH(node->pass_x);
        WR64_RENDER_HASH(node->pass_z);
    }
    WR64_RENDER_HASH(state->water);
#undef WR64_RENDER_HASH
    return hash;
}

static inline float wr64_reward_potential(WaveRace64* e) {
    float lap_progress = e->route_total > 0.f
        ? e->state.progress_total / e->route_total : 0.f;
    float potential = e->reward_progress * lap_progress
        + e->reward_checkpoint * (float)e->state.checkpoints;
    return isfinite(potential) ? potential : 0.f;
}

static inline void wr64_record_curriculum_success(WaveRace64* env) {
    if (env->curriculum_laps >= env->curriculum_max_laps) return;
    env->curriculum_successes++;
    if (env->curriculum_successes >= env->curriculum_successes_per_lap) {
        env->curriculum_laps++;
        env->curriculum_successes = 0;
    }
}

void init(WaveRace64* env) {
    if (env->frameskip <= 0) env->frameskip = 2;
    env->num_agents = 1;
    if (env->booted) return;
    int abi_status = WR_RUNTIME_ABI_CHECK();
    if (abi_status != WR_RUNTIME_ABI_OK) {
        fprintf(stderr, "[waverace64] runtime ABI mismatch (status=%d)\n",
            abi_status);
        exit(1);
    }
    const char* rom = env->rom_path && env->rom_path[0]
        ? env->rom_path : "../baserom.us.rev1.z64";
    if (wr_machine_init(&env->machine, rom) != 0) {
        fprintf(stderr, "[waverace64] cannot open ROM: %s\n", rom);
        exit(1);
    }
    wr_current = &env->machine;
    wr_init_overlay_table();
    wr_install_fault_reporter(&env->machine);
    wr_dma_copy(env->machine.rdram, 0x80046800u,
                env->machine.rom + 0x1000, 0xA95D0 - 0x1000);
    // Boot once and savestate the exact native time-zero Sunny Beach Time
    // Trial; every later reset remaps that state instead of replaying menus.
    if (wr_boot_to_race(&env->machine, 8000) < 0) {
        fprintf(stderr, "[waverace64] never reached a race\n");
        exit(1);
    }
    if (wr_snapshot_capture(&env->snap, &env->machine) != 0) {
        fprintf(stderr, "[waverace64] savestate failed\n");
        exit(1);
    }
    if (!wr64_build_route(env)) {
        fprintf(stderr, "[waverace64] invalid official course route\n");
        exit(1);
    }
    if (!wr64_reset_contract_valid(env, 3)) {
        fprintf(stderr, "[waverace64] boot did not produce the fixed Time Trial contract\n");
        exit(1);
    }
    float spawn_x, spawn_y, spawn_z;
    wr64_position(env, &spawn_x, &spawn_y, &spawn_z);
    (void)spawn_x;
    (void)spawn_z;
    if (!isfinite(spawn_y)) {
        fprintf(stderr, "[waverace64] invalid race-start vertical position\n");
        exit(1);
    }
    env->vertical_origin = spawn_y;
    env->booted = 1;
}

void puf_close(WaveRace64* env) {
#ifdef PUFFER_WAVERACE64_RENDER
    wr64_render_close(env);
#endif
    if (env->snap.valid) wr_snapshot_free(&env->snap);
    if (env->machine.rdram) wr_machine_free(&env->machine);
}

void add_log(WaveRace64* env) {
    env->log.episode_length += env->state.tick;
    env->log.episode_return += env->state.episode_return;
    env->log.distance       += env->state.dist_total;
    env->log.mean_speed     += env->state.tick
        ? (float)WR_GAME_UPDATE_HZ * env->state.dist_total / env->state.tick
        : 0.f;
    env->log.score          += env->state.progress_total;
    env->log.checkpoints    += (float)env->state.checkpoints;
    env->log.misses         += (float)env->state.misses;
    env->log.success_rate   += env->state.success ? 1.f : 0.f;
    env->log.failure_rate   += env->state.failed
        && !env->state.disqualified && !env->state.safety_timeout ? 1.f : 0.f;
    env->log.disqualification_rate += env->state.disqualified ? 1.f : 0.f;
    env->log.safety_timeout_rate += env->state.safety_timeout ? 1.f : 0.f;
    env->log.env_fault_rate += env->state.env_fault ? 1.f : 0.f;
    int32_t target_laps = wr64_target_laps(env);
    env->log.target_laps += (float)target_laps;
    env->log.three_lap_success_rate += env->state.success
        && target_laps == 3 ? 1.f : 0.f;
    float perf = env->route_total > 0.f && target_laps > 0
        ? env->state.max_progress / (env->route_total * (float)target_laps)
        : 0.f;
    if (perf < 0.f) perf = 0.f;
    if (perf > 1.f) perf = 1.f;
    env->log.perf           += perf;
    env->log.n              += 1;
}

static inline void wr64_local_target_features(float x, float z,
        float hx, float hz, float target_x, float target_z, float route_total,
        float* forward, float* lateral, float* distance_fraction) {
    float dx = target_x - x;
    float dz = target_z - z;
    float distance = hypotf(dx, dz);
    float inverse = isfinite(distance) && distance > 1e-4f
        ? 1.f / distance : 0.f;
    *forward = (hx*dx + hz*dz) * inverse;
    *lateral = (hz*dx - hx*dz) * inverse;
    *distance_fraction = route_total > 0.f ? distance / route_total : 0.f;
    if (*distance_fraction < 0.f) *distance_fraction = 0.f;
    if (*distance_fraction > 1.f) *distance_fraction = 1.f;
}

void compute_observations(WaveRace64* env) {
    float* o = env->agents[0].observations;
    float x, y, z;
    wr64_position(env, &x, &y, &z);
    float vx = env->state.velocity_x;
    float vz = env->state.velocity_z;
    float sp = sqrtf(vx*vx + vz*vz);
    float inv = sp > 1e-4f ? 1.f/sp : 0.f;
    float hx, hz;
    wr64_heading(env, vx, vz, &hx, &hz);
    float target_x, target_z;
    int32_t target_node;
    uint32_t course;
    wr64_target(env, &target_x, &target_z, &target_node, &course);
    (void)course;
    float gdx = target_x - x;
    float gdz = target_z - z;
    float gd = sqrtf(gdx*gdx + gdz*gdz);
    float gate_inv = isfinite(gd) && gd > 1e-4f ? 1.f / gd : 0.f;
    float gate_fraction = env->route_total > 0.f ? gd / env->route_total : 0.f;
    if (gate_fraction < 0.f) gate_fraction = 0.f;
    if (gate_fraction > 1.f) gate_fraction = 1.f;
    int32_t target_laps = wr64_target_laps(env);
    float lap_fraction = target_laps > 0
        ? (float)wr64_lap(env) / (float)target_laps : 0.f;
    if (lap_fraction < 0.f) lap_fraction = 0.f;
    if (lap_fraction > 1.f) lap_fraction = 1.f;

    float pass_x = target_x;
    float pass_z = target_z;
    (void)wr64_pass_point(env, target_node, &pass_x, &pass_z);
    float pass_forward, pass_lateral, pass_distance;
    wr64_local_target_features(x, z, hx, hz, pass_x, pass_z,
        env->route_total, &pass_forward, &pass_lateral, &pass_distance);

    int32_t next_node = wr64_next_node(
        env, WR64_COURSE_PRIMARY, target_node);
    float next_x = pass_x;
    float next_z = pass_z;
    (void)wr64_pass_point(env, next_node, &next_x, &next_z);
    float next_forward, next_lateral, next_distance;
    wr64_local_target_features(x, z, hx, hz, next_x, next_z,
        env->route_total, &next_forward, &next_lateral, &next_distance);
    int32_t node_type = wr64_node_type(env, target_node);

    o[0] = x * 1e-3f;  o[1] = z * 1e-3f;
    o[2] = vx / WR64_SPEED_SCALE; o[3] = vz / WR64_SPEED_SCALE;
    o[4] = hx;  o[5] = hz;
    o[6] = (y - env->vertical_origin) * 0.01f;
    o[7] = (sp > 1e-4f) ? (hx*vz - hz*vx) * inv : 0.f;
    o[8] = sp / WR64_SPEED_SCALE;
    o[9]  = (env->machine.pad_buttons & WR_BTN_A) ? 1.f : 0.f;
    o[10] = (env->machine.pad_buttons & WR_BTN_B) ? 1.f : 0.f;
    o[11] = (env->machine.pad_buttons & WR_BTN_Z) ? 1.f : 0.f;
    o[12] = (env->machine.pad_buttons & WR_BTN_R) ? 1.f : 0.f;
    o[13] = env->machine.pad_stick_x * (1.f/80.f);
    o[14] = env->machine.pad_stick_y * (1.f/80.f);
    o[15] = (float)env->state.tick / (float)WR64_MAX_STEPS;
    o[16] = wr64_buoy_side(env, target_node);
    o[17] = (hx*gdx + hz*gdz) * gate_inv;
    o[18] = (hz*gdx - hx*gdz) * gate_inv;
    o[19] = gate_fraction;
    o[20] = wr64_route_fraction(env);
    o[21] = (float)wr64_misses(env) * 0.2f;
    o[22] = lap_fraction;
    o[23] = env->route_total > 0.f && target_laps > 0
        ? env->state.progress_total / (env->route_total * (float)target_laps)
        : 0.f;
    // The first 24 slots remain compatible with the original adapter. These
    // appended features make the signed gate geometry and recovery state
    // learner-visible instead of requiring hidden RDRAM access.
    o[24] = pass_forward;
    o[25] = pass_lateral;
    o[26] = pass_distance;
    o[27] = next_forward;
    o[28] = next_lateral;
    o[29] = next_distance;
    o[30] = node_type == 4 ? 1.f : 0.f;
    o[31] = 0.5f * (float)wr64_recovery(env);
    o[32] = env->route_nodes > 0 && target_laps > 0
        ? (float)env->state.checkpoints
            / ((float)env->route_nodes * (float)target_laps)
        : 0.f;
    uint32_t physics = wr64_physics_addr(env, 0);
    o[33] = env->state.velocity_y / WR64_SPEED_SCALE;
    o[34] = wr64_f(env, physics + WR_PHYSICS_BASIS_0_X);
    o[35] = wr64_f(env, physics + WR_PHYSICS_BASIS_0_Y);
    o[36] = wr64_f(env, physics + WR_PHYSICS_BASIS_0_Z);
    o[37] = wr64_f(env, physics + WR_PHYSICS_BASIS_1_X);
    o[38] = wr64_f(env, physics + WR_PHYSICS_BASIS_1_Y);
    o[39] = wr64_f(env, physics + WR_PHYSICS_BASIS_1_Z);
    o[40] = wr64_f(env, physics + WR_PHYSICS_BASIS_2_X);
    o[41] = wr64_f(env, physics + WR_PHYSICS_BASIS_2_Y);
    o[42] = wr64_f(env, physics + WR_PHYSICS_BASIS_2_Z);
    // Dynamic water affects pitch, steering, vertical motion, and the B-button
    // damping mechanic. A small rider-local stencil keeps that state visible to
    // the learner without feeding it a rendered image or a dense surface mesh.
    static const float lateral_offsets[3] = {-96.f, 0.f, 96.f};
    static const float forward_offsets[4] = {-64.f, 64.f, 192.f, 384.f};
    int water_index = 43;
    for (int forward_i = 0; forward_i < 4; forward_i++) {
        for (int lateral_i = 0; lateral_i < 3; lateral_i++) {
            float forward = forward_offsets[forward_i];
            float lateral = lateral_offsets[lateral_i];
            float sample_x = x + forward*hx + lateral*hz;
            float sample_z = z + forward*hz - lateral*hx;
            o[water_index++] = (wr64_water_height(env, sample_x, sample_z) - y)
                * 0.01f;
        }
    }
    o[55] = wr64_physics_speed(env) / WR64_SPEED_SCALE;
    o[56] = (float)wr64_power(env) * 0.2f;
    for (int i = 0; i < WR64_OBS_SIZE; i++) {
        if (!isfinite(o[i])) o[i] = 0.f;
    }
}

void puffer_state_refresh(WaveRace64* env) { compute_observations(env); }

void puf_reset(WaveRace64* env) {
    wr_current = &env->machine;
    wr_snapshot_restore(&env->snap, &env->machine);
    if (!wr64_reset_contract_valid(env, 3)) {
        fprintf(stderr, "[waverace64] reset snapshot is not an active race\n");
        abort();
    }
    wr_wr32(env->machine.rdram, WR64_CONFIG_LAPS_ADDR,
        (uint32_t)env->curriculum_laps);
    wr_wr32(env->machine.rdram, WR64_TARGET_LAPS_ADDR,
        (uint32_t)env->curriculum_laps);
    env->machine.pad_buttons = 0;
    env->machine.pad_stick_x = 0;
    env->machine.pad_stick_y = 0;
    memset(&env->state, 0, sizeof(env->state));
    env->state.prev_lap = wr64_lap(env);
    env->state.checkpoints = 0;
    env->state.prev_misses = wr64_misses(env);
    env->state.recovery = wr64_recovery(env);
    float x, y, z;
    wr64_position(env, &x, &y, &z);
    (void)y;
    env->state.prev_node = wr64_node(env);
    env->state.prev_a = x;
    env->state.prev_y = y;
    env->state.prev_b = z;
    float fraction = 0.f;
    if (!wr64_reset_contract_valid(env, env->curriculum_laps)
            || !wr64_course_progress(env, &env->state.prev_course_progress, &fraction)
            || env->state.recovery != 0 || wr64_disqualified(env)
            || wr64_ended(env) || wr64_finished(env)) {
        fprintf(stderr, "[waverace64] reset snapshot is not an active race\n");
        abort();
    }
    compute_observations(env);
}

void puf_eval_reset(WaveRace64* env) {
    env->curriculum_laps = 3;
    env->curriculum_successes = 0;
    puf_reset(env);
#ifdef PUFFER_WAVERACE64_RENDER
    if (env->client) wr64_render_reset_episode(env);
#endif
}

void puf_step(WaveRace64* env) {
    wr_current = &env->machine;
    Agent* agent = &env->agents[0];
#ifdef PUFFER_WAVERACE64_RENDER
    if (env->client) wr64_render_human_controls(env);
#endif
    float potential_before = wr64_reward_potential(env);
    float max_progress_before = env->state.max_progress;
    WRPad pad;
    int ax = (int)agent->actions[0]; ax = ax<0?0:(ax>14?14:ax);
    int ay = (int)agent->actions[1]; ay = ay<0?0:(ay>8 ?8 :ay);
    pad.stick_x = WR64_STICK_X[ax];
    pad.stick_y = WR64_STICK_Y[ay];
    pad.a = (uint8_t)((int)agent->actions[2] & 1);
    pad.b = (uint8_t)((int)agent->actions[3] & 1);
    // The Nintendo manual specifies that Z duplicates A throttle. Keep the
    // redundant controller input fixed off rather than learning two aliases.
    pad.z = 0;
    pad.r = (uint8_t)((int)agent->actions[4] & 1);

    int elapsed_frames = env->frameskip > 0 ? env->frameskip : 1;
    uint32_t gs = wr_env_step(&env->machine, &pad, elapsed_frames);
    env->state.tick += elapsed_frames;
    int race_identity_valid = wr64_race_identity_valid(env);

    float x, y, z;
    wr64_position(env, &x, &y, &z);
    float da = x - env->state.prev_a;
    float dy = y - env->state.prev_y;
    float db = z - env->state.prev_b;
    float raw_dist = sqrtf(da*da + dy*dy + db*db);
    int teleported = !isfinite(raw_dist)
        || raw_dist > 500.f * (float)elapsed_frames;
    int recovery = wr64_recovery(env);
    int stable_motion = race_identity_valid && !teleported
        && recovery == 0 && env->state.recovery == 0;
    if (!stable_motion) {
        da = 0.f;
        dy = 0.f;
        db = 0.f;
    }
    float step_dist = sqrtf(da*da + dy*dy + db*db);
    env->state.dist_total  += step_dist;

    float frame_scale = 1.f / (float)elapsed_frames;
    float vx = da * frame_scale;
    float vy = dy * frame_scale;
    float vz = db * frame_scale;
    env->state.velocity_x = vx;
    env->state.velocity_y = vy;
    env->state.velocity_z = vz;
    float speed = sqrtf(vx*vx + vz*vz);
    float hx, hz;
    wr64_heading(env, vx, vz, &hx, &hz);
    float slip = speed > 1e-4f ? fabsf((hx*vz - hz*vx) / speed) : 0.f;

    int32_t lap = wr64_lap(env);
    int32_t node = wr64_node(env);
    float crossed_distance = 0.f;
    int gained = stable_motion ? wr64_progress_advance(
        env, env->state.prev_lap, env->state.prev_node,
        lap, node, &crossed_distance) : 0;
    (void)crossed_distance;
    int checkpoint_discontinuity = (lap != env->state.prev_lap
        || node != env->state.prev_node) && gained == 0;

    float absolute_progress = env->state.prev_course_progress;
    float route_fraction = 0.f;
    int progress_valid = wr64_course_progress(
        env, &absolute_progress, &route_fraction);
    (void)route_fraction;
    float progress = 0.f;
    if (stable_motion && progress_valid && !checkpoint_discontinuity) {
        float delta = absolute_progress - env->state.prev_course_progress;
        float max_delta = 500.f * (float)elapsed_frames;
        if (isfinite(delta) && fabsf(delta) <= max_delta) progress = delta;
    }
    env->state.progress_total += progress;
    if (env->state.progress_total > env->state.max_progress) {
        env->state.max_progress = env->state.progress_total;
    }
    float frontier_progress = env->state.max_progress - max_progress_before;
    if (!isfinite(frontier_progress) || frontier_progress < 0.f) {
        frontier_progress = 0.f;
    }

    int32_t misses = wr64_misses(env);
    int miss_events = misses - env->state.prev_misses;
    if (miss_events < 0 || miss_events > 8) miss_events = 0;
    env->state.misses += miss_events;

    // The game advances the route node for both a cleared and a missed buoy.
    // Only advances without a simultaneous miss are successful checkpoints.
    int checkpoint_events = gained - miss_events;
    if (checkpoint_events < 0) checkpoint_events = 0;
    env->state.checkpoints += checkpoint_events;

    int disqualified = wr64_disqualified(env) != 0;
    int official_finish = wr64_finished(env) != 0;
    int generic_end = wr64_ended(env) != 0;
    int success = !disqualified && official_finish;
    int failed = !success
        && (disqualified || (!official_finish && generic_end));
    int env_fault = !success && !failed
        && wr64_environment_fault(env, gs, race_identity_valid);
    if (env_fault) wr64_abort_environment_fault(env, gs);
    int safety_timeout = !success && !failed
        && env->state.tick >= WR64_MAX_STEPS;
    if (safety_timeout) failed = 1;

    int terminal = success || failed || env_fault;
    float potential_after = wr64_reward_potential(env);
    float shaping;
    if (env->reward_mode == 2) {
        shaping = env->route_total > 0.f
            ? env->reward_progress * frontier_progress / env->route_total
            : 0.f;
        shaping += env->reward_checkpoint * (float)checkpoint_events;
    } else {
        shaping = terminal && env->reward_mode == 0
            ? -potential_before
            : env->discount * potential_after - potential_before;
    }
    // Speed/slip are optional instantaneous terms. Production keeps both at
    // zero. Reward mode 0 is strict terminal-cancelled PBRS, mode 1 retains
    // terminal potential, and mode 2 credits each verified route frontier and
    // official checkpoint once.
    if (stable_motion) {
        shaping += env->reward_speed * step_dist
            - env->reward_slip * slip * (float)elapsed_frames;
    }
    if (!isfinite(shaping)) shaping = 0.f;
    float r = shaping;
    r -= env->reward_miss * (float)miss_events;
    // A failure penalty received only at the end is discounted more heavily
    // when the agent stalls. Charging F*(1-gamma) on each nonterminal
    // transition makes every zero-miss failure have discounted task return -F,
    // independent of its duration, while a faster finish remains preferable.
    if (!terminal) r -= env->reward_fail * (1.f - env->discount);
    if (success) r += env->reward_finish;
    if (failed) r -= env->reward_fail;
    if (!isfinite(r)) r = 0.f;
    agent->rewards[0] = r;
    env->state.episode_return += r;
    env->state.success = success;
    env->state.failed = failed;
    env->state.disqualified = disqualified;
    env->state.safety_timeout = safety_timeout;
    env->state.env_fault = env_fault;
    agent->terminals[0] = 0.f;

    if (terminal) {
        agent->terminals[0] = 1.f;
#ifdef PUFFER_WAVERACE64_RENDER
        if (env->client) wr64_render_capture_terminal(env);
#endif
        add_log(env);
        if (success) wr64_record_curriculum_success(env);
        puf_reset(env);
        return;
    }
    env->state.prev_a = x;
    env->state.prev_y = y;
    env->state.prev_b = z;
    env->state.prev_node = node;
    env->state.prev_lap = lap;
    env->state.prev_misses = misses;
    env->state.recovery = recovery;
    if (progress_valid) env->state.prev_course_progress = absolute_progress;
    compute_observations(env);
}

#ifdef PUFFER_WAVERACE64_RENDER
#include "waverace64_render.h"
void puf_render(WaveRace64* env) { wr64_render_draw(env); }
#else
void puf_render(WaveRace64* env) { (void)env; }
#endif

void puf_init(Env* env, Dict* kwargs) {
    env->frameskip = (int)dict_get(kwargs, "frameskip");
    env->reward_speed = (float)dict_get(kwargs, "reward_speed");
    env->reward_progress = (float)dict_get(kwargs, "reward_progress");
    env->reward_slip = (float)dict_get(kwargs, "reward_slip");
    env->reward_checkpoint = (float)dict_get(kwargs, "reward_checkpoint");
    env->reward_miss = (float)dict_get(kwargs, "reward_miss");
    env->reward_finish = (float)dict_get(kwargs, "reward_finish");
    env->reward_fail = (float)dict_get(kwargs, "reward_fail");
    env->discount = (float)dict_get(kwargs, "discount");
    if (!(env->discount > 0.f && env->discount <= 1.f)) {
        fprintf(stderr, "[waverace64] discount must be in (0, 1]\n");
        exit(1);
    }
    env->reward_mode = (int32_t)dict_get(kwargs, "reward_mode");
    if (env->reward_mode < 0 || env->reward_mode > 2) {
        fprintf(stderr, "[waverace64] reward_mode must be 0, 1, or 2\n");
        exit(1);
    }
    env->curriculum_start_laps = (int32_t)dict_get(
        kwargs, "curriculum_start_laps");
    env->curriculum_max_laps = (int32_t)dict_get(
        kwargs, "curriculum_max_laps");
    env->curriculum_successes_per_lap = (int32_t)dict_get(
        kwargs, "curriculum_successes_per_lap");
    if (env->curriculum_start_laps < 1
            || env->curriculum_start_laps > 3
            || env->curriculum_max_laps < env->curriculum_start_laps
            || env->curriculum_max_laps > 3
            || env->curriculum_successes_per_lap < 1) {
        fprintf(stderr, "[waverace64] invalid lap curriculum\n");
        exit(1);
    }
    env->curriculum_laps = env->curriculum_start_laps;
    env->curriculum_successes = 0;
    env->rom_path = (char*)dict_get_str(kwargs, "rom_path");
    env->agents[0].policy = 0;
    init(env);
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts,
        int* buffer_env_counts, Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents");
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers");
    int init_threads = (int)dict_get(vec_kwargs, "num_threads");
    if (total_agents <= 0 || num_buffers <= 0
            || total_agents % num_buffers != 0) {
        fprintf(stderr,
            "[waverace64] total_agents must be positive and divisible by num_buffers\n");
        exit(1);
    }
    if (init_threads < 1) init_threads = 1;
    if (init_threads > total_agents) init_threads = total_agents;

    // Exact allocation is required. Moving an initialized Env invalidates the
    // WRMachine address registered with its RDRAM mapping and saved ucontext.
    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    if (!envs) {
        fprintf(stderr, "[waverace64] environment allocation failed\n");
        exit(1);
    }

#ifdef __linux__
    cpu_set_t caller_affinity;
    int have_caller_affinity = sched_getaffinity(
        0, sizeof(caller_affinity), &caller_affinity) == 0;
#endif

    // Keep one canonical reset image alive while each remaining environment
    // boots. Sharing immediately bounds private snapshot-copy peak memory by
    // the initialization-team width instead of total_agents.
    envs[0].rng = 0;
    puf_init(&envs[0], env_kwargs);
    int share_error = 0;
    int share_error_env = -1;
    #pragma omp parallel for schedule(static) num_threads(init_threads)
    for (int i = 1; i < total_agents; i++) {
        envs[i].rng = (unsigned int)i;
        puf_init(&envs[i], env_kwargs);
        int rc = wr_snapshot_share_rdram(&envs[i].snap, &envs[0].snap);
        if (rc != 0) {
            #pragma omp critical
            {
                if (share_error == 0) {
                    share_error = rc;
                    share_error_env = i;
                }
            }
        }
    }

#ifdef __linux__
    // OMP_PROC_BIND may leave the thread that entered the boot region pinned to
    // one OpenMP place. Restore its incoming mask before Puffer creates rollout
    // pthreads, which inherit the creator's affinity.
    if (have_caller_affinity && sched_setaffinity(
            0, sizeof(caller_affinity), &caller_affinity) != 0) {
        perror("[waverace64] sched_setaffinity");
        exit(1);
    }
#endif

    // The deterministic boot produces one RDRAM image. Stacks and ucontexts
    // remain machine-owned, while reset RDRAM uses a shared immutable memfd.
    if (share_error != 0) {
        fprintf(stderr,
            "[waverace64] reset snapshot differs for env %d (rc=%d)\n",
            share_error_env, share_error);
        exit(1);
    }
    if (wr_snapshot_drop_rdram_copy(&envs[0].snap) != 0) {
        fprintf(stderr, "[waverace64] canonical reset backing is unavailable\n");
        exit(1);
    }

    int agents_per_buffer = total_agents / num_buffers;
    for (int buffer = 0; buffer < num_buffers; buffer++) {
        buffer_env_starts[buffer] = buffer * agents_per_buffer;
        buffer_env_counts[buffer] = agents_per_buffer;
    }
    *num_envs_out = total_agents;
    return envs;
}

void my_vec_close(Env* envs) {
    free(envs);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "distance", log->distance);
    dict_set(out, "checkpoints", log->checkpoints);
    dict_set(out, "misses", log->misses);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "failure_rate", log->failure_rate);
    dict_set(out, "disqualification_rate", log->disqualification_rate);
    dict_set(out, "safety_timeout_rate", log->safety_timeout_rate);
    dict_set(out, "env_fault_rate", log->env_fault_rate);
    dict_set(out, "mean_speed", log->mean_speed);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "target_laps", log->target_laps);
    dict_set(out, "three_lap_success_rate", log->three_lap_success_rate);
}
