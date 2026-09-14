#ifndef REK_NATIVE5_RUNTIME_API_H
#define REK_NATIVE5_RUNTIME_API_H

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#define REK_NATIVE5_OBSERVATION_SIZE 223
#define REK_NATIVE5_ACTION_COUNT 33
#define REK_NATIVE5_MOVE_COUNT 17
#define REK_NATIVE5_RUNTIME_ABI 2

/* The PufferLib reducer treats this record as an array of floats. Runtime
 * accumulates completed-round totals and increments n once for each round.
 * Finite-state failures must also fail the runtime, including before a round
 * completes, because PufferLib only reduces records whose n is nonzero. */
typedef struct RekNative5Log {
    float score;
    float episode_return;
    float episode_length;
    float hits;
    float falls;
    float wins;
    float losses;
    float draws;
    float actions_invalid;
    float finite_failures;
    float n;
} RekNative5Log;

typedef struct RekNative5Config {
    uint32_t abi_version;
    const char* model_path;
    const char* physics_export_path;
    const char* assets_path;
    const char* motion_features_path;
    const char* controller_encoder_path;
    const char* controller_decoder_path;
    int arenas;
    uint32_t seed;
    int locomotion_segment_ticks;
    uint32_t move_duration_ticks[REK_NATIVE5_MOVE_COUNT];
    float round_seconds; /* 0 preserves recovered 120 s / 30 s redo durations */
} RekNative5Config;

typedef struct RekNative5Buffers {
    float* observations;  /* device [arenas][223], learner fighter 0 */
    float* actions;       /* device [arenas], categorical index 0..32 */
    float* rewards;       /* device [arenas] */
    float* terminals;     /* device [arenas], float 0 or 1 */
    RekNative5Log* logs;  /* device, one record per arena at log_stride_bytes */
    size_t log_stride_bytes;
} RekNative5Buffers;

typedef struct RekNative5Runtime RekNative5Runtime;

/* Evaluation-only inspection. All device pointers are borrowed until close.
 * The simulation remains the owner of every physical and combat state. */
typedef struct RekNative5RoundResult {
    uint64_t completed_rounds;
    uint64_t wins[2];
    uint64_t ties;
    uint64_t redos;
    uint64_t unclassified;
    int64_t completed_points[2];
    int32_t phase;
    uint32_t round_number;
    int32_t round_result;
    int32_t round_winner;
    int32_t fight_result;
    int32_t fight_winner;
    int32_t points[2];
    uint32_t falls[2];
    float time_remaining_seconds;
    uint32_t failure_bits;
    uint8_t terminal;
} RekNative5RoundResult;

typedef struct RekNative5DeviceView {
    int arenas;
    int nq; /* 72 per arena */
    int nv; /* 70 per arena */
    const float* raw_observations; /* [arenas * 2][223] */
    const uint8_t* action_masks;   /* [arenas * 2][33] */
    const float* qpos;            /* [arenas][72], model qpos order */
    const float* qvel;            /* [arenas][70], model qvel order */
    const float* actions;         /* [arenas * 2] */
    const float* rewards;         /* [arenas * 2] */
    const float* terminals;       /* [arenas * 2] */
    const RekNative5RoundResult* rounds; /* [arenas] */
} RekNative5DeviceView;

typedef struct RekNative5Snapshot {
    int arena;
    float raw_observations[2 * REK_NATIVE5_OBSERVATION_SIZE];
    uint8_t action_masks[2 * REK_NATIVE5_ACTION_COUNT];
    float qpos[72];
    float qvel[70];
    float actions[2];
    float rewards[2];
    float terminals[2];
    RekNative5RoundResult round;
} RekNative5Snapshot;

#ifdef __cplusplus
extern "C" {
#endif

/* Creation may read files, allocate memory, and synchronize. Reset and step
 * enqueue all GPU work on stream without host reads or dynamic allocation;
 * they must also work while that stream is being captured into a CUDA graph.
 * Each step is one 50 Hz semantic tick containing ten 2 ms physics steps.
 * A successful reset writes real initial observations and clears transition
 * outputs. A successful step writes the resulting observations and outputs.
 * Paths and configuration need only remain valid through create(). */
RekNative5Runtime* rek_native5_create(const RekNative5Config* config,
    const RekNative5Buffers* buffers, cudaStream_t stream);
int rek_native5_bind_action_mask(RekNative5Runtime* runtime,
    uint8_t* action_mask, cudaStream_t stream);
int rek_native5_reset(RekNative5Runtime* runtime, cudaStream_t stream);
int rek_native5_step(RekNative5Runtime* runtime, cudaStream_t stream);
/* Bind device [arenas * 2] arrays before graph capture. A nonzero override byte
 * selects that exact external action; other rows retain their original source
 * (learner fighter 0, scripted fighter 1). Invalid/masked actions fail normally.
 * Passing two null pointers restores original behavior. Changing the binding
 * requires recapture of any graph that embeds the old pointer values. */
int rek_native5_bind_external_actions(RekNative5Runtime* runtime,
    const float* device_actions, const uint8_t* device_override, cudaStream_t stream);
int rek_native5_get_device_view(RekNative5Runtime* runtime, RekNative5DeviceView* view);
/* Exact native5 scaled_polar_xy observation transform for both fighters. */
int rek_native5_encode_fighter_observations(RekNative5Runtime* runtime,
    float* device_encoded_observations, cudaStream_t stream);
/* Inspection boundary only, not capture-safe. Includes invalid-state flags;
 * it does not reset, suppress, or repair failures. */
int rek_native5_read_snapshot(RekNative5Runtime* runtime, int arena,
    RekNative5Snapshot* snapshot, cudaStream_t stream);
/* Reporting/rollout boundary only: synchronize and fail on sticky device
 * errors. This function must never be called during stream capture. */
int rek_native5_check_status(RekNative5Runtime* runtime, cudaStream_t stream);
int rek_native5_close(RekNative5Runtime* runtime);
const char* rek_native5_error(void);

#ifdef __cplusplus
}

static_assert(sizeof(RekNative5Log) == 11 * sizeof(float),
    "Native PufferLib log reduction requires a flat float record");
#endif

#endif
