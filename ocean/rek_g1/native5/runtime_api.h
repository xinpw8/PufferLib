#ifndef REK_NATIVE5_RUNTIME_API_H
#define REK_NATIVE5_RUNTIME_API_H

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#define REK_NATIVE5_OBSERVATION_SIZE 223
#define REK_NATIVE5_ACTION_COUNT 33
#define REK_NATIVE5_MOVE_COUNT 17
#define REK_NATIVE5_RUNTIME_ABI 1

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
