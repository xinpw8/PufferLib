#include "g1_mujoco_feature_registry.h"
#include "g1_combat_metrics.h"
#include "g1_model_identity.h"
#include "g1_model_identity_generated.h"
#include "g1_semantic_action_table.h"
#include "g1_semantic_assets.h"
#include "semantic_duel_runtime.h"
#include "sonic_motion_composer_libm_candidate.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    REK_G1_OBSERVATION_FLOATS =
        (int)(sizeof(RekG1SemanticDuelObservation) / sizeof(float)),
    REK_G1_ACTION_HEADS = 1,
    REK_G1_ACTION_CATEGORIES = REK_G1_SEMANTIC_ACTION_COUNT,
    REK_G1_ERROR_CAPACITY = 1024,
};

_Static_assert(
    sizeof(RekG1SemanticDuelObservation)
        == REK_G1_SEMANTIC_DUEL_OBSERVATION_FLOATS * sizeof(float),
    "the semantic G1 observation ABI must remain 223 binary32 values");
_Static_assert(
    REK_G1_ACTION_CATEGORIES == 33,
    "the semantic G1 action ABI must remain one 33-way categorical head");
_Static_assert(
    REK_G1_GENERATED_MODEL_ROBOT_BATCH > 0u
        && REK_G1_GENERATED_MODEL_ROBOT_BATCH % GEAR_SONIC_DUEL_FIGHTERS == 0u,
    "the compiled exact-batch graphs must contain complete fighter pairs");

static const RekG1ModelIdentityContract COMPILED_MODEL_IDENTITY = {
    .schema = REK_G1_GENERATED_MODEL_SCHEMA,
    .classification = REK_G1_GENERATED_MODEL_CLASSIFICATION,
    .manifest_sha256 = REK_G1_GENERATED_MODEL_MANIFEST_SHA256,
    .robot_batch = REK_G1_GENERATED_MODEL_ROBOT_BATCH,
    .rek_parity_claim = REK_G1_GENERATED_MODEL_REK_PARITY_CLAIM,
    .current_steam_authority =
        REK_G1_GENERATED_MODEL_CURRENT_STEAM_AUTHORITY,
    .training_enabled = REK_G1_GENERATED_MODEL_TRAINING_ENABLED,
    .encoder = {
        .source_sha256 = REK_G1_GENERATED_ENCODER_SOURCE_SHA256,
        .output_sha256 = REK_G1_GENERATED_ENCODER_OUTPUT_SHA256,
        .output_bytes = REK_G1_GENERATED_ENCODER_OUTPUT_BYTES,
    },
    .decoder = {
        .source_sha256 = REK_G1_GENERATED_DECODER_SOURCE_SHA256,
        .output_sha256 = REK_G1_GENERATED_DECODER_OUTPUT_SHA256,
        .output_bytes = REK_G1_GENERATED_DECODER_OUTPUT_BYTES,
    },
};

typedef RekG1CombatMetricLog Log;

typedef struct RekG1SemanticVectorContext {
    RekG1SemanticAssets assets;
    GearSonicNativeDuelVector duel;
    RekG1MujocoFeatureRegistry feature_registry;
    RekG1SemanticDuelRuntime semantic_runtime;
    RekG1NativePufferVector puffer;
    RekG1SemanticActionTableStorage action_table;
    RekG1CombatMetricAccumulator* combat_metrics;
    size_t robot_count;
    uint64_t semantic_steps;
    uint8_t assets_open;
    uint8_t duel_open;
    uint8_t feature_registry_open;
    uint8_t semantic_runtime_open;
    uint8_t puffer_open;
    char error[REK_G1_ERROR_CAPACITY];
} RekG1SemanticVectorContext;

typedef struct RekG1SemanticEnv {
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    unsigned int rng;
    Log log;
    RekG1SemanticVectorContext* context;
    size_t row;
} RekG1SemanticEnv;

typedef RekG1SemanticEnv Env;

void c_reset(Env* env);
void c_step(Env* env);
void c_close(Env* env);
void c_render(Env* env);

#define OBS_SIZE REK_G1_OBSERVATION_FLOATS
#define NUM_ATNS REK_G1_ACTION_HEADS
#define ACT_SIZES {REK_G1_ACTION_CATEGORIES}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK REK_G1_ACTION_CATEGORIES
#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define MY_VEC_RESET rek_g1_vec_reset
#define MY_VEC_STEP rek_g1_vec_step
#define MY_VEC_STEP_RANGE rek_g1_vec_step_range
#include "vecenv.h"

static const uint32_t MIRROR_SOURCE_INDICES[GEAR_SONIC_ACTION_DIM] = {
    6u, 7u, 8u, 9u, 10u, 11u,
    0u, 1u, 2u, 3u, 4u, 5u,
    12u, 13u, 14u,
    22u, 23u, 24u, 25u, 26u, 27u, 28u,
    15u, 16u, 17u, 18u, 19u, 20u, 21u,
};

static const uint8_t MIRROR_NEGATE[GEAR_SONIC_ACTION_DIM] = {
    0u, 1u, 1u, 0u, 0u, 1u,
    0u, 1u, 1u, 0u, 0u, 1u,
    1u, 1u, 0u,
    0u, 1u, 1u, 0u, 1u, 0u, 1u,
    0u, 1u, 1u, 0u, 1u, 0u, 1u,
};

static void rek_g1_binding_fail(const char* operation, const char* detail) {
    if (operation == NULL) operation = "REK G1 semantic binding";
    if (detail == NULL || detail[0] == '\0') detail = "unknown failure";
    (void)fprintf(stderr, "%s: %s\n", operation, detail);
    (void)fflush(stderr);
    abort();
}

static int exact_u32_config(
        Dict* dictionary,
        const char* key,
        uint32_t minimum,
        uint32_t maximum,
        uint32_t* output) {
    if (dictionary == NULL || key == NULL || output == NULL) return 0;
    DictItem* item = dict_get_unsafe(dictionary, key);
    if (item == NULL || !isfinite(item->value)
            || item->value < (double)minimum
            || item->value > (double)maximum
            || floor(item->value) != item->value) {
        return 0;
    }
    *output = (uint32_t)item->value;
    return 1;
}

static const char* required_environment_path(const char* name) {
    if (name == NULL) return NULL;
    const char* value = getenv(name);
    return value != NULL && value[0] != '\0' ? value : NULL;
}

static void close_context(RekG1SemanticVectorContext* context) {
    if (context == NULL) return;
    if (context->puffer_open) {
        rek_g1_native_puffer_close(&context->puffer);
        context->puffer_open = 0u;
        context->semantic_runtime_open = 0u;
    } else if (context->semantic_runtime_open) {
        rek_g1_semantic_duel_close(&context->semantic_runtime);
        context->semantic_runtime_open = 0u;
    }
    if (context->feature_registry_open) {
        rek_g1_mujoco_feature_registry_close(&context->feature_registry);
        context->feature_registry_open = 0u;
    }
    if (context->duel_open) {
        gear_sonic_native_duel_close(&context->duel);
        context->duel_open = 0u;
    }
    if (context->assets_open) {
        rek_g1_semantic_assets_close(&context->assets);
        context->assets_open = 0u;
    }
    free(context->combat_metrics);
    context->combat_metrics = NULL;
}

/*
 * The pinned f84f1874 Windows build's finite-input
 * ComputeYawForgivenessDelta control flow returns exactly zero. The configured
 * clip and impact forgiveness rates can be nonzero, but the only correction
 * branch is unreachable after the build's wrapped-yaw clamp and sign test.
 */
static int current_build_zero_forgiveness(
        void* opaque,
        const GearSonicNativeDuelVector* duel,
        size_t robot_row,
        float* delta_radians_out) {
    RekG1SemanticVectorContext* context = opaque;
    if (context == NULL || duel != &context->duel
            || delta_radians_out == NULL
            || robot_row >= context->robot_count) {
        return 0;
    }
    *delta_radians_out = 0.0f;
    return 1;
}

static RekG1NativePufferIO vector_io(StaticVec* vector, int with_actions) {
    RekG1NativePufferIO io = {
        .actions = with_actions ? vector->actions : NULL,
        .action_rows = with_actions ? (size_t)vector->total_agents : 0u,
        .action_heads = with_actions ? REK_G1_ACTION_HEADS : 0u,
        .observations = vector->observations,
        .observation_rows = (size_t)vector->total_agents,
        .observation_stride_bytes = sizeof(RekG1SemanticDuelObservation),
        .rewards = vector->rewards,
        .reward_rows = (size_t)vector->total_agents,
        .terminals = vector->terminals,
        .terminal_rows = (size_t)vector->total_agents,
        .action_masks = vector->action_mask,
        .action_mask_rows = (size_t)vector->total_agents,
        .action_mask_stride_bytes = REK_G1_ACTION_CATEGORIES,
    };
    return io;
}

static RekG1SemanticVectorContext* vector_context(StaticVec* vector) {
    if (vector == NULL || vector->envs == NULL || vector->size <= 0) {
        rek_g1_binding_fail("resolve semantic vector", "missing environment rows");
    }
    Env* environments = vector->envs;
    RekG1SemanticVectorContext* context = environments[0].context;
    if (context == NULL || context->robot_count != (size_t)vector->total_agents) {
        rek_g1_binding_fail("resolve semantic vector", "context identity mismatch");
    }
    return context;
}

static void record_completed_round_metrics(
        StaticVec* vector,
        RekG1SemanticVectorContext* context) {
    Env* environments = vector->envs;
    const size_t arena_count = context->robot_count / GEAR_SONIC_DUEL_FIGHTERS;
    for (size_t arena = 0u; arena < arena_count; arena++) {
        RekG1CombatMetricLog completed = {0};
        const RekG1SemanticDuelRuntime* runtime = &context->semantic_runtime;
        if (!rek_g1_combat_metrics_record_step(
                &context->combat_metrics[arena],
                &runtime->combat_states[arena].fight,
                runtime->scored_contact_counts[arena],
                runtime->attributed_contact_counts[arena],
                REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS,
                runtime->arena_terminals[arena],
                &completed)) {
            rek_g1_binding_fail(
                "record REK G1 combat metrics", "metric state rejected");
        }
        if (completed.n != 0.0f
                && !rek_g1_combat_metrics_merge_completed(
                    &environments[arena * GEAR_SONIC_DUEL_FIGHTERS].log,
                    &completed)) {
            rek_g1_binding_fail(
                "record REK G1 combat metrics", "completed log overflow");
        }
    }
}

void rek_g1_vec_reset(StaticVec* vector) {
    RekG1SemanticVectorContext* context = vector_context(vector);
    context->error[0] = '\0';
    RekG1NativePufferStatus status = rek_g1_native_puffer_reset(
        &context->puffer,
        vector_io(vector, 0),
        context->error,
        sizeof(context->error));
    if (status != REK_G1_NATIVE_PUFFER_OK) {
        rek_g1_binding_fail("reset REK G1 semantic vector", context->error);
    }
    const size_t arena_count = context->robot_count / GEAR_SONIC_DUEL_FIGHTERS;
    memset(
        context->combat_metrics,
        0,
        arena_count * sizeof(*context->combat_metrics));
    context->semantic_steps = 0u;
}

void rek_g1_vec_step(StaticVec* vector) {
    RekG1SemanticVectorContext* context = vector_context(vector);
    context->error[0] = '\0';
    RekG1NativePufferStatus status = rek_g1_native_puffer_step(
        &context->puffer,
        vector_io(vector, 1),
        context->error,
        sizeof(context->error));
    if (status != REK_G1_NATIVE_PUFFER_OK) {
        rek_g1_binding_fail("step REK G1 semantic vector", context->error);
    }
    record_completed_round_metrics(vector, context);
    context->semantic_steps += 1u;
}

void rek_g1_vec_step_range(
        StaticVec* vector,
        int environment_start,
        int environment_count,
        int worker_count) {
    if (vector == NULL || environment_start != 0
            || environment_count != vector->size || worker_count <= 0) {
        rek_g1_binding_fail(
            "step REK G1 semantic vector range",
            "shared-contact runtime requires the complete single buffer");
    }
    rek_g1_vec_step(vector);
}

Env* my_vec_init(
        int* environment_count_out,
        int* buffer_environment_starts,
        int* buffer_environment_counts,
        Dict* vector_kwargs,
        Dict* environment_kwargs) {
    if (environment_count_out == NULL || buffer_environment_starts == NULL
            || buffer_environment_counts == NULL || vector_kwargs == NULL
            || environment_kwargs == NULL) {
        rek_g1_binding_fail("initialize REK G1 semantic vector", "null argument");
    }

    uint32_t total_agents = 0u;
    uint32_t num_buffers = 0u;
    uint32_t physics_workers = 0u;
    uint32_t locomotion_segment_ticks = 0u;
    uint32_t move_ticks[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {0u};
    const char* const move_keys[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {
        "move_0_duration_ticks",
        "move_1_duration_ticks",
        "move_2_duration_ticks",
        "move_3_duration_ticks",
        "move_4_duration_ticks",
        "move_5_duration_ticks",
        "move_6_duration_ticks",
        "move_7_duration_ticks",
        "move_8_duration_ticks",
        "move_9_duration_ticks",
        "move_10_duration_ticks",
        "move_11_duration_ticks",
        "move_12_duration_ticks",
        "move_13_duration_ticks",
        "move_14_duration_ticks",
        "move_15_duration_ticks",
        "move_16_duration_ticks",
    };
    if (!exact_u32_config(
            vector_kwargs, "total_agents", 2u, (uint32_t)INT_MAX, &total_agents)
            || (total_agents % GEAR_SONIC_DUEL_FIGHTERS) != 0u
            || !exact_u32_config(
                vector_kwargs, "num_buffers", 1u, 1u, &num_buffers)
            || !exact_u32_config(
                environment_kwargs,
                "physics_workers",
                1u,
                (uint32_t)INT_MAX,
                &physics_workers)
            || !exact_u32_config(
                environment_kwargs,
                "locomotion_segment_ticks",
                1u,
                UINT32_MAX,
                &locomotion_segment_ticks)) {
        rek_g1_binding_fail(
            "initialize REK G1 semantic vector",
            "invalid vector shape or required numeric configuration");
    }
    (void)num_buffers;
    for (size_t index = 0;
            index < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; index++) {
        if (!exact_u32_config(
                environment_kwargs,
                move_keys[index],
                1u,
                UINT32_MAX,
                &move_ticks[index])) {
            rek_g1_binding_fail(
                "initialize REK G1 semantic vector",
                "a required discrete-move duration is absent or invalid");
        }
    }

    const char* asset_root = required_environment_path(
        "REK_G1_SEMANTIC_ASSETS_DIR");
    const char* encoder_path = required_environment_path(
        "REK_G1_ENCODER_ONNX");
    const char* decoder_path = required_environment_path(
        "REK_G1_DECODER_ONNX");
    if (asset_root == NULL || encoder_path == NULL || decoder_path == NULL) {
        rek_g1_binding_fail(
            "initialize REK G1 semantic vector",
            "REK_G1_SEMANTIC_ASSETS_DIR, REK_G1_ENCODER_ONNX, and REK_G1_DECODER_ONNX are required");
    }

    RekG1SemanticVectorContext* context = calloc(1u, sizeof(*context));
    Env* environments = calloc((size_t)total_agents, sizeof(*environments));
    if (context == NULL || environments == NULL) {
        free(context);
        free(environments);
        rek_g1_binding_fail("initialize REK G1 semantic vector", "allocation failed");
    }
    context->robot_count = total_agents;
    context->combat_metrics = calloc(
        (size_t)total_agents / GEAR_SONIC_DUEL_FIGHTERS,
        sizeof(*context->combat_metrics));
    if (context->combat_metrics == NULL) {
        free(environments);
        free(context);
        rek_g1_binding_fail(
            "initialize REK G1 combat metrics", "allocation failed");
    }

    RekG1PufferStatus table_status = rek_g1_semantic_action_table_init(
        &context->action_table, locomotion_segment_ticks, move_ticks);
    if (table_status != REK_G1_PUFFER_OK) {
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail("initialize semantic action table", "table rejected");
    }

    RekG1SemanticAssetsStatus asset_status = rek_g1_semantic_assets_load(
        &context->assets,
        asset_root,
        move_ticks,
        context->error,
        sizeof(context->error));
    if (asset_status != REK_G1_SEMANTIC_ASSETS_OK) {
        char error_copy[REK_G1_ERROR_CAPACITY];
        (void)snprintf(error_copy, sizeof(error_copy), "%s", context->error);
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail("load REK G1 semantic assets", error_copy);
    }
    context->assets_open = 1u;

    RekG1VerifiedModelPair verified_models = {0};
    context->error[0] = '\0';
    const RekG1ModelIdentityStatus model_identity_status =
        rek_g1_model_identity_load_verified(
            &COMPILED_MODEL_IDENTITY,
            encoder_path,
            decoder_path,
            (size_t)total_agents,
            &verified_models,
            context->error,
            sizeof(context->error));
    if (model_identity_status != REK_G1_MODEL_IDENTITY_OK) {
        char error_copy[REK_G1_ERROR_CAPACITY];
        (void)snprintf(error_copy, sizeof(error_copy), "%s", context->error);
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail(
            "load verified REK G1 model bytes",
            error_copy);
    }

    const int duel_opened = gear_sonic_native_duel_open_from_memory(
            &context->duel,
            context->assets.model_xml_data,
            context->assets.model_xml_byte_count,
            verified_models.encoder.data,
            verified_models.encoder.byte_count,
            verified_models.decoder.data,
            verified_models.decoder.byte_count,
            context->assets.fixed_idle,
            (size_t)total_agents / GEAR_SONIC_DUEL_FIGHTERS,
            (int)physics_workers,
            context->error,
            sizeof(context->error));
    rek_g1_model_identity_release_verified(&verified_models);
    if (!duel_opened) {
        char error_copy[REK_G1_ERROR_CAPACITY];
        (void)snprintf(error_copy, sizeof(error_copy), "%s", context->error);
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail("open REK G1 shared-contact duel", error_copy);
    }
    context->duel_open = 1u;

    RekG1MujocoFeatureRegistryStatus feature_status =
        rek_g1_mujoco_feature_registry_open(
            &context->feature_registry,
            &context->duel,
            &context->assets,
            context->error,
            sizeof(context->error));
    if (feature_status != REK_G1_MUJOCO_FEATURE_REGISTRY_OK) {
        char error_copy[REK_G1_ERROR_CAPACITY];
        (void)snprintf(error_copy, sizeof(error_copy), "%s", context->error);
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail("open REK G1 motion feature registry", error_copy);
    }
    context->feature_registry_open = 1u;

    RekG1SemanticDuelConfig runtime_config = {
        .input_timing = {
            .elapsed_seconds = REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS,
            .yaw_ramp_seconds = REK_G1_SEMANTIC_DUEL_YAW_RAMP_SECONDS,
        },
        .command = {
            .locomotion_speed_scale = 1.0f,
            .command_yaw_rate_scale = 1.0f,
            .heading_yaw_rate_scale = 1.0f,
            .controller_rate_hz = REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ,
        },
        .locomotion = {
            .settle_linear_speed = REK_G1_SEMANTIC_DUEL_SETTLE_LINEAR_SPEED,
            .settle_yaw_rate = REK_G1_SEMANTIC_DUEL_SETTLE_YAW_RATE,
            .stop_brake_rate = REK_G1_SEMANTIC_DUEL_STOP_BRAKE_RATE,
            .transition_settle = 1u,
        },
        .composer_backends = {
            .quaternion_slerp =
                sonic_motion_composer_libm_candidate_quaternion_slerp,
            .atan2_f = sonic_motion_composer_libm_candidate_atan2_f,
            .sin_cos_f = sonic_motion_composer_libm_candidate_sin_cos_f,
            .loop_entry_matcher = NULL,
            .context = NULL,
        },
        .mirror_table = {
            .source_indices = MIRROR_SOURCE_INDICES,
            .negate = MIRROR_NEGATE,
            .source_index_count = GEAR_SONIC_ACTION_DIM,
            .negate_count = GEAR_SONIC_ACTION_DIM,
        },
        .forgiveness_delta = current_build_zero_forgiveness,
        .forgiveness_context = context,
    };
    feature_status = rek_g1_mujoco_feature_registry_bind_backends(
        &context->feature_registry,
        &runtime_config.composer_backends,
        context->error,
        sizeof(context->error));
    if (feature_status != REK_G1_MUJOCO_FEATURE_REGISTRY_OK) {
        char error_copy[REK_G1_ERROR_CAPACITY];
        (void)snprintf(error_copy, sizeof(error_copy), "%s", context->error);
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail("bind REK G1 motion feature registry", error_copy);
    }

    const RekG1NativeMotionRouteTable* routes =
        rek_g1_native_static_motion_routes();
    RekG1SemanticDuelStatus runtime_status = rek_g1_semantic_duel_open(
        &context->semantic_runtime,
        &context->duel,
        routes,
        context->assets.route_assets,
        REK_G1_STATIC_ROUTE_COUNT,
        &runtime_config,
        context->error,
        sizeof(context->error));
    if (runtime_status != REK_G1_SEMANTIC_DUEL_OK) {
        char error_copy[REK_G1_ERROR_CAPACITY];
        (void)snprintf(error_copy, sizeof(error_copy), "%s", context->error);
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail("open REK G1 semantic runtime", error_copy);
    }
    context->semantic_runtime_open = 1u;

    RekG1NativePufferStatus puffer_status = rek_g1_native_puffer_open(
        &context->puffer,
        total_agents,
        &context->action_table.table,
        routes,
        rek_g1_semantic_duel_batch_ops(),
        &context->semantic_runtime);
    if (puffer_status != REK_G1_NATIVE_PUFFER_OK) {
        close_context(context);
        free(environments);
        free(context);
        rek_g1_binding_fail("open REK G1 Puffer vector", "adapter rejected");
    }
    context->puffer_open = 1u;

    for (size_t row = 0; row < total_agents; row++) {
        environments[row].num_agents = 1;
        environments[row].rng = (unsigned int)row;
        environments[row].context = context;
        environments[row].row = row;
    }
    buffer_environment_starts[0] = 0;
    buffer_environment_counts[0] = (int)total_agents;
    *environment_count_out = (int)total_agents;

    (void)fprintf(
        stderr,
        "rek_g1 semantic candidate: rows=%u arenas=%u actions=%u observation=%u; model_manifest_sha256=%s model_classification=%s rek_parity_claim=false current_steam_authority=false authentic_l100_no_getup=user_observed candidate_can_get_up=false training_enabled=false; 2 ms contact, fall, score, round-terminal, fallen-policy suspension, and two-phase spawn-reset state are wired; score-delta reward is an unapproved candidate contract; same-tick ordering and held-out trajectory parity remain gated\n",
        total_agents,
        total_agents / GEAR_SONIC_DUEL_FIGHTERS,
        (unsigned int)REK_G1_ACTION_CATEGORIES,
        (unsigned int)REK_G1_OBSERVATION_FLOATS,
        COMPILED_MODEL_IDENTITY.manifest_sha256,
        COMPILED_MODEL_IDENTITY.classification);
    return environments;
}

void my_vec_close(Env* environments) {
    if (environments == NULL) return;
    RekG1SemanticVectorContext* context = environments[0].context;
    close_context(context);
    free(context);
    environments[0].context = NULL;
}

void my_init(Env* environment, Dict* kwargs) {
    (void)environment;
    (void)kwargs;
    rek_g1_binding_fail(
        "initialize scalar REK G1 environment",
        "shared-contact semantic runtime requires vector initialization");
}

void my_log(Log* log, Dict* output) {
    if (log == NULL || output == NULL) return;
    dict_set(output, "side0_round_win_rate", log->side0_round_win_rate);
    dict_set(output, "side1_round_win_rate", log->side1_round_win_rate);
    dict_set(output, "round_tie_rate", log->round_tie_rate);
    dict_set(output, "round_redo_result_rate", log->round_redo_result_rate);
    dict_set(output, "redo_round_rate", log->redo_round_rate);
    dict_set(output, "ko_round_rate", log->ko_round_rate);
    dict_set(output, "side0_points_per_round", log->side0_points_per_round);
    dict_set(output, "side1_points_per_round", log->side1_points_per_round);
    dict_set(output, "side0_falls_per_round", log->side0_falls_per_round);
    dict_set(output, "side1_falls_per_round", log->side1_falls_per_round);
    dict_set(output, "scored_hits_per_round", log->scored_hits_per_round);
    dict_set(
        output,
        "attributed_contacts_per_round",
        log->attributed_contacts_per_round);
    dict_set(
        output,
        "elapsed_seconds_per_round",
        log->elapsed_seconds_per_round);
    dict_set(
        output,
        "semantic_steps_per_round",
        log->semantic_steps_per_round);
}

void c_reset(Env* environment) {
    (void)environment;
    rek_g1_binding_fail(
        "reset scalar REK G1 environment",
        "shared-contact semantic runtime requires vector reset");
}

void c_step(Env* environment) {
    (void)environment;
    rek_g1_binding_fail(
        "step scalar REK G1 environment",
        "shared-contact semantic runtime requires vector stepping");
}

void c_close(Env* environment) {
    (void)environment;
}

void c_render(Env* environment) {
    (void)environment;
}
