#include "gear_sonic_native_duel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float* read_exact_floats(const char* path, size_t count) {
    FILE* stream = fopen(path, "rb");
    if (stream == NULL || count > SIZE_MAX / sizeof(float)) {
        if (stream != NULL) fclose(stream);
        return NULL;
    }
    float* values = (float*)malloc(count * sizeof(float));
    if (values == NULL) {
        fclose(stream);
        return NULL;
    }
    int ok = fread(values, sizeof(float), count, stream) == count;
    if (ok) ok = fgetc(stream) == EOF && !ferror(stream);
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(values);
        return NULL;
    }
    return values;
}

static int checked_product(size_t left, size_t right, size_t* result) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

typedef struct PostStepProbe {
    const GearSonicNativeDuelVector* vector;
    size_t arena_count;
    size_t calls;
    size_t fail_arena;
    uint32_t fail_substep;
    int fail_enabled;
    int invalid;
} PostStepProbe;

static int observe_post_step(
        void* opaque,
        const GearSonicNativeDuelPostStepObservation* observation) {
    PostStepProbe* probe = (PostStepProbe*)opaque;
    if (probe == NULL || observation == NULL || probe->arena_count == 0) {
        return 0;
    }
    const uint32_t expected_substep = (uint32_t)(probe->calls / probe->arena_count);
    const size_t expected_arena = probe->calls % probe->arena_count;
    const double expected_time = 0.002 * (double)(expected_substep + 1u);
    if (observation->vector != probe->vector
            || observation->model != probe->vector->model
            || observation->arena_index != expected_arena
            || observation->physics_substep_index != expected_substep
            || observation->data != probe->vector->data[expected_arena]
            || observation->physics_dt_seconds != 0.002
            || fabs(observation->data->time - expected_time) > 1e-12) {
        probe->invalid = 1;
        return 0;
    }
    probe->calls += 1;
    return !(probe->fail_enabled
        && observation->arena_index == probe->fail_arena
        && observation->physics_substep_index == probe->fail_substep);
}

static int fail(
        const char* message,
        GearSonicNativeDuelVector* vector,
        float* dof,
        float* position,
        float* rotation) {
    fprintf(stderr, "%s\n", message);
    gear_sonic_native_duel_close(vector);
    free(rotation);
    free(position);
    free(dof);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(
            stderr,
            "usage: %s MODEL ENCODER DECODER FRAMES DOF ROOT_POSITION "
            "ROOT_ROTATION ARENAS\n",
            argv[0]);
        return 64;
    }
    char* end = NULL;
    const unsigned long parsed_frames = strtoul(argv[4], &end, 10);
    if (end == argv[4] || *end != '\0' || parsed_frames == 0
            || parsed_frames > SIZE_MAX / GEAR_SONIC_ACTION_DIM) {
        return 64;
    }
    const size_t frames = (size_t)parsed_frames;
    end = NULL;
    const unsigned long parsed_arenas = strtoul(argv[8], &end, 10);
    if (end == argv[8] || *end != '\0' || parsed_arenas == 0
            || parsed_arenas > SIZE_MAX
                / (GEAR_SONIC_DUEL_FIGHTERS
                    * GEAR_SONIC_HISTORY_FRAMES
                    * GEAR_SONIC_ENCODER_INPUT_WIDTH
                    * sizeof(float))) {
        return 64;
    }
    const size_t arenas = (size_t)parsed_arenas;
    float* dof = read_exact_floats(
        argv[5], frames * GEAR_SONIC_ACTION_DIM);
    float* position = read_exact_floats(argv[6], frames * 3);
    float* rotation = read_exact_floats(argv[7], frames * 4);
    if (dof == NULL || position == NULL || rotation == NULL) {
        GearSonicNativeDuelVector empty = {0};
        return fail("motion read failed", &empty, dof, position, rotation);
    }
    GearSonicNativeMotion motion = {
        .dof_position_mujoco = dof,
        .root_position_m = position,
        .root_rotation_xyzw = rotation,
        .frames = frames,
        .loop = 1,
    };
    GearSonicNativeDuelVector vector = {0};
    char error[1024] = {0};
    if (!gear_sonic_native_duel_open(
            &vector,
            argv[1],
            argv[2],
            argv[3],
            motion,
            arenas,
            1,
            error,
            sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return fail("duel open failed", &vector, dof, position, rotation);
    }
    if (vector.arena_count != arenas
            || vector.robot_count != arenas * GEAR_SONIC_DUEL_FIGHTERS
            || vector.controller.batch_size != vector.robot_count
            || vector.ort.batch_size != vector.robot_count
            || !vector.spawn_prefixes_verified
            || vector.failed) {
        return fail("duel cardinality contract failed", &vector, dof, position, rotation);
    }
    const GearSonicNativeDuelVector active_snapshot = vector;
    error[0] = '\0';
    if (gear_sonic_native_duel_open(
            &vector,
            argv[1],
            argv[2],
            argv[3],
            motion,
            arenas,
            1,
            error,
            sizeof(error)) != 0
            || strstr(error, "already initialized") == NULL
            || memcmp(&vector, &active_snapshot, sizeof(vector)) != 0) {
        return fail(
            "repeated duel open changed an active vector",
            &vector,
            dof,
            position,
            rotation);
    }
    for (size_t index = 0; index < arenas; index++) {
        if (vector.data[index] == NULL) {
            return fail("arena data is missing", &vector, dof, position, rotation);
        }
        for (size_t other = index + 1; other < arenas; other++) {
            if (vector.data[index] == vector.data[other]) {
                return fail("arena data objects alias", &vector, dof, position, rotation);
            }
        }
    }
    size_t state_count = 0;
    if (!checked_product(arenas, GEAR_SONIC_DUEL_QPOS_DIM, &state_count)
            || state_count > SIZE_MAX / sizeof(double)) {
        return fail("state snapshot size overflow", &vector, dof, position, rotation);
    }
    double* before = (double*)malloc(state_count * sizeof(double));
    if (before == NULL) {
        return fail("state snapshot allocation failed", &vector, dof, position, rotation);
    }
    for (size_t index = 0; index < arenas; index++) {
        memcpy(
            before + index * GEAR_SONIC_DUEL_QPOS_DIM,
            vector.data[index]->qpos,
            GEAR_SONIC_DUEL_QPOS_DIM * sizeof(double));
    }
    const uint32_t opaque_commands[2] = {0, UINT32_MAX};
    if (gear_sonic_native_duel_step_semantic_unavailable(
            &vector,
            opaque_commands,
            2,
            error,
            sizeof(error)) != 0
            || !vector.failed
            || strstr(error, "not wired") == NULL) {
        free(before);
        return fail("semantic fail-closed contract failed", &vector, dof, position, rotation);
    }
    for (size_t index = 0; index < arenas; index++) {
        if (memcmp(
                before + index * GEAR_SONIC_DUEL_QPOS_DIM,
                vector.data[index]->qpos,
                GEAR_SONIC_DUEL_QPOS_DIM * sizeof(double)) != 0
                || vector.data[index]->time != 0.0) {
            free(before);
            return fail("semantic rejection advanced physics", &vector, dof, position, rotation);
        }
    }
    free(before);
    if (!gear_sonic_native_duel_reset(&vector, error, sizeof(error))
            || vector.failed) {
        fprintf(stderr, "%s\n", error);
        return fail("full reset did not recover rejected state", &vector, dof, position, rotation);
    }

    const size_t reference_rows
        = vector.robot_count * GEAR_SONIC_HISTORY_FRAMES;
    const size_t reference_position_count
        = reference_rows * GEAR_SONIC_ACTION_DIM;
    const size_t reference_rotation_count = reference_rows * 4;
    const size_t qpos_count = arenas * GEAR_SONIC_DUEL_QPOS_DIM;
    const size_t qvel_count = arenas * GEAR_SONIC_DUEL_QVEL_DIM;
    const size_t action_count = vector.robot_count * GEAR_SONIC_ACTION_DIM;
    const size_t encoder_count
        = vector.robot_count * GEAR_SONIC_ENCODER_INPUT_WIDTH;
    float* reference_position = (float*)malloc(
        reference_position_count * sizeof(float));
    float* reference_next_position = (float*)malloc(
        reference_position_count * sizeof(float));
    float* reference_rotation = (float*)malloc(
        reference_rotation_count * sizeof(float));
    double* fixed_qpos = (double*)malloc(qpos_count * sizeof(double));
    double* fixed_qvel = (double*)malloc(qvel_count * sizeof(double));
    float* fixed_action = (float*)malloc(action_count * sizeof(float));
    float* fixed_target = (float*)malloc(action_count * sizeof(float));
    float* fixed_encoder = (float*)malloc(encoder_count * sizeof(float));
    const char* reference_failure = NULL;
    if (reference_position == NULL || reference_next_position == NULL
            || reference_rotation == NULL || fixed_qpos == NULL
            || fixed_qvel == NULL || fixed_action == NULL
            || fixed_target == NULL || fixed_encoder == NULL) {
        reference_failure = "reference regression allocation failed";
        goto reference_cleanup;
    }
    for (size_t row = 0; row < vector.robot_count; row++) {
        for (size_t future = 0; future < GEAR_SONIC_HISTORY_FRAMES; future++) {
            const size_t frame = (future * 5) % frames;
            const size_t next = (frame + 1) % frames;
            const size_t reference_row = row * GEAR_SONIC_HISTORY_FRAMES + future;
            memcpy(
                reference_position + reference_row * GEAR_SONIC_ACTION_DIM,
                dof + frame * GEAR_SONIC_ACTION_DIM,
                GEAR_SONIC_ACTION_DIM * sizeof(float));
            memcpy(
                reference_next_position + reference_row * GEAR_SONIC_ACTION_DIM,
                dof + next * GEAR_SONIC_ACTION_DIM,
                GEAR_SONIC_ACTION_DIM * sizeof(float));
            memcpy(
                reference_rotation + reference_row * 4,
                rotation + frame * 4,
                4 * sizeof(float));
        }
    }
    if (!gear_sonic_native_duel_step_fixed(&vector, error, sizeof(error))) {
        reference_failure = "fixed-motion regression step failed";
        goto reference_cleanup;
    }
    for (size_t index = 0; index < arenas; index++) {
        memcpy(
            fixed_qpos + index * GEAR_SONIC_DUEL_QPOS_DIM,
            vector.data[index]->qpos,
            GEAR_SONIC_DUEL_QPOS_DIM * sizeof(double));
        memcpy(
            fixed_qvel + index * GEAR_SONIC_DUEL_QVEL_DIM,
            vector.data[index]->qvel,
            GEAR_SONIC_DUEL_QVEL_DIM * sizeof(double));
    }
    memcpy(
        fixed_action,
        vector.controller.actions_policy,
        action_count * sizeof(float));
    memcpy(
        fixed_target,
        vector.controller.targets_mujoco,
        action_count * sizeof(float));
    memcpy(
        fixed_encoder,
        vector.controller.encoder_observations,
        encoder_count * sizeof(float));
    if (!gear_sonic_native_duel_reset(&vector, error, sizeof(error))) {
        reference_failure = "reset before reference-window step failed";
        goto reference_cleanup;
    }
    GearSonicNativeReferenceInput references = {
        .dof_position_mujoco = reference_position,
        .dof_next_position_mujoco = reference_next_position,
        .root_rotation_xyzw = reference_rotation,
    };
    if (!gear_sonic_native_duel_step_references(
            &vector, &references, error, sizeof(error))) {
        reference_failure = "reference-window regression step failed";
        goto reference_cleanup;
    }
    for (size_t index = 0; index < arenas; index++) {
        if (memcmp(
                fixed_qpos + index * GEAR_SONIC_DUEL_QPOS_DIM,
                vector.data[index]->qpos,
                GEAR_SONIC_DUEL_QPOS_DIM * sizeof(double)) != 0
                || memcmp(
                    fixed_qvel + index * GEAR_SONIC_DUEL_QVEL_DIM,
                    vector.data[index]->qvel,
                    GEAR_SONIC_DUEL_QVEL_DIM * sizeof(double)) != 0) {
            reference_failure = "reference-window state differs from fixed wrapper";
            goto reference_cleanup;
        }
    }
    if (memcmp(
            fixed_action,
            vector.controller.actions_policy,
            action_count * sizeof(float)) != 0
            || memcmp(
                fixed_target,
                vector.controller.targets_mujoco,
                action_count * sizeof(float)) != 0) {
        reference_failure = "reference-window controller output differs from fixed wrapper";
        goto reference_cleanup;
    }
    if (!gear_sonic_native_duel_reset(&vector, error, sizeof(error))) {
        reference_failure = "reset before per-fighter reference test failed";
        goto reference_cleanup;
    }
    for (size_t arena_index = 0; arena_index < arenas; arena_index++) {
        const size_t opponent_row
            = arena_index * GEAR_SONIC_DUEL_FIGHTERS + GEAR_SONIC_DUEL_OPPONENT;
        const size_t reference_row = opponent_row * GEAR_SONIC_HISTORY_FRAMES;
        reference_position[reference_row * GEAR_SONIC_ACTION_DIM] += 0.125f;
    }
    if (!gear_sonic_native_duel_step_references(
            &vector, &references, error, sizeof(error))) {
        reference_failure = "per-fighter reference-window step failed";
        goto reference_cleanup;
    }
    for (size_t arena_index = 0; arena_index < arenas; arena_index++) {
        const size_t player_row
            = arena_index * GEAR_SONIC_DUEL_FIGHTERS + GEAR_SONIC_DUEL_PLAYER;
        const size_t opponent_row
            = arena_index * GEAR_SONIC_DUEL_FIGHTERS + GEAR_SONIC_DUEL_OPPONENT;
        if (memcmp(
                fixed_encoder + player_row * GEAR_SONIC_ENCODER_INPUT_WIDTH,
                vector.controller.encoder_observations
                    + player_row * GEAR_SONIC_ENCODER_INPUT_WIDTH,
                GEAR_SONIC_ENCODER_INPUT_WIDTH * sizeof(float)) != 0) {
            reference_failure = "opponent reference change contaminated player row";
            goto reference_cleanup;
        }
        if (memcmp(
                fixed_encoder + opponent_row * GEAR_SONIC_ENCODER_INPUT_WIDTH,
                vector.controller.encoder_observations
                    + opponent_row * GEAR_SONIC_ENCODER_INPUT_WIDTH,
                GEAR_SONIC_ENCODER_INPUT_WIDTH * sizeof(float)) == 0) {
            reference_failure = "opponent reference change was silently ignored";
            goto reference_cleanup;
        }
    }
    if (!gear_sonic_native_duel_reset(&vector, error, sizeof(error))) {
        reference_failure = "reset before successful post-step observer failed";
        goto reference_cleanup;
    }
    PostStepProbe successful_probe = {
        .vector = &vector,
        .arena_count = arenas,
    };
    if (!gear_sonic_native_duel_step_fixed_with_post_step_observer(
            &vector,
            observe_post_step,
            &successful_probe,
            error,
            sizeof(error))
            || successful_probe.invalid
            || successful_probe.calls != arenas * 10u) {
        reference_failure = "successful post-step observer contract failed";
        goto reference_cleanup;
    }
    for (size_t row = 0; row < vector.robot_count; row++) {
        if (vector.policy_ticks[row] != 1u) {
            reference_failure = "successful observed step did not commit policy tick";
            goto reference_cleanup;
        }
    }
    if (!gear_sonic_native_duel_reset(&vector, error, sizeof(error))) {
        reference_failure = "reset before rejected post-step observer failed";
        goto reference_cleanup;
    }
    const size_t rejected_arena = arenas > 1u ? 1u : 0u;
    const uint32_t rejected_substep = 3u;
    char rejected_arena_text[32];
    char rejected_substep_text[32];
    (void)snprintf(
        rejected_arena_text,
        sizeof(rejected_arena_text),
        "arena %zu",
        rejected_arena);
    (void)snprintf(
        rejected_substep_text,
        sizeof(rejected_substep_text),
        "substep %u",
        (unsigned int)rejected_substep);
    PostStepProbe rejected_probe = {
        .vector = &vector,
        .arena_count = arenas,
        .fail_arena = rejected_arena,
        .fail_substep = rejected_substep,
        .fail_enabled = 1,
    };
    if (gear_sonic_native_duel_step_references_with_post_step_observer(
            &vector,
            &references,
            observe_post_step,
            &rejected_probe,
            error,
            sizeof(error)) != 0
            || !vector.failed
            || rejected_probe.invalid
            || rejected_probe.calls
                != (size_t)rejected_substep * arenas + rejected_arena + 1u
            || strstr(error, rejected_arena_text) == NULL
            || strstr(error, rejected_substep_text) == NULL) {
        reference_failure = "rejected post-step observer contract failed";
        goto reference_cleanup;
    }
    const double rejected_time = 0.002 * (double)(rejected_substep + 1u);
    for (size_t arena_index = 0; arena_index < arenas; arena_index++) {
        if (fabs(vector.data[arena_index]->time - rejected_time) > 1e-12) {
            reference_failure = "observer rejection advanced a later substep";
            goto reference_cleanup;
        }
    }
    for (size_t row = 0; row < vector.robot_count; row++) {
        if (vector.policy_ticks[row] != 0u) {
            reference_failure = "rejected observed step committed policy tick";
            goto reference_cleanup;
        }
    }
    if (!gear_sonic_native_duel_reset(&vector, error, sizeof(error))) {
        reference_failure = "reset after rejected observer before invalid reference failed";
        goto reference_cleanup;
    }
    memset(reference_rotation, 0, 4 * sizeof(float));
    if (gear_sonic_native_duel_step_references(
            &vector, &references, error, sizeof(error)) != 0
            || !vector.failed
            || strstr(error, "unit quaternion") == NULL) {
        reference_failure = "invalid reference quaternion was not rejected";
        goto reference_cleanup;
    }
    for (size_t arena_index = 0; arena_index < arenas; arena_index++) {
        if (vector.data[arena_index]->time != 0.0) {
            reference_failure = "invalid reference rejection advanced physics";
            goto reference_cleanup;
        }
    }
    if (!gear_sonic_native_duel_reset(&vector, error, sizeof(error))) {
        reference_failure = "reset after invalid reference rejection failed";
        goto reference_cleanup;
    }

reference_cleanup:
    free(fixed_encoder);
    free(fixed_target);
    free(fixed_action);
    free(fixed_qvel);
    free(fixed_qpos);
    free(reference_rotation);
    free(reference_next_position);
    free(reference_position);
    if (reference_failure != NULL) {
        fprintf(stderr, "%s\n", error);
        return fail(reference_failure, &vector, dof, position, rotation);
    }
    printf(
        "{\"arena_count\":%zu,\"physical_robot_count\":%zu,"
        "\"distinct_mjdata\":true,\"spawn_prefixes_verified\":true,"
        "\"semantic_input_rejected_without_step\":true,"
        "\"reset_recovered\":true,\"fixed_reference_equivalent\":true,"
        "\"per_fighter_reference_isolated\":true,"
        "\"serial_post_step_observer\":true,"
        "\"observer_rejection_stopped_later_substeps\":true,"
         "\"invalid_reference_rejected_without_step\":true,"
         "\"repeated_open_rejected_without_mutation\":true,"
         "\"rek_parity_claim\":false}\n",
        vector.arena_count,
        vector.robot_count);
    gear_sonic_native_duel_close(&vector);
    free(rotation);
    free(position);
    free(dof);
    return 0;
}
