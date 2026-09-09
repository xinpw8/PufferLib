#include "sonic_motion_composer_native.h"

#include <fenv.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_FRAMES 10
#define REFERENCE_FRAMES 5

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
            return 0; \
        } \
    } while (0)

#define CHECK_STATUS(expression, expected) \
    do { \
        SonicMotionComposerNativeStatus check_status = (expression); \
        if (check_status != (expected)) { \
            fprintf(stderr, \
                "%s:%d: status %s, expected %s: %s\n", \
                __FILE__, \
                __LINE__, \
                sonic_motion_composer_native_status_string(check_status), \
                sonic_motion_composer_native_status_string(expected), \
                #expression); \
            return 0; \
        } \
    } while (0)

typedef struct TestClipStorage {
    float dofs[TEST_FRAMES * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float roots[TEST_FRAMES * 4];
    SonicMotionComposerNativeClip clip;
} TestClipStorage;

typedef struct ReferenceClipStorage {
    float dofs[REFERENCE_FRAMES * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float roots[REFERENCE_FRAMES * 4];
    SonicMotionComposerNativeClip clip;
} ReferenceClipStorage;

typedef struct TestBackendContext {
    int return_numerator;
    float matched_cursor;
    float last_atan_numerator;
    float last_atan_denominator;
    float last_sin_cos_angle;
    int matcher_calls;
} TestBackendContext;

typedef struct ReferenceFixture {
    ReferenceClipStorage old_storage;
    ReferenceClipStorage new_storage;
    TestBackendContext backend_context;
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeReferenceTiming timing;
    uint32_t mirror_indices[SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    uint8_t mirror_negate[SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    SonicMotionComposerNativeMirrorTable mirror_table;
    float current_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float next_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float root_rows[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4];
} ReferenceFixture;

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int same_float(float actual, float expected) {
    return float_bits(actual) == float_bits(expected);
}

static SonicMotionComposerNativeConfig make_config(
        int mirror,
        int loop,
        float speed,
        int32_t start,
        int32_t end,
        float blend_in,
        float blend_out,
        float yaw_blend) {
    SonicMotionComposerNativeConfig config;
    config.mirror = mirror;
    config.loop = loop;
    config.playback_speed = speed;
    config.start_frame = start;
    config.end_frame = end;
    config.blend_in_seconds = blend_in;
    config.blend_out_seconds = blend_out;
    config.yaw_blend = yaw_blend;
    return config;
}

static void init_test_clip(TestClipStorage* storage, size_t frames, float fps) {
    memset(storage, 0, sizeof(*storage));
    for (size_t frame = 0; frame < frames; frame++) {
        for (size_t joint = 0;
                joint < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
                joint++) {
            storage->dofs[
                frame * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT + joint]
                = (float)(frame * 64 + joint) * 0.25f;
        }
        storage->roots[frame * 4 + 0] = 1.0f;
        storage->roots[frame * 4 + 1] = 0.0f;
        storage->roots[frame * 4 + 2] = 0.0f;
        storage->roots[frame * 4 + 3] = (float)frame * 0.0625f;
    }
    storage->clip.dof_position_mujoco = storage->dofs;
    storage->clip.root_quaternion_wxyz = storage->roots;
    storage->clip.dof_position_count
        = frames * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    storage->clip.root_quaternion_count = frames * 4;
    storage->clip.frame_count = frames;
    storage->clip.fps = fps;
}

static void init_reference_clip(
        ReferenceClipStorage* storage,
        float dof_base,
        float root_w_base) {
    memset(storage, 0, sizeof(*storage));
    for (size_t frame = 0; frame < REFERENCE_FRAMES; frame++) {
        for (size_t joint = 0;
                joint < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
                joint++) {
            storage->dofs[
                frame * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT + joint]
                = dof_base + (float)(frame * 64 + joint) * 0.25f;
        }
        storage->roots[frame * 4 + 0]
            = root_w_base + (float)frame * 0.125f;
        storage->roots[frame * 4 + 1]
            = (float)(frame + 1) * 0.125f;
        storage->roots[frame * 4 + 2]
            = (float)(frame + 1) * 0.0625f;
        storage->roots[frame * 4 + 3]
            = (float)(frame + 1) * 0.03125f;
    }
    storage->clip.dof_position_mujoco = storage->dofs;
    storage->clip.root_quaternion_wxyz = storage->roots;
    storage->clip.dof_position_count
        = REFERENCE_FRAMES * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    storage->clip.root_quaternion_count = REFERENCE_FRAMES * 4;
    storage->clip.frame_count = REFERENCE_FRAMES;
    storage->clip.fps = 50.0f;
}

static int linear_slerp(
        void* context,
        const float a[4],
        const float b[4],
        float t,
        float output[4]) {
    (void)context;
    for (size_t index = 0; index < 4; index++) {
        output[index] = (float)(
            (double)a[index]
            + ((double)b[index] - (double)a[index]) * (double)t);
    }
    return 1;
}

static int test_atan2(
        void* context,
        float numerator,
        float denominator,
        float* output) {
    TestBackendContext* backend = context;
    backend->last_atan_numerator = numerator;
    backend->last_atan_denominator = denominator;
    *output = backend->return_numerator ? numerator : 0.4f;
    return 1;
}

static int fixed_sin_cos(
        void* context,
        float angle,
        float* sine,
        float* cosine) {
    TestBackendContext* backend = context;
    backend->last_sin_cos_angle = angle;
    *sine = 0.2f;
    *cosine = 0.8f;
    return 1;
}

static int fixed_loop_entry(
        void* context,
        const SonicMotionComposerNativeLayer* target,
        const SonicMotionComposerNativeLayer* outgoing,
        float* matched_cursor) {
    TestBackendContext* backend = context;
    CHECK(target->active);
    CHECK(outgoing->active);
    backend->matcher_calls += 1;
    *matched_cursor = backend->matched_cursor;
    return 1;
}

static SonicMotionComposerNativeBackends make_backends(
        TestBackendContext* context) {
    SonicMotionComposerNativeBackends backends;
    backends.quaternion_slerp = linear_slerp;
    backends.atan2_f = test_atan2;
    backends.sin_cos_f = fixed_sin_cos;
    backends.loop_entry_matcher = fixed_loop_entry;
    backends.context = context;
    return backends;
}

static int setup_reference_fixture(ReferenceFixture* fixture) {
    SonicMotionComposerNativeConfig old_config;
    SonicMotionComposerNativeConfig new_config;
    SonicMotionComposerNativeBackends backends;
    SonicMotionComposerNativeAdvanceResult advance;
    SonicMotionComposerNativeReferenceOutput output;
    memset(fixture, 0, sizeof(*fixture));
    init_reference_clip(&fixture->old_storage, 0.0f, 1.0f);
    init_reference_clip(&fixture->new_storage, 20.0f, 0.5f);
    fixture->backend_context.matched_cursor = 2.25f;
    backends = make_backends(&fixture->backend_context);
    CHECK_STATUS(
        sonic_motion_composer_native_init(&fixture->composer, 50, &backends),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    old_config = make_config(0, 1, 1.0f, 0, -1, 0.2f, 0.52f, 0.0f);
    new_config = make_config(1, 0, 1.0f, 0, -1, 0.04f, 0.7f, 0.5f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &fixture->composer,
            &fixture->old_storage.clip,
            &old_config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_advance(&fixture->composer, &advance),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &fixture->composer,
            &fixture->new_storage.clip,
            &new_config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(fixture->composer.w_in == 2);
    CHECK(fixture->composer.w_out == 26);
    CHECK(fixture->composer.w_total == 26);
    CHECK(fixture->composer.xt == 0);
    for (size_t row = 0;
            row < SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS;
            row++) {
        fixture->timing.current_offsets[row] = (int32_t)(row * 5);
        fixture->timing.next_offsets[row] = (int32_t)(row * 5 + 1);
    }
    for (size_t index = 0;
            index < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
            index++) {
        fixture->mirror_indices[index]
            = (uint32_t)(SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT - 1 - index);
        fixture->mirror_negate[index] = (uint8_t)(index % 2);
    }
    fixture->mirror_table.source_indices = fixture->mirror_indices;
    fixture->mirror_table.negate = fixture->mirror_negate;
    fixture->mirror_table.source_index_count
        = SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    fixture->mirror_table.negate_count
        = SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    output.dof_position_mujoco = fixture->current_rows;
    output.dof_next_position_mujoco = fixture->next_rows;
    output.root_rotation_xyzw = fixture->root_rows;
    output.dof_position_capacity
        = SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    output.dof_next_position_capacity = output.dof_position_capacity;
    output.root_rotation_capacity
        = SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4;
    CHECK_STATUS(
        sonic_motion_composer_native_build_reference_rows(
            &fixture->composer,
            &fixture->timing,
            &fixture->mirror_table,
            &output),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    return 1;
}

static int test_install_resolve_and_completion(void) {
    TestClipStorage storage;
    SonicMotionComposerNativeLayer layer;
    SonicMotionComposerNativeConfig config;
    SonicMotionComposerNativeResolvedFrames frames;
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeAdvanceResult advance;
    init_test_clip(&storage, TEST_FRAMES, 100.0f);
    memset(&layer, 0, sizeof(layer));
    config = make_config(1, 0, 0.5f, -4, 99, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_install_layer(
            &layer,
            &storage.clip,
            &config,
            50),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(layer.start_frame == 0);
    CHECK(layer.end_frame == 9);
    CHECK(same_float(layer.per_tick, 1.0f));
    CHECK_STATUS(
        sonic_motion_composer_native_resolve_frames(&layer, 2, &frames),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(frames.f0 == 2 && frames.f1 == 3 && same_float(frames.t, 0.0f));

    config = make_config(0, 0, -0.001f, 2, 7, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_install_layer(
            &layer,
            &storage.clip,
            &config,
            100),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(same_float(layer.speed, -0.01f));
    CHECK(same_float(layer.per_tick, -0.01f));
    CHECK(same_float(layer.cursor, 7.0f));

    CHECK_STATUS(
        sonic_motion_composer_native_init(&composer, 100, NULL),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    config = make_config(0, 0, 1.0f, 0, 2, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &composer,
            &storage.clip,
            &config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_advance(&composer, &advance),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(!advance.current.completed);
    CHECK_STATUS(
        sonic_motion_composer_native_advance(&composer, &advance),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(advance.current.completed);
    CHECK(!composer.action_playing);
    CHECK(composer.current_layer.active);
    CHECK_STATUS(
        sonic_motion_composer_native_advance(&composer, &advance),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(advance.current.completed);
    return 1;
}

static int test_loop_wrap_and_weight_bits(void) {
    TestClipStorage storage;
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeConfig config;
    SonicMotionComposerNativeAdvanceResult advance;
    float weight;
    init_test_clip(&storage, 5, 75.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_init(&composer, 50, NULL),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    config = make_config(0, 1, 1.0f, 0, 4, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &composer,
            &storage.clip,
            &config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    composer.current_layer.cursor = 3.75f;
    CHECK_STATUS(
        sonic_motion_composer_native_advance(&composer, &advance),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(advance.current.wrapped);
    CHECK(!advance.current.completed);
    CHECK(same_float(composer.current_layer.cursor, 0.25f));
    CHECK_STATUS(
        sonic_motion_composer_native_weight_current(1.0f, 2, 26, &weight),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(float_bits(weight) == UINT32_C(0x3eaf286c));
    CHECK_STATUS(
        sonic_motion_composer_native_xfade_at(1, -1, 2, 26, &weight),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(same_float(weight, 0.0f));
    return 1;
}

static int test_reference_fixture_values(void) {
    ReferenceFixture fixture;
    CHECK(setup_reference_fixture(&fixture));
    CHECK(same_float(fixture.current_rows[0], 16.0f));
    CHECK(same_float(fixture.current_rows[1], 16.25f));
    CHECK(float_bits(fixture.next_rows[0]) == UINT32_C(0x420f0d79));
    CHECK(isfinite(fixture.root_rows[0]));
    CHECK(same_float(
        fixture.backend_context.last_atan_numerator,
        -0.703125f));
    CHECK(same_float(fixture.backend_context.last_sin_cos_angle, -0.1f));
    return 1;
}

static int build_fractional_reference_rows(
        float current_rows[
            SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
            * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT],
        float next_rows[
            SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
            * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT],
        float root_rows[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4],
        SonicMotionComposerNative* composer,
        TestClipStorage* storage,
        TestBackendContext* context) {
    static const int32_t current_offsets[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS]
        = {-3, 0, 1, 2, 7, 8, 9, 10, 11, 23};
    static const int32_t next_offsets[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS]
        = {-1, 2, 3, 4, 9, 10, 11, 12, 13, 25};
    SonicMotionComposerNativeBackends backends;
    SonicMotionComposerNativeConfig config;
    SonicMotionComposerNativeReferenceTiming timing;
    SonicMotionComposerNativeReferenceOutput output;
    init_test_clip(storage, 5, 25.0f);
    memset(context, 0, sizeof(*context));
    backends = make_backends(context);
    CHECK_STATUS(
        sonic_motion_composer_native_init(composer, 50, &backends),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    config = make_config(0, 1, 1.0f, 0, 4, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            composer,
            &storage->clip,
            &config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    for (size_t row = 0;
            row < SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS;
            row++) {
        timing.current_offsets[row] = current_offsets[row];
        timing.next_offsets[row] = next_offsets[row];
    }
    output.dof_position_mujoco = current_rows;
    output.dof_next_position_mujoco = next_rows;
    output.root_rotation_xyzw = root_rows;
    output.dof_position_capacity
        = SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    output.dof_next_position_capacity = output.dof_position_capacity;
    output.root_rotation_capacity
        = SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4;
    CHECK_STATUS(
        sonic_motion_composer_native_build_reference_rows(
            composer,
            &timing,
            NULL,
            &output),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    return 1;
}

static int test_fractional_and_explicit_reference_timing(void) {
    float current_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float next_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float root_rows[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4];
    SonicMotionComposerNative composer;
    TestClipStorage storage;
    TestBackendContext context;
    CHECK(build_fractional_reference_rows(
        current_rows,
        next_rows,
        root_rows,
        &composer,
        &storage,
        &context));
    CHECK(same_float(composer.current_layer.per_tick, 0.5f));
    CHECK(same_float(current_rows[0], 56.0f));
    CHECK(same_float(current_rows[
        SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT], 0.0f));
    CHECK(same_float(current_rows[
        2 * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT], 8.0f));
    CHECK(same_float(next_rows[0], 32.0f));
    CHECK(isfinite(root_rows[0]));
    return 1;
}

static int test_fail_closed_backends_and_atomic_output(void) {
    TestClipStorage storage;
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeConfig config;
    SonicMotionComposerNativeReferenceTiming timing;
    SonicMotionComposerNativeReferenceOutput output;
    SonicMotionComposerNativeBackends backends;
    TestBackendContext context;
    float current_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float next_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float roots[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4];
    init_test_clip(&storage, 5, 50.0f);
    memset(&context, 0, sizeof(context));
    memset(&timing, 0, sizeof(timing));
    for (size_t index = 0;
            index < sizeof(current_rows) / sizeof(current_rows[0]);
            index++) {
        current_rows[index] = 123.0f;
        next_rows[index] = 123.0f;
    }
    for (size_t index = 0; index < sizeof(roots) / sizeof(roots[0]); index++) {
        roots[index] = 123.0f;
    }
    output.dof_position_mujoco = current_rows;
    output.dof_next_position_mujoco = next_rows;
    output.root_rotation_xyzw = roots;
    output.dof_position_capacity = sizeof(current_rows) / sizeof(current_rows[0]);
    output.dof_next_position_capacity = sizeof(next_rows) / sizeof(next_rows[0]);
    output.root_rotation_capacity = sizeof(roots) / sizeof(roots[0]);
    CHECK_STATUS(
        sonic_motion_composer_native_init(&composer, 50, NULL),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    config = make_config(0, 0, 1.0f, 0, -1, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &composer,
            &storage.clip,
            &config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_build_reference_rows(
            &composer,
            &timing,
            NULL,
            &output),
        SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_QUATERNION_SLERP);
    CHECK(same_float(current_rows[0], 123.0f));
    CHECK(same_float(next_rows[0], 123.0f));
    CHECK(same_float(roots[0], 123.0f));

    backends = make_backends(&context);
    backends.atan2_f = NULL;
    backends.sin_cos_f = NULL;
    backends.loop_entry_matcher = NULL;
    CHECK_STATUS(
        sonic_motion_composer_native_init(&composer, 50, &backends),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    config = make_config(0, 0, 1.0f, 0, -1, 0.1f, 0.1f, 1.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &composer,
            &storage.clip,
            &config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_build_reference_rows(
            &composer,
            &timing,
            NULL,
            &output),
        SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_ATAN2);
    composer.backends.atan2_f = test_atan2;
    CHECK_STATUS(
        sonic_motion_composer_native_build_reference_rows(
            &composer,
            &timing,
            NULL,
            &output),
        SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_SIN_COS);

    config = make_config(1, 0, 1.0f, 0, -1, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action_immediate(
            &composer,
            &storage.clip,
            &config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_build_reference_rows(
            &composer,
            &timing,
            NULL,
            &output),
        SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_MIRROR_TABLE);
    return 1;
}

static int test_loop_entry_matcher_and_independent_state(void) {
    TestClipStorage first_storage;
    TestClipStorage next_storage;
    SonicMotionComposerNative first;
    SonicMotionComposerNative second;
    SonicMotionComposerNativeConfig loop_config;
    SonicMotionComposerNativeBackends backends;
    SonicMotionComposerNativeAdvanceResult advance;
    TestBackendContext context;
    init_test_clip(&first_storage, 10, 50.0f);
    init_test_clip(&next_storage, 10, 50.0f);
    next_storage.dofs[0] = 99.0f;
    memset(&context, 0, sizeof(context));
    context.matched_cursor = 7.25f;
    CHECK_STATUS(
        sonic_motion_composer_native_init(&first, 50, NULL),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    loop_config = make_config(0, 1, 1.0f, 0, 9, 0.1f, 0.1f, 0.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &first,
            &first_storage.clip,
            &loop_config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    first.current_layer.cursor = 3.0f;
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &first,
            &next_storage.clip,
            &loop_config),
        SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_LOOP_ENTRY_MATCHER);
    CHECK(same_float(first.current_layer.cursor, 3.0f));
    CHECK(!first.from_layer.active);

    backends = make_backends(&context);
    first.backends = backends;
    CHECK_STATUS(
        sonic_motion_composer_native_play_action_immediate(
            &first,
            &next_storage.clip,
            &loop_config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(context.matcher_calls == 1);
    CHECK(same_float(first.current_layer.cursor, 7.25f));
    CHECK(!first.from_layer.active);

    CHECK_STATUS(
        sonic_motion_composer_native_init(&second, 50, &backends),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &second,
            &first_storage.clip,
            &loop_config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_composer_native_advance(&second, &advance),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(same_float(second.current_layer.cursor, 1.0f));
    CHECK(same_float(first.current_layer.cursor, 7.25f));
    CHECK_STATUS(
        sonic_motion_composer_native_advance(&first, &advance),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(same_float(second.current_layer.cursor, 1.0f));
    CHECK(same_float(first.current_layer.cursor, 8.25f));
    return 1;
}

static int test_heading_seam_state_and_consumption(void) {
    TestClipStorage storage;
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeConfig config;
    SonicMotionComposerNativeBackends backends;
    SonicMotionComposerNativeAdvanceResult advance;
    TestBackendContext context;
    float heading_delta;
    float ownership;
    init_test_clip(&storage, 4, 50.0f);
    for (size_t frame = 0; frame < 4; frame++) {
        storage.roots[frame * 4 + 0] = 1.0f;
        storage.roots[frame * 4 + 1] = 0.0f;
        storage.roots[frame * 4 + 2] = 0.0f;
        storage.roots[frame * 4 + 3] = (float)frame * 0.0625f;
    }
    memset(&context, 0, sizeof(context));
    context.return_numerator = 1;
    backends = make_backends(&context);
    CHECK_STATUS(
        sonic_motion_composer_native_init(&composer, 50, &backends),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    config = make_config(0, 1, 1.0f, 0, 3, 0.1f, 0.1f, 1.0f);
    CHECK_STATUS(
        sonic_motion_composer_native_play_action(
            &composer,
            &storage.clip,
            &config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    for (size_t tick = 0; tick < 5; tick++) {
        CHECK_STATUS(
            sonic_motion_composer_native_advance(&composer, &advance),
            SONIC_MOTION_COMPOSER_NATIVE_OK);
    }
    CHECK(!composer.current_layer.heading_resync);
    CHECK(same_float(composer.current_layer.last_heading_delta, 0.125f));
    CHECK(same_float(composer.pending_heading_delta, 0.5f));
    CHECK_STATUS(
        sonic_motion_composer_native_heading_clip_ownership(
            &composer,
            &ownership),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(same_float(ownership, 1.0f));
    CHECK_STATUS(
        sonic_motion_composer_native_consume_heading_delta(
            &composer,
            &heading_delta),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(same_float(heading_delta, 0.5f));
    CHECK(same_float(composer.pending_heading_delta, 0.0f));
    return 1;
}

static int test_nondefault_rounding_mode_fails_closed(void) {
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeStatus status;
    int restore_status;
    CHECK(fegetround() == FE_TONEAREST);
    CHECK(fesetround(FE_DOWNWARD) == 0);
    status = sonic_motion_composer_native_init(&composer, 50, NULL);
    restore_status = fesetround(FE_TONEAREST);
    CHECK(restore_status == 0);
    CHECK_STATUS(
        status,
        SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT);
    return 1;
}

static int run_all_tests(void) {
    CHECK(test_install_resolve_and_completion());
    CHECK(test_loop_wrap_and_weight_bits());
    CHECK(test_reference_fixture_values());
    CHECK(test_fractional_and_explicit_reference_timing());
    CHECK(test_fail_closed_backends_and_atomic_output());
    CHECK(test_loop_entry_matcher_and_independent_state());
    CHECK(test_heading_seam_state_and_consumption());
    CHECK(test_nondefault_rounding_mode_fails_closed());
    return 1;
}

static void print_bits_line(const char* label, const float* values, size_t count) {
    printf("%s", label);
    for (size_t index = 0; index < count; index++) {
        printf(" %08" PRIx32, float_bits(values[index]));
    }
    putchar('\n');
}

static int print_reference_probe(void) {
    ReferenceFixture fixture;
    float fractional_current[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float fractional_next[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float fractional_root[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4];
    SonicMotionComposerNative fractional_composer;
    TestClipStorage fractional_storage;
    TestBackendContext fractional_context;
    if (!setup_reference_fixture(&fixture)) {
        return 0;
    }
    print_bits_line(
        "current",
        fixture.current_rows,
        sizeof(fixture.current_rows) / sizeof(fixture.current_rows[0]));
    print_bits_line(
        "next",
        fixture.next_rows,
        sizeof(fixture.next_rows) / sizeof(fixture.next_rows[0]));
    print_bits_line(
        "root",
        fixture.root_rows,
        sizeof(fixture.root_rows) / sizeof(fixture.root_rows[0]));
    printf(
        "state %" PRId32 " %" PRId32 " %" PRId32 " %" PRId32
        " %08" PRIx32 " %08" PRIx32 "\n",
        fixture.composer.xt,
        fixture.composer.w_in,
        fixture.composer.w_out,
        fixture.composer.w_total,
        float_bits(fixture.composer.current_layer.cursor),
        float_bits(fixture.composer.from_layer.cursor));
    if (!build_fractional_reference_rows(
            fractional_current,
            fractional_next,
            fractional_root,
            &fractional_composer,
            &fractional_storage,
            &fractional_context)) {
        return 0;
    }
    print_bits_line(
        "fractional-current",
        fractional_current,
        sizeof(fractional_current) / sizeof(fractional_current[0]));
    print_bits_line(
        "fractional-next",
        fractional_next,
        sizeof(fractional_next) / sizeof(fractional_next[0]));
    print_bits_line(
        "fractional-root",
        fractional_root,
        sizeof(fractional_root) / sizeof(fractional_root[0]));
    printf(
        "fractional-state %08" PRIx32 " %08" PRIx32 "\n",
        float_bits(fractional_composer.current_layer.cursor),
        float_bits(fractional_composer.current_layer.per_tick));
    return 1;
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--probe") == 0) {
        return print_reference_probe() ? 0 : 1;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--probe]\n", argv[0]);
        return 2;
    }
    if (!run_all_tests()) {
        return 1;
    }
    puts("sonic_motion_composer_native tests passed");
    return 0;
}
