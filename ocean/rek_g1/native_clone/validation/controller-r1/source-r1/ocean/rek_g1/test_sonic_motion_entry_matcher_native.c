#include "sonic_motion_entry_matcher_native.h"

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_FRAMES 6

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
        SonicMotionEntryMatcherNativeStatus check_status = (expression); \
        if (check_status != (expected)) { \
            fprintf(stderr, \
                "%s:%d: status %s, expected %s: %s\n", \
                __FILE__, \
                __LINE__, \
                sonic_motion_entry_matcher_native_status_string(check_status), \
                sonic_motion_entry_matcher_native_status_string(expected), \
                #expression); \
            return 0; \
        } \
    } while (0)

#define CHECK_COMPOSER_STATUS(expression, expected) \
    do { \
        SonicMotionComposerNativeStatus check_status = (expression); \
        if (check_status != (expected)) { \
            fprintf(stderr, \
                "%s:%d: composer status %s, expected %s: %s\n", \
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

typedef struct BakeContext {
    size_t calls;
    size_t fail_at;
    int emit_non_finite;
} BakeContext;

static uint32_t float_bits(float value) {
    uint32_t result = 0;
    memcpy(&result, &value, sizeof(result));
    return result;
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
        float blend_out) {
    SonicMotionComposerNativeConfig config;
    config.mirror = mirror;
    config.loop = loop;
    config.playback_speed = speed;
    config.start_frame = start;
    config.end_frame = end;
    config.blend_in_seconds = blend_in;
    config.blend_out_seconds = blend_out;
    config.yaw_blend = 0.0f;
    return config;
}

static void init_clip(TestClipStorage* storage, float dof_bias) {
    memset(storage, 0, sizeof(*storage));
    for (size_t frame = 0; frame < TEST_FRAMES; frame++) {
        for (size_t joint = 0;
                joint < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
                joint++) {
            storage->dofs[
                frame * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT + joint]
                = dof_bias + (float)(frame * 32 + joint) * 0.125f;
        }
        storage->roots[frame * 4] = 1.0f;
    }
    storage->clip.dof_position_mujoco = storage->dofs;
    storage->clip.root_quaternion_wxyz = storage->roots;
    storage->clip.dof_position_count
        = TEST_FRAMES * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    storage->clip.root_quaternion_count = TEST_FRAMES * 4;
    storage->clip.frame_count = TEST_FRAMES;
    storage->clip.fps = 50.0f;
}

static int bake_sampler(
        void* raw_context,
        const float dof[SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT],
        float output[SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH]) {
    BakeContext* context = (BakeContext*)raw_context;
    size_t call = context->calls++;
    if (call == context->fail_at) {
        return 0;
    }
    for (size_t index = 0;
            index < SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH;
            index++) {
        output[index] = dof[index];
    }
    if (context->emit_non_finite && call == 1) {
        output[4] = NAN;
    }
    return 1;
}

static void fill_scalar_features(float output[TEST_FRAMES * 6]) {
    static const float values[TEST_FRAMES] = {
        10.0f, 3.0f, 1.0f, 0.95f, -2.0f, -5.0f
    };
    for (size_t frame = 0; frame < TEST_FRAMES; frame++) {
        for (size_t component = 0; component < 6; component++) {
            output[frame * 6 + component] = values[frame];
        }
    }
}

static void fill_linear_features(float output[TEST_FRAMES * 6]) {
    for (size_t frame = 0; frame < TEST_FRAMES; frame++) {
        for (size_t component = 0; component < 6; component++) {
            output[frame * 6 + component] = (float)frame;
        }
    }
}

static int test_make_and_sample(void) {
    const float raw[6] = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
    const float two_rows[12] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f
    };
    SonicMotionEntryMatcherNativeFootFeature feature;

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_make_feature(raw, 0, &feature),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(same_float(feature.left_xyz[0], 1.0f));
    CHECK(same_float(feature.left_xyz[1], -2.0f));
    CHECK(same_float(feature.left_xyz[2], 3.0f));
    CHECK(same_float(feature.right_xyz[0], 4.0f));
    CHECK(same_float(feature.right_xyz[1], -5.0f));
    CHECK(same_float(feature.right_xyz[2], 6.0f));

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_make_feature(raw, 1, &feature),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(same_float(feature.left_xyz[0], -4.0f));
    CHECK(same_float(feature.left_xyz[1], -5.0f));
    CHECK(same_float(feature.left_xyz[2], 6.0f));
    CHECK(same_float(feature.right_xyz[0], -1.0f));
    CHECK(same_float(feature.right_xyz[1], -2.0f));
    CHECK(same_float(feature.right_xyz[2], 3.0f));

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_sample_at(
            two_rows, 2, -17, 0, &feature),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(same_float(feature.left_xyz[0], 1.0f));
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_sample_at(
            two_rows, 2, 17, 0, &feature),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(same_float(feature.right_xyz[2], 12.0f));

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_sample_lerp(
            two_rows, 2, 0.25f, 0, &feature),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(same_float(feature.left_xyz[0], 1.25f));
    CHECK(same_float(feature.left_xyz[1], 2.5f));
    CHECK(same_float(feature.left_xyz[2], 3.75f));
    CHECK(same_float(feature.right_xyz[0], 5.0f));
    CHECK(same_float(feature.right_xyz[1], 6.25f));
    CHECK(same_float(feature.right_xyz[2], 7.5f));

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_sample_lerp(
            two_rows, 2, 0.25f, 1, &feature),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(same_float(feature.left_xyz[0], -5.0f));
    CHECK(same_float(feature.left_xyz[1], 6.25f));
    CHECK(same_float(feature.left_xyz[2], 7.5f));
    CHECK(same_float(feature.right_xyz[0], -1.25f));
    CHECK(same_float(feature.right_xyz[1], 2.5f));
    CHECK(same_float(feature.right_xyz[2], 3.75f));

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_sample_lerp(
            two_rows, 2, NAN, 0, &feature),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE);
    return 1;
}

static int test_clip_view_and_bake(void) {
    TestClipStorage storage;
    const float* dofs = NULL;
    size_t frame_count = 0;
    size_t row_width = 0;
    float features[TEST_FRAMES * 6];
    BakeContext context = {0, SIZE_MAX, 0};

    init_clip(&storage, 0.5f);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_clip_dof_positions(
            &storage.clip, &dofs, &frame_count, &row_width),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(dofs == storage.dofs);
    CHECK(frame_count == TEST_FRAMES);
    CHECK(row_width == SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT);

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_bake_features(
            &storage.clip,
            bake_sampler,
            &context,
            features,
            TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(context.calls == TEST_FRAMES);
    for (size_t frame = 0; frame < TEST_FRAMES; frame++) {
        for (size_t component = 0; component < 6; component++) {
            CHECK(same_float(
                features[frame * 6 + component],
                storage.dofs[
                    frame * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT
                    + component]));
        }
    }

    context.calls = 0;
    context.fail_at = 2;
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_bake_features(
            &storage.clip,
            bake_sampler,
            &context,
            features,
            TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_KINEMATICS_FAILURE);
    CHECK(context.calls == 3);

    context.calls = 0;
    context.fail_at = SIZE_MAX;
    context.emit_non_finite = 1;
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_bake_features(
            &storage.clip,
            bake_sampler,
            &context,
            features,
            TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE);
    CHECK(context.calls == 2);
    return 1;
}

static int setup_layers(
        TestClipStorage* outgoing_storage,
        TestClipStorage* target_storage,
        SonicMotionComposerNativeLayer* outgoing,
        SonicMotionComposerNativeLayer* target) {
    SonicMotionComposerNativeConfig outgoing_config
        = make_config(0, 1, 1.0f, 0, 5, 0.1f, 0.06f);
    SonicMotionComposerNativeConfig target_config
        = make_config(0, 1, 1.0f, 1, 4, 0.04f, 0.1f);
    init_clip(outgoing_storage, 0.0f);
    init_clip(target_storage, 100.0f);
    memset(outgoing, 0, sizeof(*outgoing));
    memset(target, 0, sizeof(*target));
    if (sonic_motion_composer_native_install_layer(
            outgoing, &outgoing_storage->clip, &outgoing_config, 50)
            != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return 0;
    }
    if (sonic_motion_composer_native_install_layer(
            target, &target_storage->clip, &target_config, 50)
            != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return 0;
    }
    outgoing->cursor = 5.75f;
    return 1;
}

static int test_registry_and_match_fixture(void) {
    TestClipStorage outgoing_storage;
    TestClipStorage target_storage;
    SonicMotionComposerNativeLayer outgoing;
    SonicMotionComposerNativeLayer target;
    SonicMotionEntryMatcherNativeFeatureSlot slots[2];
    SonicMotionEntryMatcherNative matcher;
    float outgoing_features[TEST_FRAMES * 6];
    float target_features[TEST_FRAMES * 6];
    float matched = -99.0f;
    float replacement_features[TEST_FRAMES * 6];

    CHECK(setup_layers(
        &outgoing_storage, &target_storage, &outgoing, &target));
    fill_linear_features(outgoing_features);
    fill_scalar_features(target_features);
    memcpy(replacement_features, target_features, sizeof(target_features));
    replacement_features[3 * 6] = 77.0f;

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_init(&matcher, 50, slots, 2),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_match(
            &matcher, &target, &outgoing, &matched),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURES_NOT_REGISTERED);
    CHECK(!matcher.diagnostics.valid);
    CHECK(same_float(matched, -99.0f));

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher,
            &outgoing_storage.clip,
            outgoing_features,
            TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher,
            &target_storage.clip,
            target_features,
            TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(matcher.slot_count == 2);

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_match(
            &matcher, &target, &outgoing, &matched),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(matcher.diagnostics.valid);
    CHECK(matcher.diagnostics.target_best_frame == 3);
    CHECK(float_bits(matcher.diagnostics.transition_center_ticks)
        == UINT32_C(0x3f99999a));
    CHECK(float_bits(matcher.diagnostics.outgoing_feature_cursor)
        == UINT32_C(0x3f733330));
    CHECK(float_bits(matcher.diagnostics.best_squared_distance)
        == UINT32_C(0x2a580000));
    CHECK(float_bits(matched) == UINT32_C(0x3fe66666));

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher,
            &target_storage.clip,
            replacement_features,
            TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(matcher.slot_count == 2);
    CHECK(slots[1].root_local_foot_xyz == replacement_features);

    {
        TestClipStorage third;
        float third_features[TEST_FRAMES * 6] = {0};
        init_clip(&third, 200.0f);
        CHECK_STATUS(
            sonic_motion_entry_matcher_native_register(
                &matcher, &third.clip, third_features, TEST_FRAMES * 6),
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_REGISTRY_FULL);
    }
    return 1;
}

static int test_mirror_tie_and_nonloop_clamp(void) {
    TestClipStorage outgoing_storage;
    TestClipStorage target_storage;
    SonicMotionComposerNativeLayer outgoing;
    SonicMotionComposerNativeLayer target;
    SonicMotionEntryMatcherNativeFeatureSlot slots[2];
    SonicMotionEntryMatcherNative matcher;
    float outgoing_features[TEST_FRAMES * 6];
    float target_features[TEST_FRAMES * 6];
    float matched = 0.0f;

    CHECK(setup_layers(
        &outgoing_storage, &target_storage, &outgoing, &target));
    memset(outgoing_features, 0, sizeof(outgoing_features));
    memset(target_features, 0, sizeof(target_features));
    outgoing.config.mirror = 1;
    target.config.mirror = 1;
    target.config.loop = 0;
    target.per_tick = -1.0f;
    target.start_frame = 1;
    target.end_frame = 4;

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_init(&matcher, 50, slots, 2),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher, &outgoing_storage.clip, outgoing_features, 36),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher, &target_storage.clip, target_features, 36),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_match(
            &matcher, &target, &outgoing, &matched),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(matcher.diagnostics.target_best_frame == 1);
    CHECK(same_float(matcher.diagnostics.best_squared_distance, 0.0f));
    CHECK(float_bits(matched) == UINT32_C(0x400ccccd));
    return 1;
}

static int test_distance_operation_order_and_saturation(void) {
    TestClipStorage outgoing_storage;
    TestClipStorage target_storage;
    SonicMotionComposerNativeLayer outgoing;
    SonicMotionComposerNativeLayer target;
    SonicMotionEntryMatcherNativeFeatureSlot slots[2];
    SonicMotionEntryMatcherNative matcher;
    float outgoing_features[TEST_FRAMES * 6] = {0};
    float target_features[TEST_FRAMES * 6] = {0};
    float matched = 0.0f;

    CHECK(setup_layers(
        &outgoing_storage, &target_storage, &outgoing, &target));
    target.start_frame = 1;
    target.end_frame = 1;
    target_features[6 + 3] = 13105.3857421875f;
    target_features[6 + 4] = -6566480.5f;
    target_features[6 + 5] = -2512.767578125f;

    CHECK_STATUS(
        sonic_motion_entry_matcher_native_init(&matcher, 50, slots, 2),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher, &outgoing_storage.clip, outgoing_features, 36),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher, &target_storage.clip, target_features, 36),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_match(
            &matcher, &target, &outgoing, &matched),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(float_bits(matcher.diagnostics.best_squared_distance)
        == UINT32_C(0x561cdd8e));

    for (size_t component = 0; component < 6; component++) {
        target_features[6 + component] = 2.0e19f;
    }
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_match(
            &matcher, &target, &outgoing, &matched),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(float_bits(matcher.diagnostics.best_squared_distance)
        == UINT32_C(0x7f7fffff));
    CHECK(matcher.diagnostics.target_best_frame == 1);
    return 1;
}

static int test_composer_callback_integration(void) {
    TestClipStorage outgoing_storage;
    TestClipStorage target_storage;
    SonicMotionEntryMatcherNativeFeatureSlot slots[2];
    SonicMotionEntryMatcherNative matcher;
    SonicMotionComposerNativeBackends backends;
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeConfig outgoing_config
        = make_config(0, 1, 1.0f, 0, 5, 0.1f, 0.06f);
    SonicMotionComposerNativeConfig target_config
        = make_config(0, 1, 1.0f, 1, 4, 0.04f, 0.1f);
    float outgoing_features[TEST_FRAMES * 6];
    float target_features[TEST_FRAMES * 6];

    init_clip(&outgoing_storage, 0.0f);
    init_clip(&target_storage, 100.0f);
    fill_linear_features(outgoing_features);
    fill_scalar_features(target_features);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_init(&matcher, 50, slots, 2),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher, &outgoing_storage.clip, outgoing_features, 36),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_register(
            &matcher, &target_storage.clip, target_features, 36),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    memset(&backends, 0, sizeof(backends));
    backends.loop_entry_matcher = sonic_motion_entry_matcher_native_callback;
    backends.context = &matcher;
    CHECK_COMPOSER_STATUS(
        sonic_motion_composer_native_init(&composer, 50, &backends),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK_COMPOSER_STATUS(
        sonic_motion_composer_native_play_action_immediate(
            &composer, &outgoing_storage.clip, &outgoing_config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    composer.current_layer.cursor = 5.75f;
    CHECK_COMPOSER_STATUS(
        sonic_motion_composer_native_play_action(
            &composer, &target_storage.clip, &target_config),
        SONIC_MOTION_COMPOSER_NATIVE_OK);
    CHECK(matcher.last_status == SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(float_bits(composer.current_layer.cursor) == UINT32_C(0x3fe66666));
    CHECK(composer.w_in == 2);
    CHECK(composer.w_out == 3);
    CHECK(composer.from_layer.active);

    {
        SonicMotionEntryMatcherNativeFeatureSlot missing_slots[1];
        SonicMotionEntryMatcherNative missing_matcher;
        SonicMotionComposerNative missing_composer;
        SonicMotionComposerNativeBackends missing_backends;
        CHECK_STATUS(
            sonic_motion_entry_matcher_native_init(
                &missing_matcher, 50, missing_slots, 1),
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
        CHECK_STATUS(
            sonic_motion_entry_matcher_native_register(
                &missing_matcher,
                &outgoing_storage.clip,
                outgoing_features,
                36),
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
        memset(&missing_backends, 0, sizeof(missing_backends));
        missing_backends.loop_entry_matcher
            = sonic_motion_entry_matcher_native_callback;
        missing_backends.context = &missing_matcher;
        CHECK_COMPOSER_STATUS(
            sonic_motion_composer_native_init(
                &missing_composer, 50, &missing_backends),
            SONIC_MOTION_COMPOSER_NATIVE_OK);
        CHECK_COMPOSER_STATUS(
            sonic_motion_composer_native_play_action_immediate(
                &missing_composer, &outgoing_storage.clip, &outgoing_config),
            SONIC_MOTION_COMPOSER_NATIVE_OK);
        missing_composer.current_layer.cursor = 5.75f;
        CHECK_COMPOSER_STATUS(
            sonic_motion_composer_native_play_action(
                &missing_composer, &target_storage.clip, &target_config),
            SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE);
        CHECK(missing_matcher.last_status
            == SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURES_NOT_REGISTERED);
        CHECK(missing_composer.current_layer.clip.dof_position_mujoco
            == outgoing_storage.clip.dof_position_mujoco);
        CHECK(same_float(missing_composer.current_layer.cursor, 5.75f));
    }
    return 1;
}

static int test_float_environment_guard(void) {
    int original = fegetround();
    SonicMotionEntryMatcherNative matcher;
    SonicMotionEntryMatcherNativeFeatureSlot slot;
    if (fesetround(FE_DOWNWARD) != 0) {
        return 1;
    }
    CHECK_STATUS(
        sonic_motion_entry_matcher_native_init(&matcher, 50, &slot, 1),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT);
    CHECK(fesetround(original) == 0);
    return 1;
}

static int test_match_entry_rounded_upper_boundary(void) {
    TestClipStorage outgoing_storage, target_storage;
    SonicMotionComposerNativeLayer outgoing, target;
    SonicMotionEntryMatcherNativeFeatureSlot slots[2];
    SonicMotionEntryMatcherNative matcher;
    float outgoing_features[TEST_FRAMES * 6] = {0};
    float target_features[TEST_FRAMES * 6];
    float matched = -1.0f;
    CHECK(setup_layers(&outgoing_storage, &target_storage, &outgoing, &target));
    for (int i = 0; i < TEST_FRAMES * 6; ++i) target_features[i] = 1.0f;
    memset(target_features + 6, 0, 6 * sizeof(float));
    outgoing.config.blend_out_seconds = target.config.blend_in_seconds = 0.08f;
    target.start_frame = 0;
    target.end_frame = TEST_FRAMES - 1;
    target.config.loop = 1;
    target.per_tick = nextafterf(0.5f, 1.0f);
    CHECK_STATUS(sonic_motion_entry_matcher_native_init(&matcher, 50, slots, 2),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(sonic_motion_entry_matcher_native_register(
        &matcher, &outgoing_storage.clip, outgoing_features, TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(sonic_motion_entry_matcher_native_register(
        &matcher, &target_storage.clip, target_features, TEST_FRAMES * 6),
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK_STATUS(sonic_motion_entry_matcher_native_match(
        &matcher, &target, &outgoing, &matched), SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK);
    CHECK(matcher.diagnostics.target_best_frame == 1);
    CHECK(same_float(matcher.diagnostics.transition_center_ticks, 2.0f));
    CHECK(same_float(matched, 0.0f));
    return 1;
}

int main(void) {
    CHECK(test_make_and_sample());
    CHECK(test_clip_view_and_bake());
    CHECK(test_registry_and_match_fixture());
    CHECK(test_mirror_tie_and_nonloop_clamp());
    CHECK(test_distance_operation_order_and_saturation());
    CHECK(test_composer_callback_integration());
    CHECK(test_match_entry_rounded_upper_boundary());
    CHECK(test_float_environment_guard());
    puts("sonic_motion_entry_matcher_native: all checks passed");
    return 0;
}
