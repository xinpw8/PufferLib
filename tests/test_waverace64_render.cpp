#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifndef PUFFER_WAVERACE64_RENDER
#error "compile this harness with PUFFER_WAVERACE64_RENDER"
#endif

#include "waverace64.h"

#define WR64_TEST_MASK_SIZE (15 + 9 + 2 + 2 + 2)
#define WR64_TEST_PATH_SIZE 4096

typedef struct TestEnv {
    WaveRace64 env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float reward;
    float terminal;
    unsigned char action_mask[WR64_TEST_MASK_SIZE];
} TestEnv;

typedef struct CoreDigest {
    uint64_t env;
    uint64_t rdram;
    uint64_t game_stack;
    uint64_t snapshot_rdram;
    uint64_t snapshot_stack;
    uint64_t observations;
    uint64_t actions;
    uint64_t reward;
    uint64_t terminal;
    uint64_t action_mask;
} CoreDigest;

static uint64_t hash_update(uint64_t hash, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t hash_memory(const void* data, size_t size) {
    assert(data || size == 0);
    return hash_update(UINT64_C(14695981039346656037), data, size);
}

static uint64_t hash_env_without_client(const WaveRace64* env) {
    const uint8_t* bytes = (const uint8_t*)env;
    const size_t client_offset = offsetof(WaveRace64, client);
    const size_t after_client = client_offset + sizeof(env->client);
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = hash_update(hash, bytes, client_offset);
    hash = hash_update(hash, bytes + after_client,
        sizeof(*env) - after_client);
    return hash;
}

static CoreDigest capture_core_digest(const TestEnv* test) {
    const WaveRace64* env = &test->env;
    CoreDigest digest = {};
    digest.env = hash_env_without_client(env);
    digest.rdram = hash_memory(env->machine.rdram, WR_RDRAM_SIZE);
    digest.game_stack = hash_memory(
        env->machine.game_stack, env->machine.game_stack_size);
    digest.snapshot_rdram = hash_memory(env->snap.rdram, env->snap.size);
    digest.snapshot_stack = hash_memory(
        env->snap.stack, env->snap.stack_size);
    digest.observations = hash_memory(
        test->observations, sizeof(test->observations));
    digest.actions = hash_memory(test->actions, sizeof(test->actions));
    digest.reward = hash_memory(&test->reward, sizeof(test->reward));
    digest.terminal = hash_memory(&test->terminal, sizeof(test->terminal));
    digest.action_mask = hash_memory(
        test->action_mask, sizeof(test->action_mask));
    return digest;
}

static void assert_core_equal(const CoreDigest* before,
        const CoreDigest* after) {
    assert(before->env == after->env);
    assert(before->rdram == after->rdram);
    assert(before->game_stack == after->game_stack);
    assert(before->snapshot_rdram == after->snapshot_rdram);
    assert(before->snapshot_stack == after->snapshot_stack);
    assert(before->observations == after->observations);
    assert(before->actions == after->actions);
    assert(before->reward == after->reward);
    assert(before->terminal == after->terminal);
    assert(before->action_mask == after->action_mask);
}

static void init_kwargs(Dict* kwargs, const char* rom_path) {
    dict_set_str(kwargs, "rom_path", rom_path);
    dict_set(kwargs, "frameskip", 2);
    dict_set(kwargs, "reward_speed", 0);
    dict_set(kwargs, "reward_progress", 3);
    dict_set(kwargs, "reward_slip", 0);
    dict_set(kwargs, "reward_checkpoint", 0.3);
    dict_set(kwargs, "reward_miss", 0.5);
    dict_set(kwargs, "reward_finish", 10);
    dict_set(kwargs, "reward_fail", 2);
    dict_set(kwargs, "discount", 0.9997499687421851);
    dict_set(kwargs, "reward_mode", 2);
    dict_set(kwargs, "curriculum_start_laps", 3);
    dict_set(kwargs, "curriculum_max_laps", 3);
    dict_set(kwargs, "curriculum_successes_per_lap", 1);
}

static void init_test_env(TestEnv* test, Dict* kwargs) {
    memset(test, 0, sizeof(*test));
    memset(test->action_mask, 1, sizeof(test->action_mask));
    puf_init(&test->env, kwargs);
    test->env.agents[0].observations = test->observations;
    test->env.agents[0].actions = test->actions;
    test->env.agents[0].rewards = &test->reward;
    test->env.agents[0].terminals = &test->terminal;
    test->env.agents[0].action_mask = test->action_mask;
    puf_eval_reset(&test->env);
}

static void set_action(TestEnv* test, int x, int y,
        int a, int b, int r) {
    test->actions[0] = (float)x;
    test->actions[1] = (float)y;
    test->actions[2] = (float)a;
    test->actions[3] = (float)b;
    test->actions[4] = (float)r;
}

static void set_lcg_action(TestEnv* test, uint32_t* rng) {
    *rng = *rng * UINT32_C(1664525) + UINT32_C(1013904223);
    int x = (int)(*rng % 15u);
    *rng = *rng * UINT32_C(1664525) + UINT32_C(1013904223);
    int y = (int)(*rng % 9u);
    *rng = *rng * UINT32_C(1664525) + UINT32_C(1013904223);
    set_action(test, x, y,
        (int)((*rng >> 0) & 1u),
        (int)((*rng >> 1) & 1u),
        (int)((*rng >> 2) & 1u));
}

static void test_training_client_remains_null(TestEnv* test) {
    assert(test->env.client == NULL);
    uint32_t rng = UINT32_C(0xC001D00D);
    int terminals = 0;
    for (int decision = 0; decision < 192; decision++) {
        set_lcg_action(test, &rng);
        puf_step(&test->env);
        terminals += test->terminal > 0.5f;
        assert(test->env.client == NULL);
    }
    assert(terminals > 0);
    puf_eval_reset(&test->env);
    assert(test->env.client == NULL);
    printf("PASS renderer-lazy training-decisions=192 terminals=%d\n",
        terminals);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void assert_float_bits(float actual, float expected) {
    assert(float_bits(actual) == float_bits(expected));
}

static void test_control_mode_toggle() {
    Client client = {};
    assert(client.human_control == 0);
    wr64_render_update_control_mode(&client, 0);
    assert(client.human_control == 0);
    wr64_render_update_control_mode(&client, 1);
    assert(client.human_control == 1);
    wr64_render_update_control_mode(&client, 1);
    assert(client.human_control == 1);
    wr64_render_update_control_mode(&client, 0);
    assert(client.human_control == 1);
    wr64_render_update_control_mode(&client, 1);
    assert(client.human_control == 0);
    wr64_render_update_control_mode(&client, 0);
    assert(client.human_control == 0);
    printf("PASS control-mode shift-up rising-edge toggle\n");
}

static void test_time_format() {
    char text[32];
    wr64_render_format_time(text, sizeof(text), 0);
    assert(strcmp(text, "0'00\"000") == 0);
    wr64_render_format_time(text, sizeof(text), 1000);
    assert(strcmp(text, "0'01\"000") == 0);
    wr64_render_format_time(text, sizeof(text), 72650);
    assert(strcmp(text, "1'12\"650") == 0);
    wr64_render_format_time(text, sizeof(text), -5);
    assert(strcmp(text, "0'00\"000") == 0);
    printf("PASS original-style race-clock formatting\n");
}

static void test_camera_wave_visibility() {
    Client client = {};
    WR64RenderState state = {};
    state.heading[1] = 1.f;
    state.tick = 100;
    state.position[1] = 100.f;
    wr64_render_update_camera_anchor(&client, &state);
    Camera3D first = wr64_render_camera(&client, &state);
    float first_relative_y = wr64_render_position(&state).y - first.position.y;

    state.tick += 2;
    state.position[1] = 124.066f;
    wr64_render_update_camera_anchor(&client, &state);
    Camera3D second = wr64_render_camera(&client, &state);
    float rider_y = wr64_render_position(&state).y;
    float second_relative_y = rider_y - second.position.y;
    assert(client.camera_y > 1.f);
    assert(client.camera_y < rider_y);
    assert(second.position.y - first.position.y < rider_y - 1.f);
    assert(fabsf(second_relative_y - first_relative_y) > 0.05f);

    state.tick += 2;
    state.position[1] = 250.f;
    state.recovery = 1;
    wr64_render_update_camera_anchor(&client, &state);
    assert(fabsf(client.camera_y - 2.5f) < 1e-6f);

    state.recovery = 0;
    state.tick = 0;
    state.position[1] = -50.f;
    wr64_render_update_camera_anchor(&client, &state);
    assert(fabsf(client.camera_y + 0.5f) < 1e-6f);
    printf("PASS camera exposes wave heave and snaps on recovery/reset\n");
}

static void test_final_lap_banner_state() {
    Client client = {};
    client.hud_lap = -1;
    client.final_lap_until_tick = -1;
    WR64RenderState state = {};
    state.target_laps = 3;
    state.lap = 1;
    state.tick = 100;
    wr64_render_update_hud_state(&client, &state);
    assert(client.final_lap_until_tick == -1);
    state.lap = 2;
    state.tick = 500;
    wr64_render_update_hud_state(&client, &state);
    assert(client.final_lap_until_tick == -1);
    state.lap = 3;
    state.tick = 900;
    wr64_render_update_hud_state(&client, &state);
    assert(client.final_lap_until_tick == 960);
    printf("PASS final-lap banner transition is state-driven\n");
}

static void assert_render_state_matches_authority(
        WaveRace64* env, const WR64RenderState* state) {
    assert(state->version == 2);
    assert(state->game_state == wr64_u(env, WR_ADDR_GAMESTATE));
    assert(state->course_id == wr64_u(env, WR_ADDR_COURSE_ID));
    assert(state->game_mode == wr64_u(env, WR_ADDR_GAME_MODE));
    assert(state->machine_ticks == env->machine.ticks);
    assert(state->tick == env->state.tick);
    assert(state->lap == wr64_lap(env));
    assert(state->target_laps == wr64_target_laps(env));
    assert(state->race_time_ms == wr64_race_time_ms(env));
    assert(state->lap_time_ms == wr64_lap_time_ms(env));
    for (int lap = 0; lap < 3; lap++) {
        assert(state->lap_splits_ms[lap] == wr64_lap_split_ms(env, lap));
    }
    assert(state->speed_kmh == wr64_speed_kmh(env));
    assert(state->power == wr64_power(env));
    assert(state->race_position == (int32_t)wr64_u(
        env, wr64_rider_addr(env, WR_RIDER_RACE_POSITION)));
    assert(state->target_node == wr64_sanitize_node(
        env, WR64_COURSE_PRIMARY, wr64_node(env)));
    assert(state->next_node == wr64_next_node(
        env, WR64_COURSE_PRIMARY, state->target_node));
    assert(state->misses == wr64_misses(env));
    assert(state->checkpoints == env->state.checkpoints);
    assert(state->recovery == wr64_recovery(env));
    assert(state->disqualified == (wr64_disqualified(env) != 0));
    assert(state->ended == (wr64_ended(env) != 0));
    assert(state->finished == (wr64_finished(env) != 0));
    assert(state->success == env->state.success);
    assert(state->failed == env->state.failed);
    assert(state->pad_buttons == env->machine.pad_buttons);
    assert(state->pad_stick_x == env->machine.pad_stick_x);
    assert(state->pad_stick_y == env->machine.pad_stick_y);

    uint32_t physics = wr64_physics_addr(env, 0);
    assert(state->physics_state == (int32_t)wr64_u(
        env, physics + WR64_PHYSICS_STATE));
    assert(state->physics_state_frame == (int32_t)wr64_u(
        env, physics + WR64_PHYSICS_STATE_FRAME));
    for (int i = 0; i < 3; i++) {
        assert_float_bits(state->position[i],
            wr64_f(env, physics + WR64_PHYSICS_POS + 4u*(uint32_t)i));
    }
    assert_float_bits(state->velocity[0], env->state.velocity_x);
    assert_float_bits(state->velocity[1], env->state.velocity_y);
    assert_float_bits(state->velocity[2], env->state.velocity_z);
    float heading_x, heading_z;
    wr64_heading(env, env->state.velocity_x, env->state.velocity_z,
        &heading_x, &heading_z);
    assert_float_bits(state->heading[0], heading_x);
    assert_float_bits(state->heading[1], heading_z);
    static const uint32_t basis_offsets[9] = {
        WR_PHYSICS_BASIS_0_X, WR_PHYSICS_BASIS_0_Y,
        WR_PHYSICS_BASIS_0_Z, WR_PHYSICS_BASIS_1_X,
        WR_PHYSICS_BASIS_1_Y, WR_PHYSICS_BASIS_1_Z,
        WR_PHYSICS_BASIS_2_X, WR_PHYSICS_BASIS_2_Y,
        WR_PHYSICS_BASIS_2_Z,
    };
    for (int i = 0; i < 9; i++) {
        assert_float_bits(state->basis[i],
            wr64_f(env, physics + basis_offsets[i]));
    }

    assert(state->node_count == wr64_node_count(env, WR64_COURSE_PRIMARY));
    for (int i = 0; i < state->node_count; i++) {
        const WR64RenderNode* node = &state->nodes[i];
        uint32_t address = wr64_course_addr(WR64_COURSE_PRIMARY, i, 0);
        assert(node->index == i);
        assert(node->next == wr64_next_node(env, WR64_COURSE_PRIMARY, i));
        assert(node->type == (int32_t)wr64_u(
            env, address + WR64_COURSE_NODE_TYPE));
        int valid = (int32_t)wr64_u(
            env, address + WR64_COURSE_NODE_DISABLED) == 0
            && (int32_t)wr64_u(
                env, address + WR64_COURSE_NODE_ENABLED) != 0;
        assert(node->valid == valid);
        assert_float_bits(node->live_x,
            wr64_f(env, address + WR64_COURSE_NODE_X));
        assert_float_bits(node->live_y,
            wr64_f(env, address + WR64_COURSE_NODE_Y));
        assert_float_bits(node->live_z,
            wr64_f(env, address + WR64_COURSE_NODE_Z));
        assert_float_bits(node->anchor_x,
            wr64_f(env, address + WR64_COURSE_NODE_ANCHOR_X));
        assert_float_bits(node->anchor_z,
            wr64_f(env, address + WR64_COURSE_NODE_ANCHOR_Z));
        assert_float_bits(node->tangent_x,
            wr64_f(env, address + WR64_COURSE_NODE_TANGENT_X));
        assert_float_bits(node->tangent_z,
            wr64_f(env, address + WR64_COURSE_NODE_TANGENT_Z));
        assert_float_bits(node->lateral_x,
            wr64_f(env, address + WR_COURSE_NODE_LATERAL_X));
        assert_float_bits(node->lateral_z,
            wr64_f(env, address + WR_COURSE_NODE_LATERAL_Z));
        assert_float_bits(node->length,
            wr64_f(env, address + WR64_COURSE_NODE_LENGTH));
        float pass_x, pass_z;
        assert(wr64_pass_point(env, i, &pass_x, &pass_z));
        assert_float_bits(node->pass_x, pass_x);
        assert_float_bits(node->pass_z, pass_z);
    }

    assert_float_bits(state->water_level,
        (float)(int32_t)wr64_u(env, WR64_WATER_LEVEL));
    for (int row = 0; row < WR64_RENDER_WATER_DIM; row++) {
        float z = state->water_origin_z
            + (float)row * state->water_spacing;
        for (int col = 0; col < WR64_RENDER_WATER_DIM; col++) {
            float x = state->water_origin_x
                + (float)col * state->water_spacing;
            float expected = wr64_finite_or_zero(
                wr64_water_height(env, x, z));
            assert_float_bits(
                state->water[row * WR64_RENDER_WATER_DIM + col], expected);
        }
    }
}

static void test_render_state_capture_is_pure(TestEnv* test) {
    WR64RenderState first;
    WR64RenderState second;
    memset(&first, 0xA5, sizeof(first));
    memset(&second, 0x5A, sizeof(second));
    WRMachine* tls_before = wr_current;
    CoreDigest before = capture_core_digest(test);
    wr64_capture_render_state(&test->env, &first);
    wr64_capture_render_state(&test->env, &second);
    CoreDigest after = capture_core_digest(test);
    assert_core_equal(&before, &after);
    assert(wr_current == tls_before);
    assert(memcmp(&first, &second, sizeof(first)) == 0);
    assert(wr64_render_state_hash(&first)
        == wr64_render_state_hash(&second));
    assert_render_state_matches_authority(&test->env, &first);
    printf("PASS render-state capture-pure hash=%016llx nodes=%d\n",
        (unsigned long long)wr64_render_state_hash(&first),
        first.node_count);
}

static void test_puf_render_preserves_core_state(TestEnv* test) {
    assert(test->env.client == NULL);
    WRMachine* tls_before = wr_current;
    CoreDigest before = capture_core_digest(test);
    puf_render(&test->env);
    CoreDigest after_first = capture_core_digest(test);
    assert_core_equal(&before, &after_first);
    assert(wr_current == tls_before);
    assert(test->env.client != NULL);
    assert(IsWindowReady());

    for (int frame = 0; frame < 8; frame++) {
        puf_render(&test->env);
    }
    CoreDigest after_repeated = capture_core_digest(test);
    assert_core_equal(&before, &after_repeated);
    assert(wr_current == tls_before);
    printf("PASS puf-render core-state-preserved repeated=8\n");
}

static void join_path(char* output, size_t size,
        const char* directory, const char* filename) {
    int written = snprintf(output, size, "%s/%s", directory, filename);
    assert(written > 0 && (size_t)written < size);
}

static uint64_t image_hash(const Color* colors, size_t count) {
    return hash_memory(colors, count * sizeof(*colors));
}

static uint64_t test_fixed_state_pixels(TestEnv* test,
        const char* output_dir) {
    char first_path[WR64_TEST_PATH_SIZE];
    char second_path[WR64_TEST_PATH_SIZE];
    join_path(first_path, sizeof(first_path),
        output_dir, "fixed-state-a.png");
    join_path(second_path, sizeof(second_path),
        output_dir, "fixed-state-b.png");

    puf_render(&test->env);
    Image first = LoadImageFromScreen();
    assert(IsImageValid(first));
    assert(ExportImage(first, first_path));
    for (int frame = 0; frame < 7; frame++) {
        puf_render(&test->env);
    }
    Image second = LoadImageFromScreen();
    assert(IsImageValid(second));
    assert(ExportImage(second, second_path));
    assert(first.width == second.width && first.height == second.height);
    assert(first.width == GetScreenWidth() && first.height == GetScreenHeight());
    Color* first_colors = LoadImageColors(first);
    Color* second_colors = LoadImageColors(second);
    assert(first_colors && second_colors);
    size_t pixel_count = (size_t)first.width * (size_t)first.height;
    assert(memcmp(first_colors, second_colors,
        pixel_count * sizeof(Color)) == 0);

    size_t blue = 0;
    size_t red = 0;
    size_t yellow = 0;
    int min_luma = 255;
    int max_luma = 0;
    for (size_t i = 0; i < pixel_count; i++) {
        Color color = first_colors[i];
        int luma = (int)color.r + (int)color.g + (int)color.b;
        luma /= 3;
        if (luma < min_luma) min_luma = luma;
        if (luma > max_luma) max_luma = luma;
        blue += color.b > color.r + 20 && color.b > color.g;
        red += color.r > 150 && color.r > color.g + 40
            && color.r > color.b + 40;
        yellow += color.r > 170 && color.g > 140 && color.b < 130;
    }
    assert(blue > pixel_count / 20);
    assert(red > 20);
    assert(yellow > 20);
    assert(max_luma - min_luma > 100);
    uint64_t hash = image_hash(first_colors, pixel_count);

    UnloadImageColors(first_colors);
    UnloadImageColors(second_colors);
    UnloadImage(first);
    UnloadImage(second);
    printf("PASS fixed-state pixels=%dx%d hash=%016llx "
        "blue=%zu red=%zu yellow=%zu\n",
        GetScreenWidth(), GetScreenHeight(),
        (unsigned long long)hash, blue, red, yellow);
    return hash;
}

static void assert_machine_scalars_equal(
        const WRMachine* a, const WRMachine* b) {
    assert(a->ticks == b->ticks);
    assert(a->pad_buttons == b->pad_buttons);
    assert(a->pad_stick_x == b->pad_stick_x);
    assert(a->pad_stick_y == b->pad_stick_y);
    assert(a->resident_overlay == b->resident_overlay);
    assert(a->vi_fb == b->vi_fb);
    assert(a->vi_swaps == b->vi_swaps);
    assert(a->dma_bytes == b->dma_bytes);
    assert(a->dma_count == b->dma_count);
    assert(a->cont_reads == b->cont_reads);
    assert(a->recv_calls == b->recv_calls);
    assert(a->frames_left == b->frames_left);
    assert(a->running == b->running);
    assert(a->finished == b->finished);
    assert(a->last_frame == b->last_frame);
}

static uint64_t assert_trajectory_state_equal(
        const TestEnv* headless, const TestEnv* rendered,
        uint64_t trajectory_hash) {
    const WaveRace64* a = &headless->env;
    const WaveRace64* b = &rendered->env;
    assert(memcmp(a->machine.rdram, b->machine.rdram,
        WR_RDRAM_SIZE) == 0);
    assert_machine_scalars_equal(&a->machine, &b->machine);
    assert(memcmp(&a->state, &b->state, sizeof(a->state)) == 0);
    assert(memcmp(&a->log, &b->log, sizeof(a->log)) == 0);
    assert(memcmp(headless->observations, rendered->observations,
        sizeof(headless->observations)) == 0);
    assert(memcmp(headless->actions, rendered->actions,
        sizeof(headless->actions)) == 0);
    assert_float_bits(headless->reward, rendered->reward);
    assert_float_bits(headless->terminal, rendered->terminal);
    assert(memcmp(a->route_arc, b->route_arc, sizeof(a->route_arc)) == 0);
    assert(memcmp(a->route_pred, b->route_pred, sizeof(a->route_pred)) == 0);
    assert_float_bits(a->route_total, b->route_total);
    assert(a->route_nodes == b->route_nodes);
    assert(a->route_valid == b->route_valid);
    assert(a->curriculum_laps == b->curriculum_laps);
    assert(a->curriculum_successes == b->curriculum_successes);

    WR64RenderState state_a;
    WR64RenderState state_b;
    wr64_capture_render_state((WaveRace64*)a, &state_a);
    wr64_capture_render_state((WaveRace64*)b, &state_b);
    assert(memcmp(&state_a, &state_b, sizeof(state_a)) == 0);
    uint64_t state_hash = wr64_render_state_hash(&state_a);
    return hash_update(trajectory_hash, &state_hash, sizeof(state_hash));
}

static uint64_t test_render_cadence_does_not_change_trajectory(
        TestEnv* headless, TestEnv* rendered) {
    memset(&headless->env.log, 0, sizeof(headless->env.log));
    memset(&rendered->env.log, 0, sizeof(rendered->env.log));
    puf_eval_reset(&headless->env);
    puf_eval_reset(&rendered->env);
    assert(headless->env.client == NULL);
    assert(rendered->env.client != NULL);
    assert(memcmp(headless->env.machine.rdram,
        rendered->env.machine.rdram, WR_RDRAM_SIZE) == 0);

    uint32_t rng = UINT32_C(0xC001D00D);
    uint64_t trajectory_hash = UINT64_C(14695981039346656037);
    int terminals = 0;
    for (int decision = 0; decision < 96; decision++) {
        uint32_t action_rng = rng;
        set_lcg_action(headless, &rng);
        set_lcg_action(rendered, &action_rng);
        assert(action_rng == rng);

        int before_renders = decision % 4;
        for (int frame = 0; frame < before_renders; frame++) {
            puf_render(&rendered->env);
        }
        puf_step(&headless->env);
        puf_step(&rendered->env);
        terminals += headless->terminal > 0.5f;
        if (decision % 5 == 0) {
            puf_render(&rendered->env);
            puf_render(&rendered->env);
        }
        trajectory_hash = assert_trajectory_state_equal(
            headless, rendered, trajectory_hash);
    }
    assert(terminals > 0);
    assert(headless->env.client == NULL);
    printf("PASS render-cadence decisions=96 terminals=%d hash=%016llx\n",
        terminals, (unsigned long long)trajectory_hash);
    return trajectory_hash;
}

static int nearest_stick(float desired) {
    int best = 0;
    float best_error = 1000.f;
    for (int i = 0; i < 15; i++) {
        float error = fabsf(desired - (float)WR64_STICK_X[i]);
        if (error < best_error) {
            best = i;
            best_error = error;
        }
    }
    return best;
}

static void set_scripted_route_action(TestEnv* test) {
    const float* obs = test->observations;
    const float pass_scale = 1.22524f;
    float center_x = obs[17] * obs[19];
    float center_z = -obs[18] * obs[19];
    float pass_x = obs[24] * obs[26];
    float pass_z = -obs[25] * obs[26];
    float dx = center_x + pass_scale * (pass_x - center_x);
    float dz = center_z + pass_scale * (pass_z - center_z);
    if (obs[30] > 0.5f) {
        float next_x = obs[27] * obs[29];
        float next_z = -obs[28] * obs[29];
        float blend = obs[26] < 1025.64f / test->env.route_total
            ? 0.990414f : 0.0293148f;
        dx = dx * (1.f - blend) + next_x * blend;
        dz = dz * (1.f - blend) + next_z * blend;
    }
    float angle = atan2f(dz, dx);
    int steer = (int)lrintf(angle * 75.848f);
    if (steer > 80) steer = 80;
    if (steer < -80) steer = -80;
    int throttle = fabsf(angle) <= 0.14112f;
    int dampen = fabsf(angle) > 0.49904f;
    int slide = fabsf(angle) > 0.708171f;
    set_action(test, nearest_stick((float)steer), 2,
        throttle, dampen, slide);
}

static void capture_moving_frames(TestEnv* test,
        const char* output_dir, int frame_count) {
    if (frame_count <= 0) return;
    puf_eval_reset(&test->env);
    test->env.client->has_terminal = 0;
    for (int frame = 0; frame < frame_count; frame++) {
        set_scripted_route_action(test);
        puf_step(&test->env);
        puf_render(&test->env);
        char filename[64];
        int written = snprintf(filename, sizeof(filename),
            "frame-%04d.png", frame);
        assert(written > 0 && (size_t)written < sizeof(filename));
        char path[WR64_TEST_PATH_SIZE];
        join_path(path, sizeof(path), output_dir, filename);
        Image image = LoadImageFromScreen();
        assert(IsImageValid(image));
        assert(ExportImage(image, path));
        UnloadImage(image);
    }
    WR64RenderState state;
    wr64_capture_render_state(&test->env, &state);
    printf("PASS moving-capture frames=%d tick=%d state=%016llx\n",
        frame_count, state.tick,
        (unsigned long long)wr64_render_state_hash(&state));
}

static void ensure_directory(const char* path) {
#ifdef __linux__
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        perror(path);
        exit(1);
    }
#else
    (void)path;
#endif
}

static void usage(const char* program) {
    fprintf(stderr,
        "usage: %s ROM [--output-dir PATH] [--capture-frames N]\n",
        program);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    const char* rom_path = argv[1];
    const char* output_dir = "/tmp/wr64-render-test";
    int capture_frames = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--capture-frames") == 0
                && i + 1 < argc) {
            capture_frames = atoi(argv[++i]);
            if (capture_frames < 0 || capture_frames > 10000) {
                usage(argv[0]);
                return 2;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    ensure_directory(output_dir);

#ifdef __linux__
    setenv("WR64_RENDER_WIDTH", "960", 1);
    setenv("WR64_RENDER_HEIGHT", "540", 1);
#endif
    SetConfigFlags(FLAG_WINDOW_HIDDEN | FLAG_MSAA_4X_HINT);

    Dict kwargs = {};
    init_kwargs(&kwargs, rom_path);
    TestEnv headless;
    TestEnv rendered;
    init_test_env(&headless, &kwargs);
    init_test_env(&rendered, &kwargs);
    assert(headless.env.client == NULL);
    assert(rendered.env.client == NULL);

    test_control_mode_toggle();
    test_time_format();
    test_camera_wave_visibility();
    test_final_lap_banner_state();
    test_training_client_remains_null(&headless);
    puf_eval_reset(&rendered.env);
    test_render_state_capture_is_pure(&rendered);
    test_puf_render_preserves_core_state(&rendered);
    uint64_t pixel_hash = test_fixed_state_pixels(&rendered, output_dir);
    uint64_t trajectory_hash = test_render_cadence_does_not_change_trajectory(
        &headless, &rendered);
    capture_moving_frames(&rendered, output_dir, capture_frames);

    assert(IsWindowReady());
    puf_close(&headless.env);
    assert(headless.env.client == NULL);
    assert(IsWindowReady());
    puf_close(&rendered.env);
    assert(rendered.env.client == NULL);
    assert(!IsWindowReady());
    dict_clear(&kwargs);

    printf("PASS waverace64 render regressions pixels=%016llx "
        "trajectory=%016llx output=%s\n",
        (unsigned long long)pixel_hash,
        (unsigned long long)trajectory_hash,
        output_dir);
    return 0;
}
