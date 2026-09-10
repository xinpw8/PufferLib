#define REK_G1_CUDA_DEVICE 1
#include "g1_native_combat_cuda.h"
extern "C" {
#include "g1_strike_catalog.h"
}

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

namespace {
constexpr unsigned THREADS = 128;
constexpr float PHYSICS_DELTA_SECONDS = 0.002f;
constexpr int ROUTE_COUNT = REK_G1_STATIC_ROUTE_COUNT;
constexpr int FIRST_DISCRETE_ROUTE = REK_G1_NATIVE_MOVE_6_LEFT_SIDE;
constexpr int LAST_DISCRETE_ROUTE = REK_G1_NATIVE_MOVE_16_BUTT_SMACK_EMOTE;
constexpr int FALL_FLOAT_FIELDS = 5;
constexpr int FALL_INTEGER_FIELDS = 7;
constexpr int HIT_FLOAT_FIELDS = 13;
constexpr int HIT_INTEGER_FIELDS = 12;

unsigned blocks_for(size_t count) {
    const size_t blocks = count / THREADS + (count % THREADS != 0);
    return static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
}

__device__ bool add_i32_nonnegative(int32_t left, int32_t right, int32_t* out) {
    if (right < 0 || left > INT32_MAX - right) return false;
    *out = left + right;
    return true;
}

__device__ bool add_u32(uint32_t left, uint32_t right, uint32_t* out) {
    if (left > UINT32_MAX - right) return false;
    *out = left + right;
    return true;
}

__global__ void init_kernel(
        RekG1CudaNativeCombatState* states,
        int32_t* statuses,
        size_t count) {
    for (size_t arena = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            arena < count; arena += size_t(blockDim.x) * gridDim.x) {
        RekG1CudaNativeCombatState value = {};
        int32_t status = REK_G1_CUDA_NATIVE_COMBAT_OK;
        if (rek_g1_combat_arena_init(&value.combat)
                != REK_G1_COMBAT_TICK_OK) {
            status = REK_G1_CUDA_NATIVE_COMBAT_TICK_REJECTED;
        }
        for (size_t fighter = 0; fighter < 2 && status == 0; fighter++) {
            if (rek_g1_fall_state_init(
                    &REK_G1_FALL_CONFIG_F84F1874,
                    &value.fall[fighter]) != REK_G1_FALL_OK) {
                status = REK_G1_CUDA_NATIVE_COMBAT_FALL_REJECTED;
            }
        }
        if (status == 0) states[arena] = value;
        statuses[arena] = status;
    }
}

__global__ void begin_tick_kernel(
        RekG1CudaNativeCombatState* states,
        uint32_t* tick_fall_events,
        uint32_t* tick_signals,
        uint32_t* tick_referee_calls,
        int32_t* tick_score_delta,
        uint32_t* tick_attributed_contacts,
        uint32_t* tick_scored_contacts,
        uint8_t* terminals,
        uint8_t* episode_reset,
        uint8_t* input_reset,
        int32_t* statuses,
        size_t count) {
    for (size_t arena = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            arena < count; arena += size_t(blockDim.x) * gridDim.x) {
        const size_t row = arena * 2;
        tick_fall_events[row] = 0;
        tick_fall_events[row + 1] = 0;
        tick_signals[arena] = 0;
        tick_referee_calls[arena] = 0;
        tick_score_delta[row] = 0;
        tick_score_delta[row + 1] = 0;
        tick_attributed_contacts[arena] = 0;
        tick_scored_contacts[arena] = 0;
        terminals[arena] = 0;
        episode_reset[arena] = 0;
        input_reset[row] = 0;
        input_reset[row + 1] = 0;
        if (statuses[arena] != REK_G1_CUDA_NATIVE_COMBAT_OK) continue;
        if (!states[arena].pending_episode_reset) continue;

        RekG1CudaNativeCombatState value = {};
        int32_t status = REK_G1_CUDA_NATIVE_COMBAT_OK;
        if (rek_g1_combat_arena_init(&value.combat)
                != REK_G1_COMBAT_TICK_OK) {
            status = REK_G1_CUDA_NATIVE_COMBAT_TICK_REJECTED;
        }
        for (size_t fighter = 0; fighter < 2 && status == 0; fighter++) {
            if (rek_g1_fall_state_init(
                    &REK_G1_FALL_CONFIG_F84F1874,
                    &value.fall[fighter]) != REK_G1_FALL_OK) {
                status = REK_G1_CUDA_NATIVE_COMBAT_FALL_REJECTED;
            }
        }
        if (status == 0) {
            states[arena] = value;
            episode_reset[arena] = 1;
        }
        statuses[arena] = status;
    }
}

__device__ bool assemble_intent(
        const SonicMotionComposerNative* composer,
        int32_t route,
        const RekG1ImpactEvent* impact_events,
        const int32_t* route_event_offsets,
        const int32_t* route_event_counts,
        RekG1StrikeIntent* output) {
    *output = {};
    if (route < 0 || route >= ROUTE_COUNT) return false;
    if (route < FIRST_DISCRETE_ROUTE || !composer->action_playing) return true;
    if (route > LAST_DISCRETE_ROUTE
            || !composer->current_layer.has_clip
            || !composer->current_layer.has_config
            || !composer->current_layer.active
            || composer->current_layer.config.loop
            || !isfinite(composer->current_layer.cursor)
            || composer->current_layer.cursor < 0.0f
            || composer->current_layer.clip.fps != 50.0f) {
        return false;
    }
    const int32_t offset = route_event_offsets[route];
    const int32_t count = route_event_counts[route];
    if (offset < 0 || count <= 0
            || offset > REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT - count) {
        return false;
    }
    output->impact_events = impact_events + offset;
    output->impact_event_count = static_cast<size_t>(count);
    output->clip_cursor_frames = composer->current_layer.cursor;
    output->clip_fps = composer->current_layer.clip.fps;
    output->move_id = composer->action_move_id;
    output->action_playing = 1;
    output->layer_active = 1;
    output->layer_loop = 0;
    return true;
}

__global__ void pack_contacts_kernel(
        const RekG1CudaNativeCombatState* states,
        const uint8_t* terminals,
        const int64_t* hit_integers,
        const float* hit_floats,
        const uint8_t* candidate_valid,
        const int64_t* candidate_order,
        const int64_t* candidate_offsets,
        const int64_t* candidate_counts,
        const uint8_t* arena_scan_valid,
        const float* arena_time_seconds,
        const SonicMotionComposerNative* composers,
        const int32_t* active_route_ids,
        const RekG1ImpactEvent* impact_events,
        const int32_t* route_event_offsets,
        const int32_t* route_event_counts,
        RekG1HitContact* packed_contacts,
        int32_t* statuses,
        size_t arena_count,
        size_t capacity) {
    for (size_t arena = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            arena < arena_count; arena += size_t(blockDim.x) * gridDim.x) {
        if (statuses[arena] != REK_G1_CUDA_NATIVE_COMBAT_OK) continue;
        const int64_t offset = candidate_offsets[arena];
        const int64_t count = candidate_counts[arena];
        if (!arena_scan_valid[arena] || !isfinite(arena_time_seconds[arena])
                || arena_time_seconds[arena] < 0.0f || offset < 0 || count < 0
                || static_cast<uint64_t>(offset) > capacity
                || static_cast<uint64_t>(count) > capacity - size_t(offset)) {
            statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_INPUT_INVALID;
            continue;
        }
        if (states[arena].reset_pending || terminals[arena]) continue;
        const uint8_t round_active = states[arena].combat.fight.phase
            == REK_G1_FIGHT_ROUND_ACTIVE;
        for (int64_t local_index = 0; local_index < count; local_index++) {
            const int64_t packed_index = offset + local_index;
            const int64_t source_index = candidate_order[packed_index];
            if (source_index < 0
                    || static_cast<uint64_t>(source_index) >= capacity
                    || !candidate_valid[source_index]) {
                statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_CONTACT_REJECTED;
                break;
            }
            const int64_t* integers = hit_integers
                + source_index * HIT_INTEGER_FIELDS;
            const float* floats = hit_floats + source_index * HIT_FLOAT_FIELDS;
            const int64_t striker = integers[6];
            const int64_t target = integers[7];
            if (integers[0] != static_cast<int64_t>(arena)
                    || striker < 0 || striker > 1 || target < 0 || target > 1
                    || striker == target || integers[8] < 0 || integers[8] >= 6
                    || (integers[9] != REK_G1_BODY_PART_HAND
                        && integers[9] != REK_G1_BODY_PART_FOOT
                        && integers[9] != REK_G1_BODY_PART_SHIN)
                    || (integers[10] != REK_G1_HAND_LEFT
                        && integers[10] != REK_G1_HAND_RIGHT)
                    || integers[11] < REK_G1_BODY_ZONE_UNKNOWN
                    || integers[11] > REK_G1_BODY_ZONE_RIGHT_ANKLE) {
                statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_CONTACT_REJECTED;
                break;
            }
            RekG1HitContact contact = {};
            const size_t striker_row = arena * 2 + size_t(striker);
            if (!assemble_intent(
                    &composers[striker_row], active_route_ids[striker_row],
                    impact_events, route_event_offsets, route_event_counts,
                    &contact.strike_intent)) {
                statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_CONTACT_REJECTED;
                break;
            }
            for (size_t axis = 0; axis < 3; axis++) {
                contact.striker_body_position_world[axis] = floats[axis];
                contact.target_body_position_world[axis] = floats[3 + axis];
                contact.striker_body_linear_velocity_world[axis] = floats[6 + axis];
                contact.target_body_linear_velocity_world[axis] = floats[9 + axis];
            }
            contact.relative_speed_mps = floats[12];
            contact.time_seconds = arena_time_seconds[arena];
            contact.striker_part = static_cast<RekG1BodyPartType>(integers[9]);
            contact.striker_side = static_cast<RekG1HandSide>(integers[10]);
            contact.target_zone = static_cast<RekG1BodyZone>(integers[11]);
            contact.striker_fighter = static_cast<uint32_t>(striker);
            contact.target_fighter = static_cast<uint32_t>(target);
            contact.striker_body_slot = static_cast<uint32_t>(integers[8]);
            contact.is_enter = 1;
            contact.round_active = round_active;
            contact.striker_upright = states[arena].fall[striker].phase
                != REK_G1_FALL_FALLEN;
            contact.target_upright = states[arena].fall[target].phase
                != REK_G1_FALL_FALLEN;
            contact.target_standing = contact.target_upright;
            packed_contacts[packed_index] = contact;
        }
    }
}

__global__ void post_step_kernel(
        RekG1CudaNativeCombatState* states,
        const float* fall_floats,
        const int64_t* fall_integers,
        const uint8_t* fall_valid,
        const int64_t* candidate_offsets,
        const int64_t* candidate_counts,
        const float* arena_time_seconds,
        const RekG1HitContact* packed_contacts,
        uint32_t* tick_fall_events,
        uint32_t* tick_signals,
        uint32_t* tick_referee_calls,
        int32_t* tick_score_delta,
        uint32_t* tick_attributed_contacts,
        uint32_t* tick_scored_contacts,
        uint8_t* terminals,
        uint8_t* input_reset,
        uint8_t* dampened,
        uint8_t* begin_reset,
        uint8_t* complete_reset,
        uint8_t* clear_contacts,
        int32_t* statuses,
        size_t arena_count) {
    for (size_t arena = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            arena < arena_count; arena += size_t(blockDim.x) * gridDim.x) {
        const size_t row = arena * 2;
        dampened[row] = dampened[row + 1] = 0;
        begin_reset[arena] = 0;
        complete_reset[arena] = 0;
        clear_contacts[arena] = 0;
        if (statuses[arena] != REK_G1_CUDA_NATIVE_COMBAT_OK) continue;

        RekG1CudaNativeCombatState value = states[arena];
        const float now = arena_time_seconds[arena];
        if (value.reset_pending) {
            if (!isfinite(now) || now < 0.0f) {
                statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_INPUT_INVALID;
                continue;
            }
            clear_contacts[arena] = 1;
            /* This call follows exactly one subsequent 2 ms mj_step. */
            value.reset_pending = 0;
            value.reset_complete_not_before_seconds = 0.0f;
            states[arena] = value;
            complete_reset[arena] = 1;
            input_reset[row] = input_reset[row + 1] = 1;
            continue;
        }

        RekG1FallStepResult fall_result[2] = {};
        uint32_t next_tick_fall[2] = {
            tick_fall_events[row], tick_fall_events[row + 1]};
        bool valid = true;
        for (size_t fighter = 0; fighter < 2; fighter++) {
            const size_t fighter_row = row + fighter;
            const float* floats = fall_floats + fighter_row * FALL_FLOAT_FIELDS;
            const int64_t* integers = fall_integers
                + fighter_row * FALL_INTEGER_FIELDS;
            if (!fall_valid[fighter_row]
                    || (integers[0] != 0 && integers[0] != 1)
                    || (integers[1] != 0 && integers[1] != 1)
                    || (integers[2] != 0 && integers[2] != 1)
                    || integers[3] < 0 || uint64_t(integers[3]) > UINT32_MAX
                    || (integers[4] != 0 && integers[4] != 1)
                    || (integers[5] != 0 && integers[5] != 1)
                    || (integers[6] != 0 && integers[6] != 1)) {
                valid = false;
                break;
            }
            const RekG1FallSample sample = {
                .tracking_active = static_cast<uint8_t>(integers[0]),
                .tilt_degrees = floats[0],
                .pelvis_height_ratio = floats[1],
                .both_feet_off_floor = static_cast<uint8_t>(integers[1]),
                .has_foot_body_contact = static_cast<uint8_t>(integers[2]),
                .distinct_nonfoot_body_contact_count =
                    static_cast<uint32_t>(integers[3]),
                .fixed_delta_seconds = floats[2],
                .can_get_up = static_cast<uint8_t>(integers[4]),
            };
            if (rek_g1_fall_state_step(
                    &REK_G1_FALL_CONFIG_F84F1874,
                    &value.fall[fighter], &sample,
                    &fall_result[fighter]) != REK_G1_FALL_OK) {
                valid = false;
                break;
            }
            next_tick_fall[fighter] |= fall_result[fighter].events;
        }
        if (!valid) {
            statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_FALL_REJECTED;
            continue;
        }
        value.fall[0] = fall_result[0].next_state;
        value.fall[1] = fall_result[1].next_state;
        dampened[row] = (fall_result[0].events
            & REK_G1_FALL_EVENT_BECAME_FALLEN) != 0;
        dampened[row + 1] = (fall_result[1].events
            & REK_G1_FALL_EVENT_BECAME_FALLEN) != 0;

        if (terminals[arena]) {
            states[arena] = value;
            tick_fall_events[row] = next_tick_fall[0];
            tick_fall_events[row + 1] = next_tick_fall[1];
            continue;
        }

        const int64_t offset = candidate_offsets[arena];
        const int64_t count = candidate_counts[arena];
        float remaining = value.combat.fight.time_remaining_seconds
            - PHYSICS_DELTA_SECONDS;
        if (remaining < 0.0f) remaining = 0.0f;
        const RekG1CombatSubstepInput input = {
            .delta_seconds = PHYSICS_DELTA_SECONDS,
            .time_remaining_seconds = remaining,
            .contacts = packed_contacts + offset,
            .contact_count = static_cast<size_t>(count),
            .fall_phase = {value.fall[0].phase, value.fall[1].phase},
            .fall_events = {fall_result[0].events, fall_result[1].events},
            .fighter_is_recovering = {0, 0},
            .fighter_can_get_up = {
                static_cast<uint8_t>(fall_integers[row * FALL_INTEGER_FIELDS + 4]),
                static_cast<uint8_t>(fall_integers[(row + 1) * FALL_INTEGER_FIELDS + 4]),
            },
            .force_slip_estop = {0, 0},
        };
        RekG1CombatSubstepResult combat_result = {};
        const RekG1HitDetectorConfig hit_config =
            rek_g1_current_build_hit_detector_config();
        if (rek_g1_combat_arena_substep(
                &value.combat, &hit_config,
                &input, &combat_result) != REK_G1_COMBAT_TICK_OK) {
            statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_TICK_REJECTED;
            continue;
        }

        int32_t next_score[2] = {};
        uint32_t next_attributed = 0;
        uint32_t next_scored = 0;
        if (!add_i32_nonnegative(
                tick_score_delta[row], combat_result.score_delta[0],
                &next_score[0])
                || !add_i32_nonnegative(
                    tick_score_delta[row + 1], combat_result.score_delta[1],
                    &next_score[1])
                || !add_u32(
                    tick_attributed_contacts[arena],
                    combat_result.attributed_contact_count, &next_attributed)
                || !add_u32(
                    tick_scored_contacts[arena],
                    combat_result.scored_contact_count, &next_scored)) {
            statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_OVERFLOW;
            continue;
        }
        value.combat = combat_result.next_state;
        if ((combat_result.signals
                & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0) {
            RekG1CombatArenaState reset_combat = {};
            RekG1FallState reset_fall[2] = {};
            if (rek_g1_combat_arena_apply_spawn_reset(
                    &value.combat, &reset_combat) != REK_G1_COMBAT_TICK_OK
                    || rek_g1_fall_state_apply_fight_spawn_reset(
                        &REK_G1_FALL_CONFIG_F84F1874,
                        &value.fall[0], &reset_fall[0]) != REK_G1_FALL_OK
                    || rek_g1_fall_state_apply_fight_spawn_reset(
                        &REK_G1_FALL_CONFIG_F84F1874,
                        &value.fall[1], &reset_fall[1]) != REK_G1_FALL_OK
                    || !isfinite(now + PHYSICS_DELTA_SECONDS)) {
                statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_RESET_REJECTED;
                continue;
            }
            value.combat = reset_combat;
            value.fall[0] = reset_fall[0];
            value.fall[1] = reset_fall[1];
            value.reset_pending = 1;
            value.reset_complete_not_before_seconds = now + PHYSICS_DELTA_SECONDS;
            begin_reset[arena] = 1;
            clear_contacts[arena] = 1;
        }
        if ((combat_result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0) {
            terminals[arena] = 1;
            value.pending_episode_reset = 1;
        }
        states[arena] = value;
        tick_fall_events[row] = next_tick_fall[0];
        tick_fall_events[row + 1] = next_tick_fall[1];
        tick_signals[arena] |= combat_result.signals;
        tick_referee_calls[arena] |= combat_result.referee_calls;
        tick_score_delta[row] = next_score[0];
        tick_score_delta[row + 1] = next_score[1];
        tick_attributed_contacts[arena] = next_attributed;
        tick_scored_contacts[arena] = next_scored;
    }
}

__global__ void observe_kernel(
        const RekG1CudaNativeCombatState* states,
        const float* fall_floats,
        const int64_t* fall_integers,
        const uint8_t* fall_valid,
        const uint32_t* tick_fall_events,
        const uint32_t* tick_signals,
        const uint32_t* tick_referee_calls,
        const int32_t* tick_score_delta,
        const uint32_t* tick_attributed_contacts,
        const uint32_t* tick_scored_contacts,
        const uint8_t* terminals,
        float* fall_observations,
        float* fight_observations,
        float* rewards,
        float* terminal_observations,
        int32_t* statuses,
        size_t arena_count) {
    for (size_t arena = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            arena < arena_count; arena += size_t(blockDim.x) * gridDim.x) {
        if (statuses[arena] != REK_G1_CUDA_NATIVE_COMBAT_OK) continue;
        bool valid = true;
        for (size_t fighter = 0; fighter < 2; fighter++) {
            const size_t row = arena * 2 + fighter;
            const int64_t* integers = fall_integers
                + row * FALL_INTEGER_FIELDS;
            valid = valid && fall_valid[row]
                && (integers[0] == 0 || integers[0] == 1)
                && (integers[1] == 0 || integers[1] == 1)
                && (integers[2] == 0 || integers[2] == 1)
                && integers[3] >= 0 && uint64_t(integers[3]) <= UINT32_MAX
                && (integers[4] == 0 || integers[4] == 1)
                && (integers[5] == 0 || integers[5] == 1)
                && (integers[6] == 0 || integers[6] == 1);
        }
        if (!valid) {
            statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_INPUT_INVALID;
            continue;
        }
        for (size_t fighter = 0; fighter < 2; fighter++) {
            const size_t row = arena * 2 + fighter;
            const size_t opponent = 1 - fighter;
            const float* measured = fall_floats + row * FALL_FLOAT_FIELDS;
            const int64_t* integers = fall_integers + row * FALL_INTEGER_FIELDS;
            const RekG1FallState& fall = states[arena].fall[fighter];
            float* fall_out = fall_observations + row * 15;
            fall_out[0] = static_cast<float>(integers[0]);
            fall_out[1] = measured[0];
            fall_out[2] = measured[1];
            fall_out[3] = static_cast<float>(integers[1]);
            fall_out[4] = static_cast<float>(integers[5]);
            fall_out[5] = static_cast<float>(integers[6]);
            fall_out[6] = static_cast<float>(integers[3]);
            fall_out[7] = static_cast<float>(integers[4]);
            fall_out[8] = static_cast<float>(fall.phase);
            fall_out[9] = fall.fallen_hold_seconds;
            fall_out[10] = fall.fallen_elapsed_seconds;
            fall_out[11] = fall.fallen_timer_seconds;
            fall_out[12] = fall.reset_grace_remaining_seconds;
            fall_out[13] = static_cast<float>(fall.recovery_armed);
            fall_out[14] = static_cast<float>(tick_fall_events[row]);

            const RekG1FightState& fight = states[arena].combat.fight;
            float* out = fight_observations + row * 39;
            out[0] = static_cast<float>(fighter);
            out[1] = static_cast<float>(fight.phase);
            out[2] = static_cast<float>(fight.current_round_number);
            out[3] = static_cast<float>(fight.current_round_is_redo);
            out[4] = fight.round_duration_seconds;
            out[5] = fight.time_remaining_seconds;
            out[6] = static_cast<float>(fight.clean_hits[fighter]);
            out[7] = static_cast<float>(fight.clean_hits[opponent]);
            out[8] = static_cast<float>(fight.falls[fighter]);
            out[9] = static_cast<float>(fight.falls[opponent]);
            out[10] = static_cast<float>(fight.rounds_won[fighter]);
            out[11] = static_cast<float>(fight.rounds_won[opponent]);
            out[12] = static_cast<float>(fight.last_struck_valid[fighter]);
            out[13] = static_cast<float>(fight.last_struck_valid[opponent]);
            out[14] = fight.last_struck_age_seconds[fighter];
            out[15] = fight.last_struck_age_seconds[opponent];
            out[16] = fight.last_struck_speed[fighter];
            out[17] = fight.last_struck_speed[opponent];
            out[18] = static_cast<float>(fight.fall_classification[fighter]);
            out[19] = static_cast<float>(fight.fall_classification[opponent]);
            out[20] = static_cast<float>(fight.count_active[fighter]);
            out[21] = static_cast<float>(fight.count_active[opponent]);
            out[22] = static_cast<float>(fight.count_is_slip[fighter]);
            out[23] = static_cast<float>(fight.count_is_slip[opponent]);
            out[24] = fight.count_elapsed_seconds;
            out[25] = fight.count_duration_seconds;
            out[26] = static_cast<float>(fight.round_result);
            out[27] = static_cast<float>(fight.round_winner_index);
            out[28] = static_cast<float>(fight.knockout_occurred);
            out[29] = static_cast<float>(fight.fight_result);
            out[30] = static_cast<float>(fight.fight_winner_index);
            out[31] = static_cast<float>(tick_signals[arena]);
            out[32] = static_cast<float>(tick_referee_calls[arena]);
            out[33] = static_cast<float>(tick_score_delta[arena * 2 + fighter]);
            out[34] = static_cast<float>(tick_score_delta[arena * 2 + opponent]);
            out[35] = static_cast<float>(tick_fall_events[arena * 2 + fighter]);
            out[36] = static_cast<float>(tick_fall_events[arena * 2 + opponent]);
            out[37] = static_cast<float>(tick_attributed_contacts[arena]);
            out[38] = static_cast<float>(tick_scored_contacts[arena]);
            rewards[row] = out[33] - out[34];
            terminal_observations[row] = terminals[arena] ? 1.0f : 0.0f;
        }
    }
}
}  // namespace

extern "C" size_t rek_g1_cuda_native_combat_state_size(void) {
    return sizeof(RekG1CudaNativeCombatState);
}

extern "C" size_t rek_g1_cuda_native_combat_contact_size(void) {
    return sizeof(RekG1HitContact);
}

extern "C" size_t rek_g1_cuda_native_combat_impact_event_size(void) {
    return sizeof(RekG1ImpactEvent);
}

extern "C" size_t rek_g1_cuda_native_combat_impact_event_count(void) {
    return REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT;
}

extern "C" cudaError_t rek_g1_cuda_native_combat_upload_catalog(
        RekG1ImpactEvent* impact_events,
        int32_t* route_event_offsets,
        int32_t* route_event_counts) {
    if (!impact_events || !route_event_offsets || !route_event_counts) {
        return cudaErrorInvalidValue;
    }
    const RekG1StrikeCatalog* catalog = rek_g1_current_build_strike_catalog();
    if (!rek_g1_validate_strike_catalog(catalog)) return cudaErrorInvalidValue;
    int32_t offsets[ROUTE_COUNT];
    int32_t counts[ROUTE_COUNT];
    for (int route = 0; route < ROUTE_COUNT; route++) {
        offsets[route] = -1;
        counts[route] = 0;
    }
    for (size_t index = 0; index < catalog->count; index++) {
        const RekG1StrikeCatalogEntry& entry = catalog->entries[index];
        offsets[entry.route_id] = entry.impact_event_offset;
        counts[entry.route_id] = entry.impact_event_count;
    }
    cudaError_t status = cudaMemcpy(
        impact_events, catalog->impact_events,
        catalog->impact_event_count * sizeof(*impact_events),
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    status = cudaMemcpy(
        route_event_offsets, offsets, sizeof(offsets), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    return cudaMemcpy(
        route_event_counts, counts, sizeof(counts), cudaMemcpyHostToDevice);
}

extern "C" cudaError_t rek_g1_cuda_native_combat_init(
        RekG1CudaNativeCombatState* states,
        int32_t* statuses,
        size_t arena_count,
        cudaStream_t stream) {
    if (arena_count == 0) return cudaSuccess;
    if (!states || !statuses) return cudaErrorInvalidValue;
    init_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, statuses, arena_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_native_combat_begin_tick(
        RekG1CudaNativeCombatState* states,
        uint32_t* tick_fall_events,
        uint32_t* tick_signals,
        uint32_t* tick_referee_calls,
        int32_t* tick_score_delta,
        uint32_t* tick_attributed_contacts,
        uint32_t* tick_scored_contacts,
        uint8_t* terminals,
        uint8_t* episode_reset,
        uint8_t* input_reset,
        int32_t* statuses,
        size_t arena_count,
        cudaStream_t stream) {
    if (arena_count == 0) return cudaSuccess;
    if (!states || !tick_fall_events || !tick_signals || !tick_referee_calls
            || !tick_score_delta || !tick_attributed_contacts
            || !tick_scored_contacts || !terminals || !episode_reset
            || !input_reset || !statuses) return cudaErrorInvalidValue;
    begin_tick_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, tick_fall_events, tick_signals, tick_referee_calls,
        tick_score_delta, tick_attributed_contacts, tick_scored_contacts,
        terminals, episode_reset, input_reset, statuses, arena_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_native_combat_post_step(
        RekG1CudaNativeCombatState* states,
        const float* fall_floats,
        const int64_t* fall_integers,
        const uint8_t* fall_valid,
        const int64_t* hit_integers,
        const float* hit_floats,
        const uint8_t* candidate_valid,
        const int64_t* candidate_order,
        const int64_t* candidate_offsets,
        const int64_t* candidate_counts,
        const uint8_t* arena_scan_valid,
        const float* arena_time_seconds,
        const SonicMotionComposerNative* composers,
        const int32_t* active_route_ids,
        const RekG1ImpactEvent* impact_events,
        const int32_t* route_event_offsets,
        const int32_t* route_event_counts,
        RekG1HitContact* packed_contacts,
        uint32_t* tick_fall_events,
        uint32_t* tick_signals,
        uint32_t* tick_referee_calls,
        int32_t* tick_score_delta,
        uint32_t* tick_attributed_contacts,
        uint32_t* tick_scored_contacts,
        uint8_t* terminals,
        uint8_t* input_reset,
        uint8_t* dampened,
        uint8_t* begin_reset,
        uint8_t* complete_reset,
        uint8_t* clear_contacts,
        int32_t* statuses,
        size_t arena_count,
        size_t directed_candidate_capacity,
        cudaStream_t stream) {
    if (arena_count == 0) return cudaSuccess;
    if (!states || !fall_floats || !fall_integers || !fall_valid
            || !hit_integers || !hit_floats || !candidate_valid
            || !candidate_order || !candidate_offsets || !candidate_counts
            || !arena_scan_valid || !arena_time_seconds || !composers
            || !active_route_ids || !impact_events || !route_event_offsets
            || !route_event_counts || !packed_contacts || !tick_fall_events
            || !tick_signals || !tick_referee_calls || !tick_score_delta
            || !tick_attributed_contacts || !tick_scored_contacts
            || !terminals || !input_reset || !dampened || !begin_reset
            || !complete_reset || !clear_contacts || !statuses
            || directed_candidate_capacity == 0) return cudaErrorInvalidValue;
    pack_contacts_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, terminals, hit_integers, hit_floats, candidate_valid, candidate_order,
        candidate_offsets, candidate_counts, arena_scan_valid,
        arena_time_seconds, composers, active_route_ids, impact_events,
        route_event_offsets, route_event_counts, packed_contacts, statuses,
        arena_count, directed_candidate_capacity);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    post_step_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, fall_floats, fall_integers, fall_valid, candidate_offsets,
        candidate_counts, arena_time_seconds, packed_contacts,
        tick_fall_events, tick_signals, tick_referee_calls, tick_score_delta,
        tick_attributed_contacts, tick_scored_contacts, terminals, input_reset,
        dampened, begin_reset, complete_reset, clear_contacts, statuses,
        arena_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_native_combat_observe(
        const RekG1CudaNativeCombatState* states,
        const float* fall_floats,
        const int64_t* fall_integers,
        const uint8_t* fall_valid,
        const uint32_t* tick_fall_events,
        const uint32_t* tick_signals,
        const uint32_t* tick_referee_calls,
        const int32_t* tick_score_delta,
        const uint32_t* tick_attributed_contacts,
        const uint32_t* tick_scored_contacts,
        const uint8_t* terminals,
        float* fall_observations,
        float* fight_observations,
        float* rewards,
        float* terminal_observations,
        int32_t* statuses,
        size_t arena_count,
        cudaStream_t stream) {
    if (arena_count == 0) return cudaSuccess;
    if (!states || !fall_floats || !fall_integers || !fall_valid
            || !tick_fall_events || !tick_signals || !tick_referee_calls
            || !tick_score_delta || !tick_attributed_contacts
            || !tick_scored_contacts || !terminals || !fall_observations
            || !fight_observations || !rewards || !terminal_observations
            || !statuses) return cudaErrorInvalidValue;
    observe_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, fall_floats, fall_integers, fall_valid, tick_fall_events,
        tick_signals, tick_referee_calls, tick_score_delta,
        tick_attributed_contacts, tick_scored_contacts, terminals,
        fall_observations, fight_observations, rewards, terminal_observations,
        statuses, arena_count);
    return cudaGetLastError();
}
