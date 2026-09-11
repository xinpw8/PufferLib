#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <vector>

extern "C" {
#include "g1_combat_tick.h"
int rek_g1_combat_fixture_main(void);
int rek_g1_fall_fixture_main(void);
int rek_g1_hit_fixture_main(void);
}
#include "g1_combat_cuda.h"

static size_t comparisons;
static size_t combat_fixture_calls;
static size_t fall_fixture_calls;
static size_t hit_fixture_calls;

static void require(bool value, const char* name) {
    comparisons++;
    if (!value) {
        std::fprintf(stderr, "GPU comparison failed: %s\n", name);
        std::exit(1);
    }
}

static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

template<class T> struct DeviceArray {
    T* data = nullptr;
    size_t size;
    explicit DeviceArray(size_t count, const T* source = nullptr): size(count) {
        if (!size) return;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), sizeof(T) * size));
        if (source) cuda_check(cudaMemcpy(data, source, sizeof(T) * size,
            cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { if (data) cuda_check(cudaFree(data)); }
    DeviceArray(const DeviceArray&) = delete;
    void read(T* destination) const {
        if (size) cuda_check(cudaMemcpy(destination, data, sizeof(T) * size,
            cudaMemcpyDeviceToHost));
    }
};

struct DeviceContacts {
    std::vector<RekG1HitContact> host;
    std::vector<std::unique_ptr<DeviceArray<RekG1ImpactEvent>>> events;
    std::unique_ptr<DeviceArray<RekG1HitContact>> contacts;
    DeviceContacts(const RekG1HitContact* source, size_t count) {
        if (!source || !count) return;
        host.assign(source, source + count);
        for (auto& contact : host) {
            auto& intent = contact.strike_intent;
            if (intent.impact_events) {
                events.emplace_back(new DeviceArray<RekG1ImpactEvent>(
                    intent.impact_event_count, intent.impact_events));
                intent.impact_events = events.back()->data;
            }
        }
        contacts.reset(new DeviceArray<RekG1HitContact>(count, host.data()));
    }
    const RekG1HitContact* data() const {
        return contacts ? contacts->data : nullptr;
    }
};

/* Compare every named field, with exact float bits, excluding C padding. */
#define FIELD(a, b, field) require(std::memcmp(&(a).field, &(b).field, \
    sizeof((a).field)) == 0, #field)

static void compare_hit_state(const RekG1HitDetectorState& a,
        const RekG1HitDetectorState& b) {
    FIELD(a, b, last_score_time_seconds);
    FIELD(a, b, scored_move_id);
    FIELD(a, b, scored_apex_mask);
    FIELD(a, b, cooldown_seen);
    FIELD(a, b, scored_move_seen);
}

static void compare_arena(const RekG1CombatArenaState& a,
        const RekG1CombatArenaState& b) {
    FIELD(a, b, initialized);
    compare_hit_state(a.hit_detector, b.hit_detector);
    const auto& x = a.fight;
    const auto& y = b.fight;
    FIELD(x, y, phase);
    FIELD(x, y, current_round_number);
    FIELD(x, y, current_round_is_redo);
    FIELD(x, y, round_duration_seconds);
    FIELD(x, y, time_remaining_seconds);
    FIELD(x, y, clean_hits);
    FIELD(x, y, falls);
    FIELD(x, y, round_result);
    FIELD(x, y, round_winner_index);
    FIELD(x, y, knockout_occurred);
    FIELD(x, y, rounds_won);
    FIELD(x, y, fight_result);
    FIELD(x, y, fight_winner_index);
    FIELD(x, y, last_struck_valid);
    FIELD(x, y, last_struck_age_seconds);
    FIELD(x, y, last_struck_speed);
    FIELD(x, y, fall_classification);
    FIELD(x, y, fall_forced_by_estop);
    FIELD(x, y, count_active);
    FIELD(x, y, count_is_slip);
    FIELD(x, y, count_elapsed_seconds);
    FIELD(x, y, count_duration_seconds);
    FIELD(x, y, transition_remaining_seconds);
}

static void compare_combat(const RekG1CombatSubstepResult& a,
        const RekG1CombatSubstepResult& b) {
    compare_arena(a.next_state, b.next_state);
    FIELD(a, b, signals);
    FIELD(a, b, referee_calls);
    FIELD(a, b, score_delta);
    FIELD(a, b, rounds_won_delta);
    FIELD(a, b, attributed_contact_count);
    FIELD(a, b, scored_contact_count);
}

extern "C" RekG1CombatTickStatus rek_g1_test_cuda_init(
        RekG1CombatArenaState* state) {
    DeviceArray<RekG1CombatArenaState> gpu_state(1);
    DeviceArray<RekG1CombatTickStatus> gpu_status(1);
    const auto status = rek_g1_combat_arena_init(state);
    cuda_check(rek_g1_cuda_combat_init(gpu_state.data, gpu_status.data, 1, 0));
    RekG1CombatTickStatus observed;
    RekG1CombatArenaState result;
    gpu_status.read(&observed);
    gpu_state.read(&result);
    require(status == observed, "init status");
    compare_arena(*state, result);
    return status;
}

extern "C" RekG1CombatTickStatus rek_g1_test_cuda_reset(
        const RekG1CombatArenaState* state, RekG1CombatArenaState* output) {
    DeviceArray<RekG1CombatArenaState> gpu_state(1, state), gpu_result(1);
    DeviceArray<RekG1CombatTickStatus> gpu_status(1);
    const auto status = rek_g1_combat_arena_apply_spawn_reset(state, output);
    cuda_check(rek_g1_cuda_combat_spawn_reset(
        gpu_state.data, gpu_result.data, gpu_status.data, 1, 0));
    RekG1CombatTickStatus observed;
    RekG1CombatArenaState result;
    gpu_status.read(&observed);
    gpu_result.read(&result);
    require(status == observed, "reset status");
    compare_arena(*output, result);
    return status;
}

extern "C" RekG1CombatTickStatus rek_g1_test_cuda_substep(
        const RekG1CombatArenaState* state, const RekG1HitDetectorConfig* config,
        const RekG1CombatSubstepInput* input, RekG1CombatSubstepResult* output) {
    combat_fixture_calls++;
    DeviceContacts contacts(input->contacts, input->contact_count);
    RekG1CombatSubstepInput relocated = *input;
    relocated.contacts = contacts.data();
    DeviceArray<RekG1CombatArenaState> gpu_state(1, state);
    DeviceArray<RekG1HitDetectorConfig> gpu_config(1, config);
    DeviceArray<RekG1CombatSubstepInput> gpu_input(1, &relocated);
    DeviceArray<RekG1CombatSubstepResult> gpu_result(1, output);
    DeviceArray<RekG1CombatTickStatus> gpu_status(1);
    const auto status = rek_g1_combat_arena_substep(state, config, input, output);
    cuda_check(rek_g1_cuda_combat_substep(gpu_state.data, gpu_config.data,
        gpu_input.data, gpu_result.data, gpu_status.data, 1, 0));
    RekG1CombatTickStatus observed;
    RekG1CombatSubstepResult result;
    gpu_status.read(&observed);
    gpu_result.read(&result);
    require(status == observed, "combat status");
    if (status == REK_G1_COMBAT_TICK_OK) compare_combat(*output, result);
    else require(std::memcmp(output, &result, sizeof(result)) == 0,
        "combat failure preserves output bytes");
    return status;
}

extern "C" RekG1FallStatus rek_g1_test_cuda_fall_step(
        const RekG1FallConfig* config, const RekG1FallState* state,
        const RekG1FallSample* sample, RekG1FallStepResult* output) {
    if (!config) {
        require(rek_g1_cuda_fall_step(nullptr, nullptr, nullptr, nullptr,
            nullptr, 1, 0) == cudaErrorInvalidValue, "null CUDA fall config");
        return rek_g1_fall_state_step(config, state, sample, output);
    }
    fall_fixture_calls++;
    DeviceArray<RekG1FallConfig> gpu_config(1, config);
    DeviceArray<RekG1FallState> gpu_state(1, state);
    DeviceArray<RekG1FallSample> gpu_sample(1, sample);
    DeviceArray<RekG1FallStepResult> gpu_result(1, output);
    DeviceArray<RekG1FallStatus> gpu_status(1);
    const auto status = rek_g1_fall_state_step(config, state, sample, output);
    cuda_check(rek_g1_cuda_fall_step(gpu_config.data, gpu_state.data,
        gpu_sample.data, gpu_result.data, gpu_status.data, 1, 0));
    RekG1FallStatus observed;
    RekG1FallStepResult result;
    gpu_status.read(&observed);
    gpu_result.read(&result);
    require(status == observed, "fall status");
    if (status == REK_G1_FALL_OK) {
        const auto& a = output->next_state;
        const auto& b = result.next_state;
        FIELD(a, b, phase);
        FIELD(a, b, fallen_hold_seconds);
        FIELD(a, b, fallen_elapsed_seconds);
        FIELD(a, b, fallen_timer_seconds);
        FIELD(a, b, reset_grace_remaining_seconds);
        FIELD(a, b, recovery_armed);
        require(output->events == result.events, "fall events");
    } else require(std::memcmp(output, &result, sizeof(result)) == 0,
        "fall failure preserves output bytes");
    return status;
}

extern "C" int rek_g1_test_cuda_hit_process(RekG1HitDetectorState* state,
        const RekG1HitDetectorConfig* config, const RekG1HitContact* contact,
        RekG1HitResult* output) {
    hit_fixture_calls++;
    DeviceContacts contacts(contact, 1);
    DeviceArray<RekG1HitDetectorState> gpu_state(1, state);
    DeviceArray<RekG1HitDetectorConfig> gpu_config(1, config);
    DeviceArray<RekG1HitResult> gpu_result(1, output);
    DeviceArray<int> gpu_status(1);
    const int status = rek_g1_hit_detector_process(state, config, contact, output);
    cuda_check(rek_g1_cuda_hit_process(gpu_state.data, gpu_config.data,
        contacts.data(), gpu_result.data, gpu_status.data, 1, 0));
    int observed;
    RekG1HitResult result;
    RekG1HitDetectorState next_state;
    gpu_status.read(&observed);
    gpu_result.read(&result);
    gpu_state.read(&next_state);
    require(status == observed, "hit status");
    compare_hit_state(*state, next_state);
    if (status) {
        FIELD(*output, result, points_awarded);
        FIELD(*output, result, apex_event_index);
        FIELD(*output, result, apex_ramp);
        FIELD(*output, result, attribution_accepted);
        FIELD(*output, result, score_accepted);
    } else require(std::memcmp(output, &result, sizeof(result)) == 0,
        "hit failure preserves output bytes");
    return status;
}

static void test_batch() {
    constexpr size_t count = 4099; // Multiple blocks and a partial final block.
    constexpr size_t ticks = 256;
    std::vector<RekG1CombatArenaState> states(count);
    std::vector<RekG1CombatSubstepInput> inputs(count);
    std::vector<RekG1CombatSubstepResult> expected(count), observed(count);
    std::vector<RekG1CombatTickStatus> statuses(count);
    std::vector<RekG1HitContact> contacts(count);
    RekG1ImpactEvent event{};
    event.impact_time_seconds = 0.1f;
    event.lead_time_seconds = 0.1f;
    event.release_time_seconds = 0.2f;
    event.limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY;
    const auto config = rek_g1_current_build_hit_detector_config();
    for (size_t i = 0; i < count; ++i) {
        require(rek_g1_combat_arena_init(&states[i]) == REK_G1_COMBAT_TICK_OK,
            "batch host init");
        auto& contact = contacts[i];
        contact.strike_intent.impact_events = &event;
        contact.strike_intent.impact_event_count = 1;
        contact.strike_intent.clip_cursor_frames = float(i % 16);
        contact.strike_intent.clip_fps = 50.0f;
        contact.strike_intent.move_id = int(i);
        contact.strike_intent.action_playing = 1;
        contact.strike_intent.layer_active = 1;
        contact.target_body_position_world[0] = 1;
        contact.target_body_position_world[1] = 0.5f * float(i % 3);
        contact.striker_body_linear_velocity_world[0] = 3;
        contact.relative_speed_mps = 1.5f + 0.25f * float(i % 17);
        contact.time_seconds = 0.002f;
        contact.striker_part = REK_G1_BODY_PART_FOOT;
        contact.striker_side = REK_G1_HAND_LEFT;
        contact.target_zone = (i % 3) ? REK_G1_BODY_ZONE_TORSO
            : REK_G1_BODY_ZONE_LEFT_WRIST;
        contact.striker_fighter = unsigned(i % 2);
        contact.target_fighter = 1 - contact.striker_fighter;
        contact.striker_body_slot = 2;
        contact.is_enter = contact.round_active = 1;
        contact.striker_upright = contact.target_upright = 1;
        contact.target_standing = 1;
    }
    DeviceArray<RekG1CombatArenaState> gpu_states(count, states.data());
    DeviceArray<RekG1CombatSubstepInput> gpu_inputs(count);
    DeviceArray<RekG1HitDetectorConfig> gpu_config(1, &config);
    DeviceArray<RekG1CombatSubstepResult> gpu_results(count);
    DeviceArray<RekG1CombatTickStatus> gpu_statuses(count);
    DeviceContacts gpu_contacts(contacts.data(), count);
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (size_t tick = 0; tick < ticks; ++tick) {
        for (size_t i = 0; i < count; ++i) {
            auto& input = inputs[i];
            input = {};
            input.delta_seconds = 0.02f;
            input.time_remaining_seconds = std::fmax(0.0f,
                states[i].fight.time_remaining_seconds - input.delta_seconds);
            input.contact_count = tick == 0 ? 1 : 0;
            input.contacts = tick == 0 ? &contacts[i] : nullptr;
            if (i % 5 == 0 && tick >= 1 && tick < 200) {
                input.fall_phase[i % 2] = tick == 1
                    ? REK_G1_FALL_FALLING : REK_G1_FALL_FALLEN;
                input.fall_events[i % 2] = tick == 1
                    ? REK_G1_FALL_EVENT_FALLING_STARTED : tick == 2
                    ? REK_G1_FALL_EVENT_BECAME_FALLEN : 0;
            }
            require(rek_g1_combat_arena_substep(&states[i], &config,
                &input, &expected[i]) == REK_G1_COMBAT_TICK_OK, "batch host step");
            input.contacts = tick == 0 ? gpu_contacts.data() + i : nullptr;
        }
        cuda_check(cudaMemcpyAsync(gpu_inputs.data, inputs.data(),
            sizeof(inputs[0]) * count, cudaMemcpyHostToDevice, stream));
        cuda_check(rek_g1_cuda_combat_substep(gpu_states.data, gpu_config.data,
            gpu_inputs.data, gpu_results.data, gpu_statuses.data, count, stream));
        cuda_check(cudaStreamSynchronize(stream));
        gpu_results.read(observed.data());
        gpu_statuses.read(statuses.data());
        for (size_t i = 0; i < count; ++i) {
            require(statuses[i] == REK_G1_COMBAT_TICK_OK, "batch GPU status");
            compare_combat(expected[i], observed[i]);
            states[i] = observed[i].next_state;
        }
        cuda_check(cudaMemcpy(gpu_states.data, states.data(),
            sizeof(states[0]) * count, cudaMemcpyHostToDevice));
    }
    cuda_check(cudaStreamDestroy(stream));
    std::printf("batch_arenas=%zu batch_ticks=%zu exact_steps=%zu\n",
        count, ticks, count * ticks);
}

int main() {
    cudaDeviceProp properties;
    cuda_check(cudaGetDeviceProperties(&properties, 0));
    std::printf("device=%s compute=%d.%d\n", properties.name,
        properties.major, properties.minor);
    require(rek_g1_cuda_combat_substep(nullptr, nullptr, nullptr, nullptr,
        nullptr, 0, 0) == cudaSuccess, "empty CUDA batch");
    require(rek_g1_combat_fixture_main() == 0, "combat fixtures");
    require(rek_g1_fall_fixture_main() == 0, "fall fixtures");
    require(rek_g1_hit_fixture_main() == 0, "hit fixtures");
    test_batch();
    std::printf("{\"status\":\"ok\",\"comparisons\":%zu,"
        "\"combat_fixture_calls\":%zu,\"fall_fixture_calls\":%zu,"
        "\"hit_fixture_calls\":%zu,\"arena_state_bytes\":%zu,"
        "\"combat_input_bytes\":%zu,\"combat_result_bytes\":%zu}\n",
        comparisons, combat_fixture_calls, fall_fixture_calls, hit_fixture_calls,
        sizeof(RekG1CombatArenaState), sizeof(RekG1CombatSubstepInput),
        sizeof(RekG1CombatSubstepResult));
    return 0;
}
