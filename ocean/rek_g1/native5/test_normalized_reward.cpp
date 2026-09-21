#include "normalized_reward.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace {
using rek5_normalized_reward::Result;
using rek5_normalized_reward::value;
std::uint64_t checks = 0;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance, const char* message) {
    check(std::abs(actual - expected) <= tolerance, message);
}
void verify(int own, int opponent, std::uint32_t events, const Result& result) {
    const std::int64_t difference = std::int64_t(own) - std::int64_t(opponent);
    const bool fell = (events & REK_G1_FALL_EVENT_BECAME_FALLEN) != 0;
    const std::int64_t raw = difference - int(fell);
    const double expected = std::max(-1.0, std::min(1.0, double(raw) / 100.0));
    check(std::isfinite(result.reward) && std::isfinite(result.score_reward)
        && std::isfinite(result.fall_reward), "nonfinite reward component");
    check(result.reward >= -1 && result.reward <= 1, "reward escaped fixed range");
    check(result.raw_total == raw, "raw point difference overflow or wrong event");
    check(result.reward == float(expected), "wrong normalized reward");
    check(result.score_reward == float(double(difference) / 100.0), "wrong score component");
    check(result.fall_reward == (fell ? -.01f : 0.f), "wrong fall component");
    check(result.saturated == (raw < -100 || raw > 100), "wrong saturation diagnostic");
}

void elementary_cases() {
    check(std::strcmp(rek5_normalized_reward::kMode, "normalized_points_falls_v1") == 0,
        "reward mode identifier changed");
    close(value(1, 0, 0).reward, .01f, 0, "one awarded point");
    close(value(5, 0, 0).reward, .05f, 0, "five awarded points");
    close(value(0, 5, 0).reward, -.05f, 0, "five conceded points");
    close(value(0, 0, REK_G1_FALL_EVENT_BECAME_FALLEN).reward, -.01f, 0,
        "new own confirmed fall");
    close(value(0, 5, REK_G1_FALL_EVENT_BECAME_FALLEN).reward, -.06f, 0,
        "conceded points and fall components must both count");
    close(value(5, 0, REK_G1_FALL_EVENT_BECAME_FALLEN).reward, .04f, 0,
        "simultaneous own points and own fall");
    close(value(5, 5, REK_G1_FALL_EVENT_BECAME_FALLEN).reward, -.01f, 0,
        "equal awards still charge own fall");
    close(value(0, 0, REK_G1_FALL_EVENT_FALLING_STARTED).reward, 0, 0,
        "unconfirmed falling must not charge");
    close(value(0, 0, REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE).reward, 0, 0,
        "repeated timeout must not charge");
    close(value(0, 0, 0).reward, 0, 0, "idle terminal/reset step has no bonus");
    check(value(100, 0, 0).reward == 1 && !value(100, 0, 0).saturated,
        "positive boundary unnecessarily clips");
    check(value(0, 99, REK_G1_FALL_EVENT_BECAME_FALLEN).reward == -1
        && !value(0, 99, REK_G1_FALL_EVENT_BECAME_FALLEN).saturated,
        "negative boundary unnecessarily clips");
    check(value(0, 100, REK_G1_FALL_EVENT_BECAME_FALLEN).reward == -1
        && value(0, 100, REK_G1_FALL_EVENT_BECAME_FALLEN).saturated,
        "fall crossing negative boundary does not report clipping");
}

void exhaustive_cases() {
    // Exhaust every native event-bit combination and every pair of small
    // signed deltas. Negative values are not valid awarded points, but the
    // helper must remain finite and overflow-free if a caller is incorrect.
    for (int own = -150; own <= 150; ++own) {
        for (int opponent = -150; opponent <= 150; ++opponent) {
            for (std::uint32_t events = 0; events < 16; ++events) {
                verify(own, opponent, events, value(own, opponent, events));
                // Unknown future event bits cannot create another fall.
                verify(own, opponent, events | 0xfffffff0u,
                    value(own, opponent, events | 0xfffffff0u));
            }
            close(value(own, opponent, 0).reward,
                -value(opponent, own, 0).reward, 0, "score antisymmetry");
            check(value(own, opponent, REK_G1_FALL_EVENT_BECAME_FALLEN).reward
                <= value(own, opponent, 0).reward, "own fall increases reward");
            if (own < 150) check(value(own + 1, opponent, 0).reward
                >= value(own, opponent, 0).reward, "own points reduce reward");
            if (opponent < 150) check(value(own, opponent + 1, 0).reward
                <= value(own, opponent, 0).reward, "conceded points increase reward");
        }
    }
    const int extremes[] = {INT_MIN, INT_MIN + 1, -100000001, -101, -100,
        -99, -1, 0, 1, 99, 100, 101, 100000001, INT_MAX - 1, INT_MAX};
    for (int own : extremes) for (int opponent : extremes) {
        for (std::uint32_t events = 0; events < 16; ++events)
            verify(own, opponent, events, value(own, opponent, events));
    }
}

void detector_trajectory() {
    // Run the actual recovered detector. One fall may produce many down ticks
    // and repeated reset-timeout events, but only one BECAME_FALLEN event.
    RekG1FallState state{};
    check(rek_g1_fall_state_init(&REK_G1_FALL_CONFIG_F84F1874, &state)
        == REK_G1_FALL_OK, "fall detector initialization");
    RekG1FallSample sample{};
    sample.tracking_active = 1;
    sample.tilt_degrees = 80;
    sample.pelvis_height_ratio = .2f;
    sample.both_feet_off_floor = 1;
    sample.distinct_nonfoot_body_contact_count = 3;
    sample.fixed_delta_seconds = .02f;
    double accumulated = 0;
    int confirmed = 0, timeouts = 0;
    for (int cycle = 0; cycle < 2; ++cycle) {
        for (int tick = 0; tick < 500; ++tick) {
            RekG1FallStepResult step{};
            check(rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874,
                &state, &sample, &step) == REK_G1_FALL_OK, "fall detector step");
            state = step.next_state;
            confirmed += !!(step.events & REK_G1_FALL_EVENT_BECAME_FALLEN);
            timeouts += !!(step.events & REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE);
            const auto reward = value(0, 0, step.events);
            accumulated += reward.reward;
            check(!reward.saturated, "fall detector stream saturated");
        }
        check(confirmed == cycle + 1, "detector repeated a confirmed fall while down");
        close(accumulated, -.01 * (cycle + 1), 1e-9,
            "persistent fallen state charged more than once");
        RekG1FallState reset{};
        check(rek_g1_fall_state_apply_fight_spawn_reset(&REK_G1_FALL_CONFIG_F84F1874,
            &state, &reset) == REK_G1_FALL_OK, "fight spawn reset");
        state = reset;
        close(value(0, 0, 0).reward, 0, 0, "spawn reset charged again");
    }
    check(timeouts >= 4, "detector did not exercise repeated down timeouts");

    // A transient loss of balance followed by upright recovery is not a fall.
    check(rek_g1_fall_state_init(&REK_G1_FALL_CONFIG_F84F1874, &state)
        == REK_G1_FALL_OK, "recovery trajectory initialization");
    RekG1FallStepResult step{};
    check(rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874,
        &state, &sample, &step) == REK_G1_FALL_OK, "transient falling step");
    check(step.events == REK_G1_FALL_EVENT_FALLING_STARTED, "transient event contract");
    close(value(0, 0, step.events).reward, 0, 0, "transient falling penalty");
    state = step.next_state;
    sample.tilt_degrees = 0;
    sample.pelvis_height_ratio = 1;
    sample.both_feet_off_floor = 0;
    sample.has_foot_body_contact = 1;
    sample.distinct_nonfoot_body_contact_count = 0;
    check(rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874,
        &state, &sample, &step) == REK_G1_FALL_OK, "upright recovery step");
    check(step.events == REK_G1_FALL_EVENT_FALLING_CLEARED, "recovery event contract");
    close(value(0, 0, step.events).reward, 0, 0, "recovery awarded bonus");
}

void score_and_round_trajectories() {
    double own_return = 0, opponent_return = 0;
    for (int tick = 0; tick < 6000; ++tick) {
        const int own = tick == 10 ? 1 : tick == 20 ? 5 : 0;
        const int opponent = tick == 200 ? 5 : 0;
        const std::uint32_t own_events = tick == 100
            ? REK_G1_FALL_EVENT_BECAME_FALLEN : 0;
        const std::uint32_t opponent_events = tick == 30
            ? REK_G1_FALL_EVENT_BECAME_FALLEN : 0;
        own_return += value(own, opponent, own_events).reward;
        opponent_return += value(opponent, own, opponent_events).reward;
        if (tick == 100) close(value(opponent, own, opponent_events).reward, 0, 0,
            "opponent fall earns unawarded bonus");
        if (tick == 5999) close(value(own, opponent, own_events).reward, 0, 0,
            "terminal step double counts outcome");
    }
    close(own_return, 0, 3e-9, "points and fall stream accounting");
    close(opponent_return, -.02, 3e-9, "opponent stream accounting");
    // Starting another round with 0:0 does not undo previous awarded points.
    close(value(0, 0, 0).reward, 0, 0, "round reset reward");
    close(value(1, 0, 0).reward, .01f, 0, "second round first point");
}

#if defined(__CUDACC__)
struct Input { int own, opponent; std::uint32_t events; };
void cuda_ok(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void device_values(const Input* inputs, Result* outputs, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) outputs[i] = value(inputs[i].own, inputs[i].opponent, inputs[i].events);
}
void device_equivalence() {
    std::vector<Input> inputs;
    for (int own = -150; own <= 150; ++own)
        for (int opponent = -150; opponent <= 150; ++opponent)
            for (std::uint32_t events = 0; events < 16; ++events)
                inputs.push_back({own, opponent, events});
    const int extremes[] = {INT_MIN, INT_MIN + 1, -101, -100, -1, 0, 1,
        100, 101, INT_MAX - 1, INT_MAX};
    for (int own : extremes) for (int opponent : extremes)
        for (std::uint32_t events = 0; events < 16; ++events)
            inputs.push_back({own, opponent, events | 0xfffffff0u});
    Input* device_inputs = nullptr;
    Result* device_outputs = nullptr;
    const int count = int(inputs.size());
    cuda_ok(cudaMalloc(&device_inputs, inputs.size() * sizeof(Input)));
    cuda_ok(cudaMalloc(&device_outputs, inputs.size() * sizeof(Result)));
    cuda_ok(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(Input), cudaMemcpyHostToDevice));
    std::vector<Result> outputs(inputs.size());
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t execution;
    cuda_ok(cudaStreamCreate(&stream));
    cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    device_values<<<(count + 255) / 256, 256, 0, stream>>>(device_inputs, device_outputs, count);
    cuda_ok(cudaGetLastError());
    cuda_ok(cudaStreamEndCapture(stream, &graph));
    cuda_ok(cudaGraphInstantiate(&execution, graph, 0));
    for (int replay = 0; replay < 4; ++replay) {
        cuda_ok(cudaGraphLaunch(execution, stream));
        cuda_ok(cudaStreamSynchronize(stream));
        cuda_ok(cudaMemcpy(outputs.data(), device_outputs, outputs.size() * sizeof(Result), cudaMemcpyDeviceToHost));
        for (int i = 0; i < count; ++i)
            verify(inputs[i].own, inputs[i].opponent, inputs[i].events, outputs[i]);
    }
    cuda_ok(cudaGraphExecDestroy(execution));
    cuda_ok(cudaGraphDestroy(graph));
    cuda_ok(cudaStreamDestroy(stream));
    cuda_ok(cudaFree(device_outputs));
    cuda_ok(cudaFree(device_inputs));
    std::printf("{\"test\":\"normalized_reward_cuda\",\"cases_per_replay\":%d,\"graph_replays\":4,\"status\":\"passed\"}\n", count);
}
#endif
}

int main() {
    try {
        elementary_cases();
        exhaustive_cases();
        detector_trajectory();
        score_and_round_trajectories();
        std::printf("{\"test\":\"normalized_reward_cpu\",\"checks\":%llu,\"status\":\"passed\",\"fall_signal\":\"BECAME_FALLEN\",\"physics_parity\":false}\n",
            static_cast<unsigned long long>(checks));
#if defined(__CUDACC__)
        device_equivalence();
#endif
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "normalized reward test: %s\n", error.what());
        return 1;
    }
}
