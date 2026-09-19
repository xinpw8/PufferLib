#include "test_recovered_balance_fixture.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#ifdef REK5_BALANCE_TEST_CUDA
extern "C" int rek5_balance_test_cuda(rek5_balance_test::Outcome*, unsigned);
#endif

int main() {
    using namespace rek5_balance;
    unsigned checks = 0;
#define CHECK(value) do { ++checks; if (!(value)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); return 1; } } while (0)
    State state{};
    CHECK(init_active(&state) == Ok);
    const auto original_state = state;
    Result sentinel;
    std::memset(&sentinel, 0x5a, sizeof(sentinel));
    auto reject = [&](const Input& input, Status expected) {
        Result output = sentinel;
        return step(&state, &input, &output) == expected
            && std::memcmp(&output, &sentinel, sizeof(output)) == 0
            && std::memcmp(&state, &original_state, sizeof(state)) == 0;
    };
    for (unsigned fighter = 0; fighter < 2; ++fighter) {
        for (unsigned bit = 0; bit < 10; ++bit) {
            auto input = rek5_balance_test::neutral_input(state);
            input.fighters[fighter].provenance.modeled &= ~(1u << bit);
            CHECK(reject(input, MissingInput));
        }
    }
    for (unsigned bit = 0; bit < 3; ++bit) {
        auto input = rek5_balance_test::neutral_input(state);
        input.provenance.modeled &= ~(1u << bit);
        CHECK(reject(input, MissingInput));
    }
    auto input = rek5_balance_test::neutral_input(state);
    input.fighters[0].provenance.measured = Tilt;
    CHECK(reject(input, InvalidInput));
    input = rek5_balance_test::neutral_input(state);
    input.fighters[0].provenance.modeled |= 1u << 20;
    CHECK(reject(input, InvalidInput));
    input = rek5_balance_test::neutral_input(state);
    input.fighters[1].dynamics.fixed_delta_seconds = 0.25f;
    CHECK(reject(input, InvalidInput));
    input = rek5_balance_test::neutral_input(state);
    input.fighters[1].dynamics.tilt_degrees = std::numeric_limits<float>::quiet_NaN();
    CHECK(reject(input, FallRejected));
    input.fighters[1].detector_tick_enabled = 0;
    CHECK(reject(input, FallRejected));
    input = rek5_balance_test::neutral_input(state);
    input.contact_count = 1;
    CHECK(reject(input, InvalidInput));
    input = rek5_balance_test::neutral_input(state);
    input.fighters[1].dynamics.can_get_up = 2;
    CHECK(reject(input, FallRejected));
    input = rek5_balance_test::neutral_input(state);
    input.time_remaining_seconds = 121;
    CHECK(reject(input, InvalidInput));
    CHECK(step(nullptr, &input, &sentinel) == NullArgument);

    constexpr unsigned count = 4096;
    std::vector<rek5_balance_test::Outcome> expected(count);
    for (unsigned i = 0; i < count; ++i) {
        expected[i] = rek5_balance_test::run_case(i);
        if (expected[i].failure_line) {
            std::fprintf(stderr, "Fixture %u failed line %d\n", i, expected[i].failure_line);
            return 1;
        }
        checks += expected[i].checks;
    }
#ifdef REK5_BALANCE_TEST_CUDA
    std::vector<rek5_balance_test::Outcome> actual(count);
    CHECK(rek5_balance_test_cuda(actual.data(), count) == 0);
    for (unsigned i = 0; i < count; ++i) {
        if (actual[i].failure_line || actual[i].digest != expected[i].digest
                || actual[i].checks != expected[i].checks) {
            std::fprintf(stderr, "CUDA mismatch case %u line %d digest %llu / %llu\n",
                i, actual[i].failure_line, (unsigned long long)actual[i].digest,
                (unsigned long long)expected[i].digest);
            return 1;
        }
        ++checks;
    }
    std::printf("recovered_balance_cuda_arenas=%u ", count);
#endif
    std::printf("recovered_balance_checks=%u passed\n", checks);
    return 0;
}
