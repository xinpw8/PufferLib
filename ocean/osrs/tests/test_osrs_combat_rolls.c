#include <assert.h>
#include <stdint.h>

#include "../osrs_combat.h"

static void test_hit_chance_fraction_matches_osrs_formula(void) {
    uint64_t num, den;

    osrs_hit_chance_fraction(100, 200, &num, &den);
    assert(num == 100);
    assert(den == 402);

    osrs_hit_chance_fraction(250, 100, &num, &den);
    assert(num == 400);
    assert(den == 502);
}

static void test_double_hit_chance_fraction_matches_closed_form(void) {
    uint64_t num, den;

    osrs_hit_chance_double_fraction(100, 200, &num, &den);
    assert(num == 100ull * 405ull);
    assert(den == 6ull * 101ull * 201ull);

    osrs_hit_chance_double_fraction(250, 100, &num, &den);
    assert(num == 6ull * 251ull * 251ull - 102ull * 203ull);
    assert(den == 6ull * 251ull * 251ull);
}

static void test_roll_ratio_consumes_rng_for_certain_outcomes(void) {
    uint32_t rng_zero = 12345;
    uint32_t rng_one = 12345;

    assert(encounter_roll_ratio_u16(&rng_zero, 0, 7) == 0);
    assert(encounter_roll_ratio_u16(&rng_one, 7, 7) == 1);
    assert(rng_zero == rng_one);
    assert(rng_zero != 12345);
}

int main(void) {
    test_hit_chance_fraction_matches_osrs_formula();
    test_double_hit_chance_fraction_matches_closed_form();
    test_roll_ratio_consumes_rng_for_certain_outcomes();
    return 0;
}
