#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../osrs_combat.h"

#define CHECK(name, cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", name); \
        abort(); \
    } \
} while (0)

static OsrsVenatorMonster live_monster(int slot, int x, int y, int size) {
    return osrs_venator_monster(
        slot, x, y, size, OSRS_VENATOR_MONSTER_ALIVE);
}

static void check_chain_slots(
    const char* name,
    OsrsVenatorChain chain,
    OsrsVenatorChainLength length,
    int slot0,
    int slot1,
    int slot2
) {
    CHECK(name, chain.length == length);
    CHECK(name, chain.hits[0].slot == slot0);
    if (length >= OSRS_VENATOR_CHAIN_LENGTH_TWO) {
        CHECK(name, chain.hits[1].slot == slot1);
        CHECK(name, chain.hits[1].kind == OSRS_VENATOR_HIT_BOUNCE);
    }
    if (length >= OSRS_VENATOR_CHAIN_LENGTH_THREE) {
        CHECK(name, chain.hits[2].slot == slot2);
        CHECK(name, chain.hits[2].kind == OSRS_VENATOR_HIT_BOUNCE);
    }
}

static int chains_equal(OsrsVenatorChain a, OsrsVenatorChain b) {
    if (a.length != b.length) return 0;
    for (int i = 0; i < (int)a.length; i++) {
        if (a.hits[i].slot != b.hits[i].slot) return 0;
        if (a.hits[i].kind != b.hits[i].kind) return 0;
        if (a.hits[i].footprint.sw_x != b.hits[i].footprint.sw_x) return 0;
        if (a.hits[i].footprint.sw_y != b.hits[i].footprint.sw_y) return 0;
        if (a.hits[i].footprint.size != b.hits[i].footprint.size) return 0;
    }
    return 1;
}

static void test_size1_bounce_law(void) {
    OsrsVenatorFootprint a = osrs_venator_footprint(0, 0, 1);
    for (int dx = -4; dx <= 4; dx++) {
        for (int dy = -4; dy <= 4; dy++) {
            OsrsVenatorFootprint b = osrs_venator_footprint(dx, dy, 1);
            int expected = chebyshev_distance(0, 0, dx, dy) <= 2;
            int ab = osrs_venator_can_bounce(a, b);
            int ba = osrs_venator_can_bounce(b, a);
            CHECK("size1 bounce equals cheb<=2", ab == expected);
            CHECK("size1 bounce is symmetric", ab == ba);
        }
    }
}

static void test_bounce_max_hit_law(void) {
    const int cases[] = {0, 1, 2, 3, 29, 30, 31, 98};
    const int expected[] = {0, 0, 1, 2, 19, 20, 20, 65};
    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        CHECK("bounce max is floor(original*2/3)",
              osrs_venator_bounce_max_hit(cases[i]) == expected[i]);
    }
}

static void test_chain_laws_and_determinism(void) {
    OsrsVenatorMonster primary = live_monster(5, 0, 0, 1);
    OsrsVenatorMonster candidates[] = {
        live_monster(9, 1, 1, 1),
        live_monster(3, -1, 1, 1),
    };

    OsrsVenatorChain first = osrs_venator_resolve_chain(
        primary, candidates, 2);
    OsrsVenatorChain second = osrs_venator_resolve_chain(
        primary, candidates, 2);

    CHECK("chain length is at least one", first.length >= OSRS_VENATOR_CHAIN_LENGTH_ONE);
    CHECK("chain length is at most three", first.length <= OSRS_VENATOR_CHAIN_LENGTH_THREE);
    CHECK("hit2 is not primary", first.hits[1].slot != primary.slot);
    CHECK("hit3 is not hit2", first.hits[2].slot != first.hits[1].slot);
    CHECK("lower slot wins equal-distance hit2 tie", first.hits[1].slot == 3);
    CHECK("targeting is deterministic", chains_equal(first, second));
}

static void test_warband_cluster_and_bounce_back(void) {
    OsrsVenatorMonster primary = live_monster(30, 0, 0, 1);
    OsrsVenatorMonster candidates[] = {
        live_monster(20, 1, 0, 1),
        live_monster(10, 2, 0, 1),
    };
    OsrsVenatorChain chain = osrs_venator_resolve_chain(
        primary, candidates, 2);
    check_chain_slots(
        "three-monster warband cluster covers all three slots",
        chain,
        OSRS_VENATOR_CHAIN_LENGTH_THREE,
        30,
        20,
        10);

    OsrsVenatorMonster bounce_primary = live_monster(10, 0, 0, 1);
    OsrsVenatorMonster bounce_candidates[] = {
        live_monster(20, 1, 0, 1),
        live_monster(30, 2, 0, 1),
    };
    OsrsVenatorChain bounce_chain = osrs_venator_resolve_chain(
        bounce_primary, bounce_candidates, 2);
    check_chain_slots(
        "warband tie permits deterministic bounce-back to primary",
        bounce_chain,
        OSRS_VENATOR_CHAIN_LENGTH_THREE,
        10,
        20,
        10);
}

static void test_no_overreach_past_original_centre(void) {
    OsrsVenatorMonster primary = live_monster(100, 0, 0, 1);
    OsrsVenatorMonster candidates[] = {
        live_monster(50, 2, 0, 1),
        live_monster(10, 3, 0, 1),
    };
    OsrsVenatorChain chain = osrs_venator_resolve_chain(primary, candidates, 2);
    check_chain_slots(
        "2nd bounce never reaches a target >2 tiles from the original centre",
        chain,
        OSRS_VENATOR_CHAIN_LENGTH_THREE,
        100,
        50,
        100);
    for (int i = 0; i < (int)chain.length; i++)
        CHECK("over-reach target (3,0) is never in the chain",
              chain.hits[i].slot != 10);
}

static void test_rule_matrix_representatives(void) {
    OsrsVenatorFootprint sender1 = osrs_venator_footprint(0, 0, 1);
    OsrsVenatorFootprint sender2 = osrs_venator_footprint(0, 0, 2);
    OsrsVenatorFootprint sender4 = osrs_venator_footprint(0, 0, 4);

    OsrsVenatorFootprint target2_accept_pass_send_fail =
        osrs_venator_footprint(2, 0, 2);
    OsrsVenatorFootprint target2_pass =
        osrs_venator_footprint(1, 0, 2);
    OsrsVenatorFootprint target2_fail =
        osrs_venator_footprint(3, 0, 2);
    CHECK("size2 accept pass", osrs_venator_accepts_bounce(
        sender1, target2_accept_pass_send_fail));
    CHECK("size2 accept fail", !osrs_venator_accepts_bounce(
        sender1, target2_fail));
    CHECK("size2 send pass", osrs_venator_sends_bounce(
        sender1, target2_pass));
    CHECK("size2 send fail", !osrs_venator_sends_bounce(
        sender1, target2_accept_pass_send_fail));

    OsrsVenatorFootprint target3_pass =
        osrs_venator_footprint(1, 0, 3);
    OsrsVenatorFootprint target3_fail =
        osrs_venator_footprint(2, 0, 3);
    CHECK("size3 accept pass", osrs_venator_accepts_bounce(
        sender1, target3_pass));
    CHECK("size3 accept fail", !osrs_venator_accepts_bounce(
        sender1, target3_fail));
    CHECK("size3 send pass", osrs_venator_sends_bounce(
        sender2, osrs_venator_footprint(2, 0, 3)));
    CHECK("size3 send fail", !osrs_venator_sends_bounce(
        sender2, osrs_venator_footprint(3, 0, 3)));

    OsrsVenatorFootprint target4_accept_pass =
        osrs_venator_footprint(3, 0, 4);
    OsrsVenatorFootprint target4_accept_fail =
        osrs_venator_footprint(4, 0, 4);
    CHECK("size4 accept pass", osrs_venator_accepts_bounce(
        sender4, target4_accept_pass));
    CHECK("size4 accept fail", !osrs_venator_accepts_bounce(
        sender4, target4_accept_fail));
    CHECK("size4 send pass", osrs_venator_sends_bounce(
        sender2, osrs_venator_footprint(1, 0, 4)));
    CHECK("size4 send fail", !osrs_venator_sends_bounce(
        sender2, osrs_venator_footprint(2, 0, 4)));

    OsrsVenatorFootprint target5_pass =
        osrs_venator_footprint(1, 0, 5);
    OsrsVenatorFootprint target5_fail =
        osrs_venator_footprint(2, 0, 5);
    CHECK("size5 accept pass", osrs_venator_accepts_bounce(
        sender2, target5_pass));
    CHECK("size5 accept fail", !osrs_venator_accepts_bounce(
        sender2, target5_fail));
    CHECK("size5 send pass", osrs_venator_sends_bounce(
        sender2, target5_pass));
    CHECK("size5 send fail", !osrs_venator_sends_bounce(
        sender2, target5_fail));
}

static void test_boundaries(void) {
    OsrsVenatorMonster primary = live_monster(1, 0, 0, 1);

    OsrsVenatorChain no_candidate = osrs_venator_resolve_chain(
        primary, NULL, 0);
    check_chain_slots(
        "no candidate gives one hit",
        no_candidate,
        OSRS_VENATOR_CHAIN_LENGTH_ONE,
        1,
        -1,
        -1);

    OsrsVenatorMonster one_candidate[] = {
        live_monster(2, 2, 0, 1),
    };
    OsrsVenatorChain one = osrs_venator_resolve_chain(
        primary, one_candidate, 1);
    check_chain_slots(
        "one candidate gives two hits",
        one,
        OSRS_VENATOR_CHAIN_LENGTH_TWO,
        1,
        2,
        -1);

    OsrsVenatorMonster excluded_candidate[] = {
        live_monster(2, 3, 0, 1),
    };
    OsrsVenatorChain excluded = osrs_venator_resolve_chain(
        primary, excluded_candidate, 1);
    check_chain_slots(
        "cheb three excludes candidate",
        excluded,
        OSRS_VENATOR_CHAIN_LENGTH_ONE,
        1,
        -1,
        -1);

    CHECK("cheb two can bounce",
          osrs_venator_can_bounce(
              primary.footprint, one_candidate[0].footprint));
    CHECK("cheb three cannot bounce",
          !osrs_venator_can_bounce(
              primary.footprint, excluded_candidate[0].footprint));
}

static void test_damage_rolls_all_chain_nodes(void) {
    OsrsVenatorMonster primary = live_monster(10, 0, 0, 1);
    OsrsVenatorMonster candidates[] = {
        live_monster(20, 1, 0, 1),
        live_monster(30, 2, 0, 1),
    };
    OsrsVenatorChain chain = osrs_venator_resolve_chain(
        primary, candidates, 2);
    int defence_rolls[OSRS_VENATOR_MAX_CHAIN_HITS] = {100, 200, 300};
    uint32_t rng = 12345;
    uint32_t expected_rng = rng;

    xorshift32(&expected_rng);
    xorshift32(&expected_rng);
    xorshift32(&expected_rng);

    OsrsVenatorDamageResult damage = osrs_venator_roll_chain_damage(
        &chain, defence_rolls, 0, 31, &rng);

    CHECK("damage result keeps chain length", damage.length == chain.length);
    CHECK("all chain nodes roll accuracy even when all miss",
          damage.hits[0].roll_state == OSRS_VENATOR_DAMAGE_ROLLED &&
          damage.hits[1].roll_state == OSRS_VENATOR_DAMAGE_ROLLED &&
          damage.hits[2].roll_state == OSRS_VENATOR_DAMAGE_ROLLED);
    CHECK("zero attack roll misses all chain nodes",
          damage.hits[0].accuracy == OSRS_VENATOR_ACCURACY_MISS &&
          damage.hits[1].accuracy == OSRS_VENATOR_ACCURACY_MISS &&
          damage.hits[2].accuracy == OSRS_VENATOR_ACCURACY_MISS);
    CHECK("all misses do not roll damage", damage.total_damage == 0);
    CHECK("primary max hit is original", damage.hits[0].max_hit == 31);
    CHECK("bounce max hit is two-thirds of original", damage.hits[1].max_hit == 20);
    CHECK("third max hit is two-thirds of original", damage.hits[2].max_hit == 20);
    CHECK("independent misses consume one accuracy roll per chain node",
          rng == expected_rng);
}

int main(void) {
    test_size1_bounce_law();
    test_bounce_max_hit_law();
    test_chain_laws_and_determinism();
    test_warband_cluster_and_bounce_back();
    test_no_overreach_past_original_centre();
    test_rule_matrix_representatives();
    test_boundaries();
    test_damage_rolls_all_chain_nodes();
    printf("test_osrs_venator_bow_bounce: OK\n");
    return 0;
}
