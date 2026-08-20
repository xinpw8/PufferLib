#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ocean/osrs/encounters/encounter_nh_pvp.h"


#include "ocean/osrs/tests/osrs_test_check.h"

static void assert_child_aborts(const char* label, void (*fn)(void)) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        fn();
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    tests_run++;
    if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("  FAIL: %s - child returned successfully\n", label);
    }
}

static void queue_test_hit(Player* attacker, Player* defender, int damage) {
    queue_hit(123, 0, 1, attacker, defender, damage, ATTACK_STYLE_MAGIC,
        4, 0, damage > 0, 0, 0, 0, 0, 0);
}

static void test_pvp_queue_accepts_capacity(void) {
    printf("--- pvp queue accepts capacity ---\n");

    Player attacker = {0};
    Player defender = {0};
    defender.prayer = PRAYER_NONE;

    for (int i = 0; i < MAX_PENDING_HITS; i++)
        queue_test_hit(&attacker, &defender, i + 1);

    ASSERT_INT_EQ("queue reaches capacity", attacker.num_pending_hits, MAX_PENDING_HITS);
    ASSERT_INT_EQ("first damage kept", attacker.pending_hits[0].damage, 1);
    ASSERT_INT_EQ("last damage kept", attacker.pending_hits[MAX_PENDING_HITS - 1].damage,
        MAX_PENDING_HITS);
}

static void child_pvp_queue_overflow(void) {
    Player attacker = {0};
    Player defender = {0};
    defender.prayer = PRAYER_NONE;

    for (int i = 0; i < MAX_PENDING_HITS; i++)
        queue_test_hit(&attacker, &defender, i + 1);
    queue_test_hit(&attacker, &defender, 99);
}

static void test_pvp_queue_overflow_aborts(void) {
    printf("--- pvp queue overflow aborts ---\n");

    assert_child_aborts("pvp pending-hit overflow aborts", child_pvp_queue_overflow);
}

static void test_pvp_remove_compacts_and_clears_tail(void) {
    printf("--- pvp remove compacts and clears tail ---\n");

    Player attacker = {0};
    Player defender = {0};
    defender.prayer = PRAYER_NONE;
    queue_test_hit(&attacker, &defender, 3);
    queue_test_hit(&attacker, &defender, 7);
    queue_test_hit(&attacker, &defender, 11);

    pvp_remove_pending_hit(&attacker, 1);

    ASSERT_INT_EQ("queue count after remove", attacker.num_pending_hits, 2);
    ASSERT_INT_EQ("first hit remains", attacker.pending_hits[0].damage, 3);
    ASSERT_INT_EQ("third hit compacted", attacker.pending_hits[1].damage, 11);
    ASSERT_INT_EQ("tail damage cleared", attacker.pending_hits[2].damage, 0);
    ASSERT_INT_EQ("tail timer cleared", attacker.pending_hits[2].ticks_until_hit, 0);
}

static CollisionMap* nh_map;
static NhPvpContext nh_context;
static NhPvpState* nh_state;

static void init_nh_fixture(void) {
    EncounterState* state = ENCOUNTER_NH_PVP.create();
    if (!state) abort();
    nh_state = (NhPvpState*)state;
    ENCOUNTER_NH_PVP.init_context((EncounterContext*)&nh_context);
    nh_map = collision_map_load("ocean/osrs/data/wilderness.cmap");
    if (!nh_map) abort();
    ENCOUNTER_NH_PVP.put_ptr(
        state, (EncounterContext*)&nh_context, "collision_map", nh_map);
    ENCOUNTER_NH_PVP.put_int(
        state, (EncounterContext*)&nh_context, "seed", 1);
    ENCOUNTER_NH_PVP.finalize_context(
        state, (EncounterContext*)&nh_context);
    ENCOUNTER_NH_PVP.reset(
        state, (EncounterContext*)&nh_context, 1);
    nh_state->env.auto_reset = 0;
}

static void test_nh_topology_geometry_parity(void) {
    printf("--- NH PvP topology geometry parity ---\n");
    const EncounterArenaTopology* topology = nh_context.route_topology;
    CHECK("NH PvP topology covers the 61 by 28 fight arena",
        topology->origin_x == FIGHT_AREA_BASE_X &&
        topology->origin_y == FIGHT_AREA_BASE_Y &&
        topology->width == 61 && topology->height == 28);
    CHECK("NH PvP keeps the existing open static LOS combat rule",
        topology->static_los_mode == ENCOUNTER_ARENA_TOPOLOGY_LOS_OPEN);

    int diagonal_wall_steps = 0;
    for (int x = FIGHT_AREA_BASE_X;
            x < FIGHT_AREA_BASE_X + FIGHT_AREA_WIDTH;
            x++) {
        for (int y = FIGHT_AREA_BASE_Y;
                y < FIGHT_AREA_BASE_Y + FIGHT_AREA_HEIGHT;
                y++) {
            int expected_walkable =
                is_in_wilderness(x, y) &&
                collision_tile_walkable(nh_map, 0, x, y);
            CHECK("NH PvP static player tile parity",
                pvp_topology_tile_walkable(topology, x, y) ==
                    expected_walkable);
            if (!expected_walkable) continue;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int destination_x = x + dx;
                    int destination_y = y + dy;
                    int expected_step =
                        encounter_arena_topology_contains(
                            topology, destination_x, destination_y) &&
                        collision_traversable_step(
                            nh_map, 0, x, y, dx, dy);
                    int actual_step =
                        encounter_arena_topology_step_allowed(
                            topology, x, y, 1, dx, dy);
                    CHECK("NH PvP cardinal and diagonal wall parity",
                        actual_step == expected_step);
                    if (dx != 0 && dy != 0 &&
                            !expected_step &&
                            pvp_topology_tile_walkable(
                                topology, x + dx, y + dy))
                        diagonal_wall_steps++;
                }
            }
        }
    }
    CHECK("NH PvP collision map contains diagonal wall-side blocking",
        diagonal_wall_steps > 0);
}

static void test_nh_local_move_routes_match_canonical_solver(void) {
    printf("--- NH PvP local move route cache parity ---\n");
    const EncounterArenaTopology* topology = nh_context.route_topology;
    int matches = 1;
    for (int x = topology->origin_x;
            x < topology->origin_x + topology->width && matches;
            x++) {
        for (int y = topology->origin_y;
                y < topology->origin_y + topology->height && matches;
                y++) {
            for (int action = 1; action < OSRS_PRIMARY_MOVE_ACTIONS; action++) {
                EncounterRouteInput input = {
                    .topology = topology,
                    .source_x = x,
                    .source_y = y,
                    .actor_size = 1,
                    .target_x = x + ENCOUNTER_MOVE_TARGET_DX[action],
                    .target_y = y + ENCOUNTER_MOVE_TARGET_DY[action],
                    .target_size = 1,
                    .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
                    .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
                    .cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS,
                };
                EncounterRouteResult expected = encounter_route_solve(&input);
                EncounterRouteResult actual;
                if (!pvp_local_move_route_lookup(
                        &pvp_route_topology_owner, &input, &actual) ||
                        actual.outcome != expected.outcome ||
                        actual.destination_x != expected.destination_x ||
                        actual.destination_y != expected.destination_y ||
                        actual.first_dx != expected.first_dx ||
                        actual.first_dy != expected.first_dy ||
                        actual.run_dx != expected.run_dx ||
                        actual.run_dy != expected.run_dy ||
                        actual.distance != expected.distance) {
                    printf("  mismatch source=(%d,%d) action=%d\n", x, y, action);
                    matches = 0;
                    break;
                }
            }
        }
    }
    CHECK("cached local routes equal canonical south-first BFS", matches);
}

static void test_nh_dynamic_player_occupancy(void) {
    printf("--- NH PvP player occupancy stays dynamic ---\n");
    const EncounterArenaTopology* topology = nh_context.route_topology;
    Player mover = {0};
    Player blocker = {0};
    mover.x = blocker.x = FIGHT_AREA_BASE_X + 8;
    mover.y = blocker.y = FIGHT_AREA_BASE_Y + 8;
    CHECK("opponent tile is not baked into static topology",
        pvp_topology_tile_walkable(topology, blocker.x, blocker.y));
    resolve_same_tile(&mover, &blocker, topology);
    ASSERT_INT_EQ("same-tile resolution keeps west-first priority",
        mover.x, blocker.x - 1);
    ASSERT_INT_EQ("same-tile resolution keeps the y coordinate",
        mover.y, blocker.y);
    CHECK("static topology remains unchanged after occupancy resolution",
        pvp_topology_tile_walkable(topology, blocker.x, blocker.y));
}

static void run_nh_chase_case(
    PvpEquipmentPlan equipment_plan,
    OsrsSpellAction spell,
    int expected_x,
    int expected_y,
    const char* x_label,
    const char* y_label
) {
    ENCOUNTER_NH_PVP.reset(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        1);
    nh_state->env.auto_reset = 0;
    Player* player = &nh_state->env.players[0];
    Player* target = &nh_state->env.players[1];
    pvp_apply_equipment_plan(player, equipment_plan);
    player->x = FIGHT_AREA_BASE_X;
    player->y = FIGHT_AREA_BASE_Y;
    player->dest_x = player->x;
    player->dest_y = player->y;
    target->x = FIGHT_AREA_BASE_X + 10;
    target->y = FIGHT_AREA_BASE_Y + 4;
    target->dest_x = target->x;
    target->dest_y = target->y;
    int actions[OSRS_BASE_NUM_ACTION_HEADS] = {0};
    actions[OSRS_HEAD_PRIMARY] = OSRS_PRIMARY_MOVE_ACTIONS;
    actions[OSRS_HEAD_SPELL] = spell;
    ENCOUNTER_NH_PVP.step(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        actions);
    ASSERT_INT_EQ(x_label, player->x, expected_x);
    ASSERT_INT_EQ(y_label, player->y, expected_y);
    ASSERT_INT_EQ("attack click cancels queued walk x",
        nh_state->env.pvp_runtime.walk_dest_x[0], -1);
    ASSERT_INT_EQ("attack click cancels queued walk y",
        nh_state->env.pvp_runtime.walk_dest_y[0], -1);
}

static void test_nh_attack_chase_destinations(void) {
    printf("--- NH PvP attack chase destinations ---\n");
    run_nh_chase_case(
        PVP_EQUIPMENT_MELEE, OSRS_SPELL_NONE,
        FIGHT_AREA_BASE_X + 2, FIGHT_AREA_BASE_Y + 2,
        "melee chase keeps x destination",
        "melee chase keeps y destination");
    run_nh_chase_case(
        PVP_EQUIPMENT_RANGED, OSRS_SPELL_NONE,
        FIGHT_AREA_BASE_X + 2, FIGHT_AREA_BASE_Y,
        "ranged chase keeps x destination",
        "ranged chase keeps y destination");
    run_nh_chase_case(
        PVP_EQUIPMENT_MAGIC, OSRS_SPELL_ICE_BARRAGE,
        FIGHT_AREA_BASE_X, FIGHT_AREA_BASE_Y,
        "in-range magic attack does not chase on x",
        "in-range magic attack does not chase on y");
}

static void test_nh_out_of_arena_destination_fallback(void) {
    printf("--- NH PvP out-of-arena destination fallback ---\n");
    int destination_x = FIGHT_AREA_BASE_X - 1;
    int destination_y = FIGHT_AREA_BASE_Y + 1;
    EncounterRouteInput route_input = {
        .topology = nh_context.route_topology,
        .source_x = FIGHT_AREA_BASE_X + 1,
        .source_y = FIGHT_AREA_BASE_Y + 1,
        .actor_size = 1,
        .target_x = destination_x,
        .target_y = destination_y,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
        .cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS,
    };
    EncounterRouteResult route = encounter_route_solve(&route_input);
    ASSERT_INT_EQ("out-of-arena route reaches the arena edge",
        route.outcome, ROUTE_REACHED_FALLBACK);
    ASSERT_INT_EQ("out-of-arena route clamps x to the arena edge",
        route.destination_x, FIGHT_AREA_BASE_X);
    ASSERT_INT_EQ("out-of-arena route keeps the selected y",
        route.destination_y, FIGHT_AREA_BASE_Y + 1);
    ASSERT_INT_EQ("out-of-arena route steps toward the edge on x",
        route.first_dx, -1);
    ASSERT_INT_EQ("out-of-arena route keeps y on the first step",
        route.first_dy, 0);

    route_input.source_x = FIGHT_AREA_BASE_X + 19;
    route_input.source_y = FIGHT_AREA_BASE_Y;
    route_input.target_x = route_input.source_x;
    route_input.target_y = FIGHT_AREA_BASE_Y - 1;
    route = encounter_route_solve(&route_input);
    ASSERT_INT_EQ("boundary source reaches the clamped destination",
        route.outcome, ROUTE_REACHED_FALLBACK);
    ASSERT_INT_EQ("boundary source keeps x",
        route.destination_x, route_input.source_x);
    ASSERT_INT_EQ("boundary source does not route across the arena",
        route.destination_y, route_input.source_y);
    ASSERT_INT_EQ("boundary source has zero route distance",
        route.distance, 0);
}

static void test_nh_shared_contract_and_inventory_actions(void) {
    printf("--- NH PvP shared contract and canonical inventory ---\n");
    CHECK("NH PvP exposes 18 shared heads",
        ENCOUNTER_NH_PVP.num_action_heads == OSRS_BASE_NUM_ACTION_HEADS);
    CHECK("NH PvP exposes shared action dimensions",
        ENCOUNTER_NH_PVP.action_head_dims[OSRS_HEAD_PRIMARY] ==
            OSRS_PRIMARY_DIM(1) &&
        ENCOUNTER_NH_PVP.action_head_dims[OSRS_HEAD_DRINK] ==
            OSRS_INVENTORY_CLICK_DIM);
    CHECK("NH PvP observation starts with the exact shared boundary",
        NH_PVP_NUM_OBS == OSRS_SHARED_OBS_SIZE + NH_PVP_SPECIFIC_OBS_SIZE);

    ENCOUNTER_NH_PVP.reset(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        1);
    Player* player = &nh_state->env.players[0];
    int nonempty = 0;
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++)
        nonempty += !osrs_inventory_cell_is_empty(
            &player->inventory_cells[cell]);
    CHECK("PvP reset seeds canonical inventory cells", nonempty > 0);
    ASSERT_INT_EQ("PvP reset reserves one inventory cell for two-handed swaps",
        nonempty, OSRS_INVENTORY_SIZE - 1);

    float observations[NH_PVP_NUM_OBS];
    ENCOUNTER_NH_PVP.write_obs(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        observations);
    CHECK("shared prefix starts with player hitpoints",
        observations[0] ==
            (float)player->current_hitpoints / player->base_hitpoints);
    CHECK("shared prefix exposes canonical inventory code",
        observations[OSRS_SHARED_OBS_INVENTORY_START] ==
            osrs_inventory_cell_obs_code_encode(
                player->inventory_cells[0].content_code));
    CHECK("shared prefix exposes canonical worn weapon code",
        observations[
            OSRS_SHARED_OBS_EQUIPPED_START + GEAR_SLOT_WEAPON] ==
            osrs_inventory_cell_obs_code_encode(
                osrs_inventory_content_code_from_item(
                    player->equipped[GEAR_SLOT_WEAPON])));

    int ranged_cell =
        pvp_inventory_cell_with_item(player, ITEM_RUNE_CROSSBOW);
    CHECK("reset exposes a ranged weapon cell", ranged_cell >= 0);
    uint8_t previous_weapon = player->equipped[GEAR_SLOT_WEAPON];
    int equip[OSRS_BASE_NUM_ACTION_HEADS] = {0};
    equip[OSRS_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] = ranged_cell + 1;
    ENCOUNTER_NH_PVP.step(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        equip);
    CHECK("weapon cell click equips the selected item",
        player->equipped[GEAR_SLOT_WEAPON] == ITEM_RUNE_CROSSBOW);
    CHECK("weapon cell receives displaced equipment",
        osrs_inventory_cell_item_index(
            &player->inventory_cells[ranged_cell]) == previous_weapon);

    int food_cell = human_pvp_find_consumable_cell(
        player, OSRS_CONSUMABLE_SHARK_FOOD);
    CHECK("reset exposes a food cell", food_cell >= 0);
    player->current_hitpoints = 50;
    int eat[OSRS_BASE_NUM_ACTION_HEADS] = {0};
    eat[OSRS_HEAD_EAT] = food_cell + 1;
    ENCOUNTER_NH_PVP.step(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        eat);
    CHECK("food cell click heals the player",
        player->current_hitpoints > 50);
    CHECK("food cell click consumes the canonical cell",
        osrs_inventory_cell_is_empty(&player->inventory_cells[food_cell]));
}

static void test_nh_masked_drink_is_not_executed(void) {
    printf("--- NH PvP masked drinks do not execute ---\n");
    ENCOUNTER_NH_PVP.reset(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        2);
    Player* player = &nh_state->env.players[0];
    int brew_cell = human_pvp_find_consumable_cell(
        player, OSRS_CONSUMABLE_BREW);
    CHECK("reset exposes a brew cell", brew_cell >= 0);
    player->current_hitpoints =
        player->base_hitpoints + osrs_brew_heal_amount(player->base_hitpoints);
    player->current_defence = player->base_defence;
    CHECK("full boosted player cannot use another brew",
        !pvp_drink_kind_available(player, OSRS_CONSUMABLE_BREW));

    int brew_doses_before = player->brew_doses;
    int actions[OSRS_BASE_NUM_ACTION_HEADS] = {0};
    actions[OSRS_HEAD_DRINK] = brew_cell + 1;
    execute_switches(
        &nh_state->env, 0, actions, nh_context.route_topology);

    ASSERT_INT_EQ("masked brew keeps its dose count",
        player->brew_doses, brew_doses_before);
    ASSERT_INT_EQ("masked brew does not start potion cooldown",
        player->potion_timer, 0);
}

static void test_nh_reset_keeps_two_handed_switch_usable(void) {
    printf("--- NH PvP reset keeps two-handed switches usable ---\n");
    ENCOUNTER_NH_PVP.reset(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        3);
    Player* player = &nh_state->env.players[0];
    int spec_cell = pvp_inventory_cell_with_item(
        player, ITEM_DRAGON_DAGGER);
    CHECK("reset exposes a spec weapon cell", spec_cell >= 0);
    player->inventory_cells[spec_cell] =
        osrs_inventory_cell_from_item(ITEM_AGS);

    CHECK("two-handed switch is legal from the reset inventory",
        osrs_can_equip_from_cell(
            player, player->inventory_cells, spec_cell));
}

static void test_nh_ko_supply_bonus_is_normalized_to_initial_pool(void) {
    printf("--- NH PvP KO supply bonus uses the initial pool ---\n");
    ENCOUNTER_NH_PVP.reset(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        4);
    nh_state->env.shaping = (RewardShapingConfig){
        .enabled = 1,
        .ko_supplies_bonus_coef = 1.0f,
    };
    nh_state->env.episode_over = 1;
    nh_state->env.winner = 0;

    ASSERT_FLOAT_NEAR("untouched opponent grants one normalized supply bonus",
        calculate_reward(&nh_state->env, 0), 2.0f, 0.0f);
}

static void test_nh_action_mask_overwrites_dirty_buffer(void) {
    ENCOUNTER_NH_PVP.reset(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        1);
    float clean[NH_PVP_ACTION_MASK_SIZE] = {0};
    float dirty[NH_PVP_ACTION_MASK_SIZE];
    for (int i = 0; i < NH_PVP_ACTION_MASK_SIZE; i++) dirty[i] = 7.0f;
    ENCOUNTER_NH_PVP.write_mask(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        clean);
    ENCOUNTER_NH_PVP.write_mask(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        dirty);
    CHECK("NH PvP mask overwrites every output",
        memcmp(clean, dirty, sizeof(clean)) == 0);

    unsigned char expected[NH_PVP_ACTION_MASK_SIZE];
    unsigned char actual[NH_PVP_ACTION_MASK_SIZE];
    for (int i = 0; i < NH_PVP_ACTION_MASK_SIZE; i++)
        expected[i] = clean[i] != 0.0f;
    pvp_write_action_mask_bytes(
        actual, &nh_state->env, 0, nh_context.route_topology);
    CHECK("NH PvP byte mask matches the float contract",
        memcmp(expected, actual, sizeof(expected)) == 0);
}

static void test_nh_scripted_policy_action_sequences(void) {
    printf("--- NH PvP scripted policies emit canonical action sequences ---\n");
    for (int type = OPP_TRUE_RANDOM; type < OPP_SELFPLAY; type++) {
        nh_state->env.pvp_runtime.opponent.type = (OpponentType)type;
        nh_state->env.pvp_runtime.use_c_opponent = 1;
        nh_state->env.pvp_runtime.use_external_opponent_actions = 0;
        ENCOUNTER_NH_PVP.reset(
            (EncounterState*)nh_state,
            (EncounterContext*)&nh_context,
            (uint32_t)(1000 + type));
        nh_state->env.auto_reset = 0;

        int bounded = 1;
        int actions[OSRS_BASE_NUM_ACTION_HEADS] = {0};
        for (int tick = 0; tick < 64 && !nh_state->env.episode_over; tick++) {
            ENCOUNTER_NH_PVP.step(
                (EncounterState*)nh_state,
                (EncounterContext*)&nh_context,
                actions);
            const int* opponent_actions =
                nh_state->env.last_executed_actions +
                OSRS_BASE_NUM_ACTION_HEADS;
            for (int head = 0; head < OSRS_BASE_NUM_ACTION_HEADS; head++) {
                if (opponent_actions[head] < 0 ||
                        opponent_actions[head] >= NH_PVP_ACTION_DIMS[head]) {
                    bounded = 0;
                }
            }
        }

        char label[80];
        snprintf(label, sizeof(label),
            "scripted opponent %d stays within canonical action dimensions",
            type);
        CHECK(label, bounded);
    }
}

int main(void) {
    init_nh_fixture();
    test_nh_topology_geometry_parity();
    test_nh_local_move_routes_match_canonical_solver();
    test_nh_dynamic_player_occupancy();
    test_nh_attack_chase_destinations();
    test_nh_out_of_arena_destination_fallback();
    test_nh_shared_contract_and_inventory_actions();
    test_nh_masked_drink_is_not_executed();
    test_nh_reset_keeps_two_handed_switch_usable();
    test_nh_ko_supply_bonus_is_normalized_to_initial_pool();
    test_nh_scripted_policy_action_sequences();
    test_nh_action_mask_overwrites_dirty_buffer();

    test_pvp_queue_accepts_capacity();
    test_pvp_queue_overflow_aborts();
    test_pvp_remove_compacts_and_clears_tail();

    return osrs_test_summary();
}
