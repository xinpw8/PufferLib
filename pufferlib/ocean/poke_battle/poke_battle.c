// poke_battle.c - Human vs AI terminal-based Pokemon battle for testing
// Compile: gcc -O2 -o poke_battle poke_battle.c -lm
// Run: ./poke_battle

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "poke_battle.h"

// ============================================================================
// Display Helpers
// ============================================================================

static const char* status_name(StatusCondition s) {
    switch (s) {
        case STATUS_NONE:      return "";
        case STATUS_SLEEP:     return " [SLP]";
        case STATUS_FREEZE:    return " [FRZ]";
        case STATUS_BURN:      return " [BRN]";
        case STATUS_POISON:    return " [PSN]";
        case STATUS_TOXIC:     return " [TOX]";
        case STATUS_PARALYSIS: return " [PAR]";
        default:               return "";
    }
}

static const char* type_name(Type t) {
    switch (t) {
        case TYPE_NORMAL:   return "Normal";
        case TYPE_FIRE:     return "Fire";
        case TYPE_WATER:    return "Water";
        case TYPE_ELECTRIC: return "Electric";
        case TYPE_GRASS:    return "Grass";
        case TYPE_ICE:      return "Ice";
        case TYPE_FIGHTING: return "Fighting";
        case TYPE_POISON:   return "Poison";
        case TYPE_GROUND:   return "Ground";
        case TYPE_FLYING:   return "Flying";
        case TYPE_PSYCHIC:  return "Psychic";
        case TYPE_BUG:      return "Bug";
        case TYPE_ROCK:     return "Rock";
        case TYPE_GHOST:    return "Ghost";
        case TYPE_DRAGON:   return "Dragon";
        default:            return "???";
    }
}

static void print_hp_bar(int hp, int max_hp, int width) {
    int filled = (max_hp > 0) ? (hp * width / max_hp) : 0;
    printf("[");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else printf(" ");
    }
    printf("] %d/%d", hp, max_hp);
}

static void print_pokemon_summary(Pokemon* p, int idx, int is_active) {
    if (p->species == SPECIES_NONE) return;
    const SpeciesData* sd = &SPECIES_DATA[p->species];

    printf("  %s%d. %-12s ", is_active ? ">" : " ", idx, sd->name);
    if (p->is_alive) {
        print_hp_bar(p->hp, p->max_hp, 20);
        printf("%s", status_name(p->status));
    } else {
        printf("  (fainted)");
    }
    printf("\n");
}

static void print_battle_state(Battle* battle, int human_player) {
    Player* human = &battle->players[human_player];
    Player* ai = &battle->players[1 - human_player];

    printf("\n");
    printf("========================================\n");
    printf("  TURN %d\n", battle->turn);
    printf("========================================\n");

    // Show opponent's team
    printf("\n  OPPONENT'S TEAM:\n");
    for (int i = 0; i < NUM_POKEMON; i++) {
        Pokemon* p = &ai->team[i];
        if (p->species == SPECIES_NONE) continue;
        const SpeciesData* sd = &SPECIES_DATA[p->species];
        printf("  %s%d. %-12s ", (i == ai->active_idx) ? ">" : " ", i, sd->name);
        if (p->is_alive) {
            print_hp_bar(p->hp, p->max_hp, 20);
            printf("%s", status_name(p->status));
        } else {
            printf("  (fainted)");
        }
        printf("\n");
    }

    // Show opponent's active in detail
    Pokemon* opp_active = active_pokemon(ai);
    const SpeciesData* osd = &SPECIES_DATA[opp_active->species];
    printf("\n  Opponent Active: %s (%s", osd->name, type_name(opp_active->type1));
    if (opp_active->type2 != TYPE_NONE) printf("/%s", type_name(opp_active->type2));
    printf(")\n");
    printf("  ");
    print_hp_bar(opp_active->hp, opp_active->max_hp, 30);
    printf("%s\n", status_name(opp_active->status));

    // Show stat stages if non-zero
    if (ai->atk_stage || ai->def_stage || ai->spc_stage || ai->spe_stage) {
        printf("  Stages:");
        if (ai->atk_stage) printf(" ATK%+d", ai->atk_stage);
        if (ai->def_stage) printf(" DEF%+d", ai->def_stage);
        if (ai->spc_stage) printf(" SPC%+d", ai->spc_stage);
        if (ai->spe_stage) printf(" SPE%+d", ai->spe_stage);
        printf("\n");
    }
    if (ai->is_confused) printf("  (confused)\n");
    if (ai->substitute_hp > 0) printf("  (behind substitute: %d HP)\n", ai->substitute_hp);

    printf("\n  ----------------------------------------\n");

    // Show player's active in detail
    Pokemon* my_active = active_pokemon(human);
    const SpeciesData* msd = &SPECIES_DATA[my_active->species];
    printf("\n  Your Active: %s (%s", msd->name, type_name(my_active->type1));
    if (my_active->type2 != TYPE_NONE) printf("/%s", type_name(my_active->type2));
    printf(")\n");
    printf("  ");
    print_hp_bar(my_active->hp, my_active->max_hp, 30);
    printf("%s\n", status_name(my_active->status));

    if (human->atk_stage || human->def_stage || human->spc_stage || human->spe_stage) {
        printf("  Stages:");
        if (human->atk_stage) printf(" ATK%+d", human->atk_stage);
        if (human->def_stage) printf(" DEF%+d", human->def_stage);
        if (human->spc_stage) printf(" SPC%+d", human->spc_stage);
        if (human->spe_stage) printf(" SPE%+d", human->spe_stage);
        printf("\n");
    }
    if (human->is_confused) printf("  (confused)\n");
    if (human->substitute_hp > 0) printf("  (behind substitute: %d HP)\n", human->substitute_hp);
    if (human->is_recharging) printf("  (must recharge!)\n");

    // Show player's team
    printf("\n  YOUR TEAM:\n");
    for (int i = 0; i < NUM_POKEMON; i++) {
        print_pokemon_summary(&human->team[i], i, i == human->active_idx);
    }
    printf("\n");
}

static void print_actions(Player* human, int mode, int player_idx) {
    int mask[NUM_ACTIONS];
    get_action_mask(human, mode, player_idx, mask);

    Pokemon* active = active_pokemon(human);

    if (mode != 0) {
        printf("  Your Pokemon fainted! Choose a replacement:\n");
    } else if (human->is_recharging) {
        printf("  Recharging... Press 0 to continue.\n");
        return;
    } else {
        printf("  MOVES:\n");
        for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
            if (active->moves[i].id != MOVE_NONE) {
                const MoveData* md = &MOVE_DATA[active->moves[i].id];
                printf("    %d. %-14s  %-8s  Pow:%-3d  Acc:%-3d  PP:%d/%d %s\n",
                       i, md->name, type_name(md->type),
                       md->power, md->accuracy,
                       active->moves[i].pp, active->moves[i].max_pp,
                       mask[i] ? "" : "(no PP)");
            }
        }
    }

    printf("  SWITCH:\n");
    for (int i = 0; i < NUM_POKEMON; i++) {
        Pokemon* p = &human->team[i];
        if (p->species == SPECIES_NONE) continue;
        const SpeciesData* sd = &SPECIES_DATA[p->species];
        printf("    %d. %-12s  %s  %s\n",
               i + 4, sd->name,
               p->is_alive ? "" : "(fainted)",
               (i == human->active_idx) ? "(active)" : "");
    }

    printf("\n  Valid actions: ");
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (mask[i]) printf("%d ", i);
    }
    printf("\n");
}

// ============================================================================
// Performance Test
// ============================================================================

static void performance_test(int seconds) {
    PokeBattle env;
    memset(&env, 0, sizeof(PokeBattle));

    float obs[OBS_SIZE] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};
    unsigned char terminals[1] = {0};

    env.observations = obs;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.seed = 42;

    init(&env);
    c_reset(&env);

    time_t start = time(NULL);
    long steps = 0;

    while (time(NULL) - start < seconds) {
        env.actions[0] = pb_rand_int(NUM_ACTIONS);
        c_step(&env);
        steps++;
    }

    time_t elapsed = time(NULL) - start;
    printf("Performance: %ld steps in %ld seconds = %ld SPS\n",
           steps, elapsed, steps / elapsed);
}

// ============================================================================
// Interactive Game
// ============================================================================

static void interactive() {
    PokeBattle env;
    memset(&env, 0, sizeof(PokeBattle));

    float obs[OBS_SIZE] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};
    unsigned char terminals[1] = {0};

    env.observations = obs;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.seed = (unsigned long long)time(NULL);

    init(&env);
    c_reset(&env);

    int human_player = 0;
    int wins = 0, losses = 0, draws = 0;

    printf("\n  ============================\n");
    printf("  Gen 1 OU Pokemon Battle!\n");
    printf("  ============================\n");
    printf("  Actions 0-3: Use move\n");
    printf("  Actions 4-9: Switch to Pokemon\n");
    printf("  Type 'q' to quit\n\n");

    while (1) {
        print_battle_state(&env.battle, human_player);

        int mode = env.battle.mode;
        print_actions(&env.battle.players[human_player], mode, human_player);

        printf("\n  Your choice: ");
        fflush(stdout);

        char input[16];
        if (!fgets(input, sizeof(input), stdin)) break;

        // Check for quit
        if (input[0] == 'q' || input[0] == 'Q') break;

        int choice = atoi(input);

        // Validate
        int mask[NUM_ACTIONS];
        get_action_mask(&env.battle.players[human_player], mode, human_player, mask);
        if (choice < 0 || choice >= NUM_ACTIONS || !mask[choice]) {
            printf("  Invalid choice! Try again.\n");
            continue;
        }

        env.actions[0] = choice;
        c_step(&env);

        // Check for game end
        if (env.terminals[0]) {
            if (env.rewards[0] > 0) {
                printf("\n  *** YOU WIN! ***\n");
                wins++;
            } else if (env.rewards[0] < 0) {
                printf("\n  *** YOU LOSE! ***\n");
                losses++;
            } else {
                printf("\n  *** DRAW (timeout) ***\n");
                draws++;
            }

            printf("  Record: %d W / %d L / %d D\n", wins, losses, draws);
            printf("\n  Press Enter for next battle, 'q' to quit: ");
            fflush(stdout);

            char buf[16];
            if (!fgets(buf, sizeof(buf), stdin)) break;
            if (buf[0] == 'q' || buf[0] == 'Q') break;

            // Reset already happened in c_step
        }
    }

    printf("\n  Final Record: %d W / %d L / %d D\n", wins, losses, draws);
    printf("  Thanks for playing!\n\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        int seconds = 10;
        if (argc > 2) seconds = atoi(argv[2]);
        performance_test(seconds);
    } else {
        interactive();
    }
    return 0;
}
