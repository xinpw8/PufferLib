#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_inferno.h"

#define FNV_OFFSET 1469598103934665603ULL
#define FNV_PRIME  1099511628211ULL

static inline uint64_t fnv_bytes(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= FNV_PRIME;
    }
    return h;
}

static inline uint64_t fnv_f32(uint64_t h, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return fnv_bytes(h, &bits, sizeof(bits));
}

static inline uint64_t fnv_i32(uint64_t h, int v) {
    int32_t w = (int32_t)v;
    return fnv_bytes(h, &w, sizeof(w));
}
static inline uint64_t fnv_u8(uint64_t h, uint8_t v) {
    return fnv_bytes(h, &v, sizeof(v));
}

static inline uint64_t fnv_u16(uint64_t h, uint16_t v) {
    return fnv_bytes(h, &v, sizeof(v));
}


static inline uint64_t splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

#define TRACE_NPC_SLOTS 32

typedef struct {
    uint64_t state;
    uint64_t reward;
    uint64_t mask;
    uint64_t terminal;
    uint64_t observation;
} TraceHashes;

static uint64_t hash_core_state(uint64_t h, const InfernoState* s) {
    h = fnv_bytes(h, &s->player, offsetof(Player, interaction));
    h = fnv_i32(h, s->player.interaction.target_slot);
    h = fnv_bytes(
        h,
        &s->player.item_effect_state,
        sizeof(s->player) - offsetof(Player, item_effect_state));
    h = fnv_bytes(h, s->pillars, sizeof(s->pillars));
    h = fnv_bytes(h, s->npcs, TRACE_NPC_SLOTS * sizeof(s->npcs[0]));
    h = fnv_bytes(h, &s->player_pending_hits, sizeof(s->player_pending_hits));
    h = fnv_bytes(h, s->pending_sparks, sizeof(s->pending_sparks));
    for (int cell_idx = 0; cell_idx < OSRS_INVENTORY_SIZE; cell_idx++) {
        const OsrsInventoryCell* cell = &s->player.inventory_cells[cell_idx];
        h = fnv_u8(h, osrs_inventory_cell_item_index(cell));
        h = fnv_u16(h, osrs_inventory_cell_raw_osrs_id(cell));
        h = fnv_u8(h, osrs_inventory_cell_dose_count(cell));
    }
    h = fnv_bytes(h, s->dead_mobs, sizeof(s->dead_mobs));
    h = fnv_i32(h, s->wave);
    return fnv_i32(h, s->tick);
}

static uint64_t hash_semantic_mask(
    uint64_t h,
    const InfernoState* s,
    const float* mask
) {
    int offset = 0;
    for (int head = 0; head < INF_NUM_ACTION_HEADS; head++) {
        if (head != INF_HEAD_PRIMARY) {
            for (int action = 0; action < INF_ACTION_DIMS[head]; action++)
                h = fnv_f32(h, mask[offset + action]);
            offset += INF_ACTION_DIMS[head];
            continue;
        }

        for (int action = 0; action < OSRS_PRIMARY_MOVE_ACTIONS; action++)
            h = fnv_f32(h, mask[offset + action]);
        for (int npc_idx = 0; npc_idx < TRACE_NPC_SLOTS; npc_idx++) {
            int slot = inf_find_target_obs_slot(s, npc_idx);
            h = fnv_f32(h, slot >= 0
                ? mask[offset + inf_primary_attack_action_for_obs_slot(slot)]
                : 0.0f);
        }
        offset += INF_ACTION_DIMS[head];
    }
    return h;
}

static TraceHashes run_episode(int start_wave, uint32_t seed, int max_ticks) {
    InfernoContext context;
    InfernoState state_storage;
    inf_init_context_typed(&context);
    inf_init_state_typed(&state_storage, &context);
    inf_put_int_ctx(
        (EncounterState*)&state_storage,
        (EncounterContext*)&context,
        "start_wave",
        start_wave);
    inf_finalize_route_topology(&context);
    inf_reset_ctx(
        (EncounterState*)&state_storage,
        (EncounterContext*)&context,
        seed);

    InfernoState* s = &state_storage;
    static float obs[INF_NUM_OBS];
    static float mask[INF_ACTION_MASK_SIZE];
    int actions[INF_NUM_ACTION_HEADS];
    uint64_t arng = ((uint64_t)seed << 20) ^ (uint64_t)(start_wave + 1) ^
        0xD1B54A32D192ED03ULL;
    TraceHashes hashes = {
        .state = FNV_OFFSET,
        .reward = FNV_OFFSET,
        .mask = FNV_OFFSET,
        .terminal = FNV_OFFSET,
        .observation = FNV_OFFSET,
    };

    for (int t = 0; t < max_ticks; t++) {
        inf_refresh_current_obs_slots_ctx(s, &context);
        for (int head = 0; head < INF_NUM_ACTION_HEADS; head++) {
            uint64_t random_value = splitmix64(&arng);
            actions[head] =
                (int)(random_value % (uint64_t)INF_ACTION_DIMS[head]);
        }

        inf_step_ctx(
            (EncounterState*)s, (EncounterContext*)&context, actions);
        inf_write_obs_ctx(
            (EncounterState*)s, (EncounterContext*)&context, obs);
        inf_write_mask_ctx(
            (EncounterState*)s, (EncounterContext*)&context, mask);

        for (int npc_idx = TRACE_NPC_SLOTS; npc_idx < INF_MAX_NPCS; npc_idx++) {
            if (s->npcs[npc_idx].active) {
                fprintf(stderr, "golden trace activated NPC slot %d\n", npc_idx);
                abort();
            }
        }
        hashes.state = hash_core_state(hashes.state, s);
        hashes.reward = fnv_f32(hashes.reward, s->reward);
        hashes.mask = hash_semantic_mask(hashes.mask, s, mask);
        hashes.terminal = fnv_i32(hashes.terminal, s->episode_over);
        hashes.terminal = fnv_i32(hashes.terminal, s->winner);
        hashes.terminal = fnv_i32(hashes.terminal, s->wave);
        for (int i = 0; i < INF_NUM_OBS; i++)
            hashes.observation = fnv_f32(hashes.observation, obs[i]);

        if (s->episode_over) break;
    }

    inf_destroy_context((EncounterContext*)&context);
    return hashes;
}

typedef struct {
    const char* name;
    int start_wave;
    uint32_t seed;
} GoldenConfig;

static const GoldenConfig CONFIGS[] = {
    { "wave1_a",     1, 0x0000001u },
    { "wave1_b",     1, 0x0BADF00Du },
    { "wave1_c",     1, 0x1234567u },
    { "meleer_a",    9, 0x0000001u },
    { "meleer_b",    9, 0x0BADF00Du },
    { "ranger_a",   18, 0x0000001u },
    { "ranger_b",   18, 0x0BADF00Du },
    { "mager_a",    35, 0x0000001u },
    { "mager_b",    35, 0x0BADF00Du },
    { "jad_a",      67, 0x0000001u },
    { "jad_b",      67, 0x0BADF00Du },
    { "jad_c",      67, 0x1234567u },
    { "zuk_a",      69, 0x0000001u },
    { "zuk_b",      69, 0x0BADF00Du },
    { "zuk_c",      69, 0x1234567u },
};

#define NUM_CONFIGS ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))
#define EPISODE_TICKS 2000

/* Canonical inventory cells intentionally change the serialized player state hash. */
static const uint64_t EXPECTED_STATE[NUM_CONFIGS] = {
    0xd1a32c595e0f20e4ULL, 0xd556d9d18290dbe0ULL, 0x7dc08bb94dc1420aULL,
    0x005ebedd7892aa36ULL, 0xde2fb70f9d751137ULL, 0x985dd084df5d7707ULL,
    0xed97211bab4b9a47ULL, 0x8ea493a4516b791fULL, 0x6cebcf139b64d613ULL,
    0x8dc44e7252f56c19ULL, 0xe5c5146c16159d32ULL, 0x237eb63b95cd8ee8ULL,
    0xc53b4ecacd50a647ULL, 0xda3e4d8630b526f2ULL, 0x1db55fe5533fdb5dULL,
};

static const uint64_t EXPECTED_REWARD[NUM_CONFIGS] = {
    0xef229a1a4b13b62aULL, 0x380701212d84bfbaULL, 0x4098eaaa762c034aULL,
    0xfcafb9b6e476bbaaULL, 0x23577b00d055c7daULL, 0x2366aa6e9afa77d3ULL,
    0xabf96da3254522b3ULL, 0x64778a6835f7bae3ULL, 0x3b83ee4d02793243ULL,
    0x24ccbb9772bf51d3ULL, 0x4c54b1462ee3a663ULL, 0x29916e1218e670b3ULL,
    0x6c34cdb27c0c1733ULL, 0x47a1f9c7b1d3def3ULL, 0x47a1f9c7b1d3def3ULL,
};

static const uint64_t EXPECTED_MASK[NUM_CONFIGS] = {
    0xbbb48dc73a049bf3ULL, 0x39dc7eec5cb9e2f3ULL, 0x4c106d4052a4fdbaULL,
    0x143ebc7ebbb2d72aULL, 0x7dad20ecbb02bd93ULL, 0xef4d1b6877c93acaULL,
    0x8fca3065ddbf1793ULL, 0x2623748ea2e51b33ULL, 0x09130bda5462ebf3ULL,
    0x12a49268eb13bd53ULL, 0x408fca7dc9aff313ULL, 0x1e80e464ea552a5aULL,
    0x77f5a923af526e3aULL, 0x9f3034f7b394665aULL, 0x25d8b9153b3ae063ULL,
};

static const uint64_t EXPECTED_TERMINAL[NUM_CONFIGS] = {
    0xb3480d7ee6336ae2ULL, 0x910cf79bbeeba6e3ULL, 0xe238e61f8c13c673ULL,
    0xda963a44d311c6daULL, 0xb310e5115ff5cb52ULL, 0x48994c2ef3769192ULL,
    0x2d82388999073b22ULL, 0x1b1338d1b46b5673ULL, 0x0e985be125b2f7f3ULL,
    0x258a6c0def52abe1ULL, 0x77852c52053de533ULL, 0xe2d9ed2b31cf44a1ULL,
    0x7c696af13bd0d307ULL, 0x12e57f8f9818c547ULL, 0x12e57f8f9818c547ULL,
};

static const uint64_t PRE_UNIFIED_OBSERVATION[NUM_CONFIGS] = {
    0xf2362a766a6212d1ULL, 0xa87ad065ca1666dcULL, 0xcac6b26329cadeafULL,
    0xda3602f8cf48fd1fULL, 0x9898fdcd480dc4e6ULL, 0x7c305c0b8123853fULL,
    0x9fb81d2ca3477306ULL, 0xb7e475f336034eadULL, 0x73f5ac59f227cb67ULL,
    0xda647dd6d85cfbe9ULL, 0x78ff2080d00d29c0ULL, 0x83cb4a3947e97146ULL,
    0xbf34279c73e947f4ULL, 0xdf399dc7aee3d3f3ULL, 0x989c8fc6b6aea06dULL,
};

static const uint64_t UNIFIED_OBSERVATION[NUM_CONFIGS] = {
    0x6c59b72c00728ad9ULL, 0x18a13290de7329bcULL, 0x6748854d95c23f92ULL,
    0x6825dad29da6b32dULL, 0x7c5520ac1cc5837fULL, 0xb88e7bda4fc6b77eULL,
    0x45cbf9bbfa703b70ULL, 0x89db5b7cc84b4728ULL, 0xb93f28f45f543b70ULL,
    0x3f1afd9c57acd296ULL, 0x3dd5d5e4e6289715ULL, 0x5c08d30cfb280174ULL,
    0x82bf60a965334a54ULL, 0x66a4cbc54247e358ULL, 0x338f8cdd2b9d56afULL,
};

static int check_hash(
    const char* config,
    const char* component,
    uint64_t actual,
    uint64_t expected
) {
    if (actual == expected) return 0;
    printf("  %-12s %-11s got 0x%016llx expected 0x%016llx\n",
        config,
        component,
        (unsigned long long)actual,
        (unsigned long long)expected);
    return 1;
}

int main(int argc, char** argv) {
    int print_mode = argc > 1 && strcmp(argv[1], "--print") == 0;
    inf_build_npc_stats();

    printf("inferno semantic golden (%d configs, <=%d ticks each)\n", NUM_CONFIGS,
        EPISODE_TICKS);
    printf("player contract obs%d primary%d heads%d\n\n",
        INF_NUM_OBS, INF_ACTION_DIMS[INF_HEAD_PRIMARY], INF_NUM_ACTION_HEADS);

    int failed = 0;
    for (int c = 0; c < NUM_CONFIGS; c++) {
        TraceHashes hashes =
            run_episode(CONFIGS[c].start_wave, CONFIGS[c].seed, EPISODE_TICKS);
        if (print_mode) {
            printf("%-12s %016llx %016llx %016llx %016llx %016llx\n",
                CONFIGS[c].name,
                (unsigned long long)hashes.state,
                (unsigned long long)hashes.reward,
                (unsigned long long)hashes.mask,
                (unsigned long long)hashes.terminal,
                (unsigned long long)hashes.observation);
            continue;
        }

        int config_failed = 0;
        config_failed += check_hash(
            CONFIGS[c].name, "state", hashes.state, EXPECTED_STATE[c]);
        config_failed += check_hash(
            CONFIGS[c].name, "reward", hashes.reward, EXPECTED_REWARD[c]);
        config_failed += check_hash(
            CONFIGS[c].name, "mask", hashes.mask, EXPECTED_MASK[c]);
        config_failed += check_hash(
            CONFIGS[c].name, "terminal", hashes.terminal, EXPECTED_TERMINAL[c]);
        config_failed += check_hash(
            CONFIGS[c].name, "observation", hashes.observation,
            UNIFIED_OBSERVATION[c]);
        if (PRE_UNIFIED_OBSERVATION[c] == UNIFIED_OBSERVATION[c]) {
            printf("  %-12s observation contract did not change\n", CONFIGS[c].name);
            config_failed++;
        }
        if (!config_failed) printf("  %-12s PASS\n", CONFIGS[c].name);
        failed += config_failed;
    }

    if (print_mode) return 0;
    printf("\n%d component mismatches\n", failed);
    return failed > 0 ? 1 : 0;
}
