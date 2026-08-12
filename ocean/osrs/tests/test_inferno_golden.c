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
        const OsrsInventoryCell* cell = &s->inventory_cells[cell_idx];
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
        if (head != INF_HEAD_TARGET) {
            for (int action = 0; action < INF_ACTION_DIMS[head]; action++)
                h = fnv_f32(h, mask[offset + action]);
            offset += INF_ACTION_DIMS[head];
            continue;
        }

        h = fnv_f32(h, mask[offset]);
        for (int npc_idx = 0; npc_idx < TRACE_NPC_SLOTS; npc_idx++) {
            int slot = inf_find_target_obs_slot(s, npc_idx);
            h = fnv_f32(h, slot >= 0 ? mask[offset + slot + 1] : 0.0f);
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
            if (head != INF_HEAD_TARGET) {
                actions[head] =
                    (int)(random_value % (uint64_t)INF_ACTION_DIMS[head]);
                continue;
            }
            int npc_idx = (int)(random_value % TRACE_NPC_SLOTS);
            int slot = inf_find_target_obs_slot(s, npc_idx);
            actions[head] = slot >= 0 ? slot + 1 : 0;
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

static const uint64_t EXPECTED_STATE[NUM_CONFIGS] = {
    0x3e4fbb35c1889d44ULL, 0x8c68f1dc6f54c2a6ULL, 0x71d36e99ebd433e8ULL,
    0x9c028c5906c58dbeULL, 0xd6183313a76b807fULL, 0x28b5ad296c307915ULL,
    0x1e408a9f378394e6ULL, 0x228334e4bd9b7f3aULL, 0x26463ee9c8e17cc8ULL,
    0x2762fdd6008979ddULL, 0xd9f8faf40e2d0657ULL, 0xff5ceeb4f52cb16fULL,
    0xef0befe2d848d410ULL, 0x38385138dce9f842ULL, 0x7444a507700231c1ULL,
};

static const uint64_t EXPECTED_REWARD[NUM_CONFIGS] = {
    0x7a65310944a973aaULL, 0xa122490e8a5ea4faULL, 0xd8288292f3160f5aULL,
    0x1f71cc42bc243e13ULL, 0x65af16b1d7e922caULL, 0x49a0f296876aea23ULL,
    0xc4080cd4d9a6eb13ULL, 0x9d01d9da22e0bb93ULL, 0x6c34cdb27c0c1733ULL,
    0x29916e1218e670b3ULL, 0x6494cc221d5e7db3ULL, 0x11a8e2781441fd33ULL,
    0x5d411311ad9cd553ULL, 0xe7f1c65d07654513ULL, 0x47a1f9c7b1d3def3ULL,
};

static const uint64_t EXPECTED_MASK[NUM_CONFIGS] = {
    0x9609313d75a829baULL, 0x34109efbca6fab0aULL, 0x9787e973169329baULL,
    0xe97fa334ceaf6153ULL, 0xd2dc555e39a58ba3ULL, 0x0ea03aa9e92854e3ULL,
    0xc2432dc597e3e483ULL, 0xbf39837ae26a6aaaULL, 0x7fc9129b3356ed93ULL,
    0xe54fb95b499c2003ULL, 0xd0bb60198af9b643ULL, 0xa01ad9cfc96241c3ULL,
    0x49744aaffe686a73ULL, 0x5b55145ce7296033ULL, 0xfd42b38660125883ULL,
};

static const uint64_t EXPECTED_TERMINAL[NUM_CONFIGS] = {
    0xaacea6292d1dbfd3ULL, 0x08f8c8682da946a1ULL, 0x8799762912bdf103ULL,
    0x622fee1940c541abULL, 0x82fbcdc1da23c683ULL, 0x2629f248900d0043ULL,
    0xc92dd3ae1e3522b2ULL, 0xbb44db79d20b1d81ULL, 0xa10f828fd67b1801ULL,
    0xe2d9ed2b31cf44a1ULL, 0x448b70e96c7c46a1ULL, 0xca591485885081a1ULL,
    0xd05fd043497d9b27ULL, 0xf92b462e21d67e67ULL, 0x12e57f8f9818c547ULL,
};

static const uint64_t LEGACY_OBSERVATION[NUM_CONFIGS] = {
    0x03f4682055435366ULL, 0x90b9c6fb55157aacULL, 0x0092dfb263b4768cULL,
    0x3964598751ca95a2ULL, 0x5a8ff64de7cc10feULL, 0xb8a7174941603f51ULL,
    0xb6b6e303a9e7857cULL, 0x792e4afd5831fe70ULL, 0xfd926d25fd8b4258ULL,
    0x7b738c46a083f9c5ULL, 0xcef689a6461bbfafULL, 0xd8e160b35c03dbc0ULL,
    0x2a2f884605e7a640ULL, 0x12fdce41ea3ac43fULL, 0xc8f251c4e9489e7dULL,
};

static const uint64_t COMPACT_OBSERVATION[NUM_CONFIGS] = {
    0xf2362a766a6212d1ULL, 0xa87ad065ca1666dcULL, 0xcac6b26329cadeafULL,
    0xda3602f8cf48fd1fULL, 0x9898fdcd480dc4e6ULL, 0x7c305c0b8123853fULL,
    0x9fb81d2ca3477306ULL, 0xb7e475f336034eadULL, 0x73f5ac59f227cb67ULL,
    0xda647dd6d85cfbe9ULL, 0x78ff2080d00d29c0ULL, 0x83cb4a3947e97146ULL,
    0xbf34279c73e947f4ULL, 0xdf399dc7aee3d3f3ULL, 0x989c8fc6b6aea06dULL,
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
    printf("lineage 59f90b7f4 obs3432 -> compact obs498\n\n");

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
            COMPACT_OBSERVATION[c]);
        if (LEGACY_OBSERVATION[c] == COMPACT_OBSERVATION[c]) {
            printf("  %-12s observation lineage did not change\n", CONFIGS[c].name);
            config_failed++;
        }
        if (!config_failed) printf("  %-12s PASS\n", CONFIGS[c].name);
        failed += config_failed;
    }

    if (print_mode) return 0;
    printf("\n%d component mismatches\n", failed);
    return failed > 0 ? 1 : 0;
}
