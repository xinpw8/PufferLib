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
    0x7ba3711e153095beULL, 0xb39d554acf436e5cULL, 0x6cc5844ce0c39c1bULL,
    0xf91d3612097dbd8aULL, 0xea21f6ea8e67827bULL, 0x6ae89eb3f13bae2eULL,
    0x4d866acbe988acceULL, 0x7854ed938c91eb29ULL, 0x689a7fc8d0ae318eULL,
    0x8bc8bdc32bd12465ULL, 0x77b8a8ab51ef62c8ULL, 0x89e7ddb514ed1906ULL,
    0xda40b9bdbbeb1e61ULL, 0xba99612d66277506ULL, 0x0600b75367c4e45fULL,
};

static const uint64_t EXPECTED_REWARD[NUM_CONFIGS] = {
    0xb089187e53857203ULL, 0xa2565e648c91b58aULL, 0x0e980cdddd5d24eaULL,
    0xb45661865c9132baULL, 0x51ad73340d19447aULL, 0x93d59ec47f5c8e53ULL,
    0x6c34cdb27c0c1733ULL, 0xa11671953e858723ULL, 0xdce53c1df8560f83ULL,
    0x93d59ec47f5c8e53ULL, 0x4c54b1462ee3a663ULL, 0x3d09912556a163b3ULL,
    0x6c34cdb27c0c1733ULL, 0x47a1f9c7b1d3def3ULL, 0xe7f1c65d07654513ULL,
};

static const uint64_t EXPECTED_MASK[NUM_CONFIGS] = {
    0x161ad634c13bed03ULL, 0xb0a5340f14eaa10aULL, 0x3e98eb184642444aULL,
    0xf43de0a67cfc551aULL, 0x8ec84065b245eaa3ULL, 0x8a02e785528c7e9aULL,
    0x8b7bc0262228b5daULL, 0x9436e2cbac777d3aULL, 0x70a471b5f966653aULL,
    0xa692532e12d537c3ULL, 0x969a7e2d44202a13ULL, 0xfef0e18569cb4cd3ULL,
    0xdbe87f09bbc753a3ULL, 0x4457d6e53647be0aULL, 0xa3c4b4613376930aULL,
};

static const uint64_t EXPECTED_TERMINAL[NUM_CONFIGS] = {
    0x45ce0c094429a073ULL, 0xb9029934b4c6dcc2ULL, 0x6c474b92f31109b3ULL,
    0xc4d9ee59b30eec02ULL, 0x74622ccdf2f00ca2ULL, 0xa24267dbb4c1dbd2ULL,
    0xda5efce8115c5862ULL, 0xce9f670c94cfbf73ULL, 0x565640d34d54c4f3ULL,
    0xc03f2da5d13a88e1ULL, 0x77852c52053de533ULL, 0x88ec0c389d9c42a1ULL,
    0x7c696af13bd0d307ULL, 0x12e57f8f9818c547ULL, 0xf92b462e21d67e67ULL,
};

static const uint64_t PRE_UNIFIED_OBSERVATION[NUM_CONFIGS] = {
    0xf2362a766a6212d1ULL, 0xa87ad065ca1666dcULL, 0xcac6b26329cadeafULL,
    0xda3602f8cf48fd1fULL, 0x9898fdcd480dc4e6ULL, 0x7c305c0b8123853fULL,
    0x9fb81d2ca3477306ULL, 0xb7e475f336034eadULL, 0x73f5ac59f227cb67ULL,
    0xda647dd6d85cfbe9ULL, 0x78ff2080d00d29c0ULL, 0x83cb4a3947e97146ULL,
    0xbf34279c73e947f4ULL, 0xdf399dc7aee3d3f3ULL, 0x989c8fc6b6aea06dULL,
};

static const uint64_t UNIFIED_OBSERVATION[NUM_CONFIGS] = {
    0xa51b82ede1cae21cULL, 0xe1997807c974def2ULL, 0xbc3aac20a6bf0a7fULL,
    0x3dccdcc4eda4266bULL, 0x27940921e8054ffdULL, 0x71c062e172101343ULL,
    0xf01f073a63279eb2ULL, 0x08121438552d14e6ULL, 0xdf889ff57739043aULL,
    0xc7ddbaa9c12699cdULL, 0x6dbcf3e1a501d061ULL, 0x6fc772947a63e769ULL,
    0xf08f7b50f9e6ec87ULL, 0xd60728d8a72defa3ULL, 0x97e58a9ee67f1f66ULL,
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
