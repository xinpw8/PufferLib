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
    h = fnv_bytes(h, &s->player, sizeof(s->player));
    h = fnv_bytes(h, s->pillars, sizeof(s->pillars));
    h = fnv_bytes(h, s->npcs, TRACE_NPC_SLOTS * sizeof(s->npcs[0]));
    h = fnv_bytes(h, &s->player_pending_hits, sizeof(s->player_pending_hits));
    h = fnv_bytes(h, s->pending_sparks, sizeof(s->pending_sparks));
    h = fnv_bytes(h, s->inventory_cells, sizeof(s->inventory_cells));
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
    EncounterState* state = inf_create();
    inf_put_int(state, "start_wave", start_wave);
    inf_reset(state, seed);

    InfernoState* s = (InfernoState*)state;
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
        inf_refresh_current_obs_slots(s);
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

        inf_step(state, actions);
        inf_write_obs(state, obs);
        inf_write_mask(state, mask);

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

    inf_destroy(state);
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
    0x443ad0be6700d3ccULL, 0x17bf6a8136354a13ULL, 0xe847a00300ff389aULL,
    0x24561a696000f6aaULL, 0x29ef822b2984ead4ULL, 0x37cd7f6db4d4a69bULL,
    0x843298acc61300d5ULL, 0x61df8fa3a31ce410ULL, 0xc147b4e946a9ef62ULL,
    0xb0cc89838a07a28eULL, 0x38e2a2100ce6aad0ULL, 0x68c2e25a45de3330ULL,
    0x4276518d9178c88aULL, 0x8c4d39c065b913ffULL, 0xfb06a88261725c65ULL,
};

static const uint64_t EXPECTED_REWARD[NUM_CONFIGS] = {
    0x8fb194fc3b735b9aULL, 0x91bc1d01b9d474aaULL, 0xd8288292f3160f5aULL,
    0x1f71cc42bc243e13ULL, 0x9ccbff0cc6408633ULL, 0x49a0f296876aea23ULL,
    0xc4080cd4d9a6eb13ULL, 0x9d01d9da22e0bb93ULL, 0x6c34cdb27c0c1733ULL,
    0x29916e1218e670b3ULL, 0x6494cc221d5e7db3ULL, 0x11a8e2781441fd33ULL,
    0x5d411311ad9cd553ULL, 0xe7f1c65d07654513ULL, 0x47a1f9c7b1d3def3ULL,
};

static const uint64_t EXPECTED_MASK[NUM_CONFIGS] = {
    0x07b15753e2efc64aULL, 0x48bb2a98696ebfb3ULL, 0x9787e973169329baULL,
    0xe97fa334ceaf6153ULL, 0xf4239d431bef2aeaULL, 0x0ea03aa9e92854e3ULL,
    0xc2432dc597e3e483ULL, 0xbf39837ae26a6aaaULL, 0x7fc9129b3356ed93ULL,
    0xe54fb95b499c2003ULL, 0xd0bb60198af9b643ULL, 0xa01ad9cfc96241c3ULL,
    0x49744aaffe686a73ULL, 0x5b55145ce7296033ULL, 0xfd42b38660125883ULL,
};

static const uint64_t EXPECTED_TERMINAL[NUM_CONFIGS] = {
    0x37cd079d32ad9e72ULL, 0xce955ae918947853ULL, 0x8799762912bdf103ULL,
    0x622fee1940c541abULL, 0x9c2e107bb54a1e8bULL, 0x2629f248900d0043ULL,
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
    0xf6b0c1dd0bcd0caaULL, 0xf6383c0531b500aeULL, 0x02d82f1ec0b39b31ULL,
    0xd2e90acf4a8889e3ULL, 0x92a416e84ff2ffbdULL, 0x24c8a4ade044fffbULL,
    0xe314373c080ea591ULL, 0xc6986cd9d1827d8dULL, 0x63f33b2c81af5dbfULL,
    0x318376fbc0ee582aULL, 0x85ebe0fa55f1f53bULL, 0x905f13ba928c7394ULL,
    0xbf34279c73e947f4ULL, 0xfe224b03850e2f95ULL, 0x72ee3c9abde4352dULL,
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
