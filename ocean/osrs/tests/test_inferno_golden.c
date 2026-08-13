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

static uint64_t run_episode(int start_wave, uint32_t seed, int max_ticks) {
    EncounterState* state = inf_create();
    inf_put_int(state, "start_wave", start_wave);
    inf_reset(state, seed);

    InfernoState* s = (InfernoState*)state;

    static float obs[INF_NUM_OBS];
    int actions[INF_NUM_ACTION_HEADS];

    uint64_t arng = ((uint64_t)seed << 20) ^ (uint64_t)(start_wave + 1) ^ 0xD1B54A32D192ED03ULL;

    uint64_t h = FNV_OFFSET;

    for (int t = 0; t < max_ticks; t++) {
        for (int head = 0; head < INF_NUM_ACTION_HEADS; head++) {
            int dim = INF_ACTION_DIMS[head];
            actions[head] = (int)(splitmix64(&arng) % (uint64_t)dim);
        }

        inf_step(state, actions);
        inf_write_obs(state, obs);

        for (int i = 0; i < INF_NUM_OBS; i++) {
            h = fnv_f32(h, obs[i]);
        }
        h = fnv_f32(h, s->reward);
        h = fnv_i32(h, s->wave);
        h = fnv_i32(h, s->tick);
        h = fnv_i32(h, s->episode_over);
        h = fnv_i32(h, s->winner);

        if (s->episode_over) break;
    }

    inf_destroy(state);
    return h;
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

static const uint64_t BASELINE[NUM_CONFIGS] = {

    0x9d8970300cea947aULL,
    0xefeefc062898de1bULL,
    0x300b40b9b6c32f47ULL,
    0xf600c7a9f79479faULL,
    0x267ab0fac9ad5b27ULL,
    0x999a41e1a0916ab9ULL,
    0x2ddb91b645db1e75ULL,
    0xd2a1416c4b53157fULL,
    0x84a2ab3540f37ac8ULL,
    0xb98928c437e48005ULL,
    0x3e5a885e173d4674ULL,
    0xd6577872951242fdULL,
    0xc5af52a73611f3f6ULL,
    0x7a84c874f2b7c5ceULL,
    0x28c4ccc92a588192ULL,
};

int main(int argc, char** argv) {
    int print_mode = (argc > 1 && strcmp(argv[1], "--print") == 0);

    inf_build_npc_stats();

    printf("inferno golden-master (%d configs, <=%d ticks each)\n\n",
           NUM_CONFIGS, EPISODE_TICKS);

    int failed = 0;
    for (int c = 0; c < NUM_CONFIGS; c++) {
        uint64_t h = run_episode(CONFIGS[c].start_wave, CONFIGS[c].seed, EPISODE_TICKS);
        if (print_mode) {
            printf("    0x%016llxULL,  /* %s */\n", (unsigned long long)h, CONFIGS[c].name);
        } else {
            int ok = (h == BASELINE[c]);
            printf("  %-12s 0x%016llx  %s\n", CONFIGS[c].name,
                   (unsigned long long)h, ok ? "PASS" : "FAIL");
            if (!ok) {
                printf("               expected 0x%016llx\n",
                       (unsigned long long)BASELINE[c]);
                failed++;
            }
        }
    }

    if (print_mode) {
        printf("\npaste the array above into BASELINE[].\n");
        return 0;
    }

    printf("\n%d/%d configs match baseline\n", NUM_CONFIGS - failed, NUM_CONFIGS);
    return failed > 0 ? 1 : 0;
}
