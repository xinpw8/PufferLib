#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

#define FNV_OFFSET 1469598103934665603ULL
#define FNV_PRIME 1099511628211ULL

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

typedef struct {
    const char* name;
    const char* npc_path;
    int public_start_wave;
    uint32_t env_seed;
    uint64_t action_seed;
} GoldenConfig;

static void fill_actions(
    const ColosseumState* s,
    uint64_t* action_rng,
    int actions[COLO_NUM_ACTION_HEADS]
) {
    for (int head = 0; head < COLO_NUM_ACTION_HEADS; head++) {
        actions[head] =
            (int)(splitmix64(action_rng) % (uint64_t)COLO_ACTION_DIMS[head]);
    }
    if (s->modifiers.draft_pending) {
        actions[COLO_HEAD_PRIMARY] = 0;
        actions[COLO_HEAD_MODIFIER_SELECT] =
            1 + (int)(splitmix64(action_rng) % COLO_MODIFIER_DRAFT_OPTIONS);
    }
}

static uint64_t run_episode(const GoldenConfig* cfg, int max_ticks) {
    ColosseumContext ctx;
    ColosseumState s;
    static float obs[COLO_NUM_OBS];
    int actions[COLO_NUM_ACTION_HEADS];

    col_init_context_typed(&ctx);
    ctx.config.start_wave = cfg->public_start_wave - 1;
    ctx.config.step_out_forecast_obs_enabled = 1;
    ctx.config.forecast_horizon = 4;

    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, cfg->env_seed);

    uint64_t action_rng = cfg->action_seed;
    uint64_t h = FNV_OFFSET;

    for (int t = 0; t < max_ticks; t++) {
        fill_actions(&s, &action_rng, actions);
        col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
        col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);

        for (int i = 0; i < COLO_NUM_OBS; i++) h = fnv_f32(h, obs[i]);
        h = fnv_f32(h, s.reward);
        h = fnv_i32(h, s.wave);
        h = fnv_i32(h, s.tick);
        h = fnv_i32(h, s.episode_over);
        h = fnv_i32(h, s.winner);

        if (s.episode_over) break;
    }

    return h;
}

static const GoldenConfig CONFIGS[] = {
    {
        "w01_shaman_jaguar",
        "wave 1 opens a mandatory draft, then spawns warband + serpent shaman, overrun reinforces jaguar",
        1,
        0x00010001u,
        0x6a09e667f3bcc909ULL,
    },
    {
        "w02_javelin",
        "wave 2 starts warband + serpent shaman + javelin colossus, overrun reinforces jaguar",
        2,
        0x00020003u,
        0xbb67ae8584caa73bULL,
    },
    {
        "w03_double_javelin",
        "wave 3 starts warband + serpent shaman + two javelin colossi, overrun reinforces jaguar",
        3,
        0x00030005u,
        0x3c6ef372fe94f82bULL,
    },
    {
        "w04_manticore",
        "wave 4 starts warband + serpent shaman + manticore, overrun reinforces jaguar + serpent shaman",
        4,
        0x00040007u,
        0xa54ff53a5f1d36f1ULL,
    },
    {
        "w05_javelin_manticore",
        "wave 5 starts warband + serpent shaman + javelin colossus + manticore, overrun reinforces jaguar + serpent shaman",
        5,
        0x0005000bu,
        0x510e527fade682d1ULL,
    },
    {
        "w06_double_javelin_manticore",
        "wave 6 starts warband + serpent shaman + two javelin colossi + manticore, overrun reinforces jaguar + serpent shaman",
        6,
        0x0006000du,
        0x9b05688c2b3e6c1fULL,
    },
    {
        "w07_shockwave",
        "wave 7 starts warband + javelin colossus + manticore + shockwave colossus, overrun reinforces minotaur",
        7,
        0x00070011u,
        0x1f83d9abfb41bd6bULL,
    },
    {
        "w08_double_javelin_shockwave",
        "wave 8 starts warband + two javelin colossi + manticore + shockwave colossus, overrun reinforces minotaur",
        8,
        0x00080013u,
        0x5be0cd19137e2179ULL,
    },
    {
        "w09_double_manticore",
        "wave 9 starts warband + javelin colossus + two manticores, overrun reinforces minotaur",
        9,
        0x00090017u,
        0xcbbb9d5dc1059ed8ULL,
    },
    {
        "w10_heavy_mix",
        "wave 10 starts warband + two javelin colossi + two manticores, overrun reinforces minotaur + serpent shaman",
        10,
        0x000a001du,
        0x629a292a367cd507ULL,
    },
    {
        "w11_shockwave_heavy_mix",
        "wave 11 starts warband + javelin colossus + two manticores + shockwave colossus, overrun reinforces minotaur + serpent shaman",
        11,
        0x000b001fu,
        0x9159015a3070dd17ULL,
    },
    {
        "w12_sol",
        "wave 12 starts Sol Heredit in the boss arena",
        12,
        0x000c0025u,
        0x152fecd8f70e5939ULL,
    },
};

#define NUM_CONFIGS ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))
#define EPISODE_TICKS 4000

static const uint64_t BASELINE[NUM_CONFIGS] = {
    0x6b648dbd26450b82ULL,
    0xd25dd5f73aea2df6ULL,
    0x137d8011443b61f8ULL,
    0xeb82de91da34947dULL,
    0xf10a60721a18d0ecULL,
    0xef3182c4eebbaeb5ULL,
    0x63f92c48e77deeecULL,
    0x4e15c78ede861b01ULL,
    0x4b420b3e18d846ebULL,
    0x054f146530962087ULL,
    0x4b5835b98fc7d4b0ULL,
    0xae22f0ac054585aaULL,
};

int main(int argc, char** argv) {
    int print_mode = (argc > 1 && strcmp(argv[1], "--print") == 0);

    col_build_npc_stats();

    printf("colosseum golden-master (%d configs, <=%d ticks each)\n\n",
           NUM_CONFIGS, EPISODE_TICKS);

    int failed = 0;
    for (int c = 0; c < NUM_CONFIGS; c++) {
        uint64_t h = run_episode(&CONFIGS[c], EPISODE_TICKS);
        if (print_mode) {
            printf("    0x%016llxULL,\n", (unsigned long long)h);
        } else {
            int ok = h == BASELINE[c];
            printf("  %-28s 0x%016llx  %s\n",
                   CONFIGS[c].name, (unsigned long long)h, ok ? "PASS" : "FAIL");
            if (!ok) {
                printf("                               expected 0x%016llx\n",
                       (unsigned long long)BASELINE[c]);
                failed++;
            }
        }
    }

    if (print_mode) {
        printf("\nconfig coverage:\n");
        for (int c = 0; c < NUM_CONFIGS; c++) {
            printf("  %-28s %s\n", CONFIGS[c].name, CONFIGS[c].npc_path);
        }
        printf("\npaste the array above into BASELINE[].\n");
        return 0;
    }

    printf("\n%d/%d configs match baseline\n", NUM_CONFIGS - failed, NUM_CONFIGS);
    return failed > 0 ? 1 : 0;
}
