#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_zulrah.h"

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

typedef struct {
    const char* name;
    const char* description;
    int gear_tier;
    int episode_mode;
    uint32_t env_seed;
} GoldenConfig;

typedef struct {
    int ticks;
    int kills;
    int winner;
} EpisodeStats;

static void apply_env_config(
    EncounterState* state, EncounterContext* context, const GoldenConfig* cfg
) {
    ENCOUNTER_ZULRAH.put_int(state, context, "gear_tier", cfg->gear_tier);
    ENCOUNTER_ZULRAH.put_int(state, context, "gear_tier_mode", ZUL_GEAR_TIER_FIXED);
    ENCOUNTER_ZULRAH.put_int(state, context, "episode_mode", cfg->episode_mode);

    ENCOUNTER_ZULRAH.put_float(state, context, "gear_tier_weight_0", 0.5f);
    ENCOUNTER_ZULRAH.put_float(state, context, "gear_tier_weight_1", 0.3f);
    ENCOUNTER_ZULRAH.put_float(state, context, "gear_tier_weight_2", 0.2f);
    ENCOUNTER_ZULRAH.put_float(state, context, "reward_win", 0.4510606370109398f);
    ENCOUNTER_ZULRAH.put_float(state, context, "reward_loss_penalty", 0.0f);
    ENCOUNTER_ZULRAH.put_float(state, context, "reward_damage_dealt", 0.25796024594409434f);
    ENCOUNTER_ZULRAH.put_float(state, context, "reward_correct_style", 0.2157127789623018f);
    ENCOUNTER_ZULRAH.put_float(state, context, "reward_damage_received_penalty", 0.0f);
    ENCOUNTER_ZULRAH.put_float(state, context, "reward_cloud_occupancy_penalty", 0.0f);
}

static uint64_t run_episode(const GoldenConfig* cfg, int max_ticks, EpisodeStats* stats) {
    ZulrahContext ctx;
    ZulrahState s;
    static float obs[ZUL_NUM_OBS];
    static float mask[ZUL_ACTION_MASK_SIZE];
    int actions[ZUL_NUM_ACTION_HEADS];

    EncounterState* state = (EncounterState*)&s;
    EncounterContext* context = (EncounterContext*)&ctx;

    ENCOUNTER_ZULRAH.init_context(context);
    ENCOUNTER_ZULRAH.init_state(state, context);
    apply_env_config(state, context, cfg);
    ENCOUNTER_ZULRAH.reset(state, context, cfg->env_seed);

    uint64_t h = FNV_OFFSET;

    for (int t = 0; t < max_ticks; t++) {
        zul_heuristic_actions(&s, actions);
        ENCOUNTER_ZULRAH.step(state, context, actions);
        ENCOUNTER_ZULRAH.write_obs(state, context, obs);
        ENCOUNTER_ZULRAH.write_mask(state, context, mask);

        for (int i = 0; i < ZUL_NUM_OBS; i++) h = fnv_f32(h, obs[i]);
        for (int i = 0; i < ZUL_ACTION_MASK_SIZE; i++) h = fnv_f32(h, mask[i]);
        h = fnv_f32(h, s.reward);
        h = fnv_i32(h, s.kills_this_episode);
        h = fnv_i32(h, s.tick);
        h = fnv_i32(h, s.episode_over);
        h = fnv_i32(h, s.winner);

        if (s.episode_over) break;
    }

    stats->ticks = s.tick;
    stats->kills = s.kills_this_episode;
    stats->winner = s.winner;
    return h;
}

static const GoldenConfig CONFIGS[] = {
    {
        "t0_single",
        "gear tier 0 fixed, single-kill episode ends on first kill or death",
        0, ZUL_EPISODE_SINGLE_KILL,
        0x000A0001u,
    },
    {
        "t1_single",
        "gear tier 1 fixed (thrall + augury), single-kill episode",
        1, ZUL_EPISODE_SINGLE_KILL,
        0x000A0003u,
    },
    {
        "t2_single",
        "gear tier 2 fixed (saturated heart + thrall), single-kill episode",
        2, ZUL_EPISODE_SINGLE_KILL,
        0x000A0005u,
    },
    {
        "t0_trip",
        "gear tier 0 fixed, trip episode chains kills until the tick cap",
        0, ZUL_EPISODE_TRIP,
        0x000B0007u,
    },
    {
        "t1_trip",
        "gear tier 1 fixed, trip episode chains kills until the tick cap",
        1, ZUL_EPISODE_TRIP,
        0x000B000Bu,
    },
    {
        "t2_trip",
        "gear tier 2 fixed, trip episode chains kills until the tick cap",
        2, ZUL_EPISODE_TRIP,
        0x000B000Du,
    },
};

#define NUM_CONFIGS ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))
#define EPISODE_TICKS 700

static_assert(EPISODE_TICKS > ZUL_MAX_TICKS,
    "episode budget must cover the in-sim tick cap");

static const uint64_t BASELINE[NUM_CONFIGS] = {
    0xecf8fefec11da90fULL,  /* t0_single */
    0xfaa0144d43a61972ULL,  /* t1_single */
    0x801a8acf962b8ff2ULL,  /* t2_single */
    0xdd701c2aac0eaa2fULL,  /* t0_trip */
    0xa9033115654c7a8aULL,  /* t1_trip */
    0x22e191b4b0fd500eULL,  /* t2_trip */
};

int main(int argc, char** argv) {
    int print_mode = (argc > 1 && strcmp(argv[1], "--print") == 0);

    printf("zulrah golden-master (%d configs, <=%d ticks each)\n\n",
           NUM_CONFIGS, EPISODE_TICKS);

    EpisodeStats stats[NUM_CONFIGS];

    int failed = 0;
    for (int c = 0; c < NUM_CONFIGS; c++) {
        uint64_t h = run_episode(&CONFIGS[c], EPISODE_TICKS, &stats[c]);
        if (print_mode) {
            printf("    0x%016llxULL,  /* %s */\n",
                   (unsigned long long)h, CONFIGS[c].name);
        } else {
            int ok = h == BASELINE[c];
            printf("  %-12s 0x%016llx  %s\n",
                   CONFIGS[c].name, (unsigned long long)h, ok ? "PASS" : "FAIL");
            if (!ok) {
                printf("               expected 0x%016llx\n",
                       (unsigned long long)BASELINE[c]);
                failed++;
            }
        }
    }

    if (print_mode) {
        printf("\nconfig coverage:\n");
        for (int c = 0; c < NUM_CONFIGS; c++) {
            printf("  %-12s ticks=%-4d kills=%-2d winner=%d  %s\n",
                   CONFIGS[c].name, stats[c].ticks, stats[c].kills,
                   stats[c].winner, CONFIGS[c].description);
        }
        printf("\npaste the array above into BASELINE[].\n");
        return 0;
    }

    printf("\n%d/%d configs match baseline\n", NUM_CONFIGS - failed, NUM_CONFIGS);
    return failed > 0 ? 1 : 0;
}
