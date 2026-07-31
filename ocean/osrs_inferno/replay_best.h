#ifndef OSRS_INFERNO_REPLAY_BEST_H
#define OSRS_INFERNO_REPLAY_BEST_H

#include <stdint.h>

typedef struct {
    int wave;
    int ticks;
    int min_zuk_hp;
    uint32_t rng_seed;
} InfernoReplayBest;

static inline InfernoReplayBest inferno_replay_best_initial(void) {
    return (InfernoReplayBest){
        .wave = 0,
        .ticks = 999999,
        .min_zuk_hp = 999999,
        .rng_seed = UINT32_MAX,
    };
}

static inline int inferno_replay_is_better(
    const InfernoReplayBest* best,
    int start_wave,
    int wave,
    int ticks,
    int min_zuk_hp,
    uint32_t rng_seed
) {
    if (start_wave == 0) {
        if (wave != best->wave) return wave > best->wave;
        if (ticks != best->ticks) return ticks < best->ticks;
        return rng_seed < best->rng_seed;
    }

    if (min_zuk_hp != best->min_zuk_hp) return min_zuk_hp < best->min_zuk_hp;
    if (min_zuk_hp == 0 && ticks != best->ticks) return ticks < best->ticks;
    return rng_seed < best->rng_seed;
}

static inline void inferno_replay_best_apply(
    InfernoReplayBest* best,
    int wave,
    int ticks,
    int min_zuk_hp,
    uint32_t rng_seed
) {
    best->wave = wave;
    best->ticks = ticks;
    best->min_zuk_hp = min_zuk_hp;
    best->rng_seed = rng_seed;
}

#endif
