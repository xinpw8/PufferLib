/* Hashes SIMULATION state only, never observation floats.
 *
 * test_colosseum_golden folds COLO_NUM_OBS obs floats and the sim scalars into one
 * hash, so re-seeding it after an observation change also re-seeds the simulation
 * half and a sim regression landed in the same commit is invisible. This probe is
 * the half that must not move when the observation is recut.
 *
 * Covers the whole ColosseumState plus the full 452-float action mask. Storage whose
 * SIZE is not part of the simulation contract -- the obs memo caches, and the pending-hit
 * queues' spare capacity -- is hashed by content rather than by raw bytes, so resizing or
 * renarrowing it does not move a digest.
 *
 * Re-seeding rule: prove the digests first, never re-print to make an edit pass. The
 * pending-hit narrowing was carried across by computing baselines on the old record with
 * this same hash and confirming the new one reproduced all 12.
 *
 * The player-collision re-seed could NOT use that method: replacing a 1156-byte array with
 * two ints moved the struct's padding (size fell 1152, not the 1148 the fields imply), so
 * the hashed byte SEQUENCE differs even though the content does not. That re-seed rests on
 * the golden tests, which hash observation floats every tick and would move if any NPC were
 * blocked differently, plus test_colosseum_modifiers at 10491/10491.
 *
 * ColosseumLog is skipped as of the reward-clamp telemetry: it is an observation OF the
 * simulation, not part of it, so adding a counter to it must not read as a sim regression.
 * That re-seed WAS proved by the differential method: digests printed with the log skipped
 * on the tree without the telemetry reproduced all 12 with it. Sim quantities the log
 * derives from still reach the goldens through the observation. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

#define FNV_OFFSET 1469598103934665603ULL
#define FNV_PRIME 1099511628211ULL

static inline uint64_t fnv_bytes(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= FNV_PRIME; }
    return h;
}

static inline uint64_t fnv_f32(uint64_t h, float v) {
    uint32_t bits; memcpy(&bits, &v, sizeof(bits));
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
    int public_start_wave;
    uint32_t env_seed;
    uint64_t action_seed;
    uint64_t baseline;
} SimConfig;

/* Baselines are generated on the tree that contains the pathfinding and reservoir
 * determinism fixes. Regenerate with --print only when a SIMULATION change is
 * intended, never to make an observation edit pass. */
static SimConfig CONFIGS[] = {
    {"w01",  1, 1001ULL, 0xC0FFEE01ULL, 0x9c4cbcde5d159413ULL},
    {"w02",  2, 1002ULL, 0xC0FFEE02ULL, 0x926e334416bc21f0ULL},
    {"w03",  3, 1003ULL, 0xC0FFEE03ULL, 0xcc8a421a35859befULL},
    {"w04",  4, 1004ULL, 0xC0FFEE04ULL, 0x14b1e80d759d1cb2ULL},
    {"w05",  5, 1005ULL, 0xC0FFEE05ULL, 0x22f415d9f4d6dd02ULL},
    {"w06",  6, 1006ULL, 0xC0FFEE06ULL, 0xb82309c9c86c363bULL},
    {"w07",  7, 1007ULL, 0xC0FFEE07ULL, 0x4fd42a575b3fd127ULL},
    {"w08",  8, 1008ULL, 0xC0FFEE08ULL, 0x48548d73bd727fd2ULL},
    {"w09",  9, 1009ULL, 0xC0FFEE09ULL, 0xb3f7b27247b54a07ULL},
    {"w10", 10, 1010ULL, 0xC0FFEE10ULL, 0x2b0b3f61efb8b43fULL},
    {"w11", 11, 1011ULL, 0xC0FFEE11ULL, 0x1e8ff8d2425f33a7ULL},
    {"w12", 12, 1012ULL, 0xC0FFEE12ULL, 0x210c0b6ce8537df3ULL},
};

static void fill_actions(
    const ColosseumState* s, uint64_t* rng, int actions[COLO_NUM_ACTION_HEADS]
) {
    for (int head = 0; head < COLO_NUM_ACTION_HEADS; head++)
        actions[head] = (int)(splitmix64(rng) % (uint64_t)COLO_ACTION_DIMS[head]);
    if (s->modifiers.draft_pending) {
        actions[COLO_HEAD_PRIMARY] = 0;
        actions[COLO_HEAD_MODIFIER_SELECT] =
            1 + (int)(splitmix64(rng) % COLO_MODIFIER_DRAFT_OPTIONS);
    }
}

/* Queue contents widened back to int32, so narrowing a field's storage type does not
 * move the digest. Only the live prefix is hashed: readers all loop to count, so bytes
 * past it are not simulation state. */
static uint64_t fnv_queue(uint64_t h, const EncounterPendingHitQueue* q) {
    h = fnv_i32(h, q->count);
    for (int i = 0; i < q->count; i++) {
        const EncounterPendingHit* p = &q->hits[i];
        h = fnv_i32(h, p->active);
        h = fnv_i32(h, p->damage);
        h = fnv_i32(h, p->ticks_remaining);
        h = fnv_i32(h, p->attack_style);
        h = fnv_i32(h, p->check_prayer);
        h = fnv_i32(h, p->prayer_check_delay);
        h = fnv_i32(h, p->spell_type);
        h = fnv_i32(h, p->source_npc_type);
        h = fnv_i32(h, p->source_npc_slot);
        h = fnv_i32(h, p->hit_success);
        h = fnv_i32(h, p->elysian_reduced);
    }
    return h;
}

static uint64_t fnv_npc(uint64_t h, const ColoNPC* n) {
    const uint8_t* b = (const uint8_t*)n;
    size_t off = offsetof(ColoNPC, pending_hits);
    size_t end = off + sizeof(n->pending_hits);
    h = fnv_bytes(h, b, off);
    h = fnv_bytes(h, b + end, sizeof(*n) - end);
    return fnv_queue(h, &n->pending_hits);
}

typedef struct { size_t off, len; } SkipRegion;

static int skip_cmp(const void* a, const void* b) {
    size_t x = ((const SkipRegion*)a)->off, y = ((const SkipRegion*)b)->off;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static uint64_t hash_sim(uint64_t h, const ColosseumState* s, const float* mask) {
    /* Hash AROUND storage whose SIZE is not part of the simulation contract, then hash
     * its contents semantically. FNV-1a multiplies once per byte, so leaving a memo
     * cache or a queue's spare capacity in the byte range makes resizing it move every
     * digest -- a false alarm on exactly the edits this probe exists to clear. */
    /* log and obs_memos are the last two members, skipped as ONE region running to the end
     * of the struct. Skipping them individually left the alignment padding between them in
     * the hash, and that padding is 4 bytes or 0 depending on sizeof(ColosseumLog) mod 8 --
     * so adding three floats to the log moved all twelve digests with no behaviour change.
     * Taking the whole tail removes the padding from the hash entirely. */
    SkipRegion skip[] = {
        {offsetof(ColosseumState, npcs), sizeof(s->npcs)},
        {offsetof(ColosseumState, player_pending_hits), sizeof(s->player_pending_hits)},
        {offsetof(ColosseumState, log), sizeof(*s) - offsetof(ColosseumState, log)},
    };
    int nskip = (int)(sizeof(skip) / sizeof(skip[0]));
    qsort(skip, (size_t)nskip, sizeof(skip[0]), skip_cmp);

    const uint8_t* base = (const uint8_t*)s;
    size_t cur = 0;
    for (int i = 0; i < nskip; i++) {
        h = fnv_bytes(h, base + cur, skip[i].off - cur);
        cur = skip[i].off + skip[i].len;
    }
    h = fnv_bytes(h, base + cur, sizeof(*s) - cur);

    for (int i = 0; i < COLO_MAX_NPCS; i++) h = fnv_npc(h, &s->npcs[i]);
    h = fnv_queue(h, &s->player_pending_hits);

    for (int i = 0; i < COLO_ACTION_MASK_SIZE; i++) h = fnv_f32(h, mask[i]);
    return h;
}

static uint64_t run_episode(const SimConfig* cfg, int max_ticks) {
    ColosseumContext ctx;
    ColosseumState s;
    static float mask[COLO_ACTION_MASK_SIZE];
    int actions[COLO_NUM_ACTION_HEADS];

    col_init_context_typed(&ctx);
    ctx.config.start_wave = cfg->public_start_wave - 1;

    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, cfg->env_seed);

    uint64_t rng = cfg->action_seed;
    uint64_t h = FNV_OFFSET;

    for (int t = 0; t < max_ticks; t++) {
        col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
        fill_actions(&s, &rng, actions);
        col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
        h = hash_sim(h, &s, mask);
        if (s.episode_over) break;
    }
    return h;
}

int main(int argc, char** argv) {
    int print = (argc > 1 && strcmp(argv[1], "--print") == 0);
    int n = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));
    int failed = 0;

    for (int i = 0; i < n; i++) {
        uint64_t h = run_episode(&CONFIGS[i], 4000);
        if (print) {
            printf("    {\"%s\", %2d, %luULL, 0x%08lXULL, 0x%016llxULL},\n",
                CONFIGS[i].name, CONFIGS[i].public_start_wave,
                (unsigned long)CONFIGS[i].env_seed,
                (unsigned long)CONFIGS[i].action_seed,
                (unsigned long long)h);
            continue;
        }
        int ok = (h == CONFIGS[i].baseline);
        printf("  %-6s 0x%016llx  %s\n", CONFIGS[i].name,
            (unsigned long long)h, ok ? "PASS" : "FAIL");
        if (!ok) {
            printf("         expected 0x%016llx\n",
                (unsigned long long)CONFIGS[i].baseline);
            failed++;
        }
    }
    if (print) return 0;
    printf("\n%d/%d sim invariants match baseline\n", n - failed, n);
    return failed ? 1 : 0;
}
