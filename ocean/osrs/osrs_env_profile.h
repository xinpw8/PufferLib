#include <stdlib.h>
#include <time.h>

/* Defined the same way in osrs_pathfinding.h, but this header is a standalone template that
 * env wrappers include on their own, so it cannot depend on that one being pulled in first. */
#ifndef OSRS_ENV_PROFILE_THREAD_LOCAL
#if defined(__cplusplus)
#define OSRS_ENV_PROFILE_THREAD_LOCAL thread_local
#else
#define OSRS_ENV_PROFILE_THREAD_LOCAL _Thread_local
#endif
#endif

#define OSRS_ENV_PROFILE_CAT_(a, b) a##b
#define OSRS_ENV_PROFILE_CAT(a, b) OSRS_ENV_PROFILE_CAT_(a, b)
#define OSRS_ENV_PROFILE_FN(suffix) OSRS_ENV_PROFILE_CAT(OSRS_ENV_PROFILE_PREFIX, suffix)
#define OSRS_ENV_PROFILE_G(suffix) \
    OSRS_ENV_PROFILE_CAT(OSRS_ENV_PROFILE_CAT(g_, OSRS_ENV_PROFILE_PREFIX), suffix)

typedef enum {
#define OSRS_ENV_PROFILE_X(slot, label) slot,
    OSRS_ENV_PROFILE_SLOTS(OSRS_ENV_PROFILE_X)
#undef OSRS_ENV_PROFILE_X
    OSRS_ENV_PROFILE_COUNT,
} OSRS_ENV_PROFILE_SLOT_TYPE;

static int OSRS_ENV_PROFILE_G(_profile_enabled) = -1;

/* Per-thread accumulators, summed only at report time.
 *
 * This was a single shared array updated under `#pragma omp atomic`, hit ~40 times per env
 * step by every worker thread. All slots fit in a handful of cache lines, so each mark became
 * a contended read-modify-write and the profiler spent most of its time measuring itself:
 * reported cost was ~5.7x the wall-clock truth, and inflated worst for the slots that were
 * marked most often -- which distorts the shares, not just the total.
 *
 * Each thread gets its own cache-line-aligned row, so marks are now plain local adds. */
#define OSRS_ENV_PROFILE_MAX_THREADS 128
#define OSRS_ENV_PROFILE_ROW_DOUBLES \
    (((OSRS_ENV_PROFILE_COUNT + 7) / 8) * 8)

static double OSRS_ENV_PROFILE_G(_profile_ms)
    [OSRS_ENV_PROFILE_MAX_THREADS][OSRS_ENV_PROFILE_ROW_DOUBLES];
static int OSRS_ENV_PROFILE_G(_profile_next_tid);

static int OSRS_ENV_PROFILE_FN(_profile_tid)(void) {
    static OSRS_ENV_PROFILE_THREAD_LOCAL int tid = -1;
    if (tid < 0) {
        int slot;
        #pragma omp atomic capture
        slot = OSRS_ENV_PROFILE_G(_profile_next_tid)++;
        if (slot >= OSRS_ENV_PROFILE_MAX_THREADS) abort();
        tid = slot;
    }
    return tid;
}

static const char* OSRS_ENV_PROFILE_G(_profile_names)[OSRS_ENV_PROFILE_COUNT] = {
#define OSRS_ENV_PROFILE_X(slot, label) label,
    OSRS_ENV_PROFILE_SLOTS(OSRS_ENV_PROFILE_X)
#undef OSRS_ENV_PROFILE_X
};

static int OSRS_ENV_PROFILE_FN(_profile_enabled)(void) {
    if (OSRS_ENV_PROFILE_G(_profile_enabled) < 0) {
        const char* text = getenv(OSRS_ENV_PROFILE_ENV_VAR);
        OSRS_ENV_PROFILE_G(_profile_enabled) =
            (text && text[0] && text[0] != '0') ? 1 : 0;
    }
    return OSRS_ENV_PROFILE_G(_profile_enabled);
}

static double OSRS_ENV_PROFILE_FN(_profile_now_ms)(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void OSRS_ENV_PROFILE_FN(_profile_add)(int slot, double ms) {
    if (slot < 0 || slot >= OSRS_ENV_PROFILE_COUNT) abort();
    OSRS_ENV_PROFILE_G(_profile_ms)[OSRS_ENV_PROFILE_FN(_profile_tid)()][slot] += ms;
}

static void OSRS_ENV_PROFILE_FN(_profile_mark)(int enabled, double* last_ms, int slot) {
    if (!enabled) return;
    double now = OSRS_ENV_PROFILE_FN(_profile_now_ms)();
    OSRS_ENV_PROFILE_FN(_profile_add)(slot, now - *last_ms);
    *last_ms = now;
}

OSRS_ENV_PROFILE_EXPORT int OSRS_ENV_PROFILE_FN(_env_profile_count)(void) {
    return OSRS_ENV_PROFILE_FN(_profile_enabled)() ? OSRS_ENV_PROFILE_COUNT : 0;
}

OSRS_ENV_PROFILE_EXPORT const char* OSRS_ENV_PROFILE_FN(_env_profile_name)(int slot) {
    if (slot < 0 || slot >= OSRS_ENV_PROFILE_COUNT) abort();
    return OSRS_ENV_PROFILE_G(_profile_names)[slot];
}

/* Called from the reporting thread between epochs, so a plain sum over the per-thread rows is
 * enough. Workers may still be marking; a torn double only skews one report. */
OSRS_ENV_PROFILE_EXPORT double OSRS_ENV_PROFILE_FN(_env_profile_read_reset_ms)(int slot) {
    if (slot < 0 || slot >= OSRS_ENV_PROFILE_COUNT) abort();
    double value = 0.0;
    int threads = OSRS_ENV_PROFILE_G(_profile_next_tid);
    if (threads > OSRS_ENV_PROFILE_MAX_THREADS) threads = OSRS_ENV_PROFILE_MAX_THREADS;
    for (int t = 0; t < threads; t++) {
        value += OSRS_ENV_PROFILE_G(_profile_ms)[t][slot];
        OSRS_ENV_PROFILE_G(_profile_ms)[t][slot] = 0.0;
    }
    return value;
}

#undef OSRS_ENV_PROFILE_CAT_
#undef OSRS_ENV_PROFILE_CAT
#undef OSRS_ENV_PROFILE_FN
#undef OSRS_ENV_PROFILE_G
#undef OSRS_ENV_PROFILE_PREFIX
#undef OSRS_ENV_PROFILE_COUNT
#undef OSRS_ENV_PROFILE_SLOT_TYPE
#undef OSRS_ENV_PROFILE_ENV_VAR
#undef OSRS_ENV_PROFILE_EXPORT
#undef OSRS_ENV_PROFILE_SLOTS
