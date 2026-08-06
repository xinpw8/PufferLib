/* Wall-clock cost of one colosseum env step, with no profiler marks compiled in.
 *
 * Exists to reconcile two numbers that disagree by 5x: the in-env profiler reports ~24400
 * ns/env-step, an earlier campaign recorded 4570. Both cannot be right, and every optimisation
 * decision downstream is priced against whichever one is.
 *
 * Covers exactly what puf_step covers: encounter step, observation write, action mask write.
 *
 * Equip heads are PINNED. The env memoises on a loadout signature, and uniform random actions
 * churn every equip head each tick, which is a genuine signature change any correct memo must
 * miss -- that artifact once produced a bogus "87% of env time is gear/DPT" conclusion.
 *
 * Build with -std=gnu11: plain c11 hides clock_gettime on glibc and _POSIX_C_SOURCE hides
 * snprintf on macOS. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ocean/osrs/encounters/encounter_colosseum.h"

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static inline uint64_t splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void fill_actions(const ColosseumState* s, uint64_t* rng,
        int actions[COLO_NUM_ACTION_HEADS]) {
    for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) {
        int equip = h >= COLO_HEAD_EQUIP_BASE && h < COLO_HEAD_EQUIP_BASE + NUM_GEAR_SLOTS;
        actions[h] = equip ? 0 : (int)(splitmix64(rng) % (uint64_t)COLO_ACTION_DIMS[h]);
    }
    if (s->modifiers.draft_pending) {
        actions[COLO_HEAD_PRIMARY] = 0;
        actions[COLO_HEAD_MODIFIER_SELECT] =
            1 + (int)(splitmix64(rng) % COLO_MODIFIER_DRAFT_OPTIONS);
    }
}

typedef struct { double step, obs, mask, total; long n; } Bench;

/* STREAM-triad-style scan: the machine's practical DRAM ceiling as a function of thread count.
 * If the env's parallel efficiency curve has the same shape, the env is bandwidth-bound and
 * not, say, lock-bound or oversubscribed. */
static void bandwidth_scan(void) {
    const size_t N = 24u << 20;   /* 192 MB per array, far past the 36 MiB L3 */
    double* a = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* c = (double*)malloc(N * sizeof(double));
    if (!a || !b || !c) { printf("  (allocation failed)\n"); return; }
    for (size_t i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    printf("memory bandwidth ceiling (triad, 3 x %zu MB)\n", N * sizeof(double) / (1u << 20));
    printf("  %8s %14s %12s\n", "threads", "GB/s", "vs 1 thread");
    double one = 0.0;
    int tc[] = {1, 2, 4, 8, 16, 24, 32};
    for (int i = 0; i < (int)(sizeof(tc)/sizeof(tc[0])); i++) {
        int T = tc[i];
        double t0 = now_ns();
        for (int rep = 0; rep < 4; rep++) {
#ifdef _OPENMP
            #pragma omp parallel for num_threads(T) schedule(static)
#endif
            for (size_t j = 0; j < N; j++) c[j] = a[j] + 3.0 * b[j];
        }
        double sec = (now_ns() - t0) / 1e9;
        double gbs = 4.0 * 3.0 * N * sizeof(double) / sec / 1e9;
        if (i == 0) one = gbs;
        printf("  %8d %14.1f %11.2fx\n", T, gbs, gbs / one);
    }
    printf("\n");
    free(a); free(b); free(c);
}

/* Same as run_multi but skips the observation and mask writes, removing ~7 KB of stores per
 * step while leaving the simulation identical. If the env is bandwidth-bound, dropping those
 * bytes should visibly improve parallel efficiency, not just lower the single-thread cost. */
static Bench run_multi_sim_only(int num_envs, int steps_per_env) {
    ColosseumContext* ctxs = (ColosseumContext*)calloc(num_envs, sizeof(ColosseumContext));
    ColosseumState* ss = (ColosseumState*)calloc(num_envs, sizeof(ColosseumState));
    int actions[COLO_NUM_ACTION_HEADS];
    Bench b = {0};
    uint64_t rng = 0xABCDEF01ULL;
    for (int e = 0; e < num_envs; e++) {
        col_init_context_typed(&ctxs[e]);
        col_reset_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e], (uint32_t)(e + 1));
    }
    double t0 = now_ns();
    for (int rep = 0; rep < steps_per_env; rep++) {
        for (int e = 0; e < num_envs; e++) {
            if (ss[e].episode_over)
                col_reset_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e],
                    (uint32_t)(e * 7919 + rep + 1));
            fill_actions(&ss[e], &rng, actions);
            col_step_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e], actions);
            b.n++;
        }
    }
    b.total = now_ns() - t0;
    free(ctxs); free(ss);
    return b;
}

/* Steps N independent envs round-robin, which is what the trainer does. One env fits in L2;
 * N of them do not, and the gap between those two numbers is the working-set cost that a
 * single-env benchmark cannot see. */
static Bench run_multi(int num_envs, int steps_per_env) {
    ColosseumContext* ctxs = (ColosseumContext*)calloc(num_envs, sizeof(ColosseumContext));
    ColosseumState* ss = (ColosseumState*)calloc(num_envs, sizeof(ColosseumState));
    static float obs[COLO_NUM_OBS];
    static float mask[COLO_ACTION_MASK_SIZE];
    int actions[COLO_NUM_ACTION_HEADS];
    Bench b = {0};
    uint64_t rng = 0xABCDEF01ULL;

    for (int e = 0; e < num_envs; e++) {
        col_init_context_typed(&ctxs[e]);
        col_reset_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e], (uint32_t)(e + 1));
    }
    for (int rep = 0; rep < 40; rep++) {
        for (int e = 0; e < num_envs; e++) {
            fill_actions(&ss[e], &rng, actions);
            col_step_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e], actions);
        }
    }

    double t0 = now_ns();
    for (int rep = 0; rep < steps_per_env; rep++) {
        for (int e = 0; e < num_envs; e++) {
            if (ss[e].episode_over)
                col_reset_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e],
                    (uint32_t)(e * 7919 + rep + 1));
            fill_actions(&ss[e], &rng, actions);
            col_step_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e], actions);
            col_write_obs_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e], obs);
            col_write_mask_ctx((EncounterState*)&ss[e], (EncounterContext*)&ctxs[e], mask);
            b.n++;
        }
    }
    b.total = now_ns() - t0;
    free(ctxs); free(ss);
    return b;
}

static Bench run(int warmup_steps, int measured_steps) {
    ColosseumContext ctx;
    ColosseumState s;
    static float obs[COLO_NUM_OBS];
    static float mask[COLO_ACTION_MASK_SIZE];
    int actions[COLO_NUM_ACTION_HEADS];
    Bench b = {0};

    col_init_context_typed(&ctx);
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 12345u);
    uint64_t rng = 0xBEEF1234ULL;

    for (int i = 0; i < warmup_steps + measured_steps; i++) {
        int measuring = i >= warmup_steps;
        if (s.episode_over) {
            col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, (uint32_t)(1000 + i));
        }
        fill_actions(&s, &rng, actions);

        double t0 = now_ns();
        col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
        double t1 = now_ns();
        col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
        double t2 = now_ns();
        col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
        double t3 = now_ns();

        if (measuring) {
            b.step += t1 - t0;
            b.obs  += t2 - t1;
            b.mask += t3 - t2;
            b.total += t3 - t0;
            b.n++;
        }
    }
    return b;
}

int main(void) {
    /* One untimed pass so the static arena/LOS tables are built before anything is measured. */
    run(200, 1);

#ifdef _OPENMP
    /* The trainer runs 8192 envs across 32 threads and its profiler reports ~7x the
     * single-threaded cost per step. Neither cache pressure nor profiler overhead accounts
     * for that, so scan thread count directly with the env count held fixed. */
    bandwidth_scan();
    printf("thread scan: 8192 envs, %d steps each\n", 40);
    printf("  %8s %12s %12s %12s\n", "threads", "ns/step", "speedup", "efficiency");
    double t1 = 0.0;
    int tcounts[] = {1, 2, 4, 8, 16, 24, 32};
    for (int i = 0; i < (int)(sizeof(tcounts)/sizeof(tcounts[0])); i++) {
        int T = tcounts[i];
        int per_thread = 8192 / T;
        double t0 = now_ns();
        long total = 0;
        #pragma omp parallel num_threads(T) reduction(+:total)
        {
            Bench m = run_multi(per_thread, 40);
            total += m.n;
        }
        double wall = now_ns() - t0;
        double ns = wall * T / (double)total;
        double wall_ns = wall / (double)total;
        if (i == 0) t1 = wall_ns;
        printf("  %8d %12.1f %11.2fx %11.0f%%\n", T, ns, t1 / wall_ns,
            100.0 * (t1 / wall_ns) / T);
    }
    printf("\n  sim only, no obs/mask writes (~7 KB/step less traffic)\n");
    printf("  %8s %12s %12s %12s\n", "threads", "ns/step", "speedup", "efficiency");
    double s1 = 0.0;
    for (int i = 0; i < (int)(sizeof(tcounts)/sizeof(tcounts[0])); i++) {
        int T = tcounts[i];
        int per_thread = 8192 / T;
        double t0 = now_ns();
        long total = 0;
        #pragma omp parallel num_threads(T) reduction(+:total)
        {
            Bench m = run_multi_sim_only(per_thread, 40);
            total += m.n;
        }
        double wall_ns = (now_ns() - t0) / (double)total;
        if (i == 0) s1 = wall_ns;
        printf("  %8d %12.1f %11.2fx %11.0f%%\n", T, wall_ns * T, s1 / wall_ns,
            100.0 * (s1 / wall_ns) / T);
    }
    printf("\n");
#endif

    printf("working-set scan: one thread, N envs round-robin\n");
    printf("  %6s %12s %14s %12s\n", "envs", "ns/step", "state MB", "vs 1 env");
    double base = 0.0;
    int counts[] = {1, 8, 64, 256, 1024, 4096, 8192};
    for (int i = 0; i < (int)(sizeof(counts)/sizeof(counts[0])); i++) {
        int n = counts[i];
        int per = n <= 64 ? 4000 : (n <= 1024 ? 400 : 60);
        Bench m = run_multi(n, per);
        double ns = m.total / m.n;
        if (i == 0) base = ns;
        printf("  %6d %12.1f %14.1f %11.2fx\n", n, ns,
            (double)n * (sizeof(ColosseumState) + sizeof(ColosseumContext)) / 1e6, ns / base);
    }
    printf("\n  sizeof(ColosseumState)=%zu  sizeof(ColosseumContext)=%zu\n\n",
        sizeof(ColosseumState), sizeof(ColosseumContext));

    Bench b = run(2000, 200000);
    printf("colosseum step benchmark, no profiler marks, equip heads pinned\n");
    printf("  measured steps        %ld\n", b.n);
    printf("  col_step_ctx          %8.1f ns/step\n", b.step / b.n);
    printf("  col_write_obs_ctx     %8.1f ns/step\n", b.obs / b.n);
    printf("  col_write_mask_ctx    %8.1f ns/step\n", b.mask / b.n);
    printf("  TOTAL                 %8.1f ns/step\n", b.total / b.n);
    printf("\n  obs is %d floats, mask is %d floats\n",
        COLO_NUM_OBS, COLO_ACTION_MASK_SIZE);
    return 0;
}
