/*
 * Field-path work breakdown for current WEF ranges (fish 100 / food 5 / walls).
 *
 * Counts real probe/source interactions from a stepped env pool, then
 * microbenches unit costs (range-check miss vs Coulomb hit) to estimate time.
 *
 *   clang -O3 -mavx2 -DNDEBUG -I./raylib... ocean/wef/bench_field_breakdown.c \
 *     ... -o wef_field_breakdown && ./wef_field_breakdown
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#include "wef.h"

#define TOTAL_FISH 2048
#define NUM_FISH 4
#define STEPS 16
#define WARMUP 2

typedef struct {
    uint64_t calls;
    uint64_t calls_walls;
    uint64_t calls_no_walls;

    uint64_t mono_checked;
    uint64_t mono_kept;
    uint64_t mono_pruned;

    uint64_t dip_fish_checked;
    uint64_t dip_fish_kept;
    uint64_t dip_fish_pruned;

    uint64_t dip_food_checked;
    uint64_t dip_food_kept;
    uint64_t dip_food_pruned;

    /* Coulomb evals = kept direct + (kept * num_near_walls) images */
    uint64_t coulomb_mono_direct;
    uint64_t coulomb_mono_wall;
    uint64_t coulomb_dip_fish_direct;
    uint64_t coulomb_dip_fish_wall;
    uint64_t coulomb_dip_food_direct;
    uint64_t coulomb_dip_food_wall;

    uint64_t wall_slots;  /* sum of num_near_walls over calls */
} FieldStats;

static FieldStats g_stats;

static inline uint64_t rdtsc_start(void) {
    _mm_lfence();
    return __rdtsc();
}
static inline uint64_t rdtsc_end(void) {
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

/* Instrumented copy of measure_field (same algebra; stats only). */
static Vec2 measure_field_stats(
    Env* env, Vec2 probe,
    const Mono* mono, int n_mono,
    const Dipole* dip, int n_dip, int n_agent_dipoles,
    float wall_range_cm
) {
    FieldStats* st = &g_stats;
    st->calls++;

    float pmx = probe.x * CM_TO_M;
    float pmy = probe.y * CM_TO_M;
    float arena_mx = env->arena_size_x * CM_TO_M;
    float arena_my = env->arena_size_y * CM_TO_M;
    float wall_range_m = wall_range_cm * CM_TO_M;
    float fr = env->field_fish_range_cm * CM_TO_M;
    float food_r = env->field_food_range_cm * CM_TO_M;
    float fish_range2 = fr * fr;
    float food_range2 = food_r * food_r;
    float eps_m = FIELD_EPS_M;

    float wall_dist[NUM_WALLS] = {
        pmx, arena_mx - pmx, pmy, arena_my - pmy
    };
    int near_walls[NUM_WALLS];
    int num_near_walls = 0;
    for (int wall = 0; wall < NUM_WALLS; wall++) {
        if (wall_dist[wall] <= wall_range_m) {
            near_walls[num_near_walls++] = wall;
        }
    }
    if (num_near_walls > 0) {
        st->calls_walls++;
        st->wall_slots += (uint64_t)num_near_walls;
    } else {
        st->calls_no_walls++;
    }

    Mono mono_in[WEF_MAX_MONO];
    Dipole dip_in[WEF_MAX_DIP];
    /* Track how many of dip_in are fish vs food for wall attribution */
    int n_dip_fish_in = 0;
    int n_mono_in = 0;
    int n_dip_in = 0;
    float field_x = 0.0f;
    float field_y = 0.0f;

    for (int i = 0; i < n_mono; i++) {
        st->mono_checked++;
        float sx = mono[i].p.x * CM_TO_M;
        float sy = mono[i].p.y * CM_TO_M;
        float dx = pmx - sx;
        float dy = pmy - sy;
        if (dx * dx + dy * dy > fish_range2) {
            st->mono_pruned++;
            continue;
        }
        st->mono_kept++;
        mono_in[n_mono_in++] = (Mono){{sx, sy}, mono[i].q};
        float dist = sqrtf(dx * dx + dy * dy) + eps_m;
        float inv_d = 1.0f / dist;
        float w = K_COULOMB * mono[i].q * inv_d * inv_d * inv_d;
        field_x += dx * w;
        field_y += dy * w;
        st->coulomb_mono_direct++;
    }

    for (int i = 0; i < n_dip; i++) {
        int is_fish = i < n_agent_dipoles;
        if (is_fish) {
            st->dip_fish_checked++;
        } else {
            st->dip_food_checked++;
        }
        float sx = dip[i].p.x * CM_TO_M;
        float sy = dip[i].p.y * CM_TO_M;
        float range2 = is_fish ? fish_range2 : food_range2;
        float dx = pmx - sx;
        float dy = pmy - sy;
        if (dx * dx + dy * dy > range2) {
            if (is_fish) {
                st->dip_fish_pruned++;
            } else {
                st->dip_food_pruned++;
            }
            continue;
        }
        if (is_fish) {
            st->dip_fish_kept++;
            n_dip_fish_in++;
            st->coulomb_dip_fish_direct++;
        } else {
            st->dip_food_kept++;
            st->coulomb_dip_food_direct++;
        }
        dip_in[n_dip_in++] = (Dipole){{sx, sy}, dip[i].m};
        float dist = sqrtf(dx * dx + dy * dy) + eps_m;
        float inv_d2 = 1.0f / (dist * dist);
        float inv_d3 = inv_d2 / dist;
        float mdot = dip[i].m.x * dx + dip[i].m.y * dy;
        float k = K_COULOMB * inv_d3;
        float t = 3.0f * mdot * inv_d2;
        field_x += k * (t * dx - dip[i].m.x);
        field_y += k * (t * dy - dip[i].m.y);
    }

    for (int w = 0; w < num_near_walls; w++) {
        int wall = near_walls[w];
        for (int k = 0; k < n_mono_in; k++) {
            float sx = mono_in[k].p.x;
            float sy = mono_in[k].p.y;
            if (wall <= WALL_RIGHT) {
                sx = wall == WALL_LEFT ? -sx : 2.0f * arena_mx - sx;
            } else {
                sy = wall == WALL_BOTTOM ? -sy : 2.0f * arena_my - sy;
            }
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps_m;
            float inv_d = 1.0f / dist;
            float wt = K_COULOMB * mono_in[k].q * inv_d * inv_d * inv_d;
            field_x += dx * wt;
            field_y += dy * wt;
            st->coulomb_mono_wall++;
        }
        for (int k = 0; k < n_dip_in; k++) {
            float sx = dip_in[k].p.x;
            float sy = dip_in[k].p.y;
            if (wall <= WALL_RIGHT) {
                sx = wall == WALL_LEFT ? -sx : 2.0f * arena_mx - sx;
            } else {
                sy = wall == WALL_BOTTOM ? -sy : 2.0f * arena_my - sy;
            }
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps_m;
            float inv_d2 = 1.0f / (dist * dist);
            float inv_d3 = inv_d2 / dist;
            float mx = dip_in[k].m.x;
            float my = dip_in[k].m.y;
            float mdot = mx * dx + my * dy;
            float kcoef = K_COULOMB * inv_d3;
            float t = 3.0f * mdot * inv_d2;
            field_x += kcoef * (t * dx - mx);
            field_y += kcoef * (t * dy - my);
            if (k < n_dip_fish_in) {
                st->coulomb_dip_fish_wall++;
            } else {
                st->coulomb_dip_food_wall++;
            }
        }
    }
    return (Vec2){field_x, field_y};
}

/* Force the obs path to use instrumented measure (shadow of compute_observations).
 * We step the real env for dynamics, then re-run field accounting by calling
 * a local copy of the sensor loops. */

static void account_step_fields(Wef* env) {
    /* Mirror scene build + sensor measures without writing obs noise RNG thrash
     * differently — use same measure_field_stats. */

    Mono eod[WEF_MAX_MONO];
    int n_eod = 0;
    float body_scale = conductor_scale(BODY_RADIUS_CM);
    float food_scale = conductor_scale(FOOD_RADIUS_CM);
    float max_moment = EOD_CHARGE_C * BODY_RADIUS_CM;

    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float c = cosf(agent->orientation);
        float s = sinf(agent->orientation);
        float q = agent->emits_eod ? EOD_CHARGE_C : 0.0f;
        agent->eod_pos[0] = (Vec2){
            c * EOD_POLE_OFFSET_CM + agent->pos.x,
            s * EOD_POLE_OFFSET_CM + agent->pos.y,
        };
        agent->eod_pos[1] = (Vec2){
            -c * EOD_POLE_OFFSET_CM + agent->pos.x,
            -s * EOD_POLE_OFFSET_CM + agent->pos.y,
        };
        agent->eod_charge[0] = q;
        agent->eod_charge[1] = -q;
        agent->intrinsic_moment =
            (Vec2){c * INTRINSIC_MOMENT_C_M, s * INTRINSIC_MOMENT_C_M};
        eod[n_eod++] = (Mono){agent->eod_pos[0], q};
        eod[n_eod++] = (Mono){agent->eod_pos[1], -q};
    }

    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float moment_x = 0.0f;
        float moment_y = 0.0f;
        if (!agent->emits_eod) {
            Vec2 f = measure_field_stats(
                env, agent->pos, eod, n_eod, NULL, 0, 0, 0.0f
            );
            moment_x = f.x * body_scale;
            moment_y = f.y * body_scale;
            float mag = sqrtf(moment_x * moment_x + moment_y * moment_y);
            if (mag > max_moment) {
                float s = max_moment / mag;
                moment_x *= s;
                moment_y *= s;
            }
        }
        agent->induced_moment = (Vec2){moment_x, moment_y};
    }
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) {
            env->food[i].intrinsic_moment = (Vec2){0};
            env->food[i].induced_moment = (Vec2){0};
            continue;
        }
        float fc = cosf(env->food[i].orientation);
        float fs = sinf(env->food[i].orientation);
        env->food[i].intrinsic_moment = (Vec2){
            -fs * FOOD_INTRINSIC_MOMENT_C_M,
            fc * FOOD_INTRINSIC_MOMENT_C_M,
        };
        Vec2 f = measure_field_stats(
            env, env->food[i].pos, eod, n_eod, NULL, 0, 0, 0.0f
        );
        env->food[i].induced_moment =
            (Vec2){f.x * food_scale, f.y * food_scale};
    }

    Dipole induced[MAX_AGENTS + MAX_FOOD];
    Dipole intrinsic[MAX_AGENTS + MAX_FOOD];
    int n_induced = 0;
    int n_intrinsic = 0;
    for (int a = 0; a < env->num_agents; a++) {
        induced[n_induced++] =
            (Dipole){env->fish[a].pos, env->fish[a].induced_moment};
        intrinsic[n_intrinsic++] =
            (Dipole){env->fish[a].pos, env->fish[a].intrinsic_moment};
    }
    for (int f = 0; f < env->num_food; f++) {
        if (!env->food[f].active) {
            continue;
        }
        induced[n_induced++] =
            (Dipole){env->food[f].pos, env->food[f].induced_moment};
        intrinsic[n_intrinsic++] =
            (Dipole){env->food[f].pos, env->food[f].intrinsic_moment};
    }

    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        for (int s = 0; s < NUM_MORMYROMASTS; s++) {
            Sensor w = sensor_world(&g_morm[s], agent);
            measure_field_stats(
                env, w.p, NULL, 0, induced, n_induced, env->num_agents,
                env->reflection_wall_range_cm
            );
        }
        for (int s = 0; s < NUM_AMPULLARY; s++) {
            Sensor w = sensor_world(&g_amp[s], agent);
            measure_field_stats(
                env, w.p, NULL, 0, intrinsic, n_intrinsic, env->num_agents,
                env->reflection_wall_range_cm
            );
        }
        for (int other = 0; other < env->num_agents; other++) {
            if (other == i || !env->fish[other].emits_eod) {
                continue;
            }
            Mono e2[2] = {
                {env->fish[other].eod_pos[0], env->fish[other].eod_charge[0]},
                {env->fish[other].eod_pos[1], env->fish[other].eod_charge[1]},
            };
            for (int s = 0; s < NUM_KNOLLEN; s++) {
                Sensor w = sensor_world(&g_knollen[s], agent);
                measure_field_stats(
                    env, w.p, e2, 2, NULL, 0, 0, 0.0f
                );
            }
        }
    }
}

__attribute__((noinline)) static float kernel_food_miss(
    const Vec2* probes, int n_probes, float food_range2, int iters
) {
    float sink = 0.0f;
    for (int n = 0; n < iters; n++) {
        Vec2 probe = probes[n % n_probes];
        float pmx = probe.x * CM_TO_M;
        float pmy = probe.y * CM_TO_M;
        /* many far food sources (always miss) */
        for (int j = 0; j < 16; j++) {
            float sx = (float)(j + 10) * 10.0f; /* meters-ish far */
            float sy = (float)(j + 10) * 10.0f;
            float dx = pmx - sx;
            float dy = pmy - sy;
            if (dx * dx + dy * dy > food_range2) {
                sink += 1.0f;
                continue;
            }
            sink += dx;
        }
    }
    return sink;
}

__attribute__((noinline)) static float kernel_dip_hits(
    const Vec2* probes, const Vec2* spos, const Vec2* moments,
    int n_probes, int n_src, float eps, int iters
) {
    float sink = 0.0f;
    for (int n = 0; n < iters; n++) {
        Vec2 probe = probes[n % n_probes];
        float pmx = probe.x * CM_TO_M;
        float pmy = probe.y * CM_TO_M;
        for (int j = 0; j < 16; j++) {
            Vec2 sp = spos[(n + j) % n_src];
            Vec2 m = moments[(n + j) % n_src];
            float sx = sp.x * CM_TO_M;
            float sy = sp.y * CM_TO_M;
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps;
            float inv_d2 = 1.0f / (dist * dist);
            float inv_d3 = inv_d2 / dist;
            float mdot = m.x * dx + m.y * dy;
            float k = K_COULOMB * inv_d3;
            float t = 3.0f * mdot * inv_d2;
            sink += k * (t * dx - m.x) + k * (t * dy - m.y);
        }
    }
    return sink;
}

__attribute__((noinline)) static float kernel_mono_hits(
    const Vec2* probes, const Vec2* spos, int n_probes, int n_src,
    float eps, int iters
) {
    float sink = 0.0f;
    for (int n = 0; n < iters; n++) {
        Vec2 probe = probes[n % n_probes];
        float pmx = probe.x * CM_TO_M;
        float pmy = probe.y * CM_TO_M;
        for (int j = 0; j < 16; j++) {
            Vec2 sp = spos[(n + j) % n_src];
            float sx = sp.x * CM_TO_M;
            float sy = sp.y * CM_TO_M;
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps;
            float inv_d = 1.0f / dist;
            float w = K_COULOMB * 1e-15f * inv_d * inv_d * inv_d;
            sink += dx * w + dy * w;
        }
    }
    return sink;
}

/* Unit-cost microbench: pruned food check vs full dipole coulomb. */
static void microbench_unit_costs(
    double* cyc_per_food_miss,
    double* cyc_per_dip_hit,
    double* cyc_per_mono_hit
) {
    enum { N = 1 << 18 };
    Vec2 probes[256];
    Vec2 spos[64];
    Vec2 moments[64];
    unsigned rng = 1u;
    for (int i = 0; i < 256; i++) {
        rng = rng * 1664525u + 1013904223u;
        probes[i].x = (float)(rng >> 8) * (70.0f / 16777216.0f);
        rng = rng * 1664525u + 1013904223u;
        probes[i].y = (float)(rng >> 8) * (70.0f / 16777216.0f);
    }
    for (int i = 0; i < 64; i++) {
        rng = rng * 1664525u + 1013904223u;
        spos[i].x = (float)(rng >> 8) * (70.0f / 16777216.0f);
        rng = rng * 1664525u + 1013904223u;
        spos[i].y = (float)(rng >> 8) * (70.0f / 16777216.0f);
        moments[i] = (Vec2){1e-23f, -1e-23f};
    }
    float food_range_m = 5.0f * CM_TO_M;
    float food_range2 = food_range_m * food_range_m;
    float eps = FIELD_EPS_M;
    volatile float sink = 0.0f;

    /* ops per outer iter = 16 */
    uint64_t t0 = rdtsc_start();
    sink += kernel_food_miss(probes, 256, food_range2, N);
    uint64_t t1 = rdtsc_end();
    *cyc_per_food_miss = (double)(t1 - t0) / (double)(N * 16);

    t0 = rdtsc_start();
    sink += kernel_dip_hits(probes, spos, moments, 256, 64, eps, N);
    t1 = rdtsc_end();
    *cyc_per_dip_hit = (double)(t1 - t0) / (double)(N * 16);

    t0 = rdtsc_start();
    sink += kernel_mono_hits(probes, spos, 256, 64, eps, N);
    t1 = rdtsc_end();
    *cyc_per_mono_hit = (double)(t1 - t0) / (double)(N * 16);
    (void)sink;
}

static void configure(Wef* env, unsigned seed, int nfish) {
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", nfish);
    dict_set(&kwargs, "min_arena_width", 70);
    dict_set(&kwargs, "min_arena_height", 70);
    dict_set(&kwargs, "max_arena_width", 70);
    dict_set(&kwargs, "max_arena_height", 70);
    dict_set(&kwargs, "food_distribution", FOOD_RANDOM);
    dict_set(&kwargs, "num_food", 64);
    dict_set(&kwargs, "patch_radius", 6);
    dict_set(&kwargs, "patch_radius_std", 1.5);
    dict_set(&kwargs, "patch_density", 0.001);
    dict_set(&kwargs, "electric_field_radius", 15);
    dict_set(&kwargs, "reflection_wall_range", 100);
    dict_set(&kwargs, "field_fish_range", 100);
    dict_set(&kwargs, "field_food_range", 5);
    dict_set(&kwargs, "episode_length", 4096);
    memset(env, 0, sizeof(*env));
    env->rng = seed ? seed : 1u;
    puf_init(env, &kwargs);
    dict_clear(&kwargs);
}

static float rand_act(unsigned* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

int main(void) {
    int num_envs = TOTAL_FISH / NUM_FISH;
    Wef* envs = calloc((size_t)num_envs, sizeof(Wef));
    unsigned* rngs = calloc((size_t)num_envs, sizeof(unsigned));
    size_t po = (size_t)NUM_FISH * OBS_SIZE;
    size_t pa = (size_t)NUM_FISH * NUM_ATNS;
    obs_t* obs = calloc((size_t)num_envs * po, sizeof(obs_t));
    float* act = calloc((size_t)num_envs * pa, sizeof(float));
    float* rew = calloc((size_t)num_envs * NUM_FISH, sizeof(float));
    float* term = calloc((size_t)num_envs * NUM_FISH, sizeof(float));
    if (!envs || !rngs || !obs || !act || !rew || !term) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (int e = 0; e < num_envs; e++) {
        configure(&envs[e], (unsigned)(e + 1), NUM_FISH);
        for (int i = 0; i < NUM_FISH; i++) {
            envs[e].agents[i].observations = obs + (size_t)e * po + i * OBS_SIZE;
            envs[e].agents[i].actions = act + (size_t)e * pa + i * NUM_ATNS;
            envs[e].agents[i].rewards = rew + (size_t)e * NUM_FISH + i;
            envs[e].agents[i].terminals = term + (size_t)e * NUM_FISH + i;
        }
        rngs[e] = (unsigned)(e + 1) * 0x9e3779b9u;
        puf_reset(&envs[e]);
    }

    for (int t = 0; t < WARMUP; t++) {
        for (int e = 0; e < num_envs; e++) {
            for (int a = 0; a < NUM_FISH; a++) {
                float* ac = envs[e].agents[a].actions;
                for (int k = 0; k < ACTION_SIZE; k++) {
                    ac[k] = rand_act(&rngs[e]);
                }
            }
            puf_step(&envs[e]);
        }
    }

    memset(&g_stats, 0, sizeof(g_stats));
    for (int t = 0; t < STEPS; t++) {
        for (int e = 0; e < num_envs; e++) {
            for (int a = 0; a < NUM_FISH; a++) {
                float* ac = envs[e].agents[a].actions;
                for (int k = 0; k < ACTION_SIZE; k++) {
                    ac[k] = rand_act(&rngs[e]);
                }
            }
            /* dynamics without field (still runs real measure inside puf_step) */
            puf_step(&envs[e]);
            /* recount field path with instrumented measure */
            account_step_fields(&envs[e]);
        }
    }

    FieldStats* s = &g_stats;
    double cyc_miss, cyc_dip, cyc_mono;
    microbench_unit_costs(&cyc_miss, &cyc_dip, &cyc_mono);

    printf("field path breakdown  (fish_range=100 food_range=5 walls=100)\n");
    printf("  pool: %d fish, %d envs, %d accounted steps\n\n",
        TOTAL_FISH, num_envs, STEPS);

    printf("calls:\n");
    printf("  measure_field          %llu\n", (unsigned long long)s->calls);
    printf("  with walls             %llu  (%.1f%%)\n",
        (unsigned long long)s->calls_walls,
        100.0 * (double)s->calls_walls / (double)s->calls);
    printf("  no walls               %llu  (%.1f%%)\n",
        (unsigned long long)s->calls_no_walls,
        100.0 * (double)s->calls_no_walls / (double)s->calls);
    printf("  avg walls / wall-call  %.2f\n\n",
        s->calls_walls ? (double)s->wall_slots / (double)s->calls_walls : 0.0);

    printf("source checks (per probe×source):\n");
    printf("  mono  (EOD poles)  checked=%llu kept=%llu pruned=%llu  keep=%.2f%%\n",
        (unsigned long long)s->mono_checked,
        (unsigned long long)s->mono_kept,
        (unsigned long long)s->mono_pruned,
        s->mono_checked ?
            100.0 * (double)s->mono_kept / (double)s->mono_checked : 0.0);
    printf("  dip fish           checked=%llu kept=%llu pruned=%llu  keep=%.2f%%\n",
        (unsigned long long)s->dip_fish_checked,
        (unsigned long long)s->dip_fish_kept,
        (unsigned long long)s->dip_fish_pruned,
        s->dip_fish_checked ?
            100.0 * (double)s->dip_fish_kept / (double)s->dip_fish_checked : 0.0);
    printf("  dip food           checked=%llu kept=%llu pruned=%llu  keep=%.2f%%\n\n",
        (unsigned long long)s->dip_food_checked,
        (unsigned long long)s->dip_food_kept,
        (unsigned long long)s->dip_food_pruned,
        s->dip_food_checked ?
            100.0 * (double)s->dip_food_kept / (double)s->dip_food_checked : 0.0);

    uint64_t coul_long =
        s->coulomb_mono_direct + s->coulomb_mono_wall +
        s->coulomb_dip_fish_direct + s->coulomb_dip_fish_wall;
    uint64_t coul_food =
        s->coulomb_dip_food_direct + s->coulomb_dip_food_wall;
    uint64_t coul_all = coul_long + coul_food;

    printf("Coulomb evaluations (direct + wall images):\n");
    printf("  mono direct            %llu\n", (unsigned long long)s->coulomb_mono_direct);
    printf("  mono wall images       %llu\n", (unsigned long long)s->coulomb_mono_wall);
    printf("  dip fish direct        %llu\n", (unsigned long long)s->coulomb_dip_fish_direct);
    printf("  dip fish wall images   %llu\n", (unsigned long long)s->coulomb_dip_fish_wall);
    printf("  dip food direct        %llu\n", (unsigned long long)s->coulomb_dip_food_direct);
    printf("  dip food wall images   %llu\n", (unsigned long long)s->coulomb_dip_food_wall);
    printf("  LONG-RANGE total       %llu  (%.1f%% of Coulomb)\n",
        (unsigned long long)coul_long,
        coul_all ? 100.0 * (double)coul_long / (double)coul_all : 0.0);
    printf("  FOOD total             %llu  (%.1f%% of Coulomb)\n\n",
        (unsigned long long)coul_food,
        coul_all ? 100.0 * (double)coul_food / (double)coul_all : 0.0);

    /* Estimated cycle share: miss checks + hits */
    double cyc_food_prune =
        (double)s->dip_food_pruned * cyc_miss;
    /* food hits pay miss-check + coulomb (approx: full hit cost includes convert) */
    double cyc_food_hit =
        (double)s->dip_food_kept * cyc_dip +
        (double)s->coulomb_dip_food_wall * cyc_dip;
    double cyc_long =
        (double)(s->coulomb_mono_direct + s->coulomb_mono_wall) * cyc_mono +
        (double)(s->coulomb_dip_fish_direct + s->coulomb_dip_fish_wall) * cyc_dip +
        /* fish prune checks almost never fire; still count checks as miss cost */
        (double)s->dip_fish_pruned * cyc_miss +
        (double)s->mono_pruned * cyc_miss;
    /* also count fish range checks that hit: included in cyc_dip/mono */
    double cyc_food_total = cyc_food_prune + cyc_food_hit;
    double cyc_all = cyc_long + cyc_food_total;

    printf("unit costs (rdtsc cycles / op, synthetic):\n");
    printf("  food range-check miss  %.1f cyc\n", cyc_miss);
    printf("  dipole Coulomb hit     %.1f cyc\n", cyc_dip);
    printf("  mono Coulomb hit       %.1f cyc\n\n", cyc_mono);

    printf("estimated field arithmetic share (counts × unit costs):\n");
    printf("  LONG-RANGE (fish/EOD + their walls)  %6.1f%%\n",
        cyc_all > 0 ? 100.0 * cyc_long / cyc_all : 0.0);
    printf("  FOOD Coulomb hits (+ food walls)     %6.1f%%\n",
        cyc_all > 0 ? 100.0 * cyc_food_hit / cyc_all : 0.0);
    printf("  FOOD prune checks only (misses)      %6.1f%%\n",
        cyc_all > 0 ? 100.0 * cyc_food_prune / cyc_all : 0.0);
    printf("  FOOD total (prune + hits)            %6.1f%%\n\n",
        cyc_all > 0 ? 100.0 * cyc_food_total / cyc_all : 0.0);

    printf("notes:\n");
    printf("  - account_step_fields mirrors induce + morm/amp/knollen measures.\n");
    printf("  - puf_step also runs the real measure_field (not double-counted here).\n");
    printf("  - wall images multiply kept sources by ~4 on wall-enabled calls.\n");

    free(obs); free(act); free(rew); free(term);
    free(envs); free(rngs);
    return 0;
}
