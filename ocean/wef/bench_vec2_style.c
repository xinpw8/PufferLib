/*
 * Compare component-wise float math vs functional Vec2 return-chains.
 * Same algebra; measures whether -O3 erases the intermediate structs.
 *
 *   clang -O3 -mavx2 -DNDEBUG -I... ocean/wef/bench_vec2_style.c -lm -o wef_bench_vec2
 *   ./wef_bench_vec2
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { float x, y; } Vec2;
typedef struct { Vec2 p, n; } Sensor;
typedef struct { Vec2 pos; float orientation; } Fish;

#define N_SENSORS 72
#define N_PROBES 256
#define N_SOURCES 16
#define K_COULOMB 8.99e9f
#define CM_TO_M 0.01f
#define EPS_M 1e-5f
#define PI_F 3.14159265358979323846f

static double timespec_seconds(struct timespec t0, struct timespec t1) {
    return (double)(t1.tv_sec - t0.tv_sec) +
        (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static float frand(unsigned* rng) {
    *rng = *rng * 1664525u + 1013904223u;
    return (float)(*rng >> 8) * (1.0f / 16777216.0f);
}

/* -------- functional Vec2 ops (return intermediates) -------- */

static Vec2 vadd(Vec2 a, Vec2 b) {
    return (Vec2){a.x + b.x, a.y + b.y};
}
static Vec2 vsub(Vec2 a, Vec2 b) {
    return (Vec2){a.x - b.x, a.y - b.y};
}
static Vec2 vscale(Vec2 v, float s) {
    return (Vec2){v.x * s, v.y * s};
}
static float vdot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}
static float vlen2(Vec2 v) {
    return v.x * v.x + v.y * v.y;
}
static Vec2 rotate(Vec2 v, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return (Vec2){c * v.x - s * v.y, s * v.x + c * v.y};
}

static Sensor sensor_world_fn(const Sensor* s, const Fish* fish) {
    return (Sensor){
        vadd(rotate(s->p, fish->orientation), fish->pos),
        rotate(s->n, fish->orientation),
    };
}

/* -------- component-wise (current wef style) -------- */

static Sensor sensor_world_cw(const Sensor* s, const Fish* fish) {
    float c = cosf(fish->orientation);
    float sn = sinf(fish->orientation);
    return (Sensor){
        {
            c * s->p.x - sn * s->p.y + fish->pos.x,
            sn * s->p.x + c * s->p.y + fish->pos.y,
        },
        {
            c * s->n.x - sn * s->n.y,
            sn * s->n.x + c * s->n.y,
        },
    };
}

/* Dipole field at probe from sources — functional chain in the accumulate. */
static Vec2 measure_fn(
    Vec2 probe, const Vec2* spos, const Vec2* moments, int n, float range2
) {
    Vec2 pm = vscale(probe, CM_TO_M);
    Vec2 field = {0};
    for (int i = 0; i < n; i++) {
        Vec2 s = vscale(spos[i], CM_TO_M);
        Vec2 d = vsub(pm, s);
        if (vlen2(d) > range2) {
            continue;
        }
        float dist = sqrtf(vlen2(d)) + EPS_M;
        float inv_d2 = 1.0f / (dist * dist);
        float inv_d3 = inv_d2 / dist;
        float mdot = vdot(moments[i], d);
        float k = K_COULOMB * inv_d3;
        float t = 3.0f * mdot * inv_d2;
        field = vadd(field, vscale(vsub(vscale(d, t), moments[i]), k));
    }
    return field;
}

/* Same algebra, scalar accumulators. */
static Vec2 measure_cw(
    Vec2 probe, const Vec2* spos, const Vec2* moments, int n, float range2
) {
    float pmx = probe.x * CM_TO_M;
    float pmy = probe.y * CM_TO_M;
    float field_x = 0.0f;
    float field_y = 0.0f;
    for (int i = 0; i < n; i++) {
        float sx = spos[i].x * CM_TO_M;
        float sy = spos[i].y * CM_TO_M;
        float dx = pmx - sx;
        float dy = pmy - sy;
        if (dx * dx + dy * dy > range2) {
            continue;
        }
        float dist = sqrtf(dx * dx + dy * dy) + EPS_M;
        float inv_d2 = 1.0f / (dist * dist);
        float inv_d3 = inv_d2 / dist;
        float mdot = moments[i].x * dx + moments[i].y * dy;
        float k = K_COULOMB * inv_d3;
        float t = 3.0f * mdot * inv_d2;
        field_x += k * (t * dx - moments[i].x);
        field_y += k * (t * dy - moments[i].y);
    }
    return (Vec2){field_x, field_y};
}

typedef struct {
    const char* name;
    double seconds;
    double ops_per_sec;
    float checksum;
} Result;

static Result bench_sensor(
    const char* name,
    Sensor (*fn)(const Sensor*, const Fish*),
    const Sensor* local,
    const Fish* fish,
    int n_fish,
    int iters
) {
    volatile float sink = 0.0f;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        for (int f = 0; f < n_fish; f++) {
            for (int s = 0; s < N_SENSORS; s++) {
                Sensor w = fn(&local[s], &fish[f]);
                sink += w.p.x + w.p.y + w.n.x + w.n.y;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = timespec_seconds(t0, t1);
    double ops = (double)iters * (double)n_fish * (double)N_SENSORS;
    Result r = {name, sec, ops / sec, (float)sink};
    return r;
}

static Result bench_measure(
    const char* name,
    Vec2 (*fn)(Vec2, const Vec2*, const Vec2*, int, float),
    const Vec2* probes,
    const Vec2* spos,
    const Vec2* moments,
    float range2,
    int iters
) {
    volatile float sink = 0.0f;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        for (int p = 0; p < N_PROBES; p++) {
            Vec2 f = fn(probes[p], spos, moments, N_SOURCES, range2);
            sink += f.x + f.y;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = timespec_seconds(t0, t1);
    double ops = (double)iters * (double)N_PROBES * (double)N_SOURCES;
    Result r = {name, sec, ops / sec, (float)sink};
    return r;
}

static void print_result(Result r, Result* baseline) {
    printf("  %-22s  %8.4fs  %10.1f Mops/s  checksum=%.6g",
        r.name, r.seconds, r.ops_per_sec * 1e-6, r.checksum);
    if (baseline) {
        double ratio = r.seconds / baseline->seconds;
        printf("  (%.3fx %s)",
            ratio, ratio > 1.0 ? "slower" : "faster");
    }
    printf("\n");
}

int main(void) {
    unsigned rng = 1u;
    Sensor local[N_SENSORS];
    for (int s = 0; s < N_SENSORS; s++) {
        float a = 2.0f * PI_F * (float)s / (float)N_SENSORS;
        local[s] = (Sensor){
            {cosf(a), sinf(a)},
            {cosf(a), sinf(a)},
        };
    }
    enum { N_FISH = 64 };
    Fish fish[N_FISH];
    for (int f = 0; f < N_FISH; f++) {
        fish[f].pos = (Vec2){frand(&rng) * 70.0f, frand(&rng) * 70.0f};
        fish[f].orientation = frand(&rng) * 2.0f * PI_F - PI_F;
    }
    Vec2 probes[N_PROBES];
    for (int p = 0; p < N_PROBES; p++) {
        probes[p] = (Vec2){frand(&rng) * 70.0f, frand(&rng) * 70.0f};
    }
    Vec2 spos[N_SOURCES];
    Vec2 moments[N_SOURCES];
    for (int i = 0; i < N_SOURCES; i++) {
        spos[i] = (Vec2){frand(&rng) * 70.0f, frand(&rng) * 70.0f};
        moments[i] = (Vec2){
            (frand(&rng) - 0.5f) * 1e-23f,
            (frand(&rng) - 0.5f) * 1e-23f,
        };
    }
    float range_m = 70.0f * CM_TO_M;
    float range2 = range_m * range_m;

    /* Correctness: bit-level or near for both styles on same inputs. */
    float max_abs_sw = 0.0f;
    for (int f = 0; f < N_FISH; f++) {
        for (int s = 0; s < N_SENSORS; s++) {
            Sensor a = sensor_world_cw(&local[s], &fish[f]);
            Sensor b = sensor_world_fn(&local[s], &fish[f]);
            max_abs_sw = fmaxf(max_abs_sw, fabsf(a.p.x - b.p.x));
            max_abs_sw = fmaxf(max_abs_sw, fabsf(a.p.y - b.p.y));
            max_abs_sw = fmaxf(max_abs_sw, fabsf(a.n.x - b.n.x));
            max_abs_sw = fmaxf(max_abs_sw, fabsf(a.n.y - b.n.y));
        }
    }
    float max_abs_m = 0.0f;
    for (int p = 0; p < N_PROBES; p++) {
        Vec2 a = measure_cw(probes[p], spos, moments, N_SOURCES, range2);
        Vec2 b = measure_fn(probes[p], spos, moments, N_SOURCES, range2);
        max_abs_m = fmaxf(max_abs_m, fabsf(a.x - b.x));
        max_abs_m = fmaxf(max_abs_m, fabsf(a.y - b.y));
    }
    printf("vec2 style microbench (-O3)\n");
    printf("  sensor_world max |cw-fn| = %.6g\n", max_abs_sw);
    printf("  measure      max |cw-fn| = %.6g\n\n", max_abs_m);

    const int sw_iters = 8000;
    const int m_iters = 4000;

    /* Warmup */
    bench_sensor("warmup", sensor_world_cw, local, fish, N_FISH, 200);
    bench_measure("warmup", measure_cw, probes, spos, moments, range2, 100);

    printf("sensor_world  (iters=%d fish=%d sensors=%d)\n",
        sw_iters, N_FISH, N_SENSORS);
    Result sw_cw = bench_sensor(
        "component-wise", sensor_world_cw, local, fish, N_FISH, sw_iters
    );
    Result sw_fn = bench_sensor(
        "functional-chain", sensor_world_fn, local, fish, N_FISH, sw_iters
    );
    print_result(sw_cw, NULL);
    print_result(sw_fn, &sw_cw);

    printf("\nmeasure dipole loop  (iters=%d probes=%d sources=%d)\n",
        m_iters, N_PROBES, N_SOURCES);
    Result m_cw = bench_measure(
        "component-wise", measure_cw, probes, spos, moments, range2, m_iters
    );
    Result m_fn = bench_measure(
        "functional-chain", measure_fn, probes, spos, moments, range2, m_iters
    );
    print_result(m_cw, NULL);
    print_result(m_fn, &m_cw);

    /* Second pass, swap order to catch cache asymmetry */
    printf("\n(second pass, functional first)\n");
    Result sw_fn2 = bench_sensor(
        "functional-chain", sensor_world_fn, local, fish, N_FISH, sw_iters
    );
    Result sw_cw2 = bench_sensor(
        "component-wise", sensor_world_cw, local, fish, N_FISH, sw_iters
    );
    print_result(sw_fn2, NULL);
    print_result(sw_cw2, &sw_fn2);
    Result m_fn2 = bench_measure(
        "functional-chain", measure_fn, probes, spos, moments, range2, m_iters
    );
    Result m_cw2 = bench_measure(
        "component-wise", measure_cw, probes, spos, moments, range2, m_iters
    );
    print_result(m_fn2, NULL);
    print_result(m_cw2, &m_fn2);

    printf("\nsummary (avg of both orderings, ratio = fn/cw time):\n");
    double sw_ratio =
        0.5 * (sw_fn.seconds / sw_cw.seconds + sw_fn2.seconds / sw_cw2.seconds);
    double m_ratio =
        0.5 * (m_fn.seconds / m_cw.seconds + m_fn2.seconds / m_cw2.seconds);
    printf("  sensor_world  fn/cw = %.4f  (%s)\n",
        sw_ratio, sw_ratio > 1.02 ? "fn slower" :
                  sw_ratio < 0.98 ? "fn faster" : "≈ same");
    printf("  measure loop  fn/cw = %.4f  (%s)\n",
        m_ratio, m_ratio > 1.02 ? "fn slower" :
                 m_ratio < 0.98 ? "fn faster" : "≈ same");
    return 0;
}
