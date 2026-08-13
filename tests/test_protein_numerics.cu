// test_protein_numerics.cu -- Numerical regression for protein.cu (production API)
//
// Build:
//   nvcc -O2 -o test_protein_numerics tests/test_protein_numerics.cu \
//        -I src/ -lcublas -lcusolver -lcurand
//
// Usage:
//   ./test_protein_numerics --gen tests/golden_protein_numerics.txt
//   ./test_protein_numerics tests/golden_protein_numerics.txt
//   ./test_protein_numerics --tol 1e-4 tests/golden_protein_numerics.txt

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Includers supply BLOCK_SIZE and block_reduce_sum (pufferl does in production).
#define BLOCK_SIZE 256
#ifndef PUF_BLOCK_REDUCE_SUM
#define PUF_BLOCK_REDUCE_SUM
__device__ __forceinline__ void block_reduce_sum(
        float *smem, float *out, int tid, int nthreads, int nchan) {
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + tid] += smem[c * nthreads + tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int c = 0; c < nchan; c++) {
            out[c] = smem[c * nthreads];
        }
    }
}
#endif

#include "protein.cu"

#define MAX_RECORDS 4096
#define NAME_LEN 128

typedef struct {
    char name[NAME_LEN];
    float val;
} Record;

static Record g_got[MAX_RECORDS];
static int g_n_got = 0;
static int g_fail = 0;
static int g_warn = 0;
static float g_max_abs_err = 0.0f;
static const char *g_max_err_name = "";

static void rec_f(const char *name, float v) {
    if (g_n_got >= MAX_RECORDS) {
        fprintf(stderr, "record overflow\n");
        exit(2);
    }
    snprintf(g_got[g_n_got].name, NAME_LEN, "%s", name);
    g_got[g_n_got].val = v;
    g_n_got++;
}

static void rec_i(const char *name, int v) {
    rec_f(name, (float)v);
}

static void rec_arr(const char *prefix, const float *a, int n) {
    char buf[NAME_LEN];
    for (int i = 0; i < n; i++) {
        snprintf(buf, NAME_LEN, "%s[%d]", prefix, i);
        rec_f(buf, a[i]);
    }
}

static uint32_t fbits(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

static void write_golden(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(2);
    }
    fprintf(f, "# protein.cu numerical golden  n=%d\n", g_n_got);
    for (int i = 0; i < g_n_got; i++) {
        fprintf(f, "%s %.9g 0x%08x\n", g_got[i].name, g_got[i].val, fbits(g_got[i].val));
    }
    fclose(f);
    printf("Wrote %d records to %s\n", g_n_got, path);
}

// Pass if abs(err) <= atol + rtol*|golden|  (numpy allclose). atol==tol, rtol=10*atol when tol>0.
static int load_and_compare(const char *path, float tol) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        return 1;
    }
    char line[512];
    Record *exp = (Record *)malloc((size_t)MAX_RECORDS * sizeof(Record));
    int n_exp = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        char name[NAME_LEN];
        float val;
        unsigned bits;
        if (sscanf(line, "%127s %f %x", name, &val, &bits) < 2) {
            continue;
        }
        snprintf(exp[n_exp].name, NAME_LEN, "%s", name);
        exp[n_exp].val = val;
        n_exp++;
        if (n_exp >= MAX_RECORDS) {
            break;
        }
    }
    fclose(f);

    if (n_exp != g_n_got) {
        fprintf(stderr, "FAIL: record count golden=%d got=%d\n", n_exp, g_n_got);
        g_fail++;
    }
    int n = n_exp < g_n_got ? n_exp : g_n_got;
    int bitwise_ok = 0, tol_ok = 0, bad = 0;
    for (int i = 0; i < n; i++) {
        if (strcmp(exp[i].name, g_got[i].name) != 0) {
            fprintf(stderr, "FAIL: name mismatch at %d: golden=%s got=%s\n", i, exp[i].name, g_got[i].name);
            g_fail++;
            bad++;
            continue;
        }
        float e = exp[i].val, g = g_got[i].val;
        if (fbits(e) == fbits(g) || (isnan(e) && isnan(g))) {
            bitwise_ok++;
            continue;
        }
        float abs_err = fabsf(e - g);
        float rel_err = abs_err / fmaxf(fabsf(e), 1e-30f);
        if (abs_err > g_max_abs_err) {
            g_max_abs_err = abs_err;
            g_max_err_name = g_got[i].name;
        }
        float rtol = (tol > 0.0f) ? (10.0f * tol) : 0.0f; // default tol=1e-5 → rtol=1e-4
        if (tol > 0.0f && abs_err <= tol + rtol * fabsf(e)) {
            tol_ok++;
            g_warn++;
            continue;
        }
        fprintf(stderr, "FAIL: %s  golden=%.9g got=%.9g abs=%.3g rel=%.3g\n", g_got[i].name, e, g, abs_err,
                rel_err);
        g_fail++;
        bad++;
    }
    free(exp);
    printf("Compare: bitwise=%d  within_tol=%d  fail=%d  max_abs=%.6g (%s)  atol=%g rtol=%g\n", bitwise_ok,
           tol_ok, bad, g_max_abs_err, g_max_err_name, tol, tol > 0 ? 10 * tol : 0);
    return g_fail ? 1 : 0;
}

static void section_space(void) {
    Space s = {.type = SPACE_LINEAR, .min = 0.0f, .max = 10.0f, .scale = 0.5f};
    rec_f("space.lin.norm5", space_normalize(&s, 5.0f));
    rec_f("space.lin.unorm0", space_unnormalize(&s, 0.0f));
    s = (Space){.type = SPACE_LOG, .min = 1e-3f, .max = 1.0f, .scale = 0.5f};
    rec_f("space.log.unorm0", space_unnormalize(&s, 0.0f));
    s = (Space){.type = SPACE_POW2, .min = 8.0f, .max = 256.0f, .scale = 0.5f, .is_integer = 1};
    rec_f("space.pow2.norm32", space_normalize(&s, 32.0f));
    s = (Space){.type = SPACE_LOGIT, .min = 0.5f, .max = 0.999f, .scale = 0.5f};
    rec_f("space.logit.norm0.9", space_normalize(&s, 0.9f));
}

static void section_sweep(void) {
    const int dim = 4;
    const int cost_idx = 3;
    SweepSpace *space = (SweepSpace *)calloc(1, sizeof(SweepSpace));
    space->spaces = (Space *)calloc((size_t)dim, sizeof(Space));
    space->num = dim;
    space->cost_idx = cost_idx;
    space->optimize_direction = 1;
    space->spaces[0] = (Space){.type = SPACE_LOG, .min = 1e-3f, .max = 1.0f, .scale = 0.5f};
    space->spaces[1] = (Space){.type = SPACE_POW2, .min = 8.0f, .max = 256.0f, .scale = 0.5f, .is_integer = 1};
    space->spaces[2] = (Space){.type = SPACE_LOGIT, .min = 0.5f, .max = 0.999f, .scale = 0.5f};
    space->spaces[3] = (Space){.type = SPACE_LOG, .min = 1e2f, .max = 1e6f, .scale = 0.25f};

    ProteinSweep *sw = protein_sweep_create((ProteinSweep){
        .space = space,
        .num_random_samples = 8,
        .suggestions_per_pareto = 16,
        .gp_training_iter = 15,
        .gp_learning_rate = 0.01f,
        .optimizer_reset_frequency = 5,
        .gp_max_obs = 64,
        .infer_batch_size = 64,
        .use_success_prob = 1,
        .prune_pareto = 1,
        .use_logit = 0,
        .global_search_scale = 1.0f,
        .max_suggestion_cost = 1e9f,
        .expansion_rate = 0.05f,
        .cost_random_suggestion = 0.0f,
        .early_stop_quantile = 0.3f,
        .success_cap = 128,
        .failure_cap = 64,
        .top_k = 5,
        .rng_seed = 4242ULL,
    });

    srand(2024);
    float suggestion[4];
    for (int step = 0; step < 28; step++) {
        float fixed = (step % 7 == 0) ? 0.25f : NAN;
        ProteinSweepInfo info = protein_sweep_suggest(sw, suggestion, fixed);

        char pref[64], buf[96];
        snprintf(pref, sizeof(pref), "sw.s%02d", step);
        snprintf(buf, sizeof(buf), "%s.is_random", pref);
        rec_i(buf, info.is_random);
        snprintf(buf, sizeof(buf), "%s.n_pareto", pref);
        rec_i(buf, info.n_pareto);
        snprintf(buf, sizeof(buf), "%s.n_gp_obs", pref);
        rec_i(buf, info.n_gp_obs);
        snprintf(buf, sizeof(buf), "%s.n_cands", pref);
        rec_i(buf, info.n_candidates);
        snprintf(buf, sizeof(buf), "%s.score_loss", pref);
        rec_f(buf, info.score_loss);
        snprintf(buf, sizeof(buf), "%s.cost_loss", pref);
        rec_f(buf, info.cost_loss);
        snprintf(buf, sizeof(buf), "%s.pred_score", pref);
        rec_f(buf, info.predicted_score);
        snprintf(buf, sizeof(buf), "%s.pred_cost", pref);
        rec_f(buf, info.predicted_cost);
        snprintf(buf, sizeof(buf), "%s.rating", pref);
        rec_f(buf, info.rating);
        snprintf(buf, sizeof(buf), "%s.sugg", pref);
        rec_arr(buf, suggestion, dim);

        float score = 0.0f;
        for (int d = 0; d < dim - 1; d++) {
            score += (0.3f + 0.1f * d) * suggestion[d];
        }
        score += 0.05f * ((float)rand() / (float)RAND_MAX);
        float cost = expf(3.0f + 1.5f * suggestion[cost_idx]);
        int fail = (score < -0.8f) || (step == 12);
        protein_sweep_observe(sw, suggestion, score, cost, fail);

        int stop = protein_sweep_should_stop(sw, score * 0.5f, cost);
        snprintf(buf, sizeof(buf), "%s.stop", pref);
        rec_i(buf, stop);
    }

    rec_i("sw.final_n_success", sw->succ_n);
    rec_i("sw.final_n_failure", sw->fail_n);
    rec_i("sw.final_n_top", sw->n_top);
    rec_f("sw.upper_cost_thr", sw->upper_cost_threshold);
    rec_f("sw.cm.A", sw->cm_A);
    rec_f("sw.cm.B", sw->cm_B);
    rec_i("sw.cm.fitted", sw->cm_fitted);

    ProteinSweepInfo info = protein_sweep_suggest(sw, suggestion, 0.0f);
    rec_i("sw.fixed.is_random", info.is_random);
    rec_f("sw.fixed.pred_score", info.predicted_score);
    rec_f("sw.fixed.pred_cost", info.predicted_cost);
    rec_f("sw.fixed.rating", info.rating);
    rec_arr("sw.fixed.sugg", suggestion, dim);

    free(space->spaces);
    free(space);
}

int main(int argc, char **argv) {
    int gen = 0;
    // Default abs tol (numpy-style allclose: rtol = 10*atol).
    float tol = 1e-5f;
    const char *path = "tests/golden_protein_numerics.txt";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gen") == 0) {
            gen = 1;
            if (i + 1 < argc) {
                path = argv[++i];
            }
        } else if (strcmp(argv[i], "--tol") == 0) {
            if (i + 1 < argc) {
                tol = (float)atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--bitwise") == 0) {
            tol = 0.0f;
        } else if (argv[i][0] != '-') {
            path = argv[i];
        }
    }

    printf("=== protein.cu numerical regression ===\n");
    section_space();
    section_sweep();
    printf("Collected %d records\n", g_n_got);

    if (gen) {
        write_golden(path);
        return 0;
    }
    int rc = load_and_compare(path, tol);
    if (rc) {
        printf("RESULT: FAIL (%d mismatches)\n", g_fail);
    } else if (g_warn) {
        printf("RESULT: PASS (with %d tol-only diffs, max_abs=%.6g)\n", g_warn, g_max_abs_err);
    } else {
        printf("RESULT: PASS (bitwise)\n");
    }
    return rc;
}
