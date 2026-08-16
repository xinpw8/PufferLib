#include <string.h>
#include <math.h>
typedef unsigned char obs_t;
#include "pufferenv.h"

#define ACT_SIZES {2}
#define OBS_SIZE 512
#define NUM_ATNS 1

struct Log {
    float perf;
    float score;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int bandwidth;
    int compute;
    unsigned int rng;
};
typedef Env Benchmark;

void puf_reset(Benchmark* env) {}

void puf_step(Benchmark* env) {
    float result = 0;
    for (int i=0; i<env->compute; i++) {
        result = sinf(result + 0.1f);
    }

    //memset((env->agents[0].observations), result, env->bandwidth);
}

void puf_render(Benchmark* env) { }

void puf_close(Benchmark* env) { }

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->compute = dict_get(kwargs, "compute");
    env->bandwidth = dict_get(kwargs, "bandwidth");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
}

