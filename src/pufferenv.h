#pragma once

#include "dict.h"
#ifndef Env
#error "Env must be defined before including pufferenv.h"
#endif

typedef struct Agent {
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;

void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
