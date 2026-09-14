#pragma once
/* Compile-only environment fixture for the pinned trainer inference test.
 * Creating or stepping this fixture aborts; it is not a training environment. */
#define PUF_BACKEND PUF_GPU
typedef float obs_t;
#include "pufferenv.h"
#define OBS_SIZE 223
#define NUM_ATNS 1
#define ACT_SIZES {33}
struct Log {float score,n;};
struct Env {Log log;Agent agents[1];int num_agents,tag,boundary_reached;unsigned int rng;};
Env* puf_vec_create(int,Dict*,obs_t*,float*,float*,float*){abort();}
void puf_bind_stream(cudaStream_t){}
void puf_init(Env*,Dict*){abort();}
void puf_reset(Env*){abort();}
void puf_step(Env*){abort();}
void puf_close(Env*){abort();}
void puf_render(Env*){abort();}
void puf_log(Log*,Dict*){abort();}
