#ifndef REK_NATIVE5_PUFFER_ENV_CU
#define REK_NATIVE5_PUFFER_ENV_CU

#define PUF_BACKEND PUF_GPU
#define PUFFER_ENV_UNCLIPPED_REWARDS
#define PUFFER_ENV_GPU_ACTION_MASK
#define PUFFER_ENV_GPU_ROLLOUT_CHECK

#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"
#include "runtime_api.h"
#include "native_policy.h"

#define OBS_SIZE REK_NATIVE5_OBSERVATION_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {REK_NATIVE5_ACTION_COUNT}

struct Log : RekNative5Log {};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
};

static_assert(sizeof(Log) == sizeof(RekNative5Log), "Log ABI mismatch");
static_assert(offsetof(Env, log) == 0, "Log must start the Env record");

static struct {
    RekNative5Runtime* runtime;
    Env* envs;
    cudaStream_t stream;
    RekNativePolicy* opponent;
    float* opponent_observations;
    float* external_actions;
    uint8_t* external_overrides;
    int opponent_deterministic;
    int opponent_encoded;
} rek_native5_binding;

static void rek_native5_require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "REK native 5.0 %s: %s\n", operation,
            cudaGetErrorString(result));
        abort();
    }
}

static void rek_native5_require_runtime(int result, const char* operation) {
    if (result != 0) {
        const char* error = rek_native5_error();
        fprintf(stderr, "REK native 5.0 %s: %s\n", operation,
            error ? error : "runtime failed without an error message");
        abort();
    }
}
static void rek_native5_require_policy(int result,const char* operation) {
    if(result){fprintf(stderr,"REK frozen opponent %s: %s\n",operation,rek_native_policy_error());abort();}
}

static const char* rek_native5_required_path(Dict* kwargs, const char* key) {
    DictItem* item = dict_find(kwargs, key);
    if (item == NULL || item->str == NULL || item->str[0] == '\0'
            || strcmp(item->str, "None") == 0) {
        fprintf(stderr, "REK native 5.0 requires --env.%s=PATH\n", key);
        abort();
    }
    return item->str;
}

Env* puf_vec_create(int n, Dict* kwargs, obs_t* observations,
        float* actions, float* rewards, float* terminals) {
    const char* selected_backend=getenv("REK_PHYSICS_BACKEND");
    if(selected_backend&&(strcmp(selected_backend,"mujoco_cpu_eval")==0||strcmp(selected_backend,"puffysics_cpu_eval")==0)){
        fprintf(stderr,"CPU evaluation backends are restricted to the standalone viewer and cannot run through the native trainer\n");abort();
    }
    if (n <= 0 || rek_native5_binding.runtime != NULL) {
        fprintf(stderr, "REK native 5.0 requires a positive, single active batch\n");
        abort();
    }
    RekNative5Config config = {};
    config.abi_version = REK_NATIVE5_RUNTIME_ABI;
    config.model_path = rek_native5_required_path(kwargs, "model_path");
    config.physics_export_path = rek_native5_required_path(kwargs, "physics_export_path");
    config.assets_path = rek_native5_required_path(kwargs, "assets_path");
    config.motion_features_path = rek_native5_required_path(kwargs, "motion_features_path");
    config.controller_encoder_path = rek_native5_required_path(kwargs, "controller_encoder_path");
    config.controller_decoder_path = rek_native5_required_path(kwargs, "controller_decoder_path");
    config.arenas = n;
    DictItem* round_seconds=dict_find(kwargs,"round_seconds");
    config.round_seconds=round_seconds?(float)round_seconds->value:0;
    DictItem* seed = dict_find(kwargs, "seed");
    config.seed = seed ? (uint32_t)seed->value : 73u;
    DictItem* segment = dict_find(kwargs, "locomotion_segment_ticks");
    config.locomotion_segment_ticks = segment ? (int)segment->value : 1;
    if (config.locomotion_segment_ticks != 1) {
        fprintf(stderr, "REK native 5.0 currently requires locomotion_segment_ticks=1\n");
        abort();
    }
    const uint32_t durations[REK_NATIVE5_MOVE_COUNT] = {
        35, 27, 31, 45, 32, 45, 157, 145, 158, 139, 134, 138, 73, 75, 68, 71, 103
    };
    for (int move = 0; move < REK_NATIVE5_MOVE_COUNT; ++move) {
        char key[64];
        snprintf(key, sizeof(key), "move_duration_%d", move);
        DictItem* item = dict_find(kwargs, key);
        if (item && (!(item->value >= 1.0 && item->value <= (double)UINT32_MAX)
                || item->value != (double)(uint32_t)item->value)) {
            fprintf(stderr, "REK native 5.0 env.%s must be a positive uint32\n", key);
            abort();
        }
        config.move_duration_ticks[move] = item ? (uint32_t)item->value : durations[move];
    }

    Env* host_envs = (Env*)calloc((size_t)n, sizeof(Env));
    if (host_envs == NULL) {
        fprintf(stderr, "REK native 5.0 could not allocate batch metadata\n");
        abort();
    }
    for (int i = 0; i < n; ++i) {
        host_envs[i].num_agents = 1;
        host_envs[i].rng = config.seed + (unsigned int)i;
    }
    Env* envs = NULL;
    rek_native5_require_cuda(cudaMalloc((void**)&envs, (size_t)n * sizeof(Env)),
        "allocate batch metadata");
    rek_native5_require_cuda(cudaMemcpy(envs, host_envs, (size_t)n * sizeof(Env),
        cudaMemcpyHostToDevice), "upload batch metadata");
    free(host_envs);

    RekNative5Buffers buffers = {};
    buffers.observations = observations;
    buffers.actions = actions;
    buffers.rewards = rewards;
    buffers.terminals = terminals;
    buffers.logs = reinterpret_cast<RekNative5Log*>(envs);
    buffers.log_stride_bytes = sizeof(Env);
    RekNative5Runtime* runtime = rek_native5_create(&config, &buffers, 0);
    if (runtime == NULL) {
        cudaFree(envs);
        rek_native5_require_runtime(1, "create runtime");
    }
    rek_native5_binding.runtime = runtime;
    rek_native5_binding.envs = envs;
    rek_native5_binding.stream = 0;
    DictItem* opponent_path=dict_find(kwargs,"opponent_checkpoint");
    if(opponent_path&&opponent_path->str&&opponent_path->str[0]&&strcmp(opponent_path->str,"None")!=0){
        const char* encoding=rek_native5_required_path(kwargs,"opponent_observation_encoding");
        if(strcmp(encoding,"scaled_polar_xy")!=0&&strcmp(encoding,"raw")!=0){fprintf(stderr,"Frozen opponent encoding must explicitly be raw or scaled_polar_xy\n");abort();}
        rek_native5_binding.opponent_encoded=strcmp(encoding,"scaled_polar_xy")==0;
        RekNativePolicyConfig policy={};policy.abi_version=REK_NATIVE_POLICY_ABI;
        policy.checkpoint_path=opponent_path->str;policy.expected_sha256=rek_native5_required_path(kwargs,"opponent_sha256");
        DictItem* hidden=dict_find(kwargs,"opponent_hidden_size");DictItem* layers=dict_find(kwargs,"opponent_num_layers");
        DictItem* precision=dict_find(kwargs,"opponent_precision");DictItem* deterministic=dict_find(kwargs,"opponent_deterministic");
        DictItem* legacy=dict_find(kwargs,"opponent_legacy_fast_hidden");
        policy.hidden_size=hidden?(int)hidden->value:256;policy.num_layers=layers?(int)layers->value:2;
        policy.precision=precision?(int)precision->value:REK_NATIVE_POLICY_BF16;policy.batch=n;policy.seed=config.seed;
        policy.legacy_fast_hidden=legacy?(int)legacy->value:0;
        rek_native5_binding.opponent_deterministic=deterministic?deterministic->value!=0:0;
        rek_native5_binding.opponent=rek_native_policy_create(&policy,0);
        rek_native5_require_policy(rek_native5_binding.opponent==NULL,"load checkpoint");
        rek_native5_require_cuda(cudaMalloc(&rek_native5_binding.opponent_observations,(size_t)n*2*OBS_SIZE*sizeof(float)),"allocate frozen observations");
        rek_native5_require_cuda(cudaMalloc(&rek_native5_binding.external_actions,(size_t)n*2*sizeof(float)),"allocate frozen actions");
        rek_native5_require_cuda(cudaMalloc(&rek_native5_binding.external_overrides,(size_t)n*2),"allocate frozen overrides");
        uint8_t* overrides=(uint8_t*)calloc((size_t)n*2,1);
        if(!overrides){fprintf(stderr,"Frozen opponent override allocation failed\n");abort();}
        for(int a=0;a<n;a++)overrides[2*a+1]=1;
        rek_native5_require_cuda(cudaMemcpy(rek_native5_binding.external_overrides,overrides,(size_t)n*2,cudaMemcpyHostToDevice),"upload frozen overrides");free(overrides);
        rek_native5_require_runtime(rek_native5_bind_external_actions(runtime,rek_native5_binding.external_actions,rek_native5_binding.external_overrides,0),"bind frozen opponent");
        fprintf(stderr,"native5 frozen opponent: sha256=%s precision=%d deterministic=%d\n",rek_native_policy_sha256(rek_native5_binding.opponent),policy.precision,rek_native5_binding.opponent_deterministic);
    }
    return envs;
}

void puf_bind_action_mask(uint8_t* action_mask) {
    rek_native5_require_runtime(rek_native5_bind_action_mask(rek_native5_binding.runtime,
        action_mask, rek_native5_binding.stream), "bind action mask");
}

void puf_bind_stream(cudaStream_t stream) {
    rek_native5_binding.stream = stream;
}

/* Reporting boundaries only. Preserve outcomes even if a later arena diverges. */
static int rek_native5_print_round_summary() {
    RekNative5DeviceView view={};
    rek_native5_require_runtime(rek_native5_get_device_view(rek_native5_binding.runtime,&view),"get round metrics");
    auto* rounds=(RekNative5RoundResult*)malloc((size_t)view.arenas*sizeof(RekNative5RoundResult));
    if(!rounds){fprintf(stderr,"Round metrics allocation failed\n");abort();}
    rek_native5_require_cuda(cudaMemcpyAsync(rounds,view.rounds,(size_t)view.arenas*sizeof(RekNative5RoundResult),cudaMemcpyDeviceToHost,rek_native5_binding.stream),"download round metrics");
    rek_native5_require_cuda(cudaStreamSynchronize(rek_native5_binding.stream),"synchronize round metrics");
    uint64_t completed=0,wins[2]={},ties=0,redos=0,unclassified=0;int64_t points[2]={};uint32_t failures=0;int first_failure=-1;
    for(int a=0;a<view.arenas;a++){
        completed+=rounds[a].completed_rounds;ties+=rounds[a].ties;redos+=rounds[a].redos;unclassified+=rounds[a].unclassified;failures|=rounds[a].failure_bits;
        if(rounds[a].failure_bits&&first_failure<0)first_failure=a;
        for(int s=0;s<2;s++){wins[s]+=rounds[a].wins[s];points[s]+=rounds[a].completed_points[s];}
    }
    free(rounds);
    printf("native5_round_summary={\"arenas\":%d,\"completed_rounds\":%llu,\"fighter0_wins\":%llu,\"fighter1_wins\":%llu,\"ties\":%llu,\"redos\":%llu,\"unclassified\":%llu,\"fighter0_completed_points\":%lld,\"fighter1_completed_points\":%lld,\"failure_bits\":%u}\n",view.arenas,(unsigned long long)completed,(unsigned long long)wins[0],(unsigned long long)wins[1],(unsigned long long)ties,(unsigned long long)redos,(unsigned long long)unclassified,(long long)points[0],(long long)points[1],failures);
    fflush(stdout);return first_failure;
}

static void rek_native5_print_json_floats(const float* values,size_t count) {
    putchar('[');
    for(size_t i=0;i<count;i++){
        if(i)putchar(',');
        if(isnan(values[i]))printf("\"NaN\"");
        else if(isinf(values[i]))printf(signbit(values[i])?"\"-Infinity\"":"\"Infinity\"");
        else printf("%.9g",values[i]);
    }
    putchar(']');
}

void puf_check_rollout(void) {
    int status=rek_native5_check_status(rek_native5_binding.runtime,rek_native5_binding.stream);
    if(status){
        char original_error[2048];snprintf(original_error,sizeof(original_error),"%s",rek_native5_error());
        int arena=rek_native5_print_round_summary();
        RekNative5Snapshot snapshot={};
        if(arena>=0&&rek_native5_read_snapshot(rek_native5_binding.runtime,arena,&snapshot,rek_native5_binding.stream)==0){
            printf("native5_failure_snapshot={\"arena\":%d,\"failure_bits\":%u,\"actions\":",arena,snapshot.round.failure_bits);
            rek_native5_print_json_floats(snapshot.actions,2);printf(",\"qpos\":");rek_native5_print_json_floats(snapshot.qpos,72);
            printf(",\"qvel\":");rek_native5_print_json_floats(snapshot.qvel,70);
            printf(",\"raw_observations\":");rek_native5_print_json_floats(snapshot.raw_observations,446);printf("}\n");fflush(stdout);
        }
        fprintf(stderr,"REK native 5.0 check completed rollout before training: %s\n",original_error);abort();
    }
    if(rek_native5_binding.opponent)rek_native5_require_policy(rek_native_policy_check_status(rek_native5_binding.opponent,rek_native5_binding.stream),"check rollout");
}

void puf_init(Env*, Dict*) {
    /* The GPU trainer initializes the entire batch through puf_vec_create. */
}

void puf_reset(Env*) {
    rek_native5_require_runtime(rek_native5_reset(rek_native5_binding.runtime,
        rek_native5_binding.stream), "reset runtime");
    if(rek_native5_binding.opponent)rek_native5_require_policy(rek_native_policy_reset(rek_native5_binding.opponent,rek_native5_binding.stream),"reset recurrent state");
}

void puf_step(Env*) {
    if(rek_native5_binding.opponent){
        RekNative5DeviceView view={};
        rek_native5_require_runtime(rek_native5_get_device_view(rek_native5_binding.runtime,&view),"get frozen opponent inputs");
        const float* opponent_observations=view.raw_observations;
        if(rek_native5_binding.opponent_encoded){
            rek_native5_require_runtime(rek_native5_encode_fighter_observations(rek_native5_binding.runtime,rek_native5_binding.opponent_observations,rek_native5_binding.stream),"encode frozen opponent inputs");
            opponent_observations=rek_native5_binding.opponent_observations;
        }
        rek_native5_require_policy(rek_native_policy_step_rows(rek_native5_binding.opponent,opponent_observations,view.action_masks,view.terminals,
            rek_native5_binding.external_actions,1,2,rek_native5_binding.opponent_deterministic,rek_native5_binding.stream),"infer frozen opponent");
    }
    rek_native5_require_runtime(rek_native5_step(rek_native5_binding.runtime,
        rek_native5_binding.stream), "step runtime");
}

void puf_close(Env*) {
    rek_native5_require_runtime(rek_native5_check_status(rek_native5_binding.runtime,
        rek_native5_binding.stream), "check final runtime state");
    rek_native5_print_round_summary();
    rek_native5_require_runtime(rek_native5_close(rek_native5_binding.runtime),
        "close runtime");
    rek_native5_require_cuda(cudaFree(rek_native5_binding.envs), "free batch metadata");
    if(rek_native5_binding.opponent){
        rek_native5_require_policy(rek_native_policy_check_status(rek_native5_binding.opponent,rek_native5_binding.stream),"check final opponent state");
        rek_native_policy_destroy(rek_native5_binding.opponent);
        rek_native5_require_cuda(cudaFree(rek_native5_binding.opponent_observations),"free frozen observations");
        rek_native5_require_cuda(cudaFree(rek_native5_binding.external_actions),"free frozen actions");
        rek_native5_require_cuda(cudaFree(rek_native5_binding.external_overrides),"free frozen overrides");
    }
    memset(&rek_native5_binding, 0, sizeof(rek_native5_binding));
}

void puf_render(Env*) {
    fprintf(stderr, "REK native 5.0 evaluation requires --headless\n");
    abort();
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hits", log->hits);
    dict_set(out, "falls", log->falls);
    dict_set(out, "wins", log->wins);
    dict_set(out, "losses", log->losses);
    dict_set(out, "draws", log->draws);
    dict_set(out, "actions_invalid", log->actions_invalid);
    dict_set(out, "finite_failures", log->finite_failures);
    dict_set(out, "n", log->n);
}

#endif
