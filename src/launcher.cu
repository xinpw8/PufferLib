// Native train/eval/sweep launcher. Built by ./build.sh ENV --native
// Run: ./build_native train breakout train.total_timesteps=1_000_000
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "config.h"
#include "pufferlib.cu"
#include "protein.cu"
#include "dashboard.h"
#include "logging.h"
#include "sweep.h"

typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

typedef struct {
    float score;
    float cost;
    float steps;
} TrainResult;

static Dict* config_to_dict(PufConfig* cfg) {
    Dict* out = create_dict(cfg->len);
    for (int i = 0; i < cfg->len; i++) {
        double val = 0;
        if (puf_config_parse_val(cfg->items[i].val, &val)) {
            dict_set(out, cfg->items[i].key, val);
        }
    }
    return out;
}

static HypersT config_to_hypers(PufConfigFile* cfg, TrainContext* ctx) {
    PufConfig* base = puf_config_get_section(cfg, "base");
    PufConfig* vec = puf_config_get_section(cfg, "vec");
    PufConfig* train = puf_config_get_section(cfg, "train");
    PufConfig* policy = puf_config_get_section(cfg, "policy");

    HypersT h = {};
    h.total_agents = (int)puf_config_val(vec, "total_agents");
    h.num_buffers = (int)puf_config_val(vec, "num_buffers");
    h.num_threads = (int)puf_config_val(vec, "num_threads");
    h.horizon = (int)puf_config_val(train, "horizon");
    h.hidden_size = (int)puf_config_val(policy, "hidden_size");
    h.num_layers = (int)puf_config_val(policy, "num_layers");
    h.lr = (float)puf_config_val(train, "learning_rate");
    h.min_lr_ratio = (float)puf_config_val(train, "min_lr_ratio");
    h.anneal_lr = (bool)puf_config_val(train, "anneal_lr");
    h.beta1 = (float)puf_config_val(train, "beta1");
    h.beta2 = (float)puf_config_val(train, "beta2");
    h.eps = (float)puf_config_val(train, "eps");
    h.minibatch_size = (int)puf_config_val(train, "minibatch_size");
    h.replay_ratio = (float)puf_config_val(train, "replay_ratio");
    h.total_timesteps = (long)puf_config_val(train, "total_timesteps");
    h.max_grad_norm = (float)puf_config_val(train, "max_grad_norm");
    h.clip_coef = (float)puf_config_val(train, "clip_coef");
    h.vf_clip_coef = (float)puf_config_val(train, "vf_clip_coef");
    h.vf_coef = (float)puf_config_val(train, "vf_coef");
    h.ent_coef = (float)puf_config_val(train, "ent_coef");
    h.min_ent_coef_ratio = (float)puf_config_val(train, "min_ent_coef_ratio");
    h.anneal_ent_coef = (bool)puf_config_val(train, "anneal_ent_coef");
    h.gamma = (float)puf_config_val(train, "gamma");
    h.gae_lambda = (float)puf_config_val(train, "gae_lambda");
    h.vtrace_rho_clip = (float)puf_config_val(train, "vtrace_rho_clip");
    h.vtrace_c_clip = (float)puf_config_val(train, "vtrace_c_clip");
    h.prio_alpha = (float)puf_config_val(train, "prio_alpha");
    h.prio_beta0 = (float)puf_config_val(train, "prio_beta0");
    h.reset_state = (bool)puf_config_val(base, "reset_state");
    h.cudagraphs = (int)puf_config_val(base, "cudagraphs");
    h.profile = (bool)puf_config_val(base, "profile");
    h.rank = ctx->rank;
    h.world_size = ctx->world_size;
    h.gpu_id = ctx->gpu_id;
    if (ctx->world_size > 1) {
        h.nccl_id = std::string((char*)ctx->nccl_id, sizeof(ncclUniqueId));
    } else {
        h.nccl_id = "";
    }
    h.seed = (int)puf_config_val(base, "seed");
    return h;
}

static PuffeRL* create_trainer(PufConfigFile* cfg, TrainContext* ctx) {
    HypersT hypers = config_to_hypers(cfg, ctx);
    PufConfig* base = puf_config_get_section(cfg, "base");
    PufConfig* vec = puf_config_get_section(cfg, "vec");
    PufConfig* env = puf_config_get_section(cfg, "env");
    Dict* vec_kwargs = config_to_dict(vec);
    Dict* env_kwargs = config_to_dict(env);
    PuffeRL* pufferl = create_pufferl_impl(hypers, puf_config_str(base, "env_name"),
        vec_kwargs, env_kwargs);
    if (!pufferl) {
        fprintf(stderr, "create_pufferl_impl failed\n");
        exit(1);
    }
    return pufferl;
}

static void rollouts(PuffeRL* p) {
    if (p->hypers.reset_state) {
        for (int i = 0; i < p->hypers.num_buffers; i++) {
            puf_zero(&p->buffer_states[i], p->default_stream);
        }
        for (int b = 0; b < p->num_frozen_banks; b++) {
            for (int i = 0; i < p->hypers.num_buffers; i++) {
                puf_zero(&p->frozen_banks[b].buffer_states[i], p->default_stream);
            }
        }
    }

    double t0 = wall_clock();
    static_vec_omp_step(p->vec);
    float sec = (float)(wall_clock() - t0);
    p->profile.accum[PROF_ROLLOUT] += sec * 1000.0f;

    float eval_prof[NUM_EVAL_PROF];
    static_vec_read_profile(p->vec, eval_prof);
    p->profile.accum[PROF_EVAL_GPU] += eval_prof[EVAL_GPU];
    p->profile.accum[PROF_EVAL_ENV] += eval_prof[EVAL_ENV_STEP];
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
}

static void close_trainer(PuffeRL* p) {
    close_impl(*p);
    delete p;
}

static void run_eval(PufConfigFile* cfg, TrainContext* ctx) {
    PufConfig* base = puf_config_get_section(cfg, "base");
    PufConfig* train = puf_config_get_section(cfg, "train");
    puf_config_put(base, "reset_state", "false");
    puf_config_put(train, "horizon", "1");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    char resolved_path[4096];
    const char* load_path = resolve_load_model_path(cfg, resolved_path, sizeof(resolved_path));
    if (load_path) {
        load_weights(pufferl, load_path);
        printf("Loaded weights from %s\n", load_path);
    }

    for (;;) {
        static_vec_render(pufferl->vec, 0);
        rollouts(pufferl);
        Dict* log = create_dict(128);
        trainer_eval_log(pufferl, log);
        puf_dashboard_print(cfg, pufferl, log, 0);
        dict_free(log);
    }

    close_trainer(pufferl);
}

static TrainResult run_train(PufConfigFile* cfg, TrainContext* ctx) {
    PufConfig* base = puf_config_get_section(cfg, "base");
    PufConfig* train = puf_config_get_section(cfg, "train");
    PufConfig* vec = puf_config_get_section(cfg, "vec");
    PufConfig* selfplay = puf_config_get_section(cfg, "selfplay");

    if (puf_config_val(selfplay, "enabled") == 0) {
        puf_config_put(vec, "num_frozen_banks", "0");
        puf_config_put(vec, "frozen_bank_pct", "0");
    }

    char run_id[64];
    const char* configured_run_id = puf_config_get(base, "run_id");
    if (!configured_run_id || strcmp(configured_run_id, "None") == 0) {
        snprintf(run_id, sizeof(run_id), "%ld", (long)(1000.0 * wall_clock()));
        puf_config_put(base, "run_id", run_id);
    } else {
        snprintf(run_id, sizeof(run_id), "%s", configured_run_id);
    }

    char checkpoint_dir[2048];
    char log_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s",
        puf_config_str(base, "checkpoint_dir"), puf_config_str(base, "env_name"), run_id);
    snprintf(log_dir, sizeof(log_dir), "%s/%s",
        puf_config_str(base, "log_dir"), puf_config_str(base, "env_name"));
    if (ctx->artifact_owner) {
        mkdir_p(checkpoint_dir);
        mkdir_p(log_dir);
    }

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    long total_timesteps = (long)puf_config_val(train, "total_timesteps");
    long batch_size = (long)puf_config_val(vec, "total_agents") * (long)puf_config_val(train, "horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long eval_epochs = train_epochs / 2;
    long checkpoint_interval = (long)puf_config_val(base, "checkpoint_interval");
    long eval_episodes = (long)puf_config_val(base, "eval_episodes");
    const char* target_key = "env/score";
    Dict* last_log = create_dict(128);
    TrainResult result = {0};

    for (long epoch = 0; epoch < train_epochs + eval_epochs; epoch++) {
        rollouts(pufferl);
        if (epoch < train_epochs) {
            train_impl(*pufferl);
        }

        bool is_final = epoch == train_epochs - 1;
        bool should_save = epoch < train_epochs && (epoch % checkpoint_interval == 0 || is_final);
        if (should_save && ctx->artifact_owner) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
            save_weights(pufferl, path);
        }

        if (wall_clock() < pufferl->last_log_time + 0.6 && epoch < train_epochs - 1) {
            continue;
        }

        dict_free(last_log);
        last_log = create_dict(128);
        if (epoch >= train_epochs) {
            trainer_eval_log(pufferl, last_log);
        } else {
            trainer_log(pufferl, last_log);
        }
        if (ctx->artifact_owner) {
            puf_dashboard_print(cfg, pufferl, last_log, (int)epoch);
        }

        if (puf_dict_get_or(last_log, target_key, -1) < 0) {
            continue;
        }
        if (epoch >= train_epochs && puf_dict_get_or(last_log, "env/n", 0) > eval_episodes) {
            break;
        }
    }

    result.score = (float)puf_dict_get_or(last_log, target_key, 0);
    result.cost = (float)puf_dict_get_or(last_log, "uptime", 0);
    result.steps = (float)puf_dict_get_or(last_log, "agent_steps", 0);
    if (ctx->artifact_owner) {
        char log_path[4096];
        snprintf(log_path, sizeof(log_path), "%s/%s.json", log_dir, run_id);
        puf_log_write_json(log_path, cfg, last_log);
    }
    dict_free(last_log);
    close_trainer(pufferl);
    return result;
}

static int gpu_for_rank(int rank, int world_size) {
    if (rank == 0) {
        return world_size - 1;
    }
    return rank - 1;
}

static void wait_children(pid_t* pids, int num_pids) {
    for (int i = 0; i < num_pids; i++) {
        int status = 0;
        if (waitpid(pids[i], &status, 0) < 0) {
            fprintf(stderr, "waitpid failed for child %d: %s\n", (int)pids[i], strerror(errno));
            exit(1);
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "worker pid %d failed\n", (int)pids[i]);
            exit(1);
        }
    }
}

static TrainResult launch_train(PufConfigFile* cfg) {
    PufConfig* train = puf_config_get_section(cfg, "train");
    int world_size = (int)puf_config_val(train, "gpus");
    if (world_size < 1) {
        fprintf(stderr, "config error: [train] gpus must be >= 1\n");
        exit(1);
    }

    ncclUniqueId nccl_id;
    ncclUniqueId* nccl_ptr = NULL;
    if (world_size > 1) {
        ncclGetUniqueId(&nccl_id);
        nccl_ptr = &nccl_id;
    }

    pid_t* pids = (pid_t*)calloc(world_size > 1 ? world_size - 1 : 1, sizeof(pid_t));
    for (int rank = world_size - 1; rank >= 1; rank--) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "fork failed: %s\n", strerror(errno));
            exit(1);
        }

        if (pid == 0) {
            if (!freopen("/dev/null", "w", stdout)) {
                fprintf(stderr, "failed to redirect child stdout: %s\n", strerror(errno));
                exit(1);
            }
            TrainContext child = {
                .rank = rank,
                .world_size = world_size,
                .gpu_id = gpu_for_rank(rank, world_size),
                .artifact_owner = 0,
                .nccl_id = nccl_ptr,
            };
            run_train(cfg, &child);
            puf_config_free(cfg);
            exit(0);
        }

        pids[rank - 1] = pid;
    }

    TrainContext host = {
        .rank = 0,
        .world_size = world_size,
        .gpu_id = gpu_for_rank(0, world_size),
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(cfg, &host);
    wait_children(pids, world_size - 1);
    free(pids);
    return result;
}

static void run_sweep(PufConfigFile* cfg) {
    validate_sweep_support(cfg);
    PufConfig* sweep = puf_config_get_section(cfg, "sweep");
    SweepParam* params = NULL;
    int num_params = 0;
    Hyperparameters* hypers = sweep_hypers_create(cfg, &params, &num_params);

    int max_runs = (int)puf_config_val(sweep, "max_runs");
    int downsample = (int)puf_config_val(sweep, "downsample");
    int prune_pareto = (int)puf_config_val(sweep, "prune_pareto");
    int use_logit = strcmp(puf_config_str(sweep, "metric_distribution"), "logit") == 0;
    float max_cost = (float)puf_config_val(sweep, "max_suggestion_cost");
    float early_stop_quantile = (float)puf_config_val(sweep, "early_stop_quantile");
    int success_cap = max_runs * downsample * 2;
    if (success_cap < 8192) {
        success_cap = 8192;
    }

    ProteinSweep* protein = protein_sweep_create(hypers,
        10, 256, 50, 0.001f, 50, 750, 4096,
        downsample == 1, prune_pareto, use_logit,
        1.0f, max_cost, 0.1f, -0.8f, early_stop_quantile,
        success_cap, 1024, 5, 73ULL);

    float* sample = (float*)calloc((size_t)num_params, sizeof(float));
    for (int run = 0; run < max_runs; run++) {
        ProteinSweepInfo info = protein_sweep_suggest(protein, sample, NAN);

        PufConfigFile trial = {0};
        puf_config_copy(&trial, cfg);
        sweep_apply(&trial, params, num_params, sample);

        char run_id[64];
        snprintf(run_id, sizeof(run_id), "sweep_%ld_%04d", (long)(1000.0 * wall_clock()), run);
        puf_config_put(puf_config_get_section(&trial, "base"), "run_id", run_id);
        puf_config_validate_train(&trial);

        TrainResult result = launch_train(&trial);
        protein_sweep_observe(protein, sample, result.score, result.cost, 0);
        printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
            run, result.score, result.cost, result.steps,
            info.is_random, info.n_gp_obs, info.n_pareto);

        puf_config_free(&trial);
    }

    free(sample);
    free(params);
    protein_sweep_destroy(protein);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 3) {
        fprintf(stderr, "usage: %s train|eval|sweep ENV [section.key=value ...]\n", argv[0]);
        exit(1);
    }

    const char* mode = argv[1];
    const char* env_name = argv[2];
    PufConfigFile cfg = {0};
    puf_config_load_env(&cfg, env_name, argc - 3, argv + 3);
    puf_config_validate_train(&cfg);

    if (strcmp(mode, "train") == 0) {
        launch_train(&cfg);
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(&cfg);
    } else if (strcmp(mode, "eval") == 0) {
        TrainContext ctx = {
            .rank = 0,
            .world_size = 1,
            .gpu_id = 0,
            .artifact_owner = 1,
            .nccl_id = NULL,
        };
        run_eval(&cfg, &ctx);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        exit(1);
    }

    puf_config_free(&cfg);
    return 0;
}
