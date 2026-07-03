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
#include "selfplay.h"
#include "league.h"
#include "sweep.h"

typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

static HypersT config_to_hypers(Config* cfg, TrainContext* ctx) {
    HypersT h = {};
    h.total_agents = (int)puf_config_val(cfg, "vec.total_agents");
    h.num_buffers = (int)puf_config_val(cfg, "vec.num_buffers");
    h.num_threads = (int)puf_config_val(cfg, "vec.num_threads");
    h.horizon = (int)puf_config_val(cfg, "train.horizon");
    h.hidden_size = (int)puf_config_val(cfg, "policy.hidden_size");
    h.num_layers = (int)puf_config_val(cfg, "policy.num_layers");
    h.lr = (float)puf_config_val(cfg, "train.learning_rate");
    h.min_lr_ratio = (float)puf_config_val(cfg, "train.min_lr_ratio");
    h.anneal_lr = (bool)puf_config_val(cfg, "train.anneal_lr");
    h.beta1 = (float)puf_config_val(cfg, "train.beta1");
    h.beta2 = (float)puf_config_val(cfg, "train.beta2");
    h.eps = (float)puf_config_val(cfg, "train.eps");
    h.minibatch_size = (int)puf_config_val(cfg, "train.minibatch_size");
    h.replay_ratio = (float)puf_config_val(cfg, "train.replay_ratio");
    h.total_timesteps = (long)puf_config_val(cfg, "train.total_timesteps");
    h.max_grad_norm = (float)puf_config_val(cfg, "train.max_grad_norm");
    h.clip_coef = (float)puf_config_val(cfg, "train.clip_coef");
    h.vf_clip_coef = (float)puf_config_val(cfg, "train.vf_clip_coef");
    h.vf_coef = (float)puf_config_val(cfg, "train.vf_coef");
    h.ent_coef = (float)puf_config_val(cfg, "train.ent_coef");
    h.min_ent_coef_ratio = (float)puf_config_val(cfg, "train.min_ent_coef_ratio");
    h.anneal_ent_coef = (bool)puf_config_val(cfg, "train.anneal_ent_coef");
    h.gamma = (float)puf_config_val(cfg, "train.gamma");
    h.gae_lambda = (float)puf_config_val(cfg, "train.gae_lambda");
    h.vtrace_rho_clip = (float)puf_config_val(cfg, "train.vtrace_rho_clip");
    h.vtrace_c_clip = (float)puf_config_val(cfg, "train.vtrace_c_clip");
    h.prio_alpha = (float)puf_config_val(cfg, "train.prio_alpha");
    h.prio_beta0 = (float)puf_config_val(cfg, "train.prio_beta0");
    h.reset_state = (bool)puf_config_val(cfg, "base.reset_state");
    h.cudagraphs = (int)puf_config_val(cfg, "base.cudagraphs");
    h.profile = (bool)puf_config_val(cfg, "base.profile");
    h.rank = ctx->rank;
    h.world_size = ctx->world_size;
    h.gpu_id = ctx->gpu_id;
    if (ctx->world_size > 1) {
        h.nccl_id = *ctx->nccl_id;
    }
    h.seed = (int)puf_config_val(cfg, "base.seed");
    return h;
}

static PuffeRL* create_trainer(Config* cfg, TrainContext* ctx) {
    HypersT hypers = config_to_hypers(cfg, ctx);
    PuffeRL* pufferl = create_pufferl_impl(hypers, &cfg->vec, &cfg->env);
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
    vec_step(p->vec);
    float sec = (float)(wall_clock() - t0);
    p->profile.accum[PROF_ROLLOUT] += sec * 1000.0f;

    float eval_prof[NUM_VEC_PROF] = {0};
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        float* src = &p->vec->accum[buf * NUM_VEC_PROF];
        for (int i = 0; i < NUM_VEC_PROF; i++) {
            eval_prof[i] += src[i];
        }
        memset(src, 0, NUM_VEC_PROF * sizeof(float));
    }
    p->profile.accum[PROF_EVAL_GPU] += eval_prof[VEC_GPU] / p->vec->buffers;
    p->profile.accum[PROF_EVAL_ENV] += eval_prof[VEC_ENV_STEP] / p->vec->buffers;
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
}

static void close_trainer(PuffeRL* p) {
    close_impl(*p);
    delete p;
}

static float log_value(Dict* log, const char* key, float fallback) {
    for (int i = 0; i < log->size; i++) {
        if (strcmp(log->items[i].key, key) == 0) {
            return (float)log->items[i].value;
        }
    }
    return fallback;
}

static void train_result_fill(TrainResult* result, PufLogHistory* history,
        Dict* last_log, Config* cfg, const char* target_key) {
    result->score = (float)puf_log_get_or(last_log, target_key, 0);
    result->cost = (float)puf_log_get_or(last_log, "uptime", 0);
    result->steps = (float)puf_log_get_or(last_log, "agent_steps", 0);

    int points = (int)puf_config_val(cfg, "sweep.downsample");
    if (points < 1) {
        points = 1;
    }
    if (points > TRAIN_RESULT_MAX_POINTS) {
        points = TRAIN_RESULT_MAX_POINTS;
    }
    result->points = points;

    if (history->size == 0 || points == 1) {
        result->scores[0] = result->score;
        result->costs[0] = result->cost;
        result->step_points[0] = result->steps;
        return;
    }

    float final_steps = log_value(&history->items[history->size - 1],
        "agent_steps", result->steps);
    int cursor = 0;
    for (int p = 0; p < points; p++) {
        float target = final_steps * (float)p / (float)(points - 1);
        while (cursor + 1 < history->size &&
                log_value(&history->items[cursor], "agent_steps", 0) < target) {
            cursor++;
        }
        Dict* log = &history->items[cursor];
        result->scores[p] = log_value(log, target_key, result->score);
        result->costs[p] = log_value(log, "uptime", result->cost);
        result->step_points[p] = log_value(log, "agent_steps", target);
    }
    result->scores[points - 1] = result->score;
    result->costs[points - 1] = result->cost;
    result->step_points[points - 1] = result->steps;
}

static void run_eval(Config* cfg, TrainContext* ctx) {
    puf_config_put(cfg, "base.reset_state", "false");
    puf_config_put(cfg, "train.horizon", "1");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    char resolved_path[4096];
    const char* load_path = puf_checkpoint_path(cfg, resolved_path, sizeof(resolved_path));
    if (load_path) {
        puf_load_weights(pufferl, load_path);
        printf("Loaded weights from %s\n", load_path);
    }

    for (;;) {
        puf_render(&pufferl->vec->envs[0]);
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        puf_dashboard_print(cfg, pufferl, &log, 0);
    }

    close_trainer(pufferl);
}

static long config_long_or(Config* cfg, const char* key, long fallback) {
    return puf_config_get(cfg, key) ? (long)puf_config_val(cfg, key) : fallback;
}

static const char* resolve_checkpoint_key(Config* cfg, const char* key,
        char* out, size_t out_size) {
    const char* load_path = puf_config_get(cfg, key);
    if (!load_path || strcmp(load_path, "None") == 0) {
        return NULL;
    }
    if (strcmp(load_path, "latest") != 0) {
        return load_path;
    }

    char root[2048];
    snprintf(root, sizeof(root), "%s/%s",
        puf_config_str(cfg, "base.checkpoint_dir"), puf_config_str(cfg, "base.env_name"));
    out[0] = 0;
    time_t best_time = 0;
    puf_find_latest_checkpoint(root, out, out_size, &best_time);
    if (!out[0]) {
        fprintf(stderr, "no .bin checkpoints found in %s\n", root);
        exit(1);
    }
    return out;
}

static void load_primary_if_configured(PuffeRL* pufferl, Config* cfg) {
    char resolved_path[4096];
    const char* load_path = resolve_checkpoint_key(cfg,
        "base.load_model_path", resolved_path, sizeof(resolved_path));
    if (load_path) {
        puf_load_weights(pufferl, load_path);
        printf("Loaded weights from %s\n", load_path);
    }
}

static void run_eval_bot(Config* cfg, TrainContext* ctx) {
    long num_games = config_long_or(cfg, "base.num_games",
        (long)puf_config_val(cfg, "base.eval_episodes"));
    long burnin_games = config_long_or(cfg, "base.burnin_games", 0);
    long eval_agents = config_long_or(cfg, "base.eval_agents", 0);
    if (num_games <= 0 || burnin_games < 0) {
        fprintf(stderr, "eval_bot requires positive num_games and nonnegative burnin_games\n");
        exit(1);
    }
    if (eval_agents <= 0) {
        eval_agents = num_games / 8;
        if (eval_agents < 1024) {
            eval_agents = 1024;
        }
        if (eval_agents > 4096) {
            eval_agents = 4096;
        }
        if (eval_agents > num_games && num_games >= 1024) {
            eval_agents = num_games;
        }
    } else if (eval_agents > num_games) {
        eval_agents = num_games;
    }
    eval_agents += (-eval_agents) % 2;

    char buf[64];
    puf_config_put(cfg, "base.reset_state", "false");
    puf_config_put(cfg, "train.horizon", "1");
    puf_config_put(cfg, "vec.num_buffers", "2");
    snprintf(buf, sizeof(buf), "%ld", eval_agents);
    puf_config_put(cfg, "vec.total_agents", buf);
    puf_config_put(cfg, "vec.num_frozen_banks", "0");
    puf_config_put(cfg, "vec.frozen_bank_pct", "0");
    puf_config_put(cfg, "selfplay.enabled", "0");
    puf_config_put(cfg, "env.dr", "0");
    puf_config_put(cfg, "env.num_agents", "1");
    puf_config_put(cfg, "env.num_bots", "1");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    load_primary_if_configured(pufferl, cfg);

    Dict baseline = {0};
    int has_baseline = 0;
    long baseline_n = 0;
    for (;;) {
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        long n = (long)puf_log_get_or(&log, "env/n", 0);
        if (burnin_games > 0 && !has_baseline && n >= burnin_games) {
            baseline = log;
            for (int i = 0; i < baseline.size; i++) {
                if (baseline.items[i].str) {
                    baseline.items[i].str = baseline.items[i].str_buf;
                }
            }
            has_baseline = 1;
            baseline_n = n;
            printf("\rbot_eval_burnin=%ld/%ld", n, burnin_games);
            continue;
        }

        double scored_n = n - baseline_n;
        double score = puf_log_get_or(&log, "env/score", 0);
        double perf = puf_log_get_or(&log, "env/perf", 0);
        if (has_baseline && scored_n > 0) {
            double base_n = (double)baseline_n;
            double cur_n = (double)n;
            score = (score * cur_n - puf_log_get_or(&baseline, "env/score", 0) * base_n) / scored_n;
            perf = (perf * cur_n - puf_log_get_or(&baseline, "env/perf", 0) * base_n) / scored_n;
        }
        printf("\rbot_eval=%.0f/%ld  perf=%.4f  score=%.3f",
            scored_n, num_games, perf, score);
        if ((n - baseline_n) >= num_games && (!burnin_games || has_baseline)) {
            break;
        }
    }
    printf("\n");
    close_trainer(pufferl);
}

static void run_match_eval(Config* cfg, TrainContext* ctx, int verbose,
        float* score_out, float* draw_out, int* games_out) {
    long num_games = config_long_or(cfg, "base.num_games",
        (long)puf_config_val(cfg, "base.eval_episodes"));
    long eval_agents = config_long_or(cfg, "base.eval_agents",
        (long)puf_config_val(cfg, "sweep.league_match_eval_agents"));
    if (eval_agents <= 0) {
        eval_agents = 8192;
    }
    eval_agents += (-eval_agents) % 4;

    char a_path_buf[4096];
    char b_path_buf[4096];
    const char* a_path = resolve_checkpoint_key(cfg,
        "base.load_model_path", a_path_buf, sizeof(a_path_buf));
    const char* b_path = resolve_checkpoint_key(cfg,
        "base.load_enemy_model_path", b_path_buf, sizeof(b_path_buf));
    if (!a_path || !b_path) {
        fprintf(stderr, "match requires base.load_model_path and base.load_enemy_model_path\n");
        exit(1);
    }

    char buf[64];
    puf_config_put(cfg, "base.reset_state", "false");
    puf_config_put(cfg, "train.horizon", "1");
    puf_config_put(cfg, "vec.num_buffers", "2");
    snprintf(buf, sizeof(buf), "%ld", eval_agents);
    puf_config_put(cfg, "vec.total_agents", buf);
    puf_config_put(cfg, "vec.num_frozen_banks", "1");
    puf_config_put(cfg, "vec.frozen_bank_pct", "1");
    puf_config_put(cfg, "selfplay.enabled", "0");
    puf_config_put(cfg, "env.dr", "0");
    puf_config_put(cfg, "env.num_agents", "2");
    puf_config_put(cfg, "env.num_bots", "0");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    puf_load_weights(pufferl, a_path);
    pufferl_load_frozen_bank(pufferl, 0, b_path);

    for (;;) {
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        long n = (long)puf_log_get_or(&log, "env/n", 0);
        double a = puf_log_get_or(&log, "env/slot_0_score", 0);
        double b = puf_log_get_or(&log, "env/slot_1_score", 0);
        double draw = puf_log_get_or(&log, "env/draw_rate", 0);
        if (verbose) {
            printf("\rgames=%ld/%ld  A=%.3f  B=%.3f  draw=%.3f",
                n, num_games, a, b, draw);
        }
        if (n >= num_games) {
            *score_out = (float)a;
            *draw_out = (float)draw;
            *games_out = (int)n;
            break;
        }
    }
    if (verbose) {
        printf("\n");
    }
    close_trainer(pufferl);
}

static void run_match(Config* cfg, TrainContext* ctx) {
    float score = 0;
    float draw = 0;
    int games = 0;
    run_match_eval(cfg, ctx, 1, &score, &draw, &games);
}

static void run_league_match_worker(Config* cfg, TrainContext* ctx) {
    const char* state_path = puf_config_str(cfg, "sweep.league_state_path");
    long games = config_long_or(cfg, "base.num_games",
        (long)puf_config_val(cfg, "sweep.league_match_games"));
    unsigned int rng = (unsigned int)puf_config_val(cfg, "base.seed") + 1009U;

    for (;;) {
        LeaguePlayer a;
        LeaguePlayer b;
        if (!league_choose_pair(state_path, &a, &b, &rng)) {
            usleep(500000);
            continue;
        }

        char buf[64];
        puf_config_put(cfg, "base.load_model_path", a.path);
        puf_config_put(cfg, "base.load_enemy_model_path", b.path);
        snprintf(buf, sizeof(buf), "%ld", games);
        puf_config_put(cfg, "base.num_games", buf);

        float score = 0;
        float draw = 0;
        int n = 0;
        run_match_eval(cfg, ctx, 0, &score, &draw, &n);
        league_record_match(state_path, a.id, b.id, n, score, draw);
        printf("league_match %s vs %s games=%d score=%.4f draw=%.4f\n",
            a.id, b.id, n, score, draw);
    }
}

static TrainResult run_train(Config* cfg, TrainContext* ctx) {
    if (puf_config_val(cfg, "selfplay.enabled") == 0) {
        puf_config_put(cfg, "vec.num_frozen_banks", "0");
        puf_config_put(cfg, "vec.frozen_bank_pct", "0");
    }

    char run_id[64];
    const char* configured_run_id = puf_config_get(cfg, "base.run_id");
    if (!configured_run_id || strcmp(configured_run_id, "None") == 0) {
        snprintf(run_id, sizeof(run_id), "%ld", (long)(1000.0 * wall_clock()));
        puf_config_put(cfg, "base.run_id", run_id);
    } else {
        snprintf(run_id, sizeof(run_id), "%s", configured_run_id);
    }

    char checkpoint_dir[2048];
    char log_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s",
        puf_config_str(cfg, "base.checkpoint_dir"), puf_config_str(cfg, "base.env_name"), run_id);
    snprintf(log_dir, sizeof(log_dir), "%s/%s",
        puf_config_str(cfg, "base.log_dir"), puf_config_str(cfg, "base.env_name"));
    if (ctx->artifact_owner) {
        mkdir_p(checkpoint_dir);
        mkdir_p(log_dir);
    }

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    Selfplay* selfplay = (Selfplay*)calloc(1, sizeof(Selfplay));
    selfplay_init(selfplay, cfg, pufferl, run_id, ctx->artifact_owner, ctx->world_size);
    long total_timesteps = (long)puf_config_val(cfg, "train.total_timesteps");
    long batch_size = (long)puf_config_val(cfg, "vec.total_agents")
        * (long)puf_config_val(cfg, "train.horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long eval_epochs = train_epochs / 2;
    long checkpoint_interval = (long)puf_config_val(cfg, "base.checkpoint_interval");
    long eval_episodes = (long)puf_config_val(cfg, "base.eval_episodes");
    const char* target_key = "env/score";
    Dict last_log = {0};
    PufLogHistory log_history = {0};
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
            puf_save_weights(pufferl, path);
            snprintf(result.checkpoint_path, sizeof(result.checkpoint_path), "%s", path);
        }

        if (wall_clock() < pufferl->last_log_time + 0.6 && epoch < train_epochs - 1) {
            continue;
        }

        Dict new_log = {0};
        if (epoch >= train_epochs) {
            trainer_eval_log(pufferl, &new_log);
        } else {
            trainer_log(pufferl, &new_log);
        }
        puf_log_update(&last_log, &new_log);
        if (epoch < train_epochs) {
            selfplay_step(selfplay, pufferl, &last_log, (int)epoch);
        }
        if (ctx->artifact_owner) {
            puf_dashboard_print(cfg, pufferl, &last_log, (int)epoch);
        }

        if (puf_log_get_or(&last_log, target_key, -1) < 0) {
            continue;
        }
        if (epoch < train_epochs) {
            puf_log_history_add(&log_history, &last_log);
        }
        if (epoch >= train_epochs && puf_log_get_or(&last_log, "env/n", 0) > eval_episodes) {
            break;
        }
    }

    train_result_fill(&result, &log_history, &last_log, cfg, target_key);
    if (ctx->artifact_owner) {
        puf_log_history_add(&log_history, &last_log);
        char log_path[4096];
        snprintf(log_path, sizeof(log_path), "%s/%s.ini", log_dir, run_id);
        puf_log_write(log_path, cfg, &log_history);
    }
    puf_log_history_free(&log_history);
    free(selfplay);
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

static TrainResult launch_train(Config* cfg) {
    int world_size = (int)puf_config_val(cfg, "train.gpus");
    if (world_size < 1) {
        fprintf(stderr, "config error: [train] gpus must be >= 1\n");
        exit(1);
    }
    const char* offset_item = puf_config_get(cfg, "base.gpu_offset");
    int gpu_offset = offset_item ? (int)puf_config_val(cfg, "base.gpu_offset") : 0;

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
                .gpu_id = gpu_offset + gpu_for_rank(rank, world_size),
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
        .gpu_id = gpu_offset + gpu_for_rank(0, world_size),
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(cfg, &host);
    wait_children(pids, world_size - 1);
    free(pids);
    return result;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 3) {
        fprintf(stderr, "usage: %s train|eval|eval_bot|match|sweep ENV [section.key=value ...]\n", argv[0]);
        exit(1);
    }

    const char* mode = argv[1];
    const char* env_name = argv[2];
    Config cfg = {0};
    puf_config_load_env(&cfg, env_name, argc - 3, argv + 3);
    puf_config_validate_train(&cfg);

    if (strcmp(mode, "train") == 0) {
        TrainResult result = launch_train(&cfg);
        const char* fd_item = puf_config_get(&cfg, "base.result_fd");
        if (fd_item) {
            int fd = (int)puf_config_val(&cfg, "base.result_fd");
            if (write(fd, &result, sizeof(result)) != sizeof(result)) {
                fprintf(stderr, "failed to write train result\n");
                exit(1);
            }
            close(fd);
        }
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(&cfg, argv[0]);
    } else if (strcmp(mode, "eval") == 0) {
        TrainContext ctx = {
            .rank = 0,
            .world_size = 1,
            .gpu_id = 0,
            .artifact_owner = 1,
            .nccl_id = NULL,
        };
        run_eval(&cfg, &ctx);
    } else if (strcmp(mode, "eval_bot") == 0) {
        TrainContext ctx = {
            .rank = 0,
            .world_size = 1,
            .gpu_id = 0,
            .artifact_owner = 1,
            .nccl_id = NULL,
        };
        run_eval_bot(&cfg, &ctx);
    } else if (strcmp(mode, "match") == 0) {
        TrainContext ctx = {
            .rank = 0,
            .world_size = 1,
            .gpu_id = 0,
            .artifact_owner = 1,
            .nccl_id = NULL,
        };
        run_match(&cfg, &ctx);
    } else if (strcmp(mode, "league_match_worker") == 0) {
        TrainContext ctx = {
            .rank = 0,
            .world_size = 1,
            .gpu_id = 0,
            .artifact_owner = 1,
            .nccl_id = NULL,
        };
        const char* offset_item = puf_config_get(&cfg, "base.gpu_offset");
        if (offset_item) {
            ctx.gpu_id = (int)puf_config_val(&cfg, "base.gpu_offset");
        }
        run_league_match_worker(&cfg, &ctx);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        exit(1);
    }

    puf_config_free(&cfg);
    return 0;
}
