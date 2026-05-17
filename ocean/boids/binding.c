#include "boids.h"
#define OBS_SIZE BOIDS_OBS_SIZE
#define NUM_ATNS 2
#define ACT_SIZES {3, 3}
#define OBS_TENSOR_T FloatTensor

#define Env Boids
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (unsigned int)dict_get(kwargs, "num_agents")->value;
    env->report_interval = (unsigned)dict_get(kwargs, "report_interval")->value;
    env->margin_turn_factor = (float)dict_get(kwargs, "margin_turn_factor")->value;
    env->cohesion_factor = (float)dict_get(kwargs, "cohesion_factor")->value;
    env->separation_factor = (float)dict_get(kwargs, "separation_factor")->value;
    env->alignment_factor = (float)dict_get(kwargs, "alignment_factor")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "margin_turn_reward", log->t_margin_turn_reward);
    dict_set(out, "cohesion_reward", log->t_cohesion_reward);
    dict_set(out, "separation_reward", log->t_separation_reward);
    dict_set(out, "alignment_reward", log->t_alignment_reward);
    dict_set(out, "speed_reward", log->t_speed_reward);
    dict_set(out, "action_reward", log->t_action_reward);
    dict_set(out, "avg_speed", log->avg_speed);
    dict_set(out, "avg_visual_count", log->avg_visual_count);
    dict_set(out, "avg_protected_count", log->avg_protected_count);
    dict_set(out, "n", log->n);
}
