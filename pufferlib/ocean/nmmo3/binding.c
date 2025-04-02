#include "nmmo3.h"

#define Env MMO
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_players = unpack(kwargs, "num_players");
    env->num_enemies = unpack(kwargs, "num_enemies");
    env->num_resources = unpack(kwargs, "num_resources");
    env->num_weapons = unpack(kwargs, "num_weapons");
    env->num_gems = unpack(kwargs, "num_gems");
    env->tiers = unpack(kwargs, "tiers");
    env->levels = unpack(kwargs, "levels");
    env->teleportitis_prob = unpack(kwargs, "teleportitis_prob");
    env->enemy_respawn_ticks = unpack(kwargs, "enemy_respawn_ticks");
    env->item_respawn_ticks = unpack(kwargs, "item_respawn_ticks");
    env->x_window = unpack(kwargs, "x_window");
    env->y_window = unpack(kwargs, "y_window");
    env->reward_combat_level = unpack(kwargs, "reward_combat_level");
    env->reward_prof_level = unpack(kwargs, "reward_prof_level");
    env->reward_item_level = unpack(kwargs, "reward_item_level");
    env->reward_market = unpack(kwargs, "reward_market");
    env->reward_death = unpack(kwargs, "reward_death");
    init(env);
    return 0;
}
