#include <Python.h>
#include "nmmo3.h"
#define ENV_MODULE_NAME "binding"
#define Env MMO
#define step c_step
#define render c_render
#define reset c_reset

static char *kwlist[] = {"width", "height", "num_players", "num_enemies",
    "num_resources", "num_weapons", "num_gems", "tiers", "levels",
    "teleportitis_prob", "enemy_respawn_ticks", "item_respawn_ticks",
    "x_window", "y_window", "reward_combat_level", "reward_prof_level",
    "reward_item_level", "reward_market", "reward_death", NULL
};

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiiiiiifiiiifffff", kwlist,
            &env->width, &env->height, &env->num_players, &env->num_enemies,
            &env->num_resources, &env->num_weapons, &env->num_gems,
            &env->tiers, &env->levels, &env->teleportitis_prob,
            &env->enemy_respawn_ticks, &env->item_respawn_ticks,
            &env->x_window, &env->y_window, &env->reward_combat_level,
            &env->reward_prof_level, &env->reward_item_level,
            &env->reward_market, &env->reward_death)) {
        return 1;
    }
    init(env);
    return 0;
}

#include "../env_binding.h"
DEFINE_PYINIT(binding)
