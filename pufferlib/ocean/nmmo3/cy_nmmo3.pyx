# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=True
# cython: wraparound=True
# cython: cdivision=False
# cython: nonecheck=True
# cython: profile=True

from libc.stdlib cimport calloc, free
cimport numpy as cnp
import numpy as np

cdef extern from "nmmo3.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float return_comb_lvl;
        float return_prof_lvl;
        float return_item_atk_lvl;
        float return_item_def_lvl;
        float return_market_buy;
        float return_market_sell;
        float return_death;
        float min_comb_prof;
        float purchases;
        float sales;
        float equip_attack;
        float equip_defense;
        float r;
        float c;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    int ATN_NOOP

    ctypedef struct Entity:
        int type
        int comb_lvl
        int element
        int dir
        int anim
        int hp
        int hp_max
        int prof_lvl
        int ui_mode
        int market_tier
        int sell_idx
        int gold
        int in_combat
        int equipment[5]
        int inventory[12]
        int is_equipped[12]
        int wander_range
        int ranged
        int goal
        int equipment_attack
        int equipment_defense
        int r
        int c
        int spawn_r
        int spawn_c
        int min_comb_prof[500]
        int min_comb_prof_idx
        int time_alive;
        int purchases;
        int sales;

    ctypedef struct Reward:
        float total
        float death;
        float pioneer;
        float comb_lvl;
        float prof_lvl;
        float item_atk_lvl;
        float item_def_lvl;
        float item_tool_lvl;
        float market_buy;
        float market_sell;

    ctypedef struct ItemMarket:
        int offer_idx
        int max_offers

    ctypedef struct RespawnBuffer:
        int* buffer
        int ticks
        int size

    ctypedef struct MMO:
        int width
        int height
        int num_players
        int num_enemies
        int num_resources
        int num_weapons
        int num_gems
        char* terrain
        unsigned char* rendered
        Entity* players
        Entity* enemies
        short* pids
        unsigned char* items
        Reward* rewards
        unsigned char* counts
        unsigned char* obs
        int* actions
        int tick
        int tiers
        int levels
        float teleportitis_prob
        int x_window
        int y_window
        int obs_size
        int enemy_respawn_ticks
        int item_respawn_ticks
        ItemMarket* market
        int market_buys
        int market_sells
        RespawnBuffer* resource_respawn_buffer
        RespawnBuffer* enemy_respawn_buffer
        Log* logs
        LogBuffer* log_buffer
        float reward_combat_level
        float reward_prof_level
        float reward_item_level
        float reward_market
        float reward_death

    ctypedef struct Client
    Client* make_client(MMO* env)
    #void close_client(Client* client)
    int tick(Client* client, MMO* env, float delta)

    void init_mmo(MMO* env)
    void reset(MMO* env, int seed)
    void step(MMO* env)

cpdef entity_dtype():
    '''Make a dummy entity to get the dtype'''
    cdef Entity entity
    return np.asarray(<Entity[:1]>&entity).dtype

cpdef reward_dtype():
    '''Make a dummy reward to get the dtype'''
    cdef Reward reward
    return np.asarray(<Reward[:1]>&reward).dtype

cdef class Environment:
    cdef:
        MMO* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, unsigned char[:, :] observations, int[:, :] players,
            int[:, :] enemies, float[:, :] rewards, int[:] actions,
            list width, list height, int num_envs, list num_players,
            list num_enemies, list num_resources, list num_weapons, list num_gems,
            list tiers, list levels, list teleportitis_prob, list enemy_respawn_ticks,
            list item_respawn_ticks, float reward_combat_level, float reward_prof_level,
            float reward_item_level, float reward_market, float reward_death,
            int x_window=7, int y_window=5):

        cdef:
            int total_players = 0
            int total_enemies = 0
            int n_players = 0
            int n_enemies = 0

        self.num_envs = num_envs
        self.envs = <MMO*> calloc(num_envs, sizeof(MMO))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        for i in range(num_envs):
            obs_i = observations[total_players:total_players+n_players]
            rewards_i = rewards[total_players:total_players+n_players]
            players_i = players[total_players:total_players+n_players]
            enemies_i = enemies[total_enemies:total_enemies+n_enemies]
            #counts_i = counts[total_players:total_players+n_players]
            #terrain_i = terrain[total_players:total_players+n_players]
            #rendered_i = rendered[total_players:total_players+n_players]
            actions_i = actions[total_players:total_players+n_players]

            self.envs[i] = MMO(
                obs=&observations[total_players, 0],
                rewards=<Reward*> &rewards[total_players, 0],
                players=<Entity*> &players[total_players, 0],
                enemies=<Entity*> &enemies[total_enemies, 0],
                actions=&actions[total_players],
                width=width[i],
                height=height[i],
                num_players=num_players[i],
                num_enemies=num_enemies[i],
                num_resources=num_resources[i],
                num_weapons=num_weapons[i],
                num_gems=num_gems[i],
                tiers=tiers[i],
                levels=levels[i],
                teleportitis_prob=teleportitis_prob[i],
                enemy_respawn_ticks=enemy_respawn_ticks[i],
                item_respawn_ticks=item_respawn_ticks[i],
                x_window=x_window,
                y_window=y_window,
                log_buffer=self.logs,
                reward_combat_level=reward_combat_level,
                reward_prof_level=reward_prof_level,
                reward_item_level=reward_item_level,
                reward_market=reward_market,
                reward_death=reward_death,
            )
            n_players = num_players[i]
            n_enemies = num_enemies[i]

            init_mmo(&self.envs[i])
            total_players += n_players
            total_enemies += n_enemies

        self.client = NULL

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            # TODO: Seed
            reset(&self.envs[i], i+1)
            # Do I need to reset terrain here?

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def pids(self):
        ary = np.zeros((512, 512), dtype=np.intc)
        cdef int i, j
        for i in range(512):
            for j in range(512):
                ary[i, j] = self.envs[0].pids[512*i + j]
        return ary

    def render(self):
        if self.client == NULL:
            self.client = make_client(&self.envs[0])

        cdef int i, atn
        cdef int action = ATN_NOOP;
        for i in range(36):
            atn = tick(self.client, &self.envs[0], i/36.0)
            if atn != ATN_NOOP:
                action = atn

        self.envs[0].actions[0] = action

    # TODO
    def close(self):
        if self.client != NULL:
            #close_game_renderer(self.renderer)
            self.client = NULL

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
