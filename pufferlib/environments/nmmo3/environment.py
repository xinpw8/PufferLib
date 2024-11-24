import functools
from nmmo3 import PuffEnv

def env_creator(name='nmmo3'):
    return make
 
def make(num_envs=1, reward_combat_level=1.0,
        reward_prof_level=1.0, reward_item_level=0.5,
        reward_market=0.01, reward_death_mmo=1.0, buf=None):
    return PuffEnv(
        width=4*[512],
        height=4*[512],
        num_envs=4,
        num_players=1024,
        num_enemies=2048,
        num_resources=2048,
        num_weapons=1024,
        num_gems=512,
        tiers=1,
        levels=8,
        player_respawn_prob=0.001,
        enemy_respawn_ticks=10,
        item_respawn_ticks=100,
        x_window=7,
        y_window=5,
        reward_combat_level=reward_combat_level,
        reward_prof_level=reward_prof_level,
        reward_item_level=reward_item_level,
        reward_market=reward_market,
        reward_death=reward_death_mmo,
        buf=buf,
    )
