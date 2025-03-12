import functools

import pufferlib
import pufferlib.emulation

train_levels =[
    "s/h0_weak_thrust",
    "s/h7_unicycle_left",
    "s/h3_point_the_thruster",
    "s/h4_thrust_aim",
    "s/h1_thrust_over_ball",
    "s/h5_rotate_fall",
    "s/h9_explode_then_thrust_over",
    "s/h6_unicycle_right",
    "s/h8_unicycle_balance",
    "s/h2_one_wheel_car",

    "m/h0_unicycle",
    "m/h1_car_left",
    "m/h2_car_right",
    "m/h3_car_thrust",
    "m/h4_thrust_the_needle",
    "m/h5_angry_birds",
    "m/h6_thrust_over",
    "m/h7_car_flip",
    "m/h8_weird_vehicle",
    "m/h9_spin_the_right_way",
    "m/h10_thrust_right_easy",
    "m/h11_thrust_left_easy",
    "m/h12_thrustfall_left",
    "m/h13_thrustfall_right",
    "m/h14_thrustblock",
    "m/h15_thrustshoot",
    "m/h16_thrustcontrol_right",
    "m/h17_thrustcontrol_left",
    "m/h18_thrust_right_very_easy",
    "m/h19_thrust_left_very_easy",
    "m/arm_left",
    "m/arm_right",
    "m/arm_up",
    "m/arm_hard",

    "l/h0_angrybirds",
    "l/h1_car_left",
    "l/h2_car_ramp",
    "l/h3_car_right",
    "l/h4_cartpole",
    "l/h5_flappy_bird",
    "l/h6_lorry",
    "l/h7_maze_1",
    "l/h8_maze_2",
    "l/h9_morph_direction",
    "l/h10_morph_direction_2",
    "l/h11_obstacle_avoidance",
    "l/h12_platformer_1",
    "l/h13_platformer_2",
    "l/h14_simple_thruster",
    "l/h15_swing_up",
    "l/h16_thruster_goal",
    "l/h17_unicycle",
    "l/hard_beam_balance",
    "l/hard_cartpole_thrust",
    "l/hard_cartpole_wheels",
    "l/hard_lunar_lander",
    "l/hard_pinball",
    "l/grasp_hard",
    "l/grasp_easy",
    "l/mjc_half_cheetah",
    "l/mjc_half_cheetah_easy",
    "l/mjc_hopper",
    "l/mjc_hopper_easy",
    "l/mjc_swimmer",
    "l/mjc_walker",
    "l/mjc_walker_easy",
    "l/car_launch",
    "l/car_swing_around",
    "l/chain_lander",
    "l/chain_thrust",
    "l/gears",
    "l/lever_puzzle",
    "l/pr",
    "l/rail",
]

def env_creator(name='kinetix'):
    return functools.partial(make, name)

def make(name, num_envs=2048, buf=None):
    from kinetix.environment.env import make_kinetix_env_from_name
    from kinetix.environment.env_state import EnvParams, StaticEnvParams
    from kinetix.util.learning import get_eval_levels

    env_params = EnvParams()
    static_env_params = StaticEnvParams()
    env = make_kinetix_env_from_name('Kinetix-Symbolic-MultiDiscrete-v1', static_env_params=static_env_params)

    import jax
    v = get_eval_levels(train_levels, static_env_params)
    def reset(rng):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        idx = jax.random.randint(_rng, (), 0, len(train_levels))
        return jax.tree.map(lambda x: x[idx], v)

    env.reset = reset
    return pufferlib.emulation.GymnaxPufferEnv(env, env_params, num_envs=num_envs, buf=buf)
