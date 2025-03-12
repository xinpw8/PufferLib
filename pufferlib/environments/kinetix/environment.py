import numpy as np
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
    return KinetixPufferEnv
    return functools.partial(KinetixPufferEnv, name)

def make(name, num_envs=2048, buf=None):
    from kinetix.environment.env import make_kinetix_env_from_args
    from kinetix.environment.env_state import EnvParams, StaticEnvParams
    from kinetix.environment.ued.ued_state import UEDParams
    from kinetix.environment.ued.distributions import sample_kinetix_level

    # Use default parameters
    env_params = EnvParams()
    static_env_params = StaticEnvParams()
    ued_params = UEDParams()

    # Create the environment
    env = make_kinetix_env_from_args(
        obs_type="pixels",
        action_type="multidiscrete",
        reset_type="replay",
        static_env_params=static_env_params,
    )

    # Sample a random level
    import jax
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    level = sample_kinetix_level(_rng, env.physics_engine, env_params, static_env_params, ued_params)

    # Reset the environment state to this level
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset_to_level(_rng, level, env_params)

    env.single_observation_space = env._observation_space.observation_space(env_state)
    env.single_action_space = env.action_space(env_state)

    #env._observation_space.get_obs(env_state)
    '''
    from kinetix.environment.env import make_kinetix_env_from_name
    from kinetix.environment.env_state import EnvParams, StaticEnvParams
    from kinetix.util.learning import get_eval_levels

    # Hack patch around local file path
    import kinetix.util.learning
    kinetix.util.learning.BASE_DIR = '../Kinetix/' + kinetix.util.learning.BASE_DIR

    env_params = EnvParams()
    static_env_params = StaticEnvParams()
    env = make_kinetix_env_from_name('Kinetix-Symbolic-MultiDiscrete-v1', static_env_params=static_env_params)
    #env._observation_space.get_obs
    breakpoint()

    import jax
    v = get_eval_levels(train_levels, static_env_params)
    def reset(rng):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        idx = jax.random.randint(_rng, (), 0, len(train_levels))
        return jax.tree.map(lambda x: x[idx], v)

    env.reset = reset
    '''


    return pufferlib.emulation.GymnaxPufferEnv(env, env_params, num_envs=num_envs, buf=buf)

class KinetixPufferEnv(pufferlib.environment.PufferEnv):
    def __init__(self, num_envs=1, buf=None):
        from kinetix.environment.env import make_kinetix_env_from_args
        from kinetix.environment.env_state import EnvParams, StaticEnvParams
        from kinetix.environment.ued.ued_state import UEDParams
        from kinetix.environment.ued.distributions import sample_kinetix_level
        from kinetix.environment.wrappers import AutoResetWrapper, UnderspecifiedToGymnaxWrapper

        from kinetix.util.learning import get_eval_levels
        import kinetix.util.learning
        kinetix.util.learning.BASE_DIR = '../Kinetix/' + kinetix.util.learning.BASE_DIR

        # Use default parameters
        env_params = EnvParams()
        static_env_params = StaticEnvParams()
        ued_params = UEDParams()

        # Create the environment
        env = make_kinetix_env_from_args(
            obs_type="pixels",
            action_type="discrete",
            reset_type="replay",
            static_env_params=static_env_params,
        )

        # Sample a random level
        import jax
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        level = sample_kinetix_level(_rng, env.physics_engine, env_params, static_env_params, ued_params)

        # Reset the environment state to this level
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset_to_level(_rng, level, env_params)

        from gymnax.environments.spaces import gymnax_space_to_gym_space
        self.single_observation_space = gymnax_space_to_gym_space(
            env._observation_space.observation_space(env_state))
        self.single_action_space = gymnax_space_to_gym_space(
            env.action_space(env_state))
        self.num_agents = num_envs

        v = get_eval_levels(train_levels, static_env_params)
        def reset(rng):
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            idx = jax.random.randint(_rng, (), 0, len(train_levels))
            return jax.tree.map(lambda x: x[idx], v)

        env = UnderspecifiedToGymnaxWrapper(AutoResetWrapper(env, reset))
        self.env = env

        super().__init__(buf)
        self.env_params = env_params
        self.env = env

        self.reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        self.step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        self.rng = jax.random.PRNGKey(0)

    def reset(self, rng, params=None):
        import jax
        self.rng, _rng = jax.random.split(self.rng)
        self.rngs = jax.random.split(_rng, self.num_agents)
        obs, self.state = self.reset_fn(self.rngs, params)
        obs = obs.image
        from torch.utils import dlpack as torch_dlpack
        self.observations = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(obs))
        return self.observations, []

    def step(self, action):
        import jax
        #self.rng, _rng = jax.random.split(self.rng)
        #rngs = jax.random.split(_rng, self.num_agents)
        obs, self.state, reward, done, info = self.step_fn(self.rngs, self.state, action, self.env_params)

        # Convert JAX array to DLPack, then to PyTorch tensor
        from torch.utils import dlpack as torch_dlpack
        self.observations = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(obs))
        self.rewards = np.asarray(reward)
        self.terminals = np.asarray(done)
        infos = [{k: v.mean().item() for k, v in info.items()}]
        return self.observations, self.rewards, self.terminals, self.terminals, infos
