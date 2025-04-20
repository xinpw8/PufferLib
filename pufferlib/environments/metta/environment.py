import functools

import pufferlib


def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/metta.yaml', render_mode='human', buf=None, seed=0):
    '''Crafter creation function'''
    return MettaPuff(config, render_mode, buf)

class MettaPuff(pufferlib.PufferEnv):
    def __init__(self, config, render_mode='human', buf=None, seed=0):
        self.render_mode = render_mode
        import mettagrid.mettagrid_env
        self.env = mettagrid.mettagrid_env.make_env_from_cfg(config, render_mode, buf=buf)
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.num_agents = self.env.num_agents
        super().__init__(buf)

        from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer
        self.env._renderer =  MettaGridRaylibRenderer(self.env, self.env._env_cfg['game'])


    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)

        self.tick += 1
        if self.tick % 128 == 0:
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']
        else:
            info = []

        return obs, rew, term, trunc, [info]

    def reset(self, seed=None):
        obs = self.env.reset()
        self.tick = 0
        return obs, []

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
