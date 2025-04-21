import functools
import numpy as np

import pufferlib

from src.simulation.env import TradingEnvironment

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config_path='trade_sim/config/experiment_config.yaml', render_mode='human', buf=None, seed=1):
    '''Crafter creation function'''
    from src.utils.config_manager import ConfigManager
    from src.data_ingestion.historical_data_reader import HistoricalDataReader

    config_manager = ConfigManager(config_path)
    config = config_manager.config
    data_reader = HistoricalDataReader(config_manager)
    data, _ = data_reader.preprocess_data()
    
    # Create environment
    env = TradingEnvironmentPuff(config_manager.config, data)
    return pufferlib.emulation.GymnasiumPufferEnv(env, buf=buf)

class TradingEnvironmentPuff(TradingEnvironment):
    def __init__(self, config, data):
        super().__init__(config, data)

    def reset(self):
        obs, info = super().reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if not terminated and not truncated:
            info = {}

        return obs.astype(np.float32), reward, terminated, truncated, info

