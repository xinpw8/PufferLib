# In your __init__.py (for your custom env)
from gymnasium.envs.registration import register

# Register the custom EnduroCy environment
register(
    id='EnduroCy-v0',  # Custom ID
    entry_point='pufferlib.environments.ocean.enduro_cy.py_enduro:EnduroEnv',  # Your custom environment path
    kwargs={'obs_type': 'rgb', 'frameskip': 4},  # Default arguments
    max_episode_steps=10000,
)
