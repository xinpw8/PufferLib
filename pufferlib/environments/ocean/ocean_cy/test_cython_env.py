from puffertank.pufferlib.pufferlib.environments.ocean.ocean_cy.py_ocean import make_ocean_cy

# Initialize the environment
env = make_ocean_cy()

# Reset the environment and print the initial observation
observation, info = env.reset()
print(f'Initial observation: {observation}')

# Sample an action to take
action = {'image': 1, 'flat': 0}

# Take a step in the environment
next_obs, reward, terminal, truncated, info = env.step([action])  # action should be passed as a list if num_envs > 1
print(f'Next observation: {next_obs}')
print(f'Reward: {reward}')
print(f'Terminal: {terminal}')
print(f'Truncated: {truncated}')
print(f'Info: {info}')
