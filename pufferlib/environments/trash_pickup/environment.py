import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from enum import Enum

import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers

import pufferlib

import pyray

class GridState(Enum):
    EMPTY = 0
    TRASH = 1
    TRASH_BIN = 2
    AGENT = 3

class TrashPickupEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], 'name': 'TrashPickup'}

    def __init__(self, grid_size=10, num_agents=3, num_trash=15, num_bins=2, max_steps=300):
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.num_trash = num_trash
        self.num_bins = num_bins

        self.possible_agents = ["agent_" + str(i) for i in range(self._num_agents)]
        self.agents = self.possible_agents[:]

        self.max_steps = max_steps
        self.current_step = 0

        self.total_episode_reward = 0

        self.POSITIVE_REWARD = 0.5 / num_trash  # this makes the total episode reward +1 if all trash is picked up and put into bin
        self.NEGATIVE_REWARD = 1 / (max_steps * num_agents) # this makes the max negative reward given per episode to be -1

        # Define the action space: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}

        # Observation space includes the agent's position, whether it's carrying trash, and the grid state
        self.observation_spaces = {
            agent: spaces.Dict({
                'agent_position': spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
                'carrying_trash': spaces.Discrete(2),  # 0 or 1
                'grid': spaces.Box(low=0, high=4, shape=(self.grid_size, self.grid_size), dtype=np.int32)
            })
            for agent in self.possible_agents
        }

        # Initialize rendering variables
        self.window_initialized = False
        self.cell_size = 40  # Size of each cell in pixels

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.total_episode_reward = 0

        # Initialize agents
        self.agents = self.possible_agents[:]
        self._agent_positions = {}
        self._agent_carrying = {}

        # Initialize the grid
        # 0: empty, 1: trash, 2: trash bin, 3: agent
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place trash
        trash_positions = self._random_positions(self.num_trash)
        for pos in trash_positions:
            self.grid[pos] = GridState.TRASH.value  # Trash

        # Place bins
        bin_positions = self._random_positions(self.num_bins, exclude=trash_positions)
        for pos in bin_positions:
            self.grid[pos] = GridState.TRASH_BIN.value  # Trash bin

        # Place agents
        agent_positions = self._random_positions(self._num_agents, exclude=trash_positions + bin_positions)  # Changed here
        for idx, agent in enumerate(self.agents):
            pos = agent_positions[idx]
            self._agent_positions[agent] = pos
            self._agent_carrying[agent] = 0
            self.grid[pos] = GridState.AGENT.value  # Agent

        observations = self._get_observations()
        return observations

    def step(self, actions):

        if self.current_step > self.max_steps:
            print("TrashPickup Env, environment did NOT reset after max steps reached...")
            return

        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Remove agents from the grid to prepare for movement
        for agent in self.agents:
            pos = self._agent_positions[agent]
            self.grid[pos] = GridState.EMPTY.value

        # Process each agent's action
        for agent in self.agents:
            action = actions[agent]
            current_pos = self._agent_positions[agent]
            carrying = self._agent_carrying[agent]

            new_pos = self._move(current_pos, action)
            x, y = new_pos

            # Check grid boundaries
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                new_pos = current_pos

            this_grid_state = self.grid[new_pos]

            if this_grid_state == GridState.EMPTY.value:
                # Empty cell: move agent
                self._agent_positions[agent] = new_pos
            elif this_grid_state == GridState.TRASH.value and carrying == 0:
                # Pick up trash
                self._agent_carrying[agent] = 1
                self.grid[new_pos] = GridState.EMPTY.value
                self._agent_positions[agent] = new_pos
                rewards[agent] += self.POSITIVE_REWARD  # Reward for picking up trash
                self.total_episode_reward += self.POSITIVE_REWARD
            elif this_grid_state == GridState.TRASH_BIN.value:
                if carrying == 1:
                    # Deposit trash
                    self._agent_carrying[agent] = 0
                    rewards[agent] += self.POSITIVE_REWARD  # Reward for depositing trash
                    self.total_episode_reward += self.POSITIVE_REWARD
                    # Agent does not move onto the bin
                    self._agent_positions[agent] = current_pos
                else:
                    # Push bin if possible
                    bin_new_pos = self._move(new_pos, action)
                    bx, by = bin_new_pos
                    if (0 <= bx < self.grid_size and 0 <= by < self.grid_size) and self.grid[bin_new_pos] == GridState.EMPTY.value:
                        # Move bin
                        self.grid[new_pos] = GridState.EMPTY.value
                        self.grid[bin_new_pos] = GridState.TRASH_BIN.value
                        # Move agent into bin's old position
                        # self._agent_positions[agent] = new_pos # actually lets make this cost a step by not moving the agent
                    else:
                        # Bin cannot be moved
                        self._agent_positions[agent] = current_pos
            else:
                # Invalid move or occupied cell: stay in place
                self._agent_positions[agent] = current_pos

            rewards[agent] -= self.NEGATIVE_REWARD  # small penalty per step to encourage efficiency
            self.total_episode_reward -= self.NEGATIVE_REWARD

        # Update agents on the grid
        for agent in self.agents:
            pos = self._agent_positions[agent]
            self.grid[pos] = GridState.AGENT.value  # Agent

        if self.current_step >= self.max_steps or self.is_episode_over():
            dones = {agent: True for agent in self.agents}
            infos["final_info"] = {
                "episode_reward": self.total_episode_reward, 
                "episode_length": self.current_step,
                "trash_remaining": np.sum(self.grid == GridState.TRASH.value),
                "any_agent_carrying": all(carrying == 0 for carrying in self._agent_carrying.values()),
            }
        self.current_step += 1

        observations = self._get_observations()
        return observations, rewards, dones, infos

    def render(self):
        # Initialize window if not already initialized
        if not self.window_initialized:
            window_size = self.grid_size * self.cell_size
            pyray.init_window(window_size, window_size, "Trash Pickup Environment")
            self.window_initialized = True

        # Check if window should close
        if pyray.window_should_close():
            self.close()
            return

        pyray.begin_drawing()
        pyray.clear_background(pyray.WHITE)

        # Draw grid cells
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_value = self.grid[x, y]
                screen_x = x * self.cell_size
                screen_y = y * self.cell_size

                rect = pyray.Rectangle(screen_x, screen_y, self.cell_size, self.cell_size)

                # Draw cell background
                pyray.draw_rectangle_lines(int(rect.x), int(rect.y), int(rect.width), int(rect.height), pyray.LIGHTGRAY)

                # Draw cell contents
                if cell_value == GridState.EMPTY.value:
                    pass  # Empty cell
                elif cell_value == GridState.TRASH.value:
                    # Draw trash as a brown square
                    pyray.draw_rectangle(int(rect.x + self.cell_size * 0.25), int(rect.y + self.cell_size * 0.25),
                                            int(self.cell_size * 0.5), int(self.cell_size * 0.5), pyray.BROWN)
                elif cell_value == GridState.TRASH_BIN.value:
                    # Draw trash bin as a blue square
                    pyray.draw_rectangle(int(rect.x + self.cell_size * 0.1), int(rect.y + self.cell_size * 0.1),
                                            int(self.cell_size * 0.8), int(self.cell_size * 0.8), pyray.BLUE)
                elif cell_value == GridState.AGENT.value:
                    # Draw agent as a red circle
                    pyray.draw_circle(int(rect.x + self.cell_size / 2), int(rect.y + self.cell_size / 2),
                                         self.cell_size * 0.4, pyray.RED)

        pyray.end_drawing()

    def is_episode_over(self):
        # Check if there are any trash cells remaining in the grid
        no_trash_left = not np.any(self.grid == GridState.TRASH.value)
        
        # Check if no agents are carrying trash
        no_agent_carrying = all(carrying == 0 for carrying in self._agent_carrying.values())
        
        # The episode is over if there's no trash left and no agents are carrying trash
        return no_trash_left and no_agent_carrying
    
    def close(self):
        pass

    def _move(self, position, action):
        # Map actions to grid movements
        x, y = position
        if action == 0:  # UP
            return (x, y + 1)
        elif action == 1:  # DOWN
            return (x, y - 1)
        elif action == 2:  # LEFT
            return (x - 1, y)
        elif action == 3:  # RIGHT
            return (x + 1, y)
        else:
            raise ValueError(f"TrashPickup Env - Unexpected Move Action Received: {action}")

    def _random_positions(self, count, exclude=[]):
        # Generate unique random positions on the grid
        positions = []
        while len(positions) < count:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if pos not in positions and pos not in exclude:
                positions.append(pos)
        return positions

    def _get_observations(self):
        # Provide observations for each agent
        observations = {}
        for agent in self.agents:
            obs = {
                'agent_position': np.array(self._agent_positions[agent], dtype=np.int32),
                'carrying_trash': np.array(self._agent_carrying[agent], dtype=np.int32),
                'grid': self.grid.copy()
            }
            observations[agent] = obs
        return observations

    def state(self):
        # Return the global state (optional)
        return self.grid.copy()

def env_creator():
    return functools.partial(make)

def make():
    env = TrashPickupEnv()
    # env = aec_to_parallel_wrapper(env)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
