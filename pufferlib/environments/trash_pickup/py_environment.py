# Standard libraries for utility functions and environment management
import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from enum import Enum
import numpy as np
import imageio
import os
import raylibpy as pyray  # Library for rendering the environment

# PufferLib libraries for environment emulation and wrappers
import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


class GridState(Enum):
    # Enum to represent the different types of cells in the grid
    EMPTY = 0       # An empty cell in the grid
    TRASH = 1       # Cell containing trash that agents need to pick up
    TRASH_BIN = 2   # Cell representing a trash bin where agents can deposit trash
    AGENT = 3       # Cell occupied by an agent

class PyTrashPickupEnv(ParallelEnv):
    # Metadata to describe render mode and environment name
    metadata = {'render_modes': ['human'], 'name': 'TrashPickupOld'}

    def __init__(self, grid_size=10, num_agents=3, num_trash=15, num_bins=2, max_steps=300):
        # Initialize environment parameters
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.num_trash = num_trash
        self.num_bins = num_bins

        # Define agent IDs
        self.possible_agents = ["agent_" + str(i) for i in range(self._num_agents)]
        self.agents = self.possible_agents[:]

        # Define maximum steps per episode and track the current step
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize episode reward tracking
        self.total_episode_reward = 0

        # Define rewards for positive (picking up or depositing trash) and negative (every environment step) outcomes
        self.POSITIVE_REWARD = 0.5 / num_trash  # this makes the total episode reward +1 if all trash is picked up and put into bin
        self.NEGATIVE_REWARD = 1 / (max_steps * num_agents) # this makes the max negative reward given per episode to be -1

        # Define action and observation spaces
        # Action space: Discrete with 4 actions (UP, DOWN, LEFT, RIGHT)
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
        self.render_mode = 'raylib'

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset episode parameters
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

        # Obtain and return initial observations
        observations = self._get_observations()
        return observations

    def step(self, actions):
        # Check if max steps exceeded without reset
        if self.current_step > self.max_steps:
            print("TrashPickupOldEnv, environment did NOT reset after max steps reached...")
            return

        # Initialize rewards, done flags, and info dictionaries for all agents
        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Prepare agents for movement by setting their current grid positions to empty
        for agent in self.agents:
            pos = self._agent_positions[agent]
            self.grid[pos] = GridState.EMPTY.value

        # Process each agent's action and update the grid accordingly
        for agent in self.agents:
            # Get current state of the agent
            action = actions[agent]
            current_pos = self._agent_positions[agent]
            carrying = self._agent_carrying[agent]

            # Compute new position based on the action
            new_pos = self._move(current_pos, action)
            x, y = new_pos

            # Validate grid boundaries for the new position
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                new_pos = current_pos

            # Check cell state and update agent's position or inventory
            this_grid_state = self.grid[new_pos]
            if this_grid_state == GridState.EMPTY.value:
                # Move to empty cell
                self._agent_positions[agent] = new_pos
            elif this_grid_state == GridState.TRASH.value and carrying == 0:
                # Pick up trash if agent is not already carrying
                self._agent_carrying[agent] = 1
                self.grid[new_pos] = GridState.EMPTY.value
                self._agent_positions[agent] = new_pos
                rewards[agent] += self.POSITIVE_REWARD  # Reward for picking up trash
                self.total_episode_reward += self.POSITIVE_REWARD
            elif this_grid_state == GridState.TRASH_BIN.value:
                if carrying == 1:
                    # Deposit trash in the bin
                    self._agent_carrying[agent] = 0
                    rewards[agent] += self.POSITIVE_REWARD  # Reward for depositing trash
                    self.total_episode_reward += self.POSITIVE_REWARD
                    # Agent does not move onto the bin
                    self._agent_positions[agent] = current_pos
                else:
                    # Attempt to push the bin to a new position
                    bin_new_pos = self._move(new_pos, action)
                    bx, by = bin_new_pos
                    if (0 <= bx < self.grid_size and 0 <= by < self.grid_size) and self.grid[bin_new_pos] == GridState.EMPTY.value:
                        # Move the bin to the new position
                        self.grid[new_pos] = GridState.EMPTY.value
                        self.grid[bin_new_pos] = GridState.TRASH_BIN.value
                    else:
                        # Bin can't be moved; agent stays in current position
                        self._agent_positions[agent] = current_pos
            else:
                # Invalid move or occupied cell: agent remains in place
                self._agent_positions[agent] = current_pos

            # Apply step penalty to encourage efficient actions
            rewards[agent] -= self.NEGATIVE_REWARD
            self.total_episode_reward -= self.NEGATIVE_REWARD

        # Update agents on the grid
        for agent in self.agents:
            pos = self._agent_positions[agent]
            self.grid[pos] = GridState.AGENT.value  # Agent

        # Check if the episode has ended (max steps reached or conditions met)
        if self.current_step >= self.max_steps or self.is_episode_over():
            dones = {agent: True for agent in self.agents}
            infos["final_info"] = {
                "episode_reward": self.total_episode_reward, 
                "episode_length": self.current_step,
                "trash_remaining": np.sum(self.grid == GridState.TRASH.value),
                "any_agent_carrying": all(carrying == 0 for carrying in self._agent_carrying.values()),
            }

        # Update current_step count
        self.current_step += 1

        # Return updated observations, rewards, done flags, and info
        observations = self._get_observations()
        return observations, rewards, dones, infos

    def render(self):
        # Initialize window if not already initialized
        if not self.window_initialized:
            window_size = self.grid_size * self.cell_size
            pyray.init_window(window_size, window_size, "Trash Pickup Environment")
            self.window_initialized = True

            agent_image = pyray.load_image("/workspaces/PufferTank/puffertank/pufferlib/pufferlib/environments/trash_pickup/single_puffer_128.png")

            if agent_image.width != 0 and agent_image.height != 0:
                # Resize the image to match the cell size
                pyray.image_resize(agent_image, self.cell_size, self.cell_size)

                # Convert the resized image to a Texture
                self.agent_texture = pyray.load_texture_from_image(agent_image)
                
                self.agent_image_loaded = True
            else:
                self.agent_image_loaded = False

            # Unload the Image from memory, as it is now loaded as a Texture
            pyray.unload_image(agent_image)

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
                    if self.agent_image_loaded:
                        pyray.draw_texture(self.agent_texture, int(screen_x), int(screen_y), pyray.WHITE)
                    else:
                        # Draw red circle for agent as a backup.
                        pyray.draw_circle(int(rect.x + self.cell_size / 2), int(rect.y + self.cell_size / 2),
                                                self.cell_size * 0.4, pyray.RED)
                    
                    # Draw smaller brown square in lower-right corner to show agent is carrying trash
                    for agent in self.agents:
                        agent_pos = self._agent_positions[agent]
                        if agent_pos[0] == x and agent_pos[1] == y:
                            if self._agent_carrying[agent]:
                                pyray.draw_rectangle(int(rect.x + self.cell_size * 0.5), int(rect.y + self.cell_size * 0.5),
                                            int(self.cell_size * 0.25), int(self.cell_size * 0.25), pyray.BROWN)


        pyray.end_drawing()

        # Capture screen as an Image object
        screen_image = pyray.load_image_from_screen()

        # Save the image temporarily
        temp_filename = "temp_screen_image.png"
        pyray.export_image(screen_image, temp_filename)

        # Load it as a NumPy array
        array_data = imageio.imread(temp_filename)

        # Clean up the temporary file
        os.remove(temp_filename)

        # Unload the image from memory
        pyray.unload_image(screen_image)

        # Return the array
        return array_data[:, :, :3]  # Returning only RGB channels

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
            raise ValueError(f"TrashPickupOldEnv - Unexpected Move Action Received: {action}")

    def _random_positions(self, count, exclude=[]):
        # Generate unique random positions on the grid, excluding specific locations
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

def env_creator(grid_size=10, num_agents=3, num_trash=15, num_bins=2, max_steps=300):
    return functools.partial(make,
        grid_size=grid_size,
        num_agents=num_agents,
        num_trash=num_trash,
        num_bins=num_bins,
        max_steps=max_steps
    )

def make(grid_size=10, num_agents=3, num_trash=15, num_bins=2, max_steps=300, video=False):
    env = PyTrashPickupEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_trash=num_trash,
        num_bins=num_bins,
        max_steps=max_steps
    )

    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    if video:
        env = pufferlib.wrappers.RecordVideo
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
