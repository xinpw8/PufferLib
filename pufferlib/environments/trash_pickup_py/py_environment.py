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
    metadata = {'render_modes': ['human'], 'name': 'PyTrashPickup'}

    def __init__(self, grid_size=10, num_agents=3, num_trash=15, num_bins=1, max_steps=300):
        # Validate num_agents
        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be an integer greater than 0.")
        self._num_agents = num_agents

        # Handle num_trash input
        if isinstance(num_trash, tuple):
            if len(num_trash) != 2 or not all(isinstance(x, int) for x in num_trash) or num_trash[0] > num_trash[1] or num_trash[0] < 0:
                raise ValueError("num_trash must be an int or a tuple of two non-negative integers where the first is <= the second.")
            self.num_trash = np.random.randint(num_trash[0], num_trash[1] + 1)
        elif isinstance(num_trash, int) and num_trash >= 0:
            self.num_trash = num_trash
        else:
            raise ValueError("num_trash must be an int >= 0 or a tuple of two non-negative integers.")

        # Handle num_bins input
        if isinstance(num_bins, tuple):
            if len(num_bins) != 2 or not all(isinstance(x, int) for x in num_bins) or num_bins[0] > num_bins[1] or num_bins[0] < 0:
                raise ValueError("num_bins must be an int or a tuple of two non-negative integers where the first is <= the second.")
            self.num_bins = np.random.randint(num_bins[0], num_bins[1] + 1)
        elif isinstance(num_bins, int) and num_bins >= 0:
            self.num_bins = num_bins
        else:
            raise ValueError("num_bins must be an int >= 0 or a tuple of two non-negative integers.")

        # Calculate minimum required grid size
        min_grid_size = int((num_agents + self.num_trash + self.num_bins) ** 0.5) + 1
        if not isinstance(grid_size, int) or grid_size < min_grid_size:
            raise ValueError(
                f"grid_size must be an integer >= {min_grid_size}. "
                f"Received grid_size={grid_size}, with num_agents={num_agents}, num_trash={self.num_trash}, and num_bins={self.num_bins}."
            )
        self.grid_size = grid_size

        # Define agent IDs
        self.possible_agents = ["agent_" + str(i) for i in range(self._num_agents)]
        self.agents = self.possible_agents[:]

        # Define maximum steps per episode and track the current step
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize episode reward tracking
        self.total_episode_reward = 0

        # Define rewards for positive (picking up or depositing trash) and negative (every environment step) outcomes
        self.POSITIVE_REWARD_PICKUP_OR_DROPOFF = 0.5 / num_trash  # this adds +1 the total episode reward if all trash is picked up and put into bin
        self.NEGATIVE_REWARD = - (1 / (max_steps * num_agents)) # this makes the max negative reward given per episode to be -1

        # Define action and observation spaces
        # Action space: Discrete with 4 actions (UP, DOWN, LEFT, RIGHT)
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}

        # Observation space includes the agent's position, whether it's carrying trash, and the grid state
        self.observation_spaces = {
            agent: spaces.Dict({
                'agent_position': spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
                'carrying_trash': spaces.Discrete(2),  # 0 or 1 for whether the agent is carrying trash
                'other_agent_positions': spaces.Box(low=0, high=self.grid_size - 1, shape=(self._num_agents - 1, 2), dtype=np.int32),
                'other_agent_carrying_trash': spaces.MultiBinary(self._num_agents - 1),  # Binary indicators for other agents carrying trash
                'bin_positions': spaces.Box(low=0, high=self.grid_size - 1, shape=(self.num_bins, 2), dtype=np.int32),
                'trash_positions': spaces.Box(low=-1, high=self.grid_size - 1, shape=(self.num_trash, 3), dtype=np.int32),  # Presence indicator, x, y
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
        self.rewards = {agent: 0 for agent in self.agents}

        # Initialize the grid
        # 0: empty, 1: trash, 2: trash bin, 3: agent
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place bins
        bin_positions = self._random_positions(self.num_bins)
        for pos in bin_positions:
            self.grid[pos] = GridState.TRASH_BIN.value  # Trash bin
        self._bin_positions = bin_positions

        # Place trash
        trash_positions = self._random_positions(self.num_trash, exclude=bin_positions)
        self._trash_positions = trash_positions
        for pos in trash_positions:
            self.grid[pos] = GridState.TRASH.value  # Trash

        # Place agents
        agent_positions = self._random_positions(self._num_agents, exclude=trash_positions + bin_positions)
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
        self.rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Process each agent's action and update the grid accordingly
        for agent in self.agents:
            # Get current state of the agent
            action = actions[agent]
            current_pos = self._agent_positions[agent]
            carrying = self._agent_carrying[agent]

            # Compute new potential position based on the action
            new_pos = self._get_move_pos(current_pos, action)
            new_pos_x, new_pos_y = new_pos

            # Validate grid boundaries for the new position
            if (0 <= new_pos_x < self.grid_size and 0 <= new_pos_y < self.grid_size):
                # Check cell state and update agent's position or inventory
                this_grid_state = self.grid[new_pos]
                if this_grid_state == GridState.EMPTY.value:
                    # Move to empty cell
                    self._move_agent_or_bin(current_pos, new_pos, False, agent_id=agent)
                elif this_grid_state == GridState.TRASH.value and carrying == 0:
                    # Pick up trash if agent is not already carrying
                    self._agent_carrying[agent] = 1
                    self._move_agent_or_bin(current_pos, new_pos, False, agent_id=agent)
                    self.add_reward(agent, self.POSITIVE_REWARD_PICKUP_OR_DROPOFF)
                elif this_grid_state == GridState.TRASH_BIN.value:
                    if carrying == 1:
                        # Deposit trash in the bin
                        self._agent_carrying[agent] = 0
                        self.add_reward(agent, self.POSITIVE_REWARD_PICKUP_OR_DROPOFF)
                    else:
                        # Attempt to push the bin to a new position
                        bin_new_pos = self._get_move_pos(new_pos, action)
                        bx, by = bin_new_pos
                        if (0 <= bx < self.grid_size and 0 <= by < self.grid_size) and self.grid[bin_new_pos] == GridState.EMPTY.value:
                            self._move_agent_or_bin(new_pos, bin_new_pos, True) # Move the bin to the new position
                        # Else: Bin can't be moved; agent stays in current position
                # Else: Invalid move or occupied cell: agent remains in place

            # Apply step penalty to encourage efficient actions
            self.add_reward(agent, self.NEGATIVE_REWARD)

        # Check if the episode has ended (max steps reached or conditions met)
        if self.current_step >= self.max_steps or self.is_episode_over():
            # print(f"Ending episode: {self.current_step >= self.max_steps} | {self.is_episode_over()}")
            dones = {agent: True for agent in self.agents}
            infos["final_info"] = {
                "episode_reward": self.total_episode_reward, 
                "episode_length": self.current_step,
                "trash_collected": self.num_trash - np.sum(self.grid == GridState.TRASH.value),
                "any_agent_carrying": all(carrying == 0 for carrying in self._agent_carrying.values()),
        }

        # Update current_step count
        self.current_step += 1

        # Return updated observations, rewards, done flags, and info
        observations = self._get_observations()
        return observations, self.rewards, dones, infos

    def render(self):
        # Initialize window if not already initialized
        if not self.window_initialized:
            self.header_offset = 60
            window_size = self.grid_size * self.cell_size
            pyray.init_window(window_size, window_size + self.header_offset, "Trash Pickup Environment")
            self.window_initialized = True

            # Get the directory of the current file (environment code file)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct the relative path to the image
            image_path = os.path.join(current_dir, "single_puffer_128.png")

            # Load the image using pyray
            agent_image = pyray.load_image(image_path)

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

        # Draw header with current step and total episode reward
        header_text = f"Step: {self.current_step}\nTotal Episode Reward: {self.total_episode_reward:.2f}\nTotal Trash Collected: {self.num_trash - np.sum(self.grid == GridState.TRASH.value)}"
        pyray.draw_text(header_text, 5, 2, 10, pyray.BLACK)  # Draw header at the top-left of the screen

        # Draw grid cells
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_value = self.grid[x, y]
                screen_x = x * self.cell_size
                screen_y = (y * self.cell_size) + self.header_offset

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
                                
                            reward_text = f"{self.rewards[agent]:.2f}"
                            pyray.draw_text(reward_text, int(screen_x + self.cell_size * 0.25), int(screen_y + self.cell_size * 0.25), 8, pyray.ORANGE)


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

    def add_reward(self, agent, val):
        self.rewards[agent] += val
        self.total_episode_reward += val
        
    def is_episode_over(self):
        # Check if there are any trash cells remaining in the grid
        no_trash_left = not np.any(self.grid == GridState.TRASH.value)
        
        # Check if no agents are carrying trash
        no_agent_carrying = all(carrying == 0 for carrying in self._agent_carrying.values())
        
        # The episode is over if there's no trash left and no agents are carrying trash
        return no_trash_left and no_agent_carrying
    
    def close(self):
        pass

    def _get_move_pos(self, position, action):
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
        
    def _move_agent_or_bin(self, current_pos, new_pos, is_bin, agent_id=None):
        if not is_bin and agent_id is None:
            raise AssertionError("_move_agent_or_bin() called but is_bin is false and agent_id is not passed in")
        
        # Clear the current position
        self.grid[current_pos] = GridState.EMPTY.value
        
        # Update the grid and agent/bin positions
        if is_bin:
            # Update the grid and bin positions
            self.grid[new_pos] = GridState.TRASH_BIN.value
            # Update the bin position in self._bin_positions
            self._bin_positions = [
                new_pos if bin_pos == current_pos else bin_pos
                for bin_pos in self._bin_positions
            ]
        else:
            # Update the grid and agent position
            self.grid[new_pos] = GridState.AGENT.value
            self._agent_positions[agent_id] = new_pos


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
        trash_positions = []
        
        # Prepare trash positions with presence indicator
        for pos in self._trash_positions:
            if self.grid[pos] == GridState.TRASH.value:
                trash_positions.append([1, pos[0], pos[1]])  # present, x, y
            else:
                trash_positions.append([0, -1, -1])  # not present

        # Static bin positions
        bin_positions = np.array([[pos[0], pos[1]] for pos in self._bin_positions], dtype=np.int32)

        for agent in self.agents:
            # Get other agents' positions and carrying status
            other_agents_positions = []
            other_agents_carrying = []

            for other_agent in self.agents:
                if other_agent != agent:
                    other_agents_positions.append(self._agent_positions[other_agent])
                    other_agents_carrying.append(self._agent_carrying[other_agent])

            obs = {
                'agent_position': np.array(self._agent_positions[agent], dtype=np.int32),
                'carrying_trash': np.array(self._agent_carrying[agent], dtype=np.int32),
                'other_agent_positions': np.array(other_agents_positions, dtype=np.int32),
                'other_agent_carrying_trash': np.array(other_agents_carrying, dtype=np.int32),
                'bin_positions' : bin_positions,
                'trash_positions': np.array(trash_positions, dtype=np.int32),
                'grid': self.grid.copy()
            }
            observations[agent] = obs
        return observations


def env_creator(grid_size=10, num_agents=3, num_trash=15, num_bins=1, max_steps=300):
    return functools.partial(make,
        grid_size=grid_size,
        num_agents=num_agents,
        num_trash=num_trash,
        num_bins=num_bins,
        max_steps=max_steps,
    )

def make(grid_size=10, num_agents=3, num_trash=15, num_bins=1, max_steps=300):
    env = PyTrashPickupEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_trash=num_trash,
        num_bins=num_bins,
        max_steps=max_steps,
    )

    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
