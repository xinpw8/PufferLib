# cy_enduro_cy.pyx

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t
from ale_py import ALEInterface  # Use ALEInterface as a Python object

cdef class CEnduroEnv:
    cdef:
        object ale  # ALEInterface instance as a Python object
        int frame_skip
        int terminal
        int score
        int lives

    def __init__(self, int frame_skip=4):
        # Initialize the ALE interface for the environment
        self.ale = ALEInterface()  # ALEInterface is treated as a Python object
        self.ale.loadROM("/puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/enduro_cy/roms/enduro.bin")


        self.frame_skip = frame_skip
        self.reset()

    def reset(self, int seed=-1, **kwargs):
        """Resets the environment."""
        if seed >= 0:
            self.ale.setInt("random_seed", seed)  # If seed is provided, set it.
        self.ale.reset_game()
        self.score = 0
        self.terminal = 0
        self.lives = self.ale.lives()
        return self._update_screen_buffer(), {}

    cdef cnp.ndarray[cnp.uint8_t, ndim=3] _update_screen_buffer(self):
        """Update and return the screen buffer as a NumPy array"""
        # Create a local buffer to store the screen data
        cdef cnp.ndarray[cnp.uint8_t, ndim=3] screen_buffer = np.zeros((210, 160, 3), dtype=np.uint8)

        # Retrieve the raw screen from ALE as an RGB array
        screen_array = self.ale.getScreenRGB()

        # Copy the screen data into the buffer
        np.copyto(screen_buffer, np.frombuffer(screen_array, dtype=np.uint8).reshape((210, 160, 3)))

        return screen_buffer

    def step(self, int action):
        """Perform a step in the environment and return the new state, reward, and terminal status"""
        cdef int total_reward = 0
        for i in range(self.frame_skip):
            total_reward += self.ale.act(action)  # Perform the action and accumulate reward
            if self.ale.game_over():
                self.terminal = 1  # Game over flag
                break

        truncated = False  # No truncation logic, so set this to False
        info = {}  # Placeholder for additional info, can be an empty dict

        return self._update_screen_buffer(), total_reward, self.terminal, truncated, info



    def get_screen(self):
        """Returns the current screen buffer"""
        return self._update_screen_buffer()

    def get_score(self):
        """Returns the current score"""
        return self.score

    def get_lives(self):
        """Returns the current number of lives"""
        return self.lives

    def __dealloc__(self):
        """Clean up and release the ALE interface"""
        self.ale = None
