# Direct C++ binding to ALE
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np
cimport numpy as cnp
from ale_py.libale import ALEInterface

cdef class CEnduroEnv:
    cdef:
        ALEInterface* ale
        cnp.uint8_t[:, :, :] screen_buffer  # Frame buffer (210, 160, 3)
        int frame_skip
        int terminal
        int score
        int lives

    def __init__(self, int frame_skip=4):
        # Set up the ALE interface (direct C++ binding)
        self.ale = new_ALEInterface()
        self.ale.loadROM("enduro.bin")
        
        # Create the frame buffer
        self.screen_buffer = np.zeros((210, 160, 3), dtype=np.uint8)
        
        self.frame_skip = frame_skip
        self.reset()

    def reset(self):
        self.ale.reset_game()
        self.score = 0
        self.terminal = 0
        self.lives = self.ale.lives()
        self._update_screen_buffer()

    cdef void _update_screen_buffer(self):
        # Get raw screen from ALE and store it in screen_buffer
        cdef const uint8_t* screen_ptr = self.ale.getScreenRGB()
        # Directly copy the memory into the numpy array
        memcpy(&self.screen_buffer[0, 0, 0], screen_ptr, 210 * 160 * 3)

    def step(self, int action):
        cdef int total_reward = 0
        for i in range(self.frame_skip):
            total_reward += self.ale.act(action)
            if self.ale.game_over():
                self.terminal = 1
                break

        self._update_screen_buffer()
        self.score += total_reward
        self.lives = self.ale.lives()
        return self.screen_buffer, total_reward, self.terminal

    def get_screen(self):
        return self.screen_buffer

    def get_score(self):
        return self.score

    def get_lives(self):
        return self.lives

    def __dealloc__(self):
        del self.ale
