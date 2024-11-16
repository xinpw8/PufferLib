cimport numpy as cnp
from libc.stdlib cimport calloc, free

cdef extern from "squared.h":
    ctypedef struct Squared:
        unsigned char* observations
        int* actions
        float* rewards
        unsigned char* terminals
        int size
        int tick
        int r
        int c

    ctypedef struct Client

    void reset(Squared* env)
    void step(Squared* env)

    Client* make_client(Squared* env)
    void close_client(Client* client)
    void render(Client* client, Squared* env)

cdef class CySquared:
    cdef:
        Squared* envs
        Client* client
        int num_envs
        int size

    def __init__(self, unsigned char[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs, int size):

        self.envs = <Squared*> calloc(num_envs, sizeof(Squared))
        self.num_envs = num_envs
        self.client = NULL

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Squared(
                observations = &observations[i, 0],
                actions = &actions[i],
                rewards = &rewards[i],
                terminals = &terminals[i],
                size=size,
            )

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef Squared* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)
