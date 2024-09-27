cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "racing.h":
    ctypedef struct Client
    Client* make_client(CRacing* env)
    void close_client(Client* client)

    ctypedef struct CRacing:
        float car_x
        float car_y
        float car_speed
        unsigned char* actions
        float* rewards
        unsigned char* dones
        int width
        int height

    void allocate(CRacing* env)
    void reset(CRacing* env)
    void step(CRacing* env)
    void render(Client* client, CRacing* env)
    void free_allocated(CRacing* env)

cdef class CyRacing:
    cdef CRacing env
    cdef Client* client

    def __init__(self, int width, int height):
        allocate(&self.env)
        self.env.width = width
        self.env.height = height
        self.client = NULL

    def reset(self):
        reset(&self.env)

    def step(self):
        step(&self.env)

    def render(self):
        if self.client == NULL:
            self.client = make_client(&self.env)
        render(self.client, &self.env)

    def __dealloc__(self):
        free_allocated(&self.env)
        if self.client:
            close_client(self.client)
