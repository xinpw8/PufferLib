# cy_racing.pyx
cimport numpy as cnp
from libc.stdlib cimport free
cdef extern from "racing.h":
    ctypedef struct CEnduro:
        float* observations
        unsigned int* actions
        float* rewards
        unsigned char* terminals
        float* player_x_y
        float* other_cars_x_y
        int* other_cars_active
        unsigned int* score_day
        int frameskip
    
    ctypedef struct Client
    
    void init(CEnduro* env)
    void reset(CEnduro* env)
    void step(CEnduro* env)
    Client* make_client(CEnduro* env)
    void close_client(Client* client)
    void render(Client* client, CEnduro* env)

cdef class CyEnduro:
    cdef CEnduro env
    cdef Client* client

    def __init__(self, 
                 cnp.ndarray[cnp.float32_t, ndim=1] observations,
                 cnp.ndarray[cnp.uint32_t, ndim=1] actions,
                 cnp.ndarray[cnp.float32_t, ndim=1] rewards,
                 cnp.ndarray[cnp.uint8_t, ndim=1] terminals,
                 cnp.ndarray[cnp.float32_t, ndim=1] player_x_y,
                 cnp.ndarray[cnp.float32_t, ndim=1] other_cars_x_y,
                 cnp.ndarray[cnp.int32_t, ndim=1] other_cars_active,
                 cnp.ndarray[cnp.uint32_t, ndim=1] score_day, 
                 float width, float height, float player_width, 
                 float player_height, float other_car_width, float other_car_height, 
                 float player_speed, float base_car_speed, float max_player_speed, 
                 float min_player_speed, float speed_increment, unsigned int max_score):
        
        self.client = NULL
        self.env = CEnduro(
            observations=<float*>observations.data,
            actions=<unsigned int*>actions.data,
            rewards=<float*>rewards.data,
            terminals=<unsigned char*>terminals.data,
            player_x_y=<float*>player_x_y.data,
            other_cars_x_y=<float*>other_cars_x_y.data,
            other_cars_active=<int*>other_cars_active.data,
            score_day=<unsigned int*>score_day.data,
            frameskip=4
        )
        init(&self.env)

    def reset(self):
        reset(&self.env)

    def step(self):
        step(&self.env)

    def render(self):
        if self.client == NULL:
            self.client = make_client(&self.env)
        render(self.client, &self.env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL