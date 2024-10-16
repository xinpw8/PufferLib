# cython: language_level=3

cimport numpy as cnp
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t


cdef extern from "enduro_clone.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return
        float episode_length
        float score

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Enduro:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* terminals
        LogBuffer* log_buffer
        Log log
        float player_x
        float player_y
        float speed
        float max_speed
        float min_speed
        int score
        int carsToPass
        int day
        int day_length
        int step_count
        int numEnemies
        int width
        int height
        int collision_cooldown

    ctypedef struct Client

    void init(Enduro* env, int width, int height, int hud_height, int car_width, int car_height, int max_enemies, int crash_noop_duration, int day_length, int initial_cars_to_pass, float min_speed, float max_speed)
    void free_enduro_clone(Enduro* env)

    Client* make_client(Enduro* env)
    void close_client(Client* client)
    void render(Client* client, Enduro* env)
    void reset(Enduro* env)
    void step(Enduro* env)

cdef class CyEnduro:
    cdef:
        Enduro* envs
        Client* client
        LogBuffer* logs
        int num_envs
    
    def __init__(self, cnp.ndarray[cnp.float32_t, ndim=2] observations,
                 cnp.ndarray[cnp.uint8_t, ndim=1] actions,
                 cnp.ndarray[cnp.float32_t, ndim=1] rewards,
                 cnp.ndarray[cnp.uint8_t, ndim=1] terminals,
                 int num_envs, int width, int height, int hud_height, int car_width, int car_height,
                 int max_enemies, int crash_noop_duration, int day_length, 
                 int initial_cars_to_pass, float min_speed, float max_speed):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <Enduro*> calloc(num_envs, sizeof(Enduro))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef:
            cnp.ndarray observations_i
            cnp.ndarray actions_i
            cnp.ndarray rewards_i
            cnp.ndarray terminals_i



        cdef int i
        for i in range(num_envs):
            observations_i = observations[i:i+1]
            actions_i = actions[i:i+1]
            rewards_i = rewards[i:i+1]
            terminals_i = terminals[i:i+1]
            self.envs[i] = Enduro(
                observations = <float*> observations_i.data,
                actions = <unsigned char*> actions_i.data,
                rewards = <float*> rewards_i.data,
                terminals = <unsigned char*> terminals_i.data,
                log_buffer=self.logs,
            )
            
            print(f"obs address: {observations_i.data}, actions address: {actions_i.data}")


            init(&self.envs[i], width, height, hud_height, car_width, car_height, max_enemies, crash_noop_duration, day_length, initial_cars_to_pass, min_speed, max_speed)

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        print(".cy step Step")
        cdef int i
        for i in range(self.num_envs):
            print(f"Stepping env {i}") # The last print statement that gets printed is "Stepping env 0"
            step(&self.envs[i]) # This is where the error occurs.

    def render(self):
        cdef Enduro* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        for i in range(self.num_envs):
            free_enduro_clone(&self.envs[i])
        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
