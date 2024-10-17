# cy_enduro_clone.pyx

cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

cimport numpy as cnp
from numpy cimport npy_intp

# cnp.import_array()
from libc.stdlib cimport calloc, free
import os

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

    ctypedef struct Car:
        float x
        float y
        int passed

    ctypedef struct Enduro:
        float* observations
        int actions  
        float* rewards
        unsigned char* terminals
        unsigned char* truncateds
        LogBuffer* log_buffer
        Log log

        float width
        float height
        float hud_height
        float car_width
        float car_height
        int max_enemies
        float crash_noop_duration
        float day_length
        int initial_cars_to_pass
        float min_speed
        float max_speed    

        float player_x
        float player_y
        float speed
        
        # ints
        int score
        int day
        int step_count
        int numEnemies
        int carsToPass

        float collision_cooldown
        float action_height

        Car enemyCars[10]

    ctypedef struct Client

    void init(Enduro* env)
    void reset(Enduro* env)
    void step(Enduro* env)

    Client* make_client(Enduro* env)
    void close_client(Client* client)
    void render(Client* client, Enduro* env)

cdef class CyEnduro:
    cdef:
        Enduro* envs        
        Client* client     
        LogBuffer* logs     
        int num_envs        


    def __init__(self,              cnp.float32_t[:, ::1] observations,
             cnp.int32_t[::1] actions,
             cnp.float32_t[::1] rewards,
             cnp.uint8_t[::1] terminals,
             cnp.uint8_t[::1] truncateds,
                int num_envs, float width, float height, float hud_height, float car_width, float car_height,
                int max_enemies, float crash_noop_duration, float day_length, 
                int initial_cars_to_pass, float min_speed, float max_speed):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <Enduro*> calloc(num_envs, sizeof(Enduro))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)


        # cdef cnp.ndarray observations_i
        # cdef cnp.ndarray actions_i
        # cdef cnp.ndarray rewards_i
        # cdef cnp.ndarray terminals_i
        # cdef cnp.ndarray truncateds_i

        # cdef int i
        # # each env gets a slice of the observations, actions, rewards, and terminals
        # for i in range(num_envs):
        #     observations_i = observations[i:i+1]
        #     actions_i = actions[i:i+1]
        #     rewards_i = rewards[i:i+1]
        #     terminals_i = terminals[i:i+1]
        #     truncateds_i = truncateds[i:i+1]

        #     # init each field of Enduro struct correctly
        #     self.envs[i] = Enduro(
        #         observations = <float*> observations_i.data,
        #         actions = <int> actions_i,  
        #         rewards = <float*> rewards_i.data,
        #         terminals = <unsigned char*> terminals_i.data,
        #         truncateds = <unsigned char*> truncateds_i.data,

        cdef int i
        cdef float* observations_i_data
        cdef int actions_i
        cdef float rewards_i
        cdef unsigned char terminals_i
        cdef unsigned char truncateds_i


        for i in range(num_envs):
            observations_i_data = &observations[i, 0]
            actions_i = actions[i]

            env = &self.envs[i]  # Get pointer to the struct

            # Assign fields individually using '.' since 'env' is a pointer
            env.observations = observations_i_data
            env.actions = actions_i
            env.rewards = &rewards[i]
            env.terminals = &terminals[i]
            env.truncateds = &truncateds[i]
            env.log_buffer = self.logs

            # Initialize the log struct
            env.log.episode_return = 0.0
            env.log.episode_length = 0.0
            env.log.score = 0.0

            # Assign other fields
            env.width = width
            env.height = height
            env.hud_height = hud_height
            env.car_width = car_width
            env.car_height = car_height
            env.max_enemies = max_enemies
            env.crash_noop_duration = crash_noop_duration
            env.day_length = day_length
            env.initial_cars_to_pass = initial_cars_to_pass
            env.min_speed = min_speed
            env.max_speed = max_speed

            env.player_x = 0.0
            env.player_y = 0.0
            env.speed = min_speed

            env.score = 0
            env.day = 1
            env.step_count = 0
            env.numEnemies = 0
            env.carsToPass = initial_cars_to_pass

            env.collision_cooldown = 0.0
            env.action_height = height - hud_height

            # Initialize enemyCars array
            for j in range(max_enemies):
                env.enemyCars[j].x = 0.0
                env.enemyCars[j].y = 0.0
                env.enemyCars[j].passed = 0

            init(env)  # Initialize the environment

            # # Manually initialize each field of each car in enemyCars
            # for j in range(max_enemies):
            #     self.envs[i].enemyCars[j].x = 0.0
            #     self.envs[i].enemyCars[j].y = 0.0
            #     self.envs[i].enemyCars[j].passed = 0  # Set each field of the Car struct
            # init(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])  

    def step(self):
        cdef int i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] obs_array
        cdef npy_intp dims[1]
        print(f"Step called, num_envs: {self.num_envs}")
        for i in range(self.num_envs):
            dims[0] = 5
            obs_array = cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_FLOAT32, self.envs[i].observations)    
            print(f"Observations for env {i}: {obs_array[:5]}")

            if &self.envs[i] == NULL:
                print(f"envs[{i}] is NULL")
                continue
            step(&self.envs[i])


    def render(self):
        cdef Enduro* env = &self.envs[0]  
        if self.client == NULL:
            self.client = make_client(env)  

        render(self.client, env)  

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        # for i in range(self.num_envs):
        #     free_enduro_clone(&self.envs[i])
        free(self.envs)
        free(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
