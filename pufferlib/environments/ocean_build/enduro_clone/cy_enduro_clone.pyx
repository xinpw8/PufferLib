# cython: language_level=3

cimport numpy as cnp
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t


cdef extern from "enduro_clone.h":
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    Log aggregate_and_clear(LogBuffer* logs)
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return
        float episode_length
        float score

    ctypedef struct LogBuffer:
        LogBuffer* allocate_logbuffer(int size)
        void free_logbuffer(LogBuffer* buffer)
        Log aggregate_and_clear(LogBuffer* logs)

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
    #     int width
    #     int height
    #     float player_x
    #     float player_y
    #     float speed
    #     float max_speed
    #     float min_speed
    #     int score
    #     int carsToPass
    #     int day
    #     int day_length
    #     int step_count
    #     int numEnemies
    #     int collision_cooldown

    
    def __init__(self, cnp.ndarray[cnp.float32_t, ndim=2] observations,
                 cnp.ndarray[cnp.uint8_t, ndim=1] actions,  # Ensure it's uint8_t
                 cnp.ndarray[cnp.float32_t, ndim=1] rewards,
                 cnp.ndarray[cnp.uint8_t, ndim=1] terminals,  # Ensure it's uint8_t
                 int num_envs, int width, int height, int hud_height, int car_width, int car_height,
                 int max_enemies, int crash_noop_duration, int day_length, 
                 int initial_cars_to_pass, float min_speed, float max_speed):

        self.num_envs = observations.shape[0]
        # self.report_interval = report_interval


        # Inside this function, access memoryviews from the numpy arrays.
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] actions_view, terminals_view
        cdef cnp.ndarray[cnp.float32_t, ndim=2] obs_view
        cdef cnp.ndarray[cnp.float32_t, ndim=1] rewards_view

        # Initialize the environments
        self.envs = <Enduro*> calloc(self.num_envs, sizeof(Enduro))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        for i in range(self.num_envs):
            # obs_view = <cnp.ndarray[cnp.float32_t, ndim=2]> observations[i:i+1]
            obs_view = <cnp.ndarray[cnp.float32_t, ndim=2]> observations[i:i+1, 0:28] # 28 elements
            actions_view = <cnp.ndarray[cnp.uint8_t, ndim=1]> actions[i:i+1]    # Convert to uint8_t memoryview
            rewards_view = <cnp.ndarray[cnp.float32_t, ndim=1]> rewards[i:i+1]
            terminals_view = <cnp.ndarray[cnp.uint8_t, ndim=1]> terminals[i:i+1]  # Convert to uint8_t memoryview

            # Pass pointers to the first element of each array
            self.envs[i] = Enduro(
                observations = &obs_view[0, 0],
                actions = &actions_view[0],       # Correctly pass uint8_t pointer
                rewards = &rewards_view[0],
                terminals = &terminals_view[0],   # Correctly pass uint8_t pointer
                log_buffer=self.logs
            )

                # player_x=self.player_x,
                # player_y=self.player_y,
                # speed=self.speed,
                # max_speed=self.max_speed,
                # min_speed=self.min_speed,
                # score=self.score,
                # carsToPass=self.carsToPass,
                # day=self.day,
                # day_length=self.day_length,
                # step_count=self.step_count,
                # numEnemies=self.numEnemies,
                # width=self.width,
                # height=self.height,
                # collision_cooldown=self.collision_cooldown,
            
            init(&self.envs[i], width, height, hud_height, car_width, car_height, max_enemies, crash_noop_duration, day_length, initial_cars_to_pass, min_speed, max_speed)
            self.client = NULL


    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        print(".cy step Step")
        cdef int i
        for i in range(self.num_envs):
            print(f"Stepping env {i}")  # Add print for each env
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

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
