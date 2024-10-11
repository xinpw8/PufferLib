cimport numpy as cnp
from libc.stdlib cimport calloc, free

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
        unsigned int* actions
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
        float width
        float height
        int frameskip
        int collision_cooldown

    ctypedef struct Client

    void init(Enduro* env, float width, float height, float hud_height, float car_width, float car_height, int max_enemies, int crash_noop_duration, int day_length, int initial_cars_to_pass, float min_speed, float max_speed)
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
        float width
        float height
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
        int frameskip
        int collision_cooldown

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
                cnp.ndarray rewards, cnp.ndarray terminals, int num_envs,
                float width, float height, float hud_height, float car_width, float car_height,
                int max_enemies, int crash_noop_duration, int day_length, int initial_cars_to_pass,
                float min_speed, float max_speed, int frameskip=4):

        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.frameskip = frameskip
        self.client = NULL
        self.envs = <Enduro*> calloc(num_envs, sizeof(Enduro))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            observations_i = observations[i:i+1]
            actions_i = actions[i:i+1]
            rewards_i = rewards[i:i+1]
            terminals_i = terminals[i:i+1]

            self.envs[i] = Enduro(
                observations=<float*> observations_i.data,
                actions=<unsigned int*> actions_i.data,
                rewards=<float*> rewards_i.data,
                terminals=<unsigned char*> terminals_i.data,
                log_buffer=self.logs,
                
                width=width,
                height=height,
                player_x=80.0,  # Initial player position
                player_y=180.0,
                speed=1.0,
                max_speed=10.0,
                min_speed=-1.0,
                score=0,
                carsToPass=5,
                day=1,
                day_length=2000,
                step_count=0,
                numEnemies=0,
                frameskip=frameskip,
                collision_cooldown=0,
            )
            init(&self.envs[i], self.width, self.height, hud_height, car_width, car_height, max_enemies, crash_noop_duration, day_length, initial_cars_to_pass, min_speed, max_speed)
            self.client = NULL


    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
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
