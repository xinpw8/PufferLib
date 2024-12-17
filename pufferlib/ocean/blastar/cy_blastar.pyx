# cy_blastar.pyx

cimport numpy as cnp
from libc.stdlib cimport calloc, free
from libc.math cimport fabs
from libc.string cimport memset

cdef extern from "blastar_env.h":
    int LOG_BUFFER_SIZE

    # Define the Bullet struct
    ctypedef struct Bullet:
        float x
        float y
        float last_x
        float last_y
        bint active

    # Define the Enemy struct
    ctypedef struct Enemy:
        float x
        float y
        float last_x
        float last_y
        bint active
        bint attacking
        int direction
        int width
        int height
        Bullet bullet

    # Define the Player struct
    ctypedef struct Player:
        float x
        float y
        float last_x
        float last_y
        int score
        int lives
        Bullet bullet
        bint bulletFired
        bint playerStuck
        float explosion_timer

    # Define the Log struct
    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        float lives

    # Define the LogBuffer struct
    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx

    # Define the BlastarEnv struct
    ctypedef struct BlastarEnv:
        int screen_width
        int screen_height
        float player_width
        float player_height
        float enemy_width
        float enemy_height
        float bullet_width
        float bullet_height
        bint gameOver
        int tick
        int playerExplosionTimer
        int enemyExplosionTimer
        Player player
        Enemy enemy
        Bullet bullet
        float* observations       # [25]
        int* actions              # [6]
        float* rewards            # [1]
        unsigned char* terminals  # [1]
        LogBuffer* log_buffer
        Log log

    # Function declarations
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    void add_log(LogBuffer* logs, Log* log)
    Log aggregate_and_clear(LogBuffer* logs)

    void init_blastar(BlastarEnv *env)
    void reset_blastar(BlastarEnv *env)
    void c_step(BlastarEnv *env)
    void close_client(Client* client)
    void c_render(Client* client, BlastarEnv* env)

    # Rendering functions
    ctypedef struct Client:
        pass

    Client* make_client(BlastarEnv* env)
    void close_client(Client* client)
    void c_render(Client* client, BlastarEnv* env)

cdef class CyBlastar:
    cdef BlastarEnv* envs
    cdef Client* client
    cdef LogBuffer* logs
    cdef int num_envs

    def __init__(self,
                 float[:, :] observations,
                 int[:] actions,
                 float[:] rewards,
                 unsigned char[:] terminals,
                 int num_envs):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <BlastarEnv*> calloc(num_envs, sizeof(BlastarEnv))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i].observations = &observations[i, 0]
            self.envs[i].actions = &actions[i]
            self.envs[i].rewards = &rewards[i]
            self.envs[i].terminals = &terminals[i]
            self.envs[i].log_buffer = self.logs

            # Initialize the environment without overwriting RL pointers
            init_blastar(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset_blastar(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef BlastarEnv* env = &self.envs[0]
        if self.client == NULL and self.num_envs > 0:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(env)
            os.chdir(cwd)

        if self.client != NULL:
            c_render(self.client, &self.envs[0])

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL
        
        free(self.envs)
        free(self.logs)

        if self.envs != NULL:
            for i in range(self.num_envs):
                close_client(self.client)
            free(self.envs)
            self.envs = NULL

        if self.logs != NULL:
            free_logbuffer(self.logs)
            self.logs = NULL

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return {
            'episode_return': log.episode_return,
            'episode_length': log.episode_length,
            'score': log.score,
            'lives': log.lives,
        }
        