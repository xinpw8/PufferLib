cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

cdef extern from "pong.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Pong:
        float* observations
        int* actions
        float* rewards
        unsigned char* terminals
        LogBuffer* log_buffer;
        Log log;
        float paddle_yl
        float paddle_yr
        float ball_x
        float ball_y
        float ball_vx;
        float ball_vy;
        unsigned int score_l
        unsigned int score_r
        float width
        float height
        float paddle_width
        float paddle_height
        float ball_width
        float ball_height
        float paddle_speed
        float ball_initial_speed_x
        float ball_initial_speed_y
        float ball_max_speed_y
        float ball_speed_y_increment
        unsigned int max_score
        float min_paddle_y
        float max_paddle_y
        float paddle_dir
        int tick
        int n_bounces
        int win
        int frameskip

    ctypedef struct Client

    void init(Pong* env)
    void reset(Pong* env)
    void step(Pong* env)

    Client* make_client(Pong* env)
    void close_client(Client* client)
    void render(Client* client, Pong* env)

cdef class CyPong:
    cdef:
        Pong* envs
        Client* client
        LogBuffer* logs
        int num_envs
        float width
        float height
        float paddle_width
        float paddle_height
        float ball_width
        float ball_height

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            float width, float height, float paddle_width, float paddle_height,
            float ball_width, float ball_height, float paddle_speed,
            float ball_initial_speed_x, float ball_initial_speed_y,
            float ball_max_speed_y, float ball_speed_y_increment,
            unsigned int max_score, int frameskip):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <Pong*> calloc(num_envs, sizeof(Pong))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Pong(
                observations = &observations[i, 0],
                actions = &actions[i],
                rewards = &rewards[i],
                terminals = &terminals[i],
                log_buffer=self.logs,
                width=width,
                height=height,
                paddle_width=paddle_width,
                paddle_height=paddle_height,
                ball_width=ball_width,
                ball_height=ball_height,
                paddle_speed=paddle_speed,
                ball_initial_speed_x=ball_initial_speed_x,
                ball_initial_speed_y=ball_initial_speed_y,
                ball_max_speed_y=ball_max_speed_y,
                ball_speed_y_increment=ball_speed_y_increment,
                max_score=max_score,
                frameskip=frameskip,
            )
            init(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef Pong* env = &self.envs[0]
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(env)
            os.chdir(cwd)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)
        free(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
