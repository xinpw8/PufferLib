cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

cdef extern from "go.h":

    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return
        float episode_length
        int games_played
        float score
        float winrate

    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Group:
        int parent
        int rank
        int size
        int liberties

    int find(Group*)
    void union_groups(Group*, int, int)



    ctypedef struct CGo:
        float* observations
        int* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer
        Log log
        float score
        int width
        int height
        int* board_x
        int* board_y
        int board_width
        int board_height
        int grid_square_size
        int grid_size
        int* board_states
        int* previous_board_state
        int last_capture_position
        int* temp_board_states
        int moves_made
        int* capture_count
        float komi
        int* visited
        Group* groups
        Group* temp_groups
        float reward_move_pass
        float reward_move_invalid
        float reward_move_valid
        float reward_player_capture
        float reward_opponent_capture

    ctypedef struct Client

    void init(CGo* env)
    void free_initialized(CGo* env)
    void reset(CGo* env)
    void step(CGo* env)

    Client* make_client(float width, float height)
    void close_client(Client* client)
    void render(Client* client, CGo* env)
    

cdef class CyGo:
    cdef:
        CGo* envs
        Client* client
        LogBuffer* logs
        int num_envs
    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int width, int height, int grid_size, int board_width, int board_height,
            int grid_square_size, int moves_made, float komi,
            float score, int last_capture_position, float reward_move_pass,
            float reward_move_invalid, float reward_move_valid, float reward_player_capture,  float reward_opponent_capture ):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <CGo*> calloc(num_envs, sizeof(CGo))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = CGo(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                dones=&terminals[i],
                log_buffer=self.logs,
                width=width,
                height=height,
                grid_size=grid_size,
                board_width=board_width,
                board_height=board_height,
                grid_square_size=grid_square_size,
                moves_made=moves_made,
                komi=komi,
                score=score,
                last_capture_position=last_capture_position,
                reward_move_pass=reward_move_pass,
                reward_move_invalid=reward_move_invalid,
                reward_move_valid=reward_move_valid
            )
            init(&self.envs[i])
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
        cdef CGo* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env.width,env.height)

        render(self.client, &self.envs[0])

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL
        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log