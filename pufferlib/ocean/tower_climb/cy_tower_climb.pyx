from libc.stdlib cimport calloc, free

cdef extern from "tower_climb.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float rows_cleared;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Level:
        const int* map;
        int rows;
        int cols;
        int size;
        int total_length;
        int goal_location;
        int spawn_location;

    ctypedef struct CTowerClimb:
        float* observations;
        int* actions;
        float* rewards;
        unsigned char* dones;
        LogBuffer* log_buffer;
        Log log;
        float score;
        int map_choice;
        int robot_position; 
        int robot_direction;
        int robot_state;
        int robot_orientation;
        int* board_state; 
        int* blocks_to_move;
        int* blocks_to_fall;
        int block_grabbed;
        int rows_cleared;
        Level level;
        int level_number;
        float reward_climb_row;
        float reward_fall_row;
        float reward_illegal_move;
    ctypedef struct Client

    void init(CTowerClimb* env)
    void free_allocated(CTowerClimb* env)


    Client* make_client(CTowerClimb* env)
    void close_client(Client* client)
    void render(Client* client, CTowerClimb* env)
    void reset(CTowerClimb* env)
    void step(CTowerClimb* env)

cdef class CyTowerClimb:
    cdef:
        CTowerClimb* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int map_choice, float reward_climb_row, float reward_fall_row,
            float reward_illegal_move):

        self.client = NULL
        self.num_envs = num_envs
        self.envs = <CTowerClimb*> calloc(num_envs, sizeof(CTowerClimb))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        cdef int i
        for i in range(num_envs):
            self.envs[i] = CTowerClimb(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                dones=&terminals[i],
                log_buffer=self.logs,
                map_choice=map_choice,
                reward_climb_row=reward_climb_row,
                reward_fall_row=reward_fall_row,
                reward_illegal_move=reward_illegal_move
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
        cdef CTowerClimb* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        render(self.client, &self.envs[0])

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
