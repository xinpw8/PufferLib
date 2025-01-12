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


    ctypedef struct TowerClimb:
        float* observations;
        int* actions;
        float* rewards;
        unsigned char* dones;
        LogBuffer* log_buffer;
        Log log;
        float score;
        int width;
        int height;
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

    ctypedef struct Client

    void init(TowerClimb* env)
    void free_allocated(TowerClimb* env)


    Client* make_client(TowerClimb* env)
    void close_client(Client* client)
    void render(Client* client, TowerClimb* env)
    void reset(TowerClimb* env)
    void step(TowerClimb* env)

cdef class CyTowerClimb:
    cdef:
        TowerClimb* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int width, int height, int map_choice):

        self.client = NULL
        self.num_envs = num_envs
        self.envs = <TowerClimb*> calloc(num_envs, sizeof(TowerClimb))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = TowerClimb(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                dones=&terminals[i],
                log_buffer=self.logs,
                width=width,
                height=height,
                map_choice=map_choice,
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
        cdef TowerClimb* env = &self.envs[0]
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
