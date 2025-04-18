from libc.stdlib cimport calloc, free

cdef extern from "cartpole.h":
    ctypedef struct Log:
        float perf
        float score
        float episode_return
        float episode_length
        int x_threshold_termination
        int pole_angle_termination
        int max_steps_termination

    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx

    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct CartPole:
        float* observations
        float* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer
        Log log
        float x
        float x_dot
        float theta
        float theta_dot
        int steps_beyond_done
        int steps
        int continuous

    ctypedef struct Client

    void init(CartPole* env)
    void free_initialized(CartPole* env)
    void allocate(CartPole* env)
    void free_allocated(CartPole* env)
    Client* make_client(CartPole* env)
    void close_client(Client* client)
    void c_render(Client* client, CartPole* env)
    void c_reset(CartPole* env)
    void c_step(CartPole* env)

cdef class CyCartPole:
    cdef:
        Client* client
        CartPole* envs
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, float[:] actions, float[:] rewards,
                 unsigned char[:] dones, int num_envs, int continuous=0):
        self.num_envs = num_envs
        self.envs = <CartPole*> calloc(num_envs, sizeof(CartPole))
        self.logs = allocate_logbuffer(1024)
        self.client = NULL

        cdef int i
        for i in range(num_envs):
            self.envs[i].observations = &observations[i, 0]
            self.envs[i].actions = &actions[i]
            self.envs[i].rewards = &rewards[i]
            self.envs[i].dones = &dones[i]
            self.envs[i].log_buffer = self.logs
            self.envs[i].continuous = continuous
            init(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef CartPole* env = &self.envs[0]
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(env)
            os.chdir(cwd)
        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL
        if self.envs != NULL:
            free(self.envs)
            self.envs = NULL
        if self.logs != NULL:
            free_logbuffer(self.logs)
            self.logs = NULL

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
