from libc.stdlib cimport calloc, free

cdef extern from "cartpole.h":
    int LOG_BUFFER_SIZE
    int MAX_STEPS

    ctypedef struct Log:
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
        int* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer
        Log log
        float x
        float x_dot
        float theta
        float theta_dot
        int steps
        int width
        int height
        int frameskip
        int continuous
        int num_obs

    ctypedef struct Client:
        float width
        float height
        float scale

    void init(CartPole* env)
    void free_initialized(CartPole* env)
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

    def __init__(self, float[:, :] observations, int[:] actions, float[:] rewards,
                 unsigned char[:] dones, int num_envs, int frameskip=1,
                 int width=800, int height=600, int continuous=0):

        self.num_envs = num_envs
        self.envs = <CartPole*> calloc(num_envs, sizeof(CartPole))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        self.client = NULL

        cdef int i
        for i in range(num_envs):
            self.envs[i].observations = &observations[i, 0]
            self.envs[i].actions = &actions[i]
            self.envs[i].rewards = &rewards[i]
            self.envs[i].dones = &dones[i]
            self.envs[i].log_buffer = self.logs
            self.envs[i].width = width
            self.envs[i].height = height
            self.envs[i].frameskip = frameskip
            self.envs[i].continuous = 0
            init(&self.envs[i])
            self.client = NULL

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

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log