from libc.stdlib cimport calloc, free
import numpy as np
cdef extern from "gpudrive.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;


    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Entity:
        int type;
        int road_object_id;
        int road_point_id;
        int array_size;
        float* traj_x;
        float* traj_y;
        float* traj_z;
        float* traj_vx;
        float* traj_vy;
        float* traj_vz;
        float* traj_heading;
        int* traj_valid;
        float width;
        float length;
        float height;
        float goal_position_x;
        float goal_position_y;
        float goal_position_z;
        int collision_state;
        float x;
        float y;
        float z;
        float vx;
        float vy;
        float vz;
        float heading;
        int valid;

    ctypedef struct GPUDrive:
        float* observations;
        int* actions;
        float* rewards;
        unsigned char* dones;
        LogBuffer* log_buffer;
        Log* logs;
        int num_agents;
        int active_agent_count;
        int* active_agent_indices;
        int human_agent_idx;
        Entity* entities;
        int num_entities;
        int timestep;
        int dynamics_model;
        float* fake_data;

    ctypedef struct Client

    void init(GPUDrive* env)
    void free_allocated(GPUDrive* env)


    Client* make_client(GPUDrive* env)
    void close_client(Client* client)
    void c_render(Client* client, GPUDrive* env)
    void c_reset(GPUDrive* env)
    void c_step(GPUDrive* env)

cpdef entity_dtype():
    '''Make a dummy entity to get the dtype'''
    # Create a numpy structured dtype that matches the Entity struct
    return np.dtype([
        ('type', np.int32),
        ('road_object_id', np.int32),
        ('road_point_id', np.int32),
        ('array_size', np.int32),
        # For pointer fields, we use intp (integer large enough to hold a pointer)
        ('traj_x', np.intp),
        ('traj_y', np.intp),
        ('traj_z', np.intp),
        ('traj_vx', np.intp),
        ('traj_vy', np.intp),
        ('traj_vz', np.intp),
        ('traj_heading', np.intp),
        ('traj_valid', np.intp),
        ('width', np.float32),
        ('length', np.float32),
        ('height', np.float32),
        ('goal_position_x', np.float32),
        ('goal_position_y', np.float32),
        ('goal_position_z', np.float32),
        ('collision_state', np.int32),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('vz', np.float32),
        ('heading', np.float32),
        ('valid', np.int32)
    ])

cdef class CyGPUDrive:
    cdef:
        GPUDrive* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int active_agent_count, int human_agent_idx):

        self.client = NULL
        self.num_envs = num_envs
        self.envs = <GPUDrive*> calloc(num_envs, sizeof(GPUDrive))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int inc = active_agent_count

        cdef int i
        for i in range(num_envs):
            self.envs[i] = GPUDrive(
                observations=&observations[inc*i, 0],
                actions=&actions[inc*i*2],
                rewards=&rewards[inc*i],
                dones=&terminals[inc*i],
                log_buffer=self.logs,
                human_agent_idx=human_agent_idx,
            )
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
        cdef GPUDrive* env = &self.envs[0]
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
