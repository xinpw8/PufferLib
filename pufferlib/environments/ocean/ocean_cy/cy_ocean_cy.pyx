import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

# Define the C struct directly in the Cython file
cdef struct COceanCyEnv:
    float* image_observations
    int* flat_observations
    # float* flat_observations
    unsigned int* actions
    float* rewards
    unsigned char* dones
    int* scores
    int num_agents
    float image_sign
    float flat_sign

# Implement the C functions directly in the Cython file
cdef COceanCyEnv* init_c_ocean_cy(float* image_observations, 
                                int* flat_observations,
                                # float* flat_observations,
                                  unsigned int* actions, 
                                  float* rewards, 
                                  unsigned char* dones, 
                                  int* scores, 
                                  int num_agents
                                  ):
    # Allocate memory for the COceanCyEnv structure
    cdef COceanCyEnv* env = <COceanCyEnv*> malloc(sizeof(COceanCyEnv))
    
    # Assign the data buffers to the C environment
    env.image_observations = image_observations
    env.flat_observations = flat_observations
    env.actions = actions
    env.rewards = rewards
    env.dones = dones
    env.scores = scores
    env.num_agents = num_agents

    return env

cdef void reset(COceanCyEnv* env):
    # Initialize the environment's observations with random values
    cdef int i, j
    for i in range(5):
        for j in range(5):
            env.image_observations[i * 5 + j] = np.random.randn()  # Random float for image
        env.flat_observations[i] = np.random.randint(-1, 2)  # Random int for flat (-1, 0, 1)
    
    # Calculate the sum of the image and flat observations manually
    cdef float image_sum = 0
    # cdef int image_sum = 0
    # cdef int flat_sum = 0
    cdef float flat_sum = 0
    for i in range(5):
        for j in range(5):
           image_sum += env.image_observations[i * 5 + j]
        flat_sum += env.flat_observations[i]
    
    # Store the sign of the sums
    env.image_sign = image_sum > 0
    env.flat_sign = flat_sum > 0


cdef void step(COceanCyEnv* env):
    # Initialize the reward to 0
    env.rewards[0] = 0.0

    # Compare the action with the sign of the image observations
    if env.actions[0] == env.image_sign:
        env.rewards[0] += 0.5

    # Compare the action with the sign of the flat observations
    if env.actions[1] == env.flat_sign:
        env.rewards[0] += 0.5

    # Mark the environment as done after one step (this mimics the original environment)
    env.dones[0] = 1


cdef void free_c_ocean_cy(COceanCyEnv* env):
    # Free the allocated environment
    free(env)

# Cython wrapper class for the C environment
cdef class COceanCy:
    cdef:
        COceanCyEnv* env  # Pointer to the C environment

    def __init__(self, 
                cnp.ndarray[cnp.float32_t, ndim=3] image_observations, 
                 cnp.ndarray[cnp.int8_t, ndim=2] flat_observations,
                #  cnp.ndarray[cnp.float32_t, ndim=2] flat_observations,
                 cnp.ndarray[cnp.uint32_t, ndim=2] actions,
                 cnp.ndarray[cnp.float32_t, ndim=2] rewards, 
                 cnp.ndarray[cnp.uint8_t, ndim=2] dones, 
                 cnp.ndarray[cnp.int32_t, ndim=2] scores, 
                 int num_agents
                 ):
        # Initialize the C environment with the data buffers
        self.env = init_c_ocean_cy(<float*>image_observations.data,
                                    <int*>flat_observations.data, 
                                #    <float*>flat_observations.data,
                                   <unsigned int*>actions.data, 
                                   <float*>rewards.data, 
                                   <unsigned char*>dones.data,
                                   <int*>scores.data, 
                                   num_agents
                                   )

    # Pufferlib does this already
    # def set_seed(self, int seed):
    #     # Set the seed for the Cython environment
    #     np.random.seed(seed)
    
    def reset(self):
        # Reset the C environment
        reset(self.env)

    def step(self):
        # Step the C environment
        step(self.env)

    def __dealloc__(self):
        # Free the C environment when the Python object is deallocated
        free_c_ocean_cy(self.env)
