# #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# # cython: language_level=3
# # cython: boundscheck=True
# # cython: initializedcheck=False
# # cython: cdivision=True
# # cython: wraparound=False
# # cython: nonecheck=False
# # cython: profile=False
# #include <numpy/arrayobject.h>

# cimport numpy as cnp
# import numpy as np
# from libc.math cimport pi, sin, cos
# from libc.stdlib cimport malloc, free

# ctypedef dict NativeDType

# # Initialize NumPy C API
# def init_numpy():
#     cnp.import_array()

# init_numpy()

# cdef int round_to(int x, int base):
#     return base * ((x + base - 1) // base)


# # Helper function to nativize dtype
# cdef NativeDType _nativize_dtype(cnp.dtype sample_dtype, cnp.dtype structured_dtype, int offset=0):
#     cdef dict subviews
#     cdef int delta, all_delta, start_offset

#     if structured_dtype.fields is None:
#         # Handle non-structured types
#         if structured_dtype.subdtype is not None:
#             dtype, shape = structured_dtype.subdtype
#         else:
#             dtype = structured_dtype
#             shape = (1,)

#         delta = int(np.prod(shape))

#         if sample_dtype.base.itemsize == 1:
#             offset = round_to(offset, dtype.alignment)
#             delta *= dtype.itemsize
#         else:
#             if dtype.itemsize != sample_dtype.base.itemsize:
#                 raise ValueError(f"Itemsize mismatch: dtype.itemsize={dtype.itemsize}, sample_dtype.base.itemsize={sample_dtype.base.itemsize}")

#         return (dtype, shape, offset, delta)
#     else:
#         # Handle structured dtypes
#         subviews = {}
#         start_offset = offset
#         all_delta = 0
        
#         for name, (dtype, _) in structured_dtype.fields.items():
#             views, dtype, shape, offset, delta = _nativize_dtype(sample_dtype, dtype, offset)
#             if views is not None:
#                 subviews[name] = views
#             else:
#                 subviews[name] = (dtype, shape, offset, delta)

#             offset += delta
#             all_delta += delta

#         return subviews, None, None, start_offset, all_delta


# def nativize_dtype(object emulated):
#     cdef cnp.dtype sample_dtype = emulated.observation_dtype
#     cdef cnp.dtype structured_dtype = emulated.emulated_observation_dtype

#     # Attempt to nativize the dtype
#     result = _nativize_dtype(sample_dtype, structured_dtype)

#     # If result is a tuple (non-structured), return it directly, otherwise return subviews (a dict)
#     if isinstance(result[0], dict):
#         print(f"Structured dtype detected: {result[0]}")
#         return result[0]  # subviews as a dict
#     else:
#         print(f"Unstructured dtype detected: {result}")
#         return result  # tuple for non-structured types

# def nativize_tensor(cnp.ndarray observation, NativeDType native_dtype):
#     try:
#         print(f"Nativizing observation - Shape: {observation.shape}, Native dtype: {native_dtype}")
#         return _nativize_tensor(observation, native_dtype)
#     except Exception as e:
#         print(f"Error during nativize_tensor: {e}")
#         raise


# cdef _nativize_tensor(cnp.ndarray observation, NativeDType native_dtype):
#     try:
#         # If the native_dtype is a tuple (non-structured type)
#         if isinstance(native_dtype, tuple):
#             dtype, shape, offset, delta = native_dtype
#             print(f"Nativizing tensor - dtype: {dtype}, shape: {shape}, offset: {offset}, delta: {delta}")

#             # Ensure we respect alignment requirements
#             aligned_offset = round_to(offset, dtype.alignment)  # Align offset to dtype alignment
#             slice = observation[:, aligned_offset:aligned_offset + delta]
#             print(f"Slice shape: {slice.shape}, expected dtype: {dtype}")

#             # View and reshape the slice into the native dtype and target shape
#             slice = slice.view(dtype).reshape(observation.shape[0], *shape)
#             return slice

#         # If the native_dtype is a dict (structured dtype)
#         else:
#             subviews = {}
#             for name, dtype in native_dtype.items():
#                 subviews[name] = _nativize_tensor(observation, dtype)
#             return subviews

#     except Exception as e:
#         print(f"Error in nativizing tensor: {e}")
#         raise


# cdef class CSpacesCy:
#     cdef:
#         float[:, :, :] buffero_image_observations
#         signed char[:, :] buffero_flat_observations
#         unsigned char[:] dones
#         float[:] rewards
#         int[:] scores
#         float[:] episodic_returns
#         int num_agents
#         int reset_count
#         int step_count
#         int[:] image_sign
#         int[:] flat_sign
#         object emulated

#     def __init__(self, 
#                 cnp.ndarray buffero_image_observations,
#                 cnp.ndarray buffero_flat_observations,
#                 cnp.ndarray rewards, 
#                 cnp.ndarray scores, 
#                 cnp.ndarray episodic_returns, 
#                 cnp.ndarray dones, 
#                 int num_agents,
#                 object emulated):
        
#         self.image_sign = np.zeros(num_agents, dtype=np.int32)
#         self.flat_sign = np.zeros(num_agents, dtype=np.int32)

#         self.buffero_image_observations = buffero_image_observations
#         self.buffero_flat_observations = buffero_flat_observations
#         self.rewards = rewards
#         self.scores = scores
#         self.episodic_returns = episodic_returns
#         self.dones = dones
#         self.num_agents = num_agents
#         self.emulated = emulated

#         for agent_idx in range(self.num_agents):
#             self.reset(agent_idx)

#     cdef void compute_observations(self, int agent_idx):
#         cdef float[:, :] image_obs = self.buffero_image_observations[agent_idx, :, :]  # Image buffer
#         cdef signed char[:] flat_obs = self.buffero_flat_observations[agent_idx, :]    # Flat buffer
#         cdef int i, j

#         # Generate image observations
#         for i in range(5):
#             for j in range(5):
#                 image_obs[i, j] = np.random.randn()

#         # Calculate the image sign (sum over the entire 5x5 observation)
#         image_sum = np.sum(image_obs)
#         self.image_sign[agent_idx] = 1 if image_sum > 0 else 0

#         # Generate flat observations
#         for i in range(5):  # flat_obs has 5 elements
#             flat_obs[i] = np.random.randint(-1, 2)

#         # Calculate the flat sign (sum over the entire flat observation)
#         flat_sum = np.sum(flat_obs)
#         self.flat_sign[agent_idx] = 1 if flat_sum > 0 else 0

#         print(f'obs_dtype: {self.emulated}')
#         print(f'buffero_image_observations: {self.buffero_image_observations}')
#         print(f'buffero_flat_observations: {self.buffero_flat_observations}')

#         # Nativize dtype and observation
#         try:
#             native_dtype = nativize_dtype(self.emulated)
#             # Ensure we're working with memory views rather than converting
#             observation = nativize_tensor(self.buffero_image_observations, native_dtype)
#         except Exception as e:
#             print(f"Error in nativizing observation: {e}")
#             raise

#     cdef void reset(self, int agent_idx):
#         cdef int reset_count
        
#         # returns image_sign and flat_sign (0 or 1) for each agent
#         self.compute_observations(agent_idx)
#         self.dones[agent_idx] = 0
#         self.reset_count += 1

#         # self.scores[agent_idx] = 0


#     def step(self, cnp.ndarray[unsigned char, ndim=2] actions):
#         cdef int step_count
#         cdef int action
#         cdef int agent_idx = 0

#         self.rewards[:] = 0.0
#         self.scores[agent_idx] = 0
#         self.dones[agent_idx] = 0

#         cdef int i, j, k
#         cdef int flat_dim = self.buffero_flat_observations.shape[1]
#         cdef int image_dim_1 = self.buffero_image_observations.shape[1]
#         cdef int image_dim_2 = self.buffero_image_observations.shape[2]

#         # Prepare space for the flattened observations
#         cdef float[:, :] concatenated_observations = np.zeros(
#             (self.num_agents, image_dim_1 * image_dim_2 + flat_dim),
#             dtype=np.float32
#         )

#         # Concatenate manually
#         for agent_idx in range(self.num_agents):
#             # Flatten the image observations manually
#             for i in range(image_dim_1):
#                 for j in range(image_dim_2):
#                     concatenated_observations[agent_idx, i * image_dim_2 + j] = self.buffero_image_observations[agent_idx, i, j]

#             # Append the flat observations
#             for k in range(flat_dim):
#                 concatenated_observations[agent_idx, image_dim_1 * image_dim_2 + k] = self.buffero_flat_observations[agent_idx, k]

#         # Process actions and rewards
#         for agent_idx in range(self.num_agents):
#             image_action = actions[agent_idx, 0]
#             flat_action = actions[agent_idx, 1]

#             if self.image_sign[agent_idx] == image_action:
#                 self.rewards[agent_idx] += 0.5
            
#             if self.flat_sign[agent_idx] == flat_action:
#                 self.rewards[agent_idx] += 0.5

#         # # Debug prints to ensure the logic works
#         # if self.step_count % 1000 == 0:
#         #     print(f'Agent {agent_idx}: Image action: {image_action}, Flat action: {flat_action}')
#         #     print(f'Agent {agent_idx} rewards: {self.rewards[agent_idx]}')


#         self.step_count += 1
#         self.reset(agent_idx)

#         return concatenated_observations, self.rewards, self.dones, self.scores


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
#include <numpy/arrayobject.h>

cimport cython
from cpython cimport array
from cython cimport view
cimport numpy as cnp
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.math cimport abs

ctypedef object NativeDType

# Initialize NumPy C API (even though we won't use NumPy calls)
def init_numpy():
    cnp.import_array()

init_numpy()

# Manual rounding function for alignment
cdef int round_to(int x, int base):
    return base * ((x + base - 1) // base)

# Helper function to manually compute product of elements in an array (shape)
cdef int manual_prod(int* shape, int n):
    cdef int prod = 1
    cdef int i
    for i in range(n):
        prod *= shape[i]
    return prod

# Random float generator between 0 and 1 using C stdlib
cdef float random_float():
    return rand() / float(RAND_MAX)

# Random integer generator between low and high using C stdlib
cdef int random_int(int low, int high):
    return low + rand() % (high - low)

cdef str safe_repr(object obj):
    try:
        return repr(obj)
    except:
        return f"<unprintable object of type {type(obj).__name__}>"

# print("Defined safe_repr function")

cdef tuple _nativize_dtype(cnp.dtype sample_dtype, object structured_dtype, int offset=0):

    # print(f"Entering _nativize_dtype with sample_dtype: {safe_repr(sample_dtype)}, structured_dtype: {safe_repr(structured_dtype)}, offset: {offset}")
    cdef dict subviews
    cdef int delta, all_delta, start_offset
    cdef list shape_list

    if isinstance(structured_dtype, tuple):
        # print(f"Handling tuple dtype: {safe_repr(structured_dtype)}")
        dtype_str, shape = structured_dtype
        shape_list = list(shape) if isinstance(shape, tuple) else [shape]
        delta = 1
        for dim in shape_list:
            delta *= dim
        return (None, cnp.dtype(dtype_str), shape_list, offset, delta)

    if not isinstance(structured_dtype, cnp.dtype):
        # print(f"Converting structured_dtype to numpy dtype: {safe_repr(structured_dtype)}")
        structured_dtype = cnp.dtype(structured_dtype)

    if structured_dtype.fields is None:
        # Handle non-structured types
        if structured_dtype.subdtype is not None:
            dtype, subdtype_shape = structured_dtype.subdtype
            shape_list = list(subdtype_shape)
        else:
            dtype = structured_dtype
            shape_list = [1]

        delta = 1
        for dim in shape_list:
            delta *= dim

        if sample_dtype.base.itemsize == 1:
            offset = round_to(offset, dtype.alignment)
            delta *= dtype.itemsize
        else:
            if dtype.itemsize != sample_dtype.base.itemsize:
                raise ValueError(f"Itemsize mismatch: dtype.itemsize={dtype.itemsize}, sample_dtype.base.itemsize={sample_dtype.base.itemsize}")

        return (None, dtype, shape_list, offset, delta)

    else:
        # Handle structured dtypes
        subviews = {}
        start_offset = offset
        all_delta = 0
        
        for name, (field_dtype, field_offset) in structured_dtype.fields.items():
            # print(f"Processing field: {name}, dtype: {safe_repr(field_dtype)}, offset: {field_offset}")
            views, dtype, shape, new_offset, delta = _nativize_dtype(sample_dtype, field_dtype, offset + field_offset)
            if views is not None:
                subviews[name] = views
            else:
                subviews[name] = (dtype, shape, new_offset, delta)

            all_delta = max(all_delta, field_offset + delta)

        return (subviews, None, None, start_offset, all_delta)

# print("Defined _nativize_dtype function")


# Main nativize function to convert observation dtype (no numpy)
cdef object nativize_dtype(object emulated):
    # print("Entering nativize_dtype")
    cdef cnp.dtype sample_dtype = emulated.observation_dtype
    # print(f"sample_dtype: {safe_repr(sample_dtype)}")
    cdef object structured_dtype = emulated.emulated_observation_dtype
    # print(f"structured_dtype: {safe_repr(structured_dtype)}")

    try:
        result = _nativize_dtype(sample_dtype, structured_dtype)
        # print(f"Result from _nativize_dtype: {safe_repr(result)}")
        return result
    except Exception as e:
        print(f"Error in _nativize_dtype: {safe_repr(e)}")
        import traceback
        traceback.print_exc()
        raise


# print("Defined nativize_dtype function")



cdef object nativize_tensor(float[:, :, :] image_obs, object native_dtype):
    cdef dict subviews
    cdef Py_ssize_t offset, sub_offset, sub_delta
    cdef list shape

    if isinstance(native_dtype, tuple) and len(native_dtype) == 5:
        subviews, _, _, offset, delta = native_dtype
        if isinstance(subviews, dict):
            result = {}
            for name, value in subviews.items():
                if isinstance(value, (tuple, list)) and len(value) == 4:
                    dtype, shape, sub_offset, sub_delta = value
                    if name == 'flat':
                        result[name] = image_obs[0, sub_offset:sub_offset+sub_delta]
                    elif name == 'image':
                        result[name] = image_obs[0, sub_offset:sub_offset+sub_delta].reshape(shape)
                else:
                    raise TypeError(f"Unexpected value type for {name}: {type(value)}")
            return result
        else:
            raise TypeError(f"Unexpected subviews type: {type(subviews)}")
    else:
        raise TypeError(f"Unexpected native_dtype structure: {native_dtype}")

print("Defined nativize_tensor function")





# CSpacesCy class for environment processing
cdef class CSpacesCy:
    cdef:
        float[:, :, :] buffero_image_observations
        signed char[:, :] buffero_flat_observations
        unsigned char[:] dones
        float[:] rewards
        int[:] scores
        float[:] episodic_returns
        int num_agents
        int reset_count
        int step_count
        int[:] image_sign
        int[:] flat_sign
        object emulated

    def __init__(self, 
                float[:, :, :] buffero_image_observations,
                signed char[:, :] buffero_flat_observations,
                float[:] rewards, 
                int[:] scores, 
                float[:] episodic_returns, 
                unsigned char[:] dones, 
                int num_agents,
                object emulated):
        
        # Initialize memoryviews instead of lists (use calloc for zero-initialization)
        self.image_sign = <int[:num_agents]> malloc(num_agents * sizeof(int))
        self.flat_sign = <int[:num_agents]> malloc(num_agents * sizeof(int))

        self.buffero_image_observations = buffero_image_observations
        self.buffero_flat_observations = buffero_flat_observations
        self.rewards = rewards
        self.scores = scores
        self.episodic_returns = episodic_returns
        self.dones = dones
        self.num_agents = num_agents
        self.emulated = emulated

        for agent_idx in range(self.num_agents):
            self.reset(agent_idx)

    cdef void compute_observations(self, int agent_idx):
        cdef float[:, :] image_obs = self.buffero_image_observations[agent_idx, :, :]  # Image buffer
        cdef signed char[:] flat_obs = self.buffero_flat_observations[agent_idx, :]    # Flat buffer
        cdef int i, j

        # Generate image observations using C-style rand()
        for i in range(5):
            for j in range(5):
                image_obs[i, j] = random_float()

        # Calculate the image sign (sum over the entire 5x5 observation)
        cdef float image_sum = 0.0
        for i in range(5):
            for j in range(5):
                image_sum += abs(image_obs[i, j])

        self.image_sign[agent_idx] = 1 if image_sum > 0 else 0

        # Generate flat observations using C-style rand()
        for i in range(5):  # flat_obs has 5 elements
            flat_obs[i] = random_int(-1, 2)

        # Calculate the flat sign (sum over the entire flat observation)
        cdef int flat_sum = 0
        for i in range(5):
            flat_sum += flat_obs[i]

        self.flat_sign[agent_idx] = 1 if flat_sum > 0 else 0

        print(f"buffero_image_observations type: {type(self.buffero_image_observations)}")
        print(f"buffero_image_observations shape: {self.buffero_image_observations.shape}")

        # Nativize dtype and observation
        native_dtype = nativize_dtype(self.emulated)
        print(f"Native dtype: {native_dtype}")
        
        try:
            observation = nativize_tensor(self.buffero_image_observations, native_dtype)
            print(f"Observation type: {type(observation)}")
            if isinstance(observation, dict):
                for key, value in observation.items():
                    print(f"Observation[{key}] type: {type(value)}")
                    print(f"Observation[{key}] shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                    if hasattr(value, '__getitem__'):
                        if value.ndim == 1:
                            print(f"Observation[{key}] (first few elements): {value[:5]}")
                        elif value.ndim == 2:
                            print(f"Observation[{key}] (first few elements):")
                            for row in value[:2]:
                                print(row[:5])
                    else:
                        print(f"Observation[{key}] (first few elements): N/A")
            elif hasattr(observation, 'shape'):
                print(f"Observation shape: {observation.shape}")
                print(f"Observation (first few elements): {observation[:5] if hasattr(observation, '__getitem__') else 'N/A'}")
            else:
                print("Observation has no shape attribute")
        except Exception as e:
            print(f"Error in nativize_tensor: {e}")
            import traceback
            traceback.print_exc()

        # Here you can use the 'observation' dictionary as needed
        # For example:
        # flat_obs = observation['flat']
        # image_obs = observation['image']

    cdef void reset(self, int agent_idx):
        # Reset observations for the given agent
        self.compute_observations(agent_idx)
        self.dones[agent_idx] = 0
        self.reset_count += 1


    def step(self, unsigned char[:, :] actions):
        cdef int step_count
        cdef int action
        cdef int agent_idx = 0

        # Initialize rewards and scores for this step
        for agent_idx in range(self.num_agents):
            self.rewards[agent_idx] = 0.0
            self.scores[agent_idx] = 0
            self.dones[agent_idx] = 0

        cdef int i, j, k
        cdef int flat_dim = self.buffero_flat_observations.shape[1]
        cdef int image_dim_1 = self.buffero_image_observations.shape[1]
        cdef int image_dim_2 = self.buffero_image_observations.shape[2]

        # Total number of columns (flattened image + flat observations)
        cdef int num_columns = image_dim_1 * image_dim_2 + flat_dim

        # Allocate memory using malloc for the 2D memoryview
        cdef float[:, :] concatenated_observations = <float[:self.num_agents, :num_columns]> malloc(self.num_agents * num_columns * sizeof(float))

        # Concatenate observations manually (image and flat observations)
        for agent_idx in range(self.num_agents):
            for i in range(image_dim_1):
                for j in range(image_dim_2):
                    concatenated_observations[agent_idx, i * image_dim_2 + j] = self.buffero_image_observations[agent_idx, i, j]
            for k in range(flat_dim):
                concatenated_observations[agent_idx, image_dim_1 * image_dim_2 + k] = self.buffero_flat_observations[agent_idx, k]

        # Process actions and rewards
        for agent_idx in range(self.num_agents):
            image_action = actions[agent_idx, 0]
            flat_action = actions[agent_idx, 1]

            if self.image_sign[agent_idx] == image_action:
                self.rewards[agent_idx] += 0.5
            
            if self.flat_sign[agent_idx] == flat_action:
                self.rewards[agent_idx] += 0.5

        self.step_count += 1
        self.reset(agent_idx)

        return concatenated_observations, self.rewards, self.dones, self.scores