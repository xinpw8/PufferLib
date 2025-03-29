import time
import numpy as np

from pufferlib import squared_bind

N = 2048
TIME = 1.0

# Create NumPy arrays with varying dtypes and sizes
obs = np.zeros((N, 11, 11), dtype=np.uint8)
atn = np.zeros((N), dtype=np.int32)
rew = np.zeros((N), dtype=np.float32)
term = np.zeros((N), dtype=np.uint8)
trunc = np.zeros((N), dtype=np.uint8)

def make_envs():
    env_ptrs = []
    for i in range(N):
        ptr = squared_bind.env_init(obs[i].ravel(), atn[i:i+1], rew[i:i+1], term[i:i+1], trunc[i:i+1], size=11)
        env_ptrs.append(ptr)

    return env_ptrs

def time_loop():
    env_ptrs = make_envs()
    for ptr in env_ptrs:
        squared_bind.env_reset(ptr)

    start = time.time()
    atn[:] = np.random.randint(0, 5, (N))
    steps = 0
    while time.time() - start < TIME:
        steps += N
        for i in range(N):
            squared_bind.env_step(env_ptrs[i])

    print("Loop SPS:", steps / (time.time() - start))

def time_vec():
    env_ptrs = make_envs()
    vec_ptr = squared_bind.init_vec(obs, atn, rew, term, trunc, N, size=11)
    squared_bind.vec_reset(vec_ptr)
    start = time.time()
    atn[:] = np.random.randint(0, 5, (N))

    steps = 0
    while time.time() - start < TIME:
        squared_bind.vec_step(vec_ptr)
        steps += N

    print("Vec SPS:", steps / (time.time() - start))

    for ptr in env_ptrs:
        squared_bind.env_close(ptr)

def test_loop():
    env_ptrs = make_envs()
    for ptr in env_ptrs:
        squared_bind.env_reset(ptr)

    while True:
        atn[:] = np.random.randint(0, 5, (N))
        for i in range(N):
            squared_bind.env_step(env_ptrs[i])

        squared_bind.env_render(env_ptrs[0])

    for ptr in env_ptrs:
        squared_bind.env_close(ptr)

def test_vec():
    vec_ptr = squared_bind.init_vec(obs, atn, rew, term, trunc, N, size=11)
    squared_bind.vec_reset(vec_ptr)
    while True:
        atn[:] = np.random.randint(0, 5, (N))
        squared_bind.vec_step(vec_ptr)
        squared_bind.vec_render(vec_ptr, 0)

    squared_bind.vec_close(vec_ptr)

def test_env_binding():
    ptr = squared_bind.env_init(obs[0], atn[0:1], rew[0:1], term[0:1], trunc[0:1], size=11)
    squared_bind.env_reset(ptr)
    squared_bind.env_step(ptr)
    squared_bind.env_close(ptr)

def test_vectorize_binding():
    ptr = squared_bind.env_init(obs[0], atn[0:1], rew[0:1], term[0:1], trunc[0:1], size=11)
    vec_ptr = squared_bind.vectorize(ptr)
    squared_bind.vec_reset(vec_ptr)
    squared_bind.vec_step(vec_ptr)
    squared_bind.vec_close(vec_ptr)

def test_vec_binding():
    vec_ptr = squared_bind.init_vec(obs, atn, rew, term, trunc, N, size=11)
    squared_bind.vec_reset(vec_ptr)
    squared_bind.vec_step(vec_ptr)
    squared_bind.vec_close(vec_ptr)



if __name__ == '__main__':
    #test_loop()
    #test_vec()
    #time_loop()
    #time_vec()

    test_env_binding()
    test_vectorize_binding()
    test_vec_binding()

