from multiprocessing import Array, Process, Manager
import numpy as np
import time

def mp_array_test(shared_array, result):
    # Convert to numpy array for manipulation
    np_data = np.random.rand(100_000)
    np_array = np.frombuffer(shared_array.get_obj(), dtype=np.float64)
    start_time = time.time()
    for i in range(10000):
        np_array[:] = np_data
    result['duration'] = time.time() - start_time

if __name__ == '__main__':
    mp_array = Array('d', 100_000)
    manager = Manager()
    result = manager.dict()

    p = Process(target=mp_array_test, args=(mp_array, result))
    p.start()
    p.join()
    perf = result['duration']
    print(f"Multiprocessing.Array duration: {perf:.4f} seconds")

from multiprocessing import shared_memory, Process
import numpy as np
import time

def shared_memory_test(shm_name, result):
    np_data = np.random.rand(100_000)
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    np_array = np.ndarray((100_000,), dtype=np.float64, buffer=existing_shm.buf)
    start_time = time.time()
    for i in range(10000):
        np_array[:] = np_data
    end_time = time.time()
    existing_shm.close()
    result['duration'] = time.time() - start_time

if __name__ == '__main__':
    shm = shared_memory.SharedMemory(create=True, size=100_000 * np.float64().itemsize)
    manager = Manager()
    result = manager.dict()

    p = Process(target=shared_memory_test, args=(shm.name, result))
    p.start()
    p.join()

    perf = result['duration']
    print(f"Shared Memory duration: {perf:.4f} seconds")
    # Cleanup
    shm.unlink()