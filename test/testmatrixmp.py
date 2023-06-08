import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import time
import logging

def calculate_row_sum(result_queue, matrix, start_row, end_row, process_index):
    start = time.time()

    result = {}
    for i in range(10): # just simulation of computation
        for row in range(start_row, end_row):
            row_sum = np.sum(matrix[row])
            result[row] = row_sum

        for col in range(start_row, end_row):
            col_sum = np.sum(matrix[:, col])
            result[col] -= col_sum

    result_queue.put((start_row, result))

    time_taken = time.time() - start
    print("process " + str(process_index) + ": " + str(time_taken))

def worker(params):
    result_queue, matrix, start_row, end_row, process_index = params

    calculate_row_sum(result_queue, matrix, start_row, end_row, process_index)

def worker_shm(params):
    result_queue, shm_name, shape, dtype, start_row, end_row, process_index = params
    shm_matrix = shm.SharedMemory(name=shm_name)
    matrix = np.ndarray(shape, dtype=dtype, buffer=shm_matrix.buf)
    calculate_row_sum(result_queue, matrix, start_row, end_row, process_index)

# class CustomPool(mp.Pool):
#     def __init__(self, result_queue, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.result_queue = result_queue

#     def apply_async(self, func, args=(), kwds={}, callback=None):
#         # Wrap the original function and pass the result_queue as an additional argument
#         func_wrapper = lambda args: func(self.result_queue, *args)
#         super().apply_async(func_wrapper, args, kwds, callback)

def main_single(result_queue, num_rows, int_range):
    # Create a 1024 x 1024 matrix with random integers
    matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))

    # Create a result queue to store the results from child processes
    # result_queue = mp.Queue()

    start = time.time()
    calculate_row_sum(result_queue, matrix, 0, num_rows-1, 0)

    results = {}
    for _ in range(1):
        start_index, result = result_queue.get()
        results[start_index] = result

    time_taken = time.time() - start
    print("all: " + str(time_taken))

    start = time.time()
    start_indices_keys = sorted(list(results.keys()))
    combined_result = [result for start_index in start_indices_keys for result in results[start_index].values()]
    # print(combined_result)

    time_taken = time.time() - start
    print("combined: " + str(time_taken))

def main_processes(result_queue, num_rows, num_processes, int_range):
    # mp.log_to_stderr(level=logging.CRITICAL)

    # Create a 1024 x 1024 matrix with random integers
    matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))

    # Number of processes
    # num_processes = mp.cpu_count()

    # Calculate number of rows per process
    rows_per_process = num_rows // num_processes

    # Create a list to store the processes
    processes = []

    # Create a result queue to store the results from child processes
    # result_queue = mp.Queue()

    start = time.time()

    # Spawn processes to calculate row sums
    for process_index in range(num_processes):
        start_row = process_index * rows_per_process
        end_row = start_row + rows_per_process
        if process_index == num_processes - 1:
            # Adjust end row for the last process to include remaining rows
            end_row = num_rows
        process = mp.Process(target=calculate_row_sum, args=(result_queue, matrix, start_row, end_row, process_index))
        process.start()
        processes.append(process)
    
    results = {}
    for _ in range(num_processes):
        start_index, result = result_queue.get()
        results[start_index] = result

    for process in processes:
        process.join()

    time_taken = time.time() - start
    print("all: " + str(time_taken))

    start = time.time()
    start_indices_keys = sorted(list(results.keys()))
    combined_result = [result for start_index in start_indices_keys for result in results[start_index].values()]
    # print(combined_result)

    time_taken = time.time() - start
    print("combined: " + str(time_taken))

def main_processes_shared_mem(result_queue, num_rows, num_processes, int_range):
    # mp.log_to_stderr(level=logging.CRITICAL)

    # Create a 1024 x 1024 matrix with random integers
    matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))

    # Create shared memory to hold the matrix data
    shm_matrix = shm.SharedMemory(create=True, size=matrix.nbytes)

    # Map the shared memory to a numpy array
    shared_array = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm_matrix.buf)

    # Copy the matrix data to the shared memory
    np.copyto(shared_array, matrix)

    # Get the name of the shared memory
    shm_name = shm_matrix.name

    # Number of processes
    # num_processes = mp.cpu_count()

    # Calculate number of rows per process
    rows_per_process = num_rows // num_processes

    # Create a list to store the processes
    processes = []

    # Create a result queue to store the results from child processes
    # result_queue = mp.Queue()

    start = time.time()

    # Spawn processes to calculate row sums
    for process_index in range(num_processes):
        start_row = process_index * rows_per_process
        end_row = start_row + rows_per_process
        if process_index == num_processes - 1:
            # Adjust end row for the last process to include remaining rows
            end_row = num_rows
        process = mp.Process(target=worker_shm, args=((result_queue, shm_name,  matrix.shape, matrix.dtype, start_row, end_row, process_index),))
        process.start()
        processes.append(process)
    
    results = {}
    for _ in range(num_processes):
        start_index, result = result_queue.get()
        results[start_index] = result

    for process in processes:
        process.join()

    time_taken = time.time() - start
    print("all: " + str(time_taken))

    start = time.time()
    start_indices_keys = sorted(list(results.keys()))
    combined_result = [result for start_index in start_indices_keys for result in results[start_index].values()]
    # print(combined_result)

    time_taken = time.time() - start
    print("combined: " + str(time_taken))

def main_pool(pool, result_queue, num_rows, num_processes, int_range, use_map=True):
    # Create a 1024 x 1024 matrix with random integers
    matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))

    # Number of processes
    # num_processes = mp.cpu_count()

    # Calculate number of rows per process
    rows_per_process = num_rows // num_processes

    start = time.time()

    if use_map:
        # Create a pool of processes
        pool = mp.Pool(num_processes)

    params = []
    # Spawn processes to calculate row sums
    for process_index in range(num_processes):
        start_row = process_index * rows_per_process
        end_row = start_row + rows_per_process
        if process_index == num_processes - 1:
            # Adjust end row for the last process to include remaining rows
            end_row = num_rows

        if use_map:
            params.append((result_queue, matrix, start_row, end_row, process_index))
        else:
            pool.apply_async(worker, args=((result_queue, matrix, start_row, end_row, process_index),))

    if use_map:
        # Call the worker method in parallel
        results = pool.map(worker, iter(params))

    results = {}
    for _ in range(num_processes):
        start_index, result = result_queue.get()
        results[start_index] = result

    # Close the pool of processes
    pool.close()
    pool.join()

    time_taken = time.time() - start
    print("all: " + str(time_taken))

    start = time.time()
    start_indices_keys = sorted(list(results.keys()))
    combined_result = [result for start_index in start_indices_keys for result in results[start_index].values()]
    # print(combined_result)

    time_taken = time.time() - start
    print("combined: " + str(time_taken))

def main_pool_sharedmem(pool, result_queue, num_rows, num_processes, int_range, use_map=True):
    # Create a 1024 x 1024 matrix with random integers
    matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))

    # Number of processes
    # num_processes = mp.cpu_count()

    # Calculate number of rows per process
    rows_per_process = num_rows // num_processes

    # Create shared memory to hold the matrix data
    shm_matrix = shm.SharedMemory(create=True, size=matrix.nbytes)

    # Map the shared memory to a numpy array
    shared_array = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm_matrix.buf)

    # Copy the matrix data to the shared memory
    np.copyto(shared_array, matrix)

    # Get the name of the shared memory
    shm_name = shm_matrix.name

    start = time.time()

    # Create a pool of processes
    # pool = mp.Pool(num_processes)

    params = []
    # Spawn processes to calculate row sums
    for process_index in range(num_processes):
        start_row = process_index * rows_per_process
        end_row = start_row + rows_per_process
        if process_index == num_processes - 1:
            # Adjust end row for the last process to include remaining rows
            end_row = num_rows

        if use_map:
            params.append((result_queue, shm_name, matrix.shape, matrix.dtype, start_row, end_row, process_index))
        else:
            pool.apply_async(worker_shm, args=((result_queue, shm_name,  matrix.shape, matrix.dtype, start_row, end_row, process_index),))

    if use_map:
        # Call the worker method in parallel
        results = pool.map(worker_shm, iter(params))

    results = {}
    for _ in range(num_processes):
        start_index, result = result_queue.get()
        results[start_index] = result

    # Close the pool of processes
    pool.close()
    pool.join()

    # Close the shared memory
    shm_matrix.close()

    # Optionally, unlink and remove the shared memory
    shm_matrix.unlink()

    time_taken = time.time() - start
    print("all: " + str(time_taken))

    start = time.time()
    start_indices_keys = sorted(list(results.keys()))
    combined_result = [result for start_index in start_indices_keys for result in results[start_index].values()]
    # print(combined_result)

    time_taken = time.time() - start
    print("combined: " + str(time_taken))

if __name__ == '__main__':
    manager = mp.Manager()

    result_queue = manager.Queue()

    pool = mp.Pool()
    # pool = CustomPool(result_queue=result_queue)

    # main_single(result_queue, 16384, 10)
    # main_processes(result_queue, 16384, 12, 10) # by reference
    # main_pool(pool, result_queue, 16384, 12, 10) # by value
    main_processes_shared_mem(result_queue, 16384, 12, 10) # by reference (shared memory)
    # main_pool_sharedmem(pool, result_queue, 16384, 12, 10, use_map=False) # by value (shared memory)