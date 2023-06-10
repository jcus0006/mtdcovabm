import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import time
import logging
import traceback

term_flag = None
shared_res = None

# def calculate_row_sum(result_queue, matrix, start_row, end_row, process_index, is_shared=False):
#     start = time.time()

#     result = {}
#     for i in range(10): # simulation of computation
#         for row in range(start_row, end_row):
#             row_sum = np.sum(matrix[row])

#             result[row] = row_sum

#         for col in range(start_row, end_row):
#             col_sum = np.sum(matrix[:, col])

#             result[col] -= col_sum

#     result_queue.put((start_row, result))

#     time_taken = time.time() - start
#     print("process " + str(process_index) + ": " + str(time_taken))

def calculate_row_sum(result_queue, matrix, start_row, end_row, process_index, is_shared=False):
    global shared_res
    start = time.time()

    result = {}
    for i in range(10): # simulation of computation
        for row in range(start_row, end_row):
            row_sum = np.sum(matrix[row])

            if not is_shared:
                result[row] = row_sum
            else:
                shared_res[row] = row_sum

        for col in range(start_row, end_row):
            col_sum = np.sum(matrix[:, col])

            if not is_shared:
                result[col] -= col_sum
            else:
                shared_res[col] -= col_sum

    if not is_shared:
        result_queue.put((start_row, result))

    time_taken = time.time() - start
    print("process " + str(process_index) + ": " + str(time_taken))

def worker(params):
    result_queue, matrix, start_row, end_row, process_index = params

    calculate_row_sum(result_queue, matrix, start_row, end_row, process_index)

def worker_shm(params):
    try:
        result_queue, shm_name, shape, dtype, start_row, end_row, process_index, is_shared = params
        shm_matrix = shm.SharedMemory(name=shm_name)
        matrix = np.ndarray(shape, dtype=dtype, buffer=shm_matrix.buf)
        calculate_row_sum(result_queue, matrix, start_row, end_row, process_index, is_shared)
    except:
        with open('testmp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)

# def init_worker_keepalive(termination_flag):
#     global term_flag

#     term_flag = termination_flag

def init_worker_keepalive(termination_flag, shared_results):
    global term_flag
    global shared_res

    term_flag = termination_flag
    shared_res = shared_results

def worker_keepalive(params): # start_row, end_row, process_index
    worker_queue, result_queue, shm_name, shape, dtype, process_index = params
    shm_matrix = shm.SharedMemory(name=shm_name)
    matrix = np.ndarray(shape, dtype=dtype, buffer=shm_matrix.buf)

    global term_flag

    while not term_flag.value: # True
        try:
            start_row, end_row = worker_queue.get(timeout=1)  # Poll the queue with a timeout

            calculate_row_sum(result_queue, matrix, start_row, end_row, process_index)
        except mp.queues.Empty:
            continue  # Queue is empty, continue polling

    # The termination flag is set, exit the process gracefully
    print("Termination flag received, exiting worker process " + str(process_index))

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
        process = mp.Process(target=worker_shm, args=((result_queue, shm_name,  matrix.shape, matrix.dtype, start_row, end_row, process_index, False),))
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

def main_pool_sharedmem(pool, result_queue, num_rows, num_processes, int_range, use_map=True, use_shared_results=False, shared_results=None):
    try:
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
                params.append((result_queue, shm_name, matrix.shape, matrix.dtype, start_row, end_row, process_index, use_shared_results))
            else:
                pool.apply_async(worker_shm, args=((result_queue, shm_name,  matrix.shape, matrix.dtype, start_row, end_row, process_index, use_shared_results),))

        if use_map:
            # Call the worker method in parallel
            results = pool.map(worker_shm, iter(params))

        results = {}
        if not use_shared_results:
            for _ in range(num_processes):
                start_index, result = result_queue.get()
                results[start_index] = result

        # Close the pool of processes
        pool.close()
        pool.join()

        time_taken = time.time() - start
        print("all: " + str(time_taken))

        if not use_shared_results:
            start = time.time()
            start_indices_keys = sorted(list(results.keys()))
            combined_result = [result for start_index in start_indices_keys for result in results[start_index].values()]
            # print(combined_result)

            time_taken = time.time() - start
            print("combined: " + str(time_taken))
        else:
            print("shared results can be used. length: " + str(len(shared_results)))

        # Close the shared memory
        shm_matrix.close()

        # Optionally, unlink and remove the shared memory
        shm_matrix.unlink()

    except:
        with open('testmp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)

def main_pool_keepalive_sharedmem(pool, worker_queue, result_queue, termination_flag, num_rows, num_processes, row_steps_per_msg, int_range, use_map=True, create_new_processes=True, terminate_now=True, shm_matrix=None):   
    use_map = False # pool.map is blocking, use apply_async
    
    if create_new_processes:
        # Create a 1024 x 1024 matrix with random integers
        matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))

        # Number of processes
        # num_processes = mp.cpu_count()

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

    if create_new_processes:
        params = []
        # Spawn processes to calculate row sums
        for process_index in range(num_processes):
            if use_map:
                params.append((worker_queue, result_queue, shm_name, matrix.shape, matrix.dtype, process_index))
            else:
                pool.apply_async(worker_keepalive, args=((worker_queue, result_queue, shm_name, matrix.shape, matrix.dtype, process_index),))

        if use_map:
            # Call the worker method in parallel
            results = pool.map(worker_keepalive, iter(params))

    num_messages = 0
    for start_row in range(0, num_rows, row_steps_per_msg):
        end_row = start_row + row_steps_per_msg
        if end_row > num_rows -1:
            # Adjust end row for the last process to include remaining rows
            end_row = num_rows-1

        worker_queue.put((start_row, end_row))
        num_messages += 1

    results = {}
    for _ in range(num_messages):
        start_index, result = result_queue.get()
        results[start_index] = result

    if terminate_now:
        termination_flag.value = 1

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

    return shm_matrix

if __name__ == '__main__':
    num_rows = 16384
    int_range = 10
    num_processes = 4

    manager = mp.Manager()

    result_queue = manager.Queue()

    worker_queue = manager.Queue()

    # to be used with keep_alive
    termination_flag = mp.Value("i", 0)

    # to be used with shared data structures
    shared_results = mp.Array('i', num_rows)

    # pool = mp.Pool(initializer=init_worker_keepalive, initargs=(termination_flag,))
    pool = mp.Pool(initializer=init_worker_keepalive, initargs=(termination_flag, shared_results,))

    # pool = mp.Pool()

    # main_single(result_queue, num_rows, int_range)
    # main_processes(result_queue, num_rows, num_processes, int_range) # by reference
    # main_pool(pool, result_queue, num_rows, num_processes, int_range) # by value
    # main_processes_shared_mem(result_queue, num_rows, num_processes, int_range) # by reference (shared memory)
    # main_pool_sharedmem(pool, result_queue, num_rows, num_processes, int_range, use_map=False) # by value (shared memory)
    # main_pool_keepalive_sharedmem(pool, worker_queue, result_queue, termination_flag, num_rows, num_processes, 2048, int_range, use_map=False) # by value (keep alive / shared memory)

    # shared_mem keep alive
    # sum_time_taken = 0
    # n = 10
    # shm_matrix = ""

    # for i in range(n):
    #     start = time.time()

    #     create_new_processes = False
    #     terminate_now = False

    #     if i == 0:
    #         create_new_processes = True

    #     if i == n-1:
    #         terminate_now = True

    #     shm_matrix = main_pool_keepalive_sharedmem(pool, worker_queue, result_queue, termination_flag, num_rows, num_processes, 2048, int_range, use_map=False, create_new_processes=create_new_processes, terminate_now=terminate_now, shm_matrix=shm_matrix) # by value (keep alive / shared memory)

    #     time_taken = time.time() - start
    #     sum_time_taken += time_taken
        
    # avg_time_taken = sum_time_taken / n
    # print("average time taken: " + str(avg_time_taken))
    # end - shared_mem keep alive

    main_pool_sharedmem(pool, result_queue, num_rows, num_processes, int_range, use_map=False, use_shared_results=True, shared_results=shared_results) # by value (shared memory)