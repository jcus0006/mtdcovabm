import time
import traceback
import numpy as np
import dask
from dask.distributed import Client, LocalCluster, as_completed
import dask.array as da

def calculate_row_sum_map(matrix, start_row, end_row, process_index):
    start = time.time()

    result = {}

    result_queue = []
    for i in range(10): # simulation of computation
        for row in range(start_row, end_row):
            row_sum = np.sum(matrix[row])
            result[row] = row_sum

        for col in range(start_row, end_row):
            col_sum = np.sum(matrix[:, col])
            result[col] -= col_sum

    result_queue.append((start_row, result))

    time_taken = time.time() - start
    print("process " + str(process_index) + ": " + str(time_taken))

    return result_queue

def worker_map(params):
    try:
        matrix, start_row, end_row, process_index = params

        return calculate_row_sum_map(matrix, start_row, end_row, process_index)
    except:
        with open("testdask.txt", 'w') as f:
            traceback.print_exc(file=f)

if __name__ == '__main__':
    start = time.time()
    # 4 processes, 1 thread each, 8192, 8192 shape
    # -------------------------------------------------------------------
    # 0 client.submit (also futures but lazy evaluation / slowest - 13.88s) 
    # 1 delayed (lazy evaluation / fastest - 11.27s) 
    # 2 submit with gather - futures (non lazy evaluation (similar to 0) / second fastest - 13.39s), 
    # 3 client.submit with scattering of matrix into distributed memory  - 11.76s
    # 4 client map - 11.69s
    # 5 delayed with da.array - 11.36s (7.89 seconds with 4 threads per worker & (512, 512) chunks)
    # 6 client submit scattered with da.array (exremely slow with large data) - 213.48s
    # 7 delayed da.array with client.persist - 11.02s (8.10s with 4 threads per worker & (512, 512) chunks)

    dask_method = 5 
    num_rows = 8192 # 16384
    int_range = 10
    num_processes = 4
    threads_per_worker = 4 # 1

    matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))
    rows_per_process = num_rows // num_processes

    da_matrix = None
    if dask_method == 5 or dask_method == 6 or dask_method == 7:
        da_matrix = da.from_array(matrix, chunks=(512, 512)) # 16 x 16 smaller arrays along each dimension
        matrix = da_matrix

    with LocalCluster(n_workers=num_processes,
        processes=True,
        threads_per_worker=threads_per_worker
    ) as cluster, Client(cluster) as client:
        # cluster_start = time.time()
        # # Create a local cluster with num_processes workers
        # cluster = LocalCluster(n_workers= num_processes)
        # client = Client(cluster
        # cluster_time_taken = time.time() - cluster_start
        # print("create cluster " + str(cluster_time_taken))

        params_start = time.time()
        params = []
        if dask_method != 3 and dask_method != 6 and dask_method != 7:
            # Spawn processes to calculate row sums
            for process_index in range(num_processes):
                start_row = process_index * rows_per_process
                end_row = start_row + rows_per_process
                if process_index == num_processes - 1:
                    # Adjust end row for the last process to include remaining rows
                    end_row = num_rows

                params.append((matrix, start_row, end_row, process_index))
        else:
            # Spawn processes to calculate row sums
            for process_index in range(num_processes):
                start_row = process_index * rows_per_process
                end_row = start_row + rows_per_process
                if process_index == num_processes - 1:
                    # Adjust end row for the last process to include remaining rows
                    end_row = num_rows

                params.append((start_row, end_row, process_index))

        params_time_taken = time.time() - params_start
        print("params " + str(params_time_taken))

        delayed_start = time.time()
        delayed_results = []
        # Create a list of delayed objects
        if dask_method == 0 or dask_method == 2:
            for param in params:
                future = client.submit(worker_map, param)
                delayed_results.append(future)
        elif dask_method == 1 or dask_method == 5:
            delayed_results = [dask.delayed(worker_map)(params[i]) for i in range(num_processes)]
        elif dask_method == 3 or dask_method == 6:
            x_future = client.scatter(matrix) # in 3 case its a numpy array. in 6 its a dask.array
            for param in params:
                future = client.submit(calculate_row_sum_map, x_future, param[0], param[1], param[2])
                delayed_results.append(future)
        elif dask_method == 7:
            x_future = client.persist(matrix) # dask array matrix
            delayed_results = [dask.delayed(worker_map)([x_future, params[i][0], params[i][1], params[i][2]]) for i in range(num_processes)]
        else:
            delayed_results = client.map(worker_map, [params[i] for i in range(num_processes)])  

        delayed_time_taken = time.time() - start
        print("delayed " + str(delayed_time_taken))

        if dask_method == 1 or dask_method == 5 or dask_method == 7:
            compute_start = time.time()
            computed_results = dask.compute(*delayed_results)
            computed_time_taken = time.time() - compute_start
            print("compute " + str(computed_time_taken))
        elif dask_method == 2:
            computed_results = client.gather(delayed_results)

        results_start = time.time()
        results = {}
        # Now you can iterate over the computed results
        if dask_method == 0 or dask_method == 3 or dask_method == 4 or dask_method == 6:
            for future in as_completed(delayed_results):
                res = future.result()
                start_index, result = res[0][0], res[0][1]
                results[start_index] = result      
        else:
            for res in computed_results:
                start_index, result = res[0][0], res[0][1]
                results[start_index] = result

        results_time_taken = time.time() - results_start
        print("results " + str(results_time_taken))

        # close_start = time.time()
        # # Close the Dask client and cluster
        # cluster.close()
        # client.close()
        # close_time_taken = time.time() - close_start
        # print("close " + str(close_time_taken))

        combined_start = time.time()
        start_indices_keys = sorted(list(results.keys()))
        combined_result = [result for start_index in start_indices_keys for result in results[start_index].values()]
        # print(combined_result)

        combined_time_taken = time.time() - combined_start
        print("combined: " + str(combined_time_taken))

        time_taken = time.time() - start
        print("all " + str(time_taken))
