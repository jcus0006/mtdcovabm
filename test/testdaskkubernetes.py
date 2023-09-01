import time
import traceback
import numpy as np
import dask
from dask.distributed import Client, as_completed, SSHCluster
from dask_kubernetes.operator import KubeCluster

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
        with open("testdaskkubernetes_worker.txt", 'w') as f:
            traceback.print_exc(file=f)

if __name__ == '__main__':
    # start = time.time()
    num_processes = 6
    dask_method = 1 # 0 clientsubmit (also futures but lazy evaluation / slowest) 1 delayed (lazy evaluation / fastest) 2 futures (non lazy evaluation (similar to 0) / second fastest)
    num_rows = 16384
    int_range = 10

    cluster = SSHCluster(["localhost", "localhost"], # LAPTOP-FDQJ136P / localhost
                        connect_options={"known_hosts": None},
                        worker_options={"n_workers": num_processes},
                        scheduler_options={"port": 0, "dashboard_address": ":8797"})

    # cluster = KubeCluster(name="my-dask-cluster", image='ghcr.io/dask/dask:2023.8.1-py3.11') # ghcr.io/dask/dask:2023.8.1-py3.11 // ghcr.io/dask/dask:latest
    # cluster.scale(num_processes)

    client = Client(cluster)

    try:
        start = time.time()
        matrix = np.random.randint(0, int_range, size=(num_rows, num_rows))
        rows_per_process = num_rows // num_processes

        params_start = time.time()
        params = []
        # Spawn processes to calculate row sums
        for process_index in range(num_processes):
            start_row = process_index * rows_per_process
            end_row = start_row + rows_per_process
            if process_index == num_processes - 1:
                # Adjust end row for the last process to include remaining rows
                end_row = num_rows

            params.append((matrix, start_row, end_row, process_index))

        params_time_taken = time.time() - params_start
        print("params " + str(params_time_taken))

        delayed_start = time.time()
        delayed_results = []
        # Create a list of delayed objects
        if dask_method == 0:
            for param in params:
                future = client.submit(worker_map, param)
                delayed_results.append(future)

        elif dask_method == 1:
            delayed_results = [dask.delayed(worker_map)(params[i]) for i in range(num_processes)]
        else:
            delayed_results = client.map(worker_map, [params[i] for i in range(num_processes)])  

        delayed_time_taken = time.time() - start
        print("delayed " + str(delayed_time_taken))

        if dask_method == 1:
            compute_start = time.time()
            computed_results = dask.compute(*delayed_results)
            computed_time_taken = time.time() - compute_start
            print("compute " + str(computed_time_taken))
        elif dask_method == 2:
            computed_results = client.gather(delayed_results)

        results_start = time.time()
        results = {}
        # Now you can iterate over the computed results
        if dask_method == 0:
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
    except:
        with open("testdaskkubernetes_master.txt", 'w') as f:
            traceback.print_exc(file=f)
    finally:
        client.shutdown()
