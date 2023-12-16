from dask.distributed import Client, SSHCluster, as_completed
import time

# run on cluster
def create_groups(n):
    groups = []
    for i in range(n):
        group = [j for j in range(i * 10, (i + 1) * 10)]
        groups.append(group)
    return groups

def process_data(group_ids, data):
    result = []
    for group_id in group_ids:
        person_ids = data[group_id]

        for temp in person_ids:
            temp = temp * temp
            result.append(temp)

    return result

def return_data(data):
    return data

# run on client
def submit_futures(n, method, data, ids=None): # data may be future
    futures = []

    for i in range(n):
        if ids is None:
            j = i + 1
            ids = [1 * j, 11 * j, 22 * j, 33 * j, 44 * j, 55 * j, 66 * j, 77 * j, 88* j, 99 * j]

        future = client.submit(method, ids, data)
        futures.append(future)

    return futures

def read_futures(futures):
    results = []
    for future in as_completed(futures):
        result = future.result()
        results.append(result)

    return len(results)

num_people = 50000
num_workers = 1
num_threads = None
directory = "/home/jurgen/AppsPy/mtdcovabm"
dask_nodes = ["localhost", "localhost", "localhost", "localhost", "localhost"]

start = time.time()
cluster = SSHCluster(dask_nodes, 
                    connect_options={"known_hosts": None},
                    worker_options={"n_workers": num_workers, "nthreads": num_threads, "local_directory": directory, }, 
                    scheduler_options={"port": 0, "dashboard_address": ":8797", "local_directory": directory,},) 
client = Client(cluster)
time_taken = time.time() - start

print("start cluster and client {0}".format(str(time_taken)))

start = time.time()
groups = create_groups(num_people)
time_taken = time.time() - start

print("create groups on main process {0}".format(str(time_taken)))

start = time.time()
groups_future = client.submit(create_groups, num_people)
time_taken = time.time() - start

print("create groups remotely & return future {0}".format(str(time_taken)))

start = time.time()
groups_scattered_future = client.scatter(groups)
time_taken = time.time() - start

print("scatter groups (serialize) {0}".format(str(time_taken)))

start = time.time()
groups_submitted_future = client.submit(return_data, groups)
time_taken = time.time() - start

print("submitted groups (serialize) x 1 {0}".format(str(time_taken)))

# raw data x 1
start = time.time()
future_result = client.submit(process_data, [1, 11, 22, 33, 44, 55, 66, 77, 88, 99], groups)
time_taken = time.time() - start

print("process_data raw data x 1 {0}".format(str(time_taken)))

start = time.time()
results = future_result.result()
time_taken = time.time() - start

print("downloading results (raw data), num_groups: {0}, {1}".format(len(results), str(time_taken)))

# future x 1
start = time.time()
future_result = client.submit(process_data, [1, 11, 22, 33, 44, 55, 66, 77, 88, 99], groups_future)
time_taken = time.time() - start

print("process_data future {0}".format(str(time_taken)))

start = time.time()
results = future_result.result()
time_taken = time.time() - start

print("downloading results (future), num_groups: {0}, {1}".format(len(results), str(time_taken)))

# scatter x 1
start = time.time()
future_result = client.submit(process_data, [1, 11, 22, 33, 44, 55, 66, 77, 88, 99], groups_scattered_future)
time_taken = time.time() - start

print("process_data (scatter) {0}".format(str(time_taken)))

start = time.time()
results = future_result.result()
time_taken = time.time() - start

print("downloading results (scatter), num_groups: {0}, {1}".format(len(results), str(time_taken)))

# submitted x 1 
start = time.time()
future_result = client.submit(process_data, [1, 11, 22, 33, 44, 55, 66, 77, 88, 99], groups_submitted_future)
time_taken = time.time() - start

print("process_data (submitted) {0}".format(str(time_taken)))

start = time.time()
results = future_result.result()
time_taken = time.time() - start

print("downloading results (submitted), num_groups: {0}, {1}".format(len(results), str(time_taken)))

# raw data x 16
start = time.time()
futures = submit_futures(16, process_data, groups)
time_taken = time.time() - start
print("process_data (raw data) x 16, {0}".format(str(time_taken)))

start = time.time()
num_results = read_futures(futures)
time_taken = time.time() - start
print("downloading results (raw data) x 16, num_groups: {0}, {1}".format(str(num_results), str(time_taken)))

# future x 16
start = time.time()
futures = submit_futures(16, process_data, groups_future)
time_taken = time.time() - start
print("process_data (future) x 16, {0}".format(str(time_taken)))

start = time.time()
num_results = read_futures(futures)
time_taken = time.time() - start
print("downloading results (future) x 16, num_groups: {0}, {1}".format(str(num_results), str(time_taken)))

# scattered x 16
start = time.time()
futures = submit_futures(16, process_data, groups_scattered_future)
time_taken = time.time() - start
print("process_data (scattered) x 16, {0}".format(str(time_taken)))

start = time.time()
num_results = read_futures(futures)
time_taken = time.time() - start
print("downloading results (scattered) x 16, num_groups: {0}, {1}".format(str(num_results), str(time_taken)))

# submitted x 16
start = time.time()
futures = submit_futures(16, process_data, groups_submitted_future)
time_taken = time.time() - start
print("process_data (submitted) x 16, {0}".format(str(time_taken)))

start = time.time()
num_results = read_futures(futures)
time_taken = time.time() - start
print("downloading results (submitted) x 16, num_groups: {0}, {1}".format(str(num_results), str(time_taken)))

