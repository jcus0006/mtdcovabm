from dask.distributed import Client, SSHCluster, as_completed
import time
import traceback
from copy import deepcopy

def return_data(data):
    return data

def process_data(params):
    res, persons, data = params

    res_persons = {}
    res_data = {}

    for id in res:
        res_persons[id] = deepcopy(persons[id])
        res_data[id] = data[id]

    # some computation which is not relevant happens now. will be simplified
    for id in res:
        res_persons[id]["it"] = [[1, 2, 3], [0, 1, 2], [1, 3, 5]]
        res_persons[id]["nda"] = [0, 1, 2, 3, 4]
        res_persons[id]["epia"] = [[1, 2, 3], [5, 6, 7]]
        res_data[id] = res_data[id] + res_persons[id]["a"]
    
    time.sleep(0.1)
    
    return res, res_persons, res_data
    
num_res = 50000
num_people = 500000
num_workers = 1
num_threads = None
batch_size = 1000
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

residences = []
for i in range(num_res):
    group = [j for j in range(i * 10, (i + 1) * 10)]
    residences.append(group)

persons = {i: {"a": 1, "it": None, "nda": None, "epia": None} for i in range(num_people)}
my_dict = {i: 0 for i in range(num_people)}

persons_future = client.submit(return_data, persons)
my_dict_future = client.submit(return_data, my_dict)

params = []
for i in range(num_res):
    param = residences[i], persons_future, my_dict_future
    params.append(param)

futures = client.map(process_data, params, batch_size=batch_size)

for batch in as_completed(futures, with_results=True).batches():
    try:
        for future, result in batch:
            try:
                res, res_persons, res_result = result

                for id in res:
                    persons[id] = res_persons[id]
                    my_dict[id] = res_result[id]
            except:
                with open("logs/dask_sre_dict_log.txt", 'a') as f:
                    traceback.print_exc(file=f)
            finally:
                future.release()
    except:
        with open("logs/dask_sre_dict_as_completed_log.txt", 'a') as f:
            traceback.print_exc(file=f)

print("random value: " + str(my_dict[222]))



