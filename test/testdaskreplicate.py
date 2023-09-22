import json
import os
import time
import numpy as np
import dask
from dask.distributed import Client, SSHCluster, get_worker, Worker, WorkerPlugin, SchedulerPlugin
from functools import partial

# def CustomWorkerPlugin(WorkerPlugin):
#     def __init__(self, path):
#         self.path = path
        
#     async def setup(self, worker: Worker):
#         self.worker = worker
        
#         self.worker.plugins["read-only-data"] = read_only_data(self.path)

#     def transition(self, key, start, finish, **kwargs):
#         if start == "waiting":
#             self.agents = read_only_data(self.path)

# def CustomWorkerPlugin(WorkerPlugin):
#     def __init__(self, path):
#         self.path = path

#     def setup(self, worker):
#         read_only_data(worker, self.path)

# def read_only_data(jsonfilepath):
#     with open(jsonfilepath, "r") as readfile:
#         return json.load(readfile)
    
def read_only_data(dask_worker: Worker, jsonfilepath):
    with open(jsonfilepath, "r") as readfile:
        dask_worker.data["agents"] = json.load(readfile)
    
def does_the_work():
    worker = get_worker()

    agents = worker.data["agents"]

    # rod = worker.plugins["read-only-data"]
    # agents = rod.agents

    ages = []
    for i in range(1):
        ages = []
        for _, agent in agents.items():
            ages.append(agent["age"])

    ages = np.array(ages)

    return ages.mean()

def inc(x):
    return x + 1

def add(x, y):
    return x + y

n_workers = 6
current_directory = os.getcwd()
# populationsubfolder = "500kagents2mtourists2019"
populationsubfolder = "1kagents2ktourists2019"
mypath = os.path.join(current_directory, "population", populationsubfolder, "agents.json")

# start = time.time()
# # no dask
# agents = read_only_data(mypath)
# ages_average_0 = does_the_work(agents)
# print("ages average {0}".format(ages_average_0))
# time_taken = time.time() - start
# print("no dask: " + str(time_taken))

start = time.time()
cluster = SSHCluster(["localhost", "localhost"], # LAPTOP-FDQJ136P / localhost
                            connect_options={"known_hosts": None},
                            worker_options={"n_workers": n_workers, },
                            scheduler_options={"port": 0, "dashboard_address": ":8797",},) 

print("create client")
client = Client(cluster)

# client.register_worker_callbacks(lambda worker: read_only_data(worker, mypath))
# client.register_worker_plugin(CustomWorkerPlugin(mypath), name="read-only-data")
# client.register_worker_plugin(read_only_data(mypath), name="read-only-data")
callback = partial(read_only_data, jsonfilepath=mypath)
client.register_worker_callbacks(callback)

# print("submit read_only_data with param " + mypath)
# future = client.submit(read_only_data, mypath)

# cr_start = time.time()
# client.replicate(future)
# cr_timetaken = time.time() - start
# print("client replicate " + str(cr_timetaken))

future2 = client.submit(does_the_work)
ages_average = future2.result()
print("ages average {0}".format(ages_average))

time_taken = time.time() - start
print("dask: " + str(time_taken))

# start = time.time()
# a = client.submit(add, 1, 2)
# time_taken = time.time() - start
# print("a " + str(time_taken))

# start = time.time()
# b = client.submit(add, 5, 10)
# time_taken = time.time() - start
# print("b " + str(time_taken))

# start = time.time()
# c = client.map(inc, range(1000))
# time_taken = time.time() - start
# print("c " + str(time_taken))

# start = time.time()
# print(a.result())
# time_taken = time.time() - start
# print("a result " + str(time_taken))

# start = time.time()
# print(b.result())
# time_taken = time.time() - start
# print("b result " + str(time_taken))

# start = time.time()
# print("c length: " + str(len(c)))
# time_taken = time.time() - start
# print("c result " + str(time_taken))