from dask.distributed import Client, SSHCluster, as_completed
import time
from dask_stateful_actor import Actor
import asyncio
import tornado.ioloop

def main():
    num_workers = 1
    num_threads = None
    directory = "/home/jurgen/AppsPy/mtdcovabm/test/logs"
    log_files = ["/home/jurgen/AppsPy/mtdcovabm/test/logs/actor_stateful_test_0.txt", "/home/jurgen/AppsPy/mtdcovabm/test/logs/actor_stateful_test_1.txt"]
    dask_nodes = ["localhost", "localhost", "192.168.1.19"]

    cluster = SSHCluster(dask_nodes, 
                        connect_options={"known_hosts": None},
                        worker_options={"n_workers": num_workers, "nthreads": num_threads, "local_directory": directory, }, 
                        scheduler_options={"port": 0, "dashboard_address": ":8797", "local_directory": directory,},
                        worker_class="distributed.Worker",) 
    client = Client(cluster)

    client.upload_file('test/dask_stateful_actor.py')

    workers = client.scheduler_info()["workers"]
    print("num_workers: " + str(len(workers)))
    workers_keys = list(workers.keys())
    print("workers_keys: " + str(workers_keys))

    print("creating actors")
    actors = []
    for worker_index in range(len(workers_keys)):
        params = (worker_index, directory)
        actor_future = client.submit(Actor, params, workers=workers_keys[worker_index], actor=True)

        actor = actor_future.result()
        actors.append(actor)

    print("actors created")

    results = []

    actor0_future = actors[0].send_message(workers_keys[1], "Hello from Actor 0 to Actor 1!", log_files[1])
    results.append(actor0_future)

    actor1_future = actors[1].send_message(workers_keys[0], "Hello from Actor 1 to Actor 0!", log_files[0])
    results.append(actor1_future)

    actor2_future = actors[0].send_message(workers_keys[1], {0: {"age": 20, "cellid": 0}, 1: {"age": 30, "cellid": 1}}, log_files[1])
    results.append(actor2_future)

    actor3_future = actors[1].send_message(workers_keys[0], {2: {"age": 40, "cellid": 2}, 3: {"age": 50, "cellid": 3}}, log_files[0])
    results.append(actor3_future)

    result_index = 0
    for future in as_completed(results):
        result = future.result()
        print("Message Result {0}: {1}".format(str(result_index), str(result)))
        result_index += 1

    print("done")

if __name__ == "__main__":
    main()


