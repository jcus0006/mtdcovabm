import multiprocessing as mp
import multiprocessing.shared_memory as shm
import threading
import numpy as np
import traceback
from simulator import contactnetwork
import time

agents_main = None
vars_util_main = None
def contactnetwork_parallel(day, 
                            weekday, 
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop, 
                            agents,
                            vars_util,
                            cells,
                            cells_households, 
                            cells_institutions, 
                            cells_accommodation, 
                            contactnetworkparams, 
                            epidemiologyparams, 
                            dynparams, 
                            contact_network_sum_time_taken, 
                            num_processes=10,
                            num_threads=2):
    global agents_main
    global vars_util_main

    agents_main = agents
    vars_util_main = vars_util

    manager = mp.Manager()
    sync_queue = manager.Queue()
    process_counter = manager.Value("i", num_processes)

    if num_processes > 1:
        # pool = mp.Pool(initializer=init_worker, initargs=(process_counter,))
        pool = mp.Pool()

        mp_cells_keys = []

        cells_agents_timesteps_keys = list(vars_util.cells_agents_timesteps.keys())
        np.random.shuffle(cells_agents_timesteps_keys)
        mp_cells_keys = np.array_split(cells_agents_timesteps_keys, num_processes)

        for process_index in range(num_processes):
            # cells_partial = {}
            cells_agents_timesteps_partial = {}

            cells_keys = mp_cells_keys[process_index]

            for cell_key in cells_keys:
                # cell = cells[cell_key]
                cell_agents_timesteps = vars_util.cells_agents_timesteps[cell_key]

                # cells_partial[cell_key] = cell
                cells_agents_timesteps_partial[cell_key] = cell_agents_timesteps

            print("starting process index " + str(process_index) + " at " + str(time.time()))
        
            # pool.apply_async(contactnetwork_worker, args=((sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cells_agents_timesteps_partial, tourists_active_ids, cells_mp, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter),))
            pool.apply_async(contactnetwork_worker, args=((sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents, vars_util, cells, cells_agents_timesteps_partial, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter),))

        # update memory from multiprocessing.queue
        
        start = time.time()
        # while process_counter.value > 0 or not sync_queue.empty(): # True
        #     try:
        #         type, index, attr_name, value = sync_queue.get(timeout=0.001)  # Poll the queue with a timeout (0.01 / 0 might cause problems)

        #         if type == "a":
        #             if attr_name is not None and attr_name != "":
        #                 agents[index][attr_name] = value
        #             else:
        #                 agents[index] = value
        #         elif type == "c":
        #             vars_util.update_cells_agents_timesteps(index, value)
        #         elif type == "v":
        #             vars_util.update(attr_name, value)
        #     except mp.queues.Empty:
        #         continue  # Queue is empty, continue polling

        # option 1 - single thread
        # sync_state_info(sync_queue, process_counter)

        # option 2 - multiple threads
        # Create multiple threads to process items from the queue
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=sync_state_info, args=(sync_queue, process_counter))
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # option 3 - multiple processes
        # processes = []
        # for process_index in range(6):
        #     process = mp.Process(target=sync_state_info, args=(sync_queue, process_counter))
        #     process.start()
        #     processes.append(process)

        # for process in processes:
        #     process.join()

        sync_time_end = time.time()
        time_taken = sync_time_end - start
        print("contact network state info sync (combined). time taken " + str(time_taken) + ", ended at " + str(sync_time_end))
        
        start = time.time()
        pool.close()
        manager.shutdown()
        time_taken = time.time() - start
        print("pool close time taken " + str(time_taken))

        start = time.time()
        pool.join()
        time_taken = time.time() - start
        print("pool join time taken " + str(time_taken))
    else:
        params = sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents, vars_util, cells, cell_agents_timesteps, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, -1, process_counter

        contactnetwork_worker(params)

def contactnetwork_worker(params):
    # still to see how to handle (whether mp.Array or local in process and then sync with main memory)
    # in the case of Agents, these are purely read only however, 
    #  - using mp.dict will introduce speed degradation from locking, 
    #  - passing large coll as param creates overhead from ser/deserialisation, and
    #  - splitting the dict into smaller pieces may be impossible
    # agents, agents_seir_state, agent_seir_state_transition_for_day, agents_infection_type, agents_infection_severity
    # and to check more params from Epidemiology ctor (especially optional params)

    try:
        print("process started " + str(time.time()))

        # sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cell_agents_timesteps, tourists_active_ids, cells_mp, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params
        sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents, vars_util, cells, cell_agents_timesteps, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params

        print("process " + str(process_index))

        contact_network_util = contactnetwork.ContactNetwork(n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            agents,
                                                            vars_util,
                                                            cells, 
                                                            cell_agents_timesteps,
                                                            cells_households, 
                                                            cells_institutions, 
                                                            cells_accommodation, 
                                                            contactnetworkparams, 
                                                            epidemiologyparams, 
                                                            dynparams, 
                                                            contact_network_sum_time_taken, 
                                                            process_index=process_index, 
                                                            sync_queue=sync_queue)

        contact_network_util.simulate_contact_network(day, weekday)
        
        # agents_mp_cn = None
        # contact_network_util = None
        # global proc_counter
        print("process " + str(process_index) + ", ended at " + str(time.time()))
    except:
        with open('cn_mp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)
    finally:
        process_counter.value -= 1

def sync_state_info(sync_queue, process_counter):
    global agents_main
    global vars_util_main

    start = time.time()
    while process_counter.value > 0 or not sync_queue.empty(): # True
        try:
            type, index, attr_name, value = sync_queue.get(timeout=0.001)  # Poll the queue with a timeout (0.01 / 0 might cause problems)

            if type is not None:
                if type == "a":
                    if attr_name is not None and attr_name != "":
                        agents_main[index][attr_name] = value
                    else:
                        agents_main[index] = value
                elif type == "c":
                    vars_util_main.update_cells_agents_timesteps(index, value)
                elif type == "v":
                    vars_util_main.update(attr_name, value)
        except mp.queues.Empty:
            continue  # Queue is empty, continue polling
    
    sync_time_end = time.time()
    time_taken = sync_time_end - start
    print("contact network state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))