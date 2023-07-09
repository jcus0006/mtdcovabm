import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import traceback
from simulator import contactnetwork
from simulator.agents_mp import Agents
from simulator.cells_mp import CellType
import time

def contactnetwork_parallel(day, 
                            weekday, 
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop, 
                            agents_mp, 
                            agents_mp_cn, 
                            vars_mp, 
                            cat_util,
                            tourists_active_ids, 
                            cells,
                            cells_households, 
                            cells_institutions, 
                            cells_accommodation, 
                            contactnetworkparams, 
                            epidemiologyparams, 
                            dynparams, 
                            contact_network_sum_time_taken, 
                            num_processes=4):
    manager = mp.Manager()
    sync_queue = manager.Queue()
    process_counter = manager.Value("i", num_processes)

    if num_processes > 1:
        # pool = mp.Pool(initializer=init_worker, initargs=(process_counter,))
        pool = mp.Pool()

        mp_cells_keys = []

        cells_agents_timesteps_keys = list(cat_util.cells_agents_timesteps.keys())
        np.random.shuffle(cells_agents_timesteps_keys)
        mp_cells_keys = np.array_split(cells_agents_timesteps_keys, num_processes)

        for process_index in range(num_processes):
            # cells_partial = {}
            cells_agents_timesteps_partial = {}

            cells_keys = mp_cells_keys[process_index]

            for cell_key in cells_keys:
                # cell = cells[cell_key]
                cell_agents_timesteps = cat_util.cells_agents_timesteps[cell_key]

                # cells_partial[cell_key] = cell
                cells_agents_timesteps_partial[cell_key] = cell_agents_timesteps

            print("starting process index " + str(process_index) + " at " + str(time.time()))
        
            # pool.apply_async(contactnetwork_worker, args=((sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cells_agents_timesteps_partial, tourists_active_ids, cells_mp, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter),))
            pool.apply_async(contactnetwork_worker, args=((sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cells_agents_timesteps_partial, tourists_active_ids, cells, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter),))

        # update memory from multiprocessing.queue
        
        start = time.time()
        while process_counter.value > 0 or not sync_queue.empty(): # True
            try:
                type, agent_index, attr_name, value = sync_queue.get(timeout=0.001)  # Poll the queue with a timeout (0 for now - might cause problems)

                if type == "a":
                    agents_mp.set(agent_index, attr_name, value)
                elif type == "v":
                    vars_mp.update(attr_name, value)
            except mp.queues.Empty:
                continue  # Queue is empty, continue polling
        
        sync_time_end = time.time()
        time_taken = sync_time_end - start
        print("cn/epi state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))
        
        start = time.time()
        pool.close()
        time_taken = time.time() - start
        print("pool close time taken " + str(time_taken))

        start = time.time()
        pool.join()
        time_taken = time.time() - start
        print("pool join time taken " + str(time_taken))

        start = time.time()
        agents_mp_cn.cleanup_shared_memory_dynamic(contactnetwork=True)
        time_taken = time.time() - start
        print("clean up time taken " + str(time_taken))
    else:
        params = sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cat_util.cells_agents_timesteps, tourists_active_ids, cells, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, -1, process_counter

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
        sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cell_agents_timesteps, tourists_active_ids, cells, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params

        print("process " + str(process_index))
        # agents_mp_cn = None
        # if process_index == -1:
        #     agents_mp_cn = agents_temp
        # else:
        #     shared_memory_names = agents_temp

        #     agents_mp_cn = Agents()
        #     agents_mp_cn.convert_from_shared_memory(shared_memory_names)

        agents_mp_cn.convert_from_shared_memory_readonly(contactnetwork=True)
        agents_mp_cn.convert_from_shared_memory_dynamic(contactnetwork=True)

        # start = time.time()
        # cells_mp.convert_from_shared_memory_readonly()
        # time_taken = time.time() - start
        # print("cells_mp convert_to_shared_memory_readonly time taken " + str(time_taken))

        # start = time.time()
        # cells_households = cells_mp.get_keys(type=CellType.Household)
        # cells_institutions = cells_mp.get_keys(type=CellType.Institution)
        # cells_accommodation = cells_mp.get_keys(type=CellType.Accommodation)
        # time_taken = time.time() - start
        # print("cells types generation (contact network) time taken " + str(time_taken))

        contact_network_util = contactnetwork.ContactNetwork(n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            agents_mp_cn, 
                                                            cell_agents_timesteps,
                                                            tourists_active_ids,
                                                            cells, 
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