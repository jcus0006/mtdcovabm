import sys
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import threading
import numpy as np
import traceback
import contactnetwork, util, vars
import time

def contactnetwork_parallel(manager,
                            pool,
                            day, 
                            weekday, 
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop, 
                            agents_dynamic,
                            vars_util,
                            cells_type,
                            indids_by_cellid,
                            cells_households, 
                            cells_institutions, 
                            cells_accommodation, 
                            contactnetworkparams, 
                            epidemiologyparams, 
                            dynparams, 
                            contact_network_sum_time_taken, 
                            num_processes=10,
                            num_threads=2,
                            keep_processes_open=True,
                            log_file_name="output.txt"):
    # manager = mp.Manager()
    # sync_queue = manager.Queue()
    process_counter = manager.Value("i", num_processes)

    if num_processes > 1:
        # pool = mp.Pool(initializer=init_worker, initargs=(process_counter,))
        # pool = mp.Pool()

        # mp_cells_keys = []
        # cells_agents_timesteps_keys = list(vars_util.cells_agents_timesteps.keys())
        # np.random.shuffle(cells_agents_timesteps_keys)
        # mp_cells_keys = np.array_split(cells_agents_timesteps_keys, num_processes)

        cells_agents_timesteps_dicts = util.split_cellsagentstimesteps_balanced(vars_util.cells_agents_timesteps, num_processes)

        imap_params, imap_results = [], []

        for process_index in range(num_processes):
            # cells_partial = {}

            cells_agents_timesteps_partial = {}
            agents_partial, agents_ids_by_ages_partial = {}, {}
            vars_util_partial = vars.Vars()
            vars_util_partial.agents_seir_state = vars_util.agents_seir_state # may be optimized by sending only specific day
            vars_util_partial.agents_seir_state_transition_for_day = vars_util.agents_seir_state_transition_for_day # to check re comment in above line for this line

            # cells_keys = mp_cells_keys[process_index]

            # for cell_key in cells_keys:
            #     cell_agents_timesteps = vars_util.cells_agents_timesteps[cell_key]
            #     cells_agents_timesteps_partial[cell_key] = cell_agents_timesteps

            cells_agents_timesteps_partial = cells_agents_timesteps_dicts[process_index]

            unique_agent_ids = set()
            for cell_vals in cells_agents_timesteps_partial.values():
                for cell_agent_timesteps in cell_vals:
                    unique_agent_ids.add(cell_agent_timesteps[0])

            unique_agent_ids = list(unique_agent_ids)

            agents_partial, _, vars_util_partial, _ = util.split_dicts_by_agentsids(unique_agent_ids, agents_dynamic, vars_util, agents_partial, vars_util_partial)

            vars_util_partial.cells_agents_timesteps = cells_agents_timesteps_partial

            print("starting process index " + str(process_index) + " at " + str(time.time()))

            params = (day, 
                      weekday, 
                      n_locals, 
                      n_tourists, 
                      locals_ratio_to_full_pop, 
                      agents_partial, 
                      vars_util_partial, 
                      cells_type, 
                      indids_by_cellid,
                      cells_households, 
                      cells_institutions, 
                      cells_accommodation, 
                      contactnetworkparams, 
                      epidemiologyparams, 
                      dynparams, 
                      contact_network_sum_time_taken, 
                      process_index, 
                      process_counter,
                      log_file_name)
            
            imap_params.append(params)
            # pool.apply_async(contactnetwork_worker, args=(params,))

        imap_results = pool.imap(contactnetwork_worker, imap_params)

        # update memory from multiprocessing.queue
        
        # start = time.time()

        # option 1 - single thread
        # sync_state_info(sync_queue, process_counter)

        # option 2 - multiple threads
        # Create multiple threads to process items from the queue
        # threads = []
        # for _ in range(num_threads):
        #     t = threading.Thread(target=sync_state_info, args=(sync_queue, process_counter))
        #     t.start()
        #     threads.append(t)

        # # Wait for all threads to complete
        # for t in threads:
        #     t.join()

        # option 3 - multiple processes
        # processes = []
        # for process_index in range(6):
        #     process = mp.Process(target=sync_state_info, args=(sync_queue, process_counter))
        #     process.start()
        #     processes.append(process)

        # for process in processes:
        #     process.join()

        # sync_time_end = time.time()
        # time_taken = sync_time_end - start
        # print("contact network state info sync (combined). time taken " + str(time_taken) + ", ended at " + str(sync_time_end))

        start = time.time()

        for result in imap_results:
            process_index, updated_agent_ids, agents_partial, vars_util_partial = result

            print("processing results for process " + str(process_index) + ", received " + str(len(updated_agent_ids)) + " agent ids to sync")
            # print(working_schedule_times_by_resid_ordered)
            # print(itinerary_times_by_resid_ordered)

            agents_dynamic, vars_util = util.sync_state_info_by_agentsids(updated_agent_ids, agents_dynamic, vars_util, agents_partial, vars_util_partial)

            vars_util = util.sync_state_info_sets(vars_util, vars_util_partial)

        vars_util.cells_agents_timesteps = {}
        
        time_taken = time.time() - start
        print("syncing pool imap results back with main process. time taken " + str(time_taken))
        
        if not keep_processes_open:
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
        params = day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_dynamic, vars_util, cells_type, indids_by_cellid, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, -1, process_counter, log_file_name

        contactnetwork_worker(params)

def contactnetwork_worker(params):
    from shared_mp import agents_static

    # still to see how to handle (whether mp.Array or local in process and then sync with main memory)
    # in the case of Agents, these are purely read only however, 
    #  - using mp.dict will introduce speed degradation from locking, 
    #  - passing large coll as param creates overhead from ser/deserialisation, and
    #  - splitting the dict into smaller pieces may be impossible
    # agents, agents_seir_state, agent_seir_state_transition_for_day, agents_infection_type, agents_infection_severity
    # and to check more params from Epidemiology ctor (especially optional params)

    original_stdout = None
    f = None
    stack_trace_log_file_name = ""

    try:
        start = time.time()

        # sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cell_agents_timesteps, tourists_active_ids, cells_mp, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params
        day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_dynamic, vars_util, cells_type, indids_by_cellid, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter, log_file_name = params

        original_stdout = sys.stdout
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_mp_stack_trace_" + str(process_index) + ".txt"
        log_file_name = log_file_name.replace(".txt", "") + "_cn_" + str(process_index) + ".txt"
        f = open(log_file_name, "w")
        sys.stdout = f

        print("process " + str(process_index) + " started at " + str(start))

        contact_network_util = contactnetwork.ContactNetwork(n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            agents_static,
                                                            agents_dynamic,
                                                            vars_util,
                                                            cells_type, 
                                                            indids_by_cellid,
                                                            cells_households, 
                                                            cells_institutions, 
                                                            cells_accommodation, 
                                                            contactnetworkparams, 
                                                            epidemiologyparams, 
                                                            dynparams, 
                                                            contact_network_sum_time_taken, 
                                                            process_index=process_index)

        process_index, updated_agent_ids, agents_partial, vars_util = contact_network_util.simulate_contact_network(day, weekday)
        
        # agents_mp_cn = None
        # contact_network_util = None
        # global proc_counter
        print("process " + str(process_index) + " ended at " + str(time.time()))

        return process_index, updated_agent_ids, agents_partial, vars_util
    except:
        with open(stack_trace_log_file_name, 'w') as f: # cn_mp_stack_trace.txt
            traceback.print_exc(file=f)
    finally:
        process_counter.value -= 1

        if original_stdout is not None:
            sys.stdout = original_stdout

            if f is not None:
                # Close the file
                f.close()

# def sync_state_info(sync_queue, process_counter):
#     global agents_main
#     global vars_util_main

#     start = time.time()
#     while process_counter.value > 0 or not sync_queue.empty(): # True
#         try:
#             type, index, attr_name, value = sync_queue.get(timeout=0.001)  # Poll the queue with a timeout (0.01 / 0 might cause problems)

#             if type is not None:
#                 if type == "a":
#                     if attr_name is not None and attr_name != "":
#                         agents_main[index][attr_name] = value
#                     else:
#                         agents_main[index] = value
#                 elif type == "c":
#                     vars_util_main.update_cells_agents_timesteps(index, value)
#                 elif type == "v":
#                     vars_util_main.update(attr_name, value)
#         except mp.queues.Empty:
#             continue  # Queue is empty, continue polling
    
#     sync_time_end = time.time()
#     time_taken = sync_time_end - start
#     print("contact network state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))