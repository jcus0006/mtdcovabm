import sys
import os
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import threading
import numpy as np
import traceback
import contactnetwork, util, daskutil, vars
import time
from copy import deepcopy
from util import MethodType
import gc
import psutil
import customdict

def contactnetwork_parallel(manager,
                            pool,
                            day, 
                            weekday, 
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop, 
                            agents_epi,
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
                            log_file_name="output.txt",
                            agents_static=None,
                            static_agents_dict=None):
    # manager = mp.Manager()
    # sync_queue = manager.Queue()
    stack_trace_log_file_name = ""
    # original_stdout = sys.stdout

    try:
        process_counter = None
        if num_processes > 1:
            process_counter = manager.Value("i", num_processes)

        folder_name = ""
        if log_file_name != "output.txt":
            folder_name = os.path.dirname(log_file_name)
        else:
            folder_name = os.getcwd()

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_main_mp_stack_trace_" + str(day) + ".txt"
        task_results_stack_trace_log_file_name = os.path.join(folder_name, "cn_main_res_task_results_stack_trace_" + str(day) + ".txt")

        if num_processes > 1:
            # pool = mp.Pool(initializer=init_worker, initargs=(process_counter,))
            # pool = mp.Pool()

            # mp_cells_keys = []
            # cells_agents_timesteps_keys = list(vars_util.cells_agents_timesteps.keys())
            # np.random.shuffle(cells_agents_timesteps_keys)
            # mp_cells_keys = np.array_split(cells_agents_timesteps_keys, num_processes)

            cells_agents_timesteps_dicts = util.split_cellsagentstimesteps_balanced(vars_util.cells_agents_timesteps, num_processes)

            imap_params, imap_results = [], []

            cat_size = util.asizeof_formatted(vars_util.cells_agents_timesteps)
            agents_epi_size = util.asizeof_formatted(agents_epi)
            vars_util_size = util.asizeof_formatted(vars_util)
            print(f"cat size: {cat_size}, agents_epi size: {agents_epi_size}, vars_util size: {vars_util_size}")

            for process_index in range(num_processes):
                # cells_partial = {}

                cells_agents_timesteps_partial = customdict.CustomDict()
                agents_epi_partial = {}
                vars_util_partial = vars.Vars()
                # vars_util_partial.agents_seir_state = vars_util.agents_seir_state # may be optimized by sending only specific day
                # vars_util_partial.agents_seir_state_transition_for_day = vars_util.agents_seir_state_transition_for_day # to check re comment in above line for this line

                # cells_keys = mp_cells_keys[process_index]

                # for cell_key in cells_keys:
                #     cell_agents_timesteps = vars_util.cells_agents_timesteps[cell_key]
                #     cells_agents_timesteps_partial[cell_key] = cell_agents_timesteps

                cells_agents_timesteps_partial = cells_agents_timesteps_dicts[process_index]

                uai_start = time.time()
                unique_agent_ids = set()
                for cell_vals in cells_agents_timesteps_partial.values():
                    for cell_agent_timesteps in cell_vals:
                        unique_agent_ids.add(cell_agent_timesteps[0])

                unique_agent_ids = list(unique_agent_ids)
                uai_time_taken = time.time() - uai_start
                print("generating unique_agent_ids: " + str(uai_time_taken))

                split_agents_start = time.time()
                agents_epi_partial, _, vars_util_partial, _ = util.split_dicts_by_agentsids(unique_agent_ids, agents_epi, vars_util, agents_epi_partial, vars_util_partial)
                split_time_taken = time.time() - split_agents_start
                print("split agents_ids: " + str(split_time_taken))

                vars_util_partial.cells_agents_timesteps = cells_agents_timesteps_partial

                print("starting process index " + str(process_index) + " at " + str(time.time()) + " with " + str(len(cells_agents_timesteps_partial)) + " cells and " + str(len(unique_agent_ids)) + " agents")

                # params = (day, 
                #         weekday, 
                #         n_locals, 
                #         n_tourists, 
                #         locals_ratio_to_full_pop, 
                #         agents_partial, 
                #         vars_util_partial, # deepcopy(vars_util_partial) 
                #         cells_type, 
                #         indids_by_cellid,
                #         cells_households, 
                #         cells_institutions, 
                #         cells_accommodation, 
                #         contactnetworkparams, 
                #         epidemiologyparams, 
                #         dynparams, 
                #         contact_network_sum_time_taken, 
                #         process_index, 
                #         process_counter,
                #         log_file_name,
                #         static_agents_dict)

                params = (day, weekday, agents_epi_partial, vars_util_partial, dynparams, contact_network_sum_time_taken, process_index, process_counter, log_file_name, static_agents_dict)

                params_size = util.asizeof_formatted(params)
                print(f"params for {process_index} size: {params_size}")

                cat_partial_size = util.asizeof_formatted(vars_util_partial.cells_agents_timesteps)
                agents_epi_partial_size = util.asizeof_formatted(agents_epi_partial)
                vars_util_partial_size = util.asizeof_formatted(vars_util_partial)
                static_agents_dict_size = util.asizeof_formatted(static_agents_dict)
                dyn_params_size = util.asizeof_formatted(dynparams)
                print(f"cat size: {cat_partial_size}, agents_epi size: {agents_epi_partial_size}, vars_util size: {vars_util_partial_size}, dyn_params size: {dyn_params_size}, static agents dict size: {static_agents_dict_size}")
                
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
            # util.log_memory_usage(prepend_text= "Before syncing contact network results ")
            _, agents_epi, vars_util, _, _, _ = daskutil.handle_futures(MethodType.ContactNetworkMP, day, imap_results, None, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, True, False, None)
            # util.log_memory_usage(prepend_text= "After syncing contact network results ")
            
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
            if len(vars_util.directcontacts_by_simcelltype_by_day) > 0:
                current_index = len(vars_util.directcontacts_by_simcelltype_by_day)

                if day not in vars_util.directcontacts_by_simcelltype_by_day_start_marker: # sync_state_info_sets is called multiple times, but start index must only be set the first time
                    vars_util.directcontacts_by_simcelltype_by_day_start_marker[day] = current_index
            else:
                vars_util.directcontacts_by_simcelltype_by_day_start_marker[day] = 0

            params = day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_epi, vars_util, cells_type, indids_by_cellid, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, -1, process_counter, log_file_name, agents_static, True

            result = contactnetwork_worker(params, True)

            if not type(result) is dict:
                process_index, agents_epi, vars_util, _ = result
            else:
                exception_info = result

                with open(exception_info["logfilename"], "a") as fi:
                    fi.write(f"Exception Type: {exception_info['type']}\n")
                    fi.write(f"Exception Message: {exception_info['message']}\n")
                    fi.write(f"Traceback: {exception_info['traceback']}\n")
    except:
        with open(stack_trace_log_file_name, 'w') as fi2:
            traceback.print_exc(file=fi2)
        raise
    finally:
        gc.collect()

def contactnetwork_worker(params, single_proc=False):
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
        pre_mem_info = psutil.virtual_memory()
        static_agents_dict = None
        is_single_proc = False

        # sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cell_agents_timesteps, tourists_active_ids, cells_mp, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params
        if len(params) >= 20:  # sp
            if len(params) == 21:
                day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_epi, vars_util, cells_type, indids_by_cellid, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter, log_file_name, agents_static, is_single_proc = params
            else:
                day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_epi, vars_util, cells_type, indids_by_cellid, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter, log_file_name, static_agents_dict = params
        else:
            day, weekday, agents_epi, vars_util, dynparams, contact_network_sum_time_taken, process_index, process_counter, log_file_name, static_agents_dict = params

        if not is_single_proc:
            from shared_mp import n_locals
            from shared_mp import n_tourists
            from shared_mp import locals_ratio_to_full_pop
            from shared_mp import contactnetworkparams
            from shared_mp import epidemiologyparams
            from shared_mp import cells_type
            from shared_mp import indids_by_cellid
            from shared_mp import cells_households
            from shared_mp import cells_institutions
            from shared_mp import cells_accommodation
            from shared_mp import agents_static

        if static_agents_dict is not None:
            agents_static.static_agents_dict = static_agents_dict

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_mp_stack_trace_" + str(day) + "_" + str(process_index) + ".txt"
        
        if not single_proc:
            original_stdout = sys.stdout
            log_file_name = log_file_name.replace(".txt", "") + "_cn_" + str(day) + "_" + str(process_index) + ".txt"
            f = open(log_file_name, "w")
            sys.stdout = f

        print("process " + str(process_index) + " started at " + str(start))
        # util.log_memory_usage(f, "Start: ", pre_mem_info)

        contact_network_util = contactnetwork.ContactNetwork(n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            agents_static,
                                                            agents_epi,
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
        main_time_taken = time.time() - start

        print("process " + str(process_index) + " ended at " + str(time.time()))

        return process_index, agents_partial, vars_util, main_time_taken
    except:
        with open(stack_trace_log_file_name, 'w') as fi: # cn_mp_stack_trace.txt
            traceback.print_exc(file=fi)
    finally:
        gc.collect()

        # util.log_memory_usage(f, "End: ")

        if process_counter is not None:
            process_counter.value -= 1

        if f is not None:
            # Close the file
            f.close()

        if original_stdout is not None:
            sys.stdout = original_stdout   

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