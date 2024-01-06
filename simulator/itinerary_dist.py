import sys
# sys.path.insert(0, '~/AppsPy/mtdcovabm/simulator')
import os
import dask
import json
# import asyncio
# from queue import Empty
# import threading
import math
import numpy as np
# import numpy.ma as ma
import traceback
import itinerary, itinerary_mp, actor_dist_mp, vars, shared_mp, jsonutil, customdict, daskutil, util
import time
from copy import copy, deepcopy
from dask.distributed import Client, get_worker, as_completed, Variable, performance_report
import multiprocessing as mp
from memory_profiler import profile
from util import MethodType

# fp = open("memory_profiler_dist.log", "w+")
# @profile(stream=fp)

def localitinerary_distributed_finegrained(client: Client, 
                                           hh_insts, 
                                           day, 
                                           weekday, 
                                           weekdaystr, 
                                           it_agents,
                                           agents_epi, 
                                           vars_util, 
                                           dyn_params,
                                           keep_workers_open=True, 
                                           dask_mode=0, # 0 client.submit, 1 dask.delayed (client.compute), 2 dask.delayed (dask.compute), 3 client.map
                                           dask_scatter=False,
                                           dask_batch_size=-1,
                                           dask_batch_recurring=False,
                                           log_file_name="output.txt"):
    original_log_file_name = log_file_name
    stack_trace_log_file_name = ""
    task_results_stack_trace_log_file_name = ""
    try:
        start = time.time()
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_res_stack_trace" + ".txt"
        task_results_stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_res_task_results_stack_trace" + ".txt"

        delayed_computations = []
        futures = []

        agents_future, agents_epi_future, agents_seir_state_future, agents_infection_type_future, agents_infection_severity_future, dyn_params_future = None, None, None, None, None, None
        if dask_scatter:
            scatter_start_time = time.time()

            agents_future = client.scatter(it_agents, broadcast=False) # broadcast=True, direct=True

            agents_epi_future = client.scatter(agents_epi, broadcast=False)
            
            agents_seir_state_future = client.scatter(vars_util.agents_seir_state, broadcast=False) # broadcast=True, direct=True

            if len(vars_util.agents_infection_type) > 0:
                agents_infection_type_future = client.scatter(vars_util.agents_infection_type, broadcast=False) # broadcast=True, direct=True
            else:
                agents_infection_type_future = vars_util.agents_infection_type

            if len(vars_util.agents_infection_severity) > 0:
                agents_infection_severity_future = client.scatter(vars_util.agents_infection_severity, broadcast=False) # broadcast=True, direct=True
            else:
                agents_infection_severity_future = vars_util.agents_infection_severity

            dyn_params_future = client.scatter(dyn_params, broadcast=False) # broadcast=True, direct=True

            scatter_time_taken = time.time() - scatter_start_time
            print("scatter/upload time_taken: " + str(scatter_time_taken))

        map_params = []
        count = 0
        for hh_inst in hh_insts:
            agents_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict()
            vars_util_partial = vars.Vars()

            if not dask_scatter: # and not dask_persist
                # vars_util_partial.agents_seir_state = [] # to be populated hereunder
                vars_util_partial.cells_agents_timesteps = customdict.CustomDict()
                
                agents_partial, _, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, None, None, True, True, agents_epi, agents_epi_partial)               
                
                params = day, weekday, weekdaystr, hh_inst, agents_partial, agents_epi_partial, vars_util_partial, dyn_params, dask_scatter, True, log_file_name
            else:
                params = day, weekday, weekdaystr, hh_inst, agents_future, agents_epi_future, agents_seir_state_future, agents_infection_type_future, agents_infection_severity_future, dyn_params_future, True, True, log_file_name

            if dask_mode == 0:
                if dask_batch_size == -1 or count < dask_batch_size:
                    future = client.submit(localitinerary_worker_res, params)
                    futures.append(future)
                else:
                    map_params.append(params)
            elif dask_mode == 1 or dask_mode == 2:
                delayed = dask.delayed(localitinerary_worker_res)(params)
                delayed_computations.append(delayed)
            elif dask_mode == 3:
                map_params.append(params)

            count += 1

        tasks_still_to_assign = len(hh_insts) - len(futures)
        safe_recurion_call_count = 964
        num_batches_required = math.ceil(tasks_still_to_assign / dask_batch_size) 
        num_recursive_calls_required = math.ceil(num_batches_required / safe_recurion_call_count)

        results = []
        if dask_mode == 1:
            futures = client.compute(delayed_computations)
        elif dask_mode == 2:
            results = dask.compute(*delayed_computations)
        elif dask_mode == 3:
            futures = client.map(localitinerary_worker_res, map_params)
        
        if (dask_mode == 0 and dask_batch_size == -1) or dask_mode == 1 or dask_mode == 3:
            it_agents, agents_epi, vars_util, _, _ = daskutil.handle_futures(MethodType.ItineraryDist, day, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name)
        elif dask_mode == 0 and not dask_batch_recurring:
            it_agents, agents_epi, vars_util = handle_futures_non_recurring_batches(MethodType.ItineraryDist, day, client, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, map_params)
        elif dask_mode == 0 and dask_batch_recurring:
            for i in range(num_recursive_calls_required):
                it_agents, agents_epi, vars_util = handle_futures_recurring_batches(MethodType.ItineraryDist, day, client, dask_batch_size, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, map_params, safe_recurion_call_count, 0)
        else:
            for result in results:
                agents_partial_result, agents_epi_partial_result, vars_util_partial_result = result
                
                # was sync_results_res
                it_agents, agents_epi, vars_util = daskutil.sync_results_it(day, it_agents, agents_epi, vars_util, agents_partial_result, agents_epi_partial_result, vars_util_partial_result)
        
        if not keep_workers_open:
            start = time.time()
            client.shutdown()
            time_taken = time.time() - start
            print("client shutdown time taken " + str(time_taken))

        time_taken = time.time() - start
        print("localitinerary_finegrained_distributed time_taken: " + str(time_taken))
    except:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_distributed_finegrained_chunks(client: Client, 
                                                hh_insts, 
                                                day, 
                                                weekday, 
                                                weekdaystr, 
                                                it_agents,
                                                agents_epi, 
                                                vars_util, 
                                                dyn_params,
                                                keep_workers_open=True, 
                                                dask_mode=0, # 0 client.submit, 1 dask.delayed (client.compute), 2 dask.delayed (dask.compute), 3 client.map
                                                dask_chunk_size=1,
                                                dask_single_item_per_task=True,
                                                dask_full_array_mapping=False,
                                                log_file_name="output.txt"): 
    original_log_file_name = log_file_name
    stack_trace_log_file_name = ""
    task_results_stack_trace_log_file_name = ""
    try:
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_dist_stack_trace" + ".txt"
        task_results_stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_res_task_results_stack_trace" + ".txt"

        if dask_single_item_per_task:
            for hh_insts_partial in util.yield_chunks(hh_insts, dask_chunk_size):
                dask_params = []
                delayed_computations = []
                futures = []

                for hh_inst in hh_insts_partial:
                    agents_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict()
                    vars_util_partial = vars.Vars()

                    # if not dask_full_array_mapping:
                    #     vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                    # else:
                    #     vars_util_partial.agents_seir_state = [] # to be populated hereunder
                    
                    vars_util_partial.cells_agents_timesteps = customdict.CustomDict()

                    agents_partial, _, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, None, None, is_itinerary=True, is_dask_task=dask_full_array_mapping, agents_epi=agents_epi, agents_epi_partial=agents_epi_partial)                  
                    
                    params = day, weekday, weekdaystr, hh_inst, agents_partial, agents_epi_partial, vars_util_partial, dyn_params, False, dask_full_array_mapping, log_file_name  

                    # start = time.time()
                    if dask_mode == 0:
                        future = client.submit(localitinerary_worker_res, params)
                        futures.append(future)
                    elif dask_mode == 1 or dask_mode == 2:
                        delayed_computation = dask.delayed(localitinerary_worker_res)(params)
                        delayed_computations.append(delayed_computation)
                        # delayed_computations = [dask.delayed(itinerary_mp.localitinerary_worker)(dask_params[i]) for i in range(len(dask_params))]
                    elif dask_mode == 3:
                        dask_params.append(params)
                
                    # time_taken = time.time() - start
                    # print("delayed_results generation: " + str(time_taken))

                start = time.time()

                if dask_mode == 1:
                    futures = client.compute(delayed_computations) # client.compute delays computation (and doesn't seem to work with the large dataset)
                elif dask_mode == 2:
                    futures = dask.compute(delayed_computations)
                elif dask_mode == 3:
                    futures = client.map(localitinerary_worker_res, dask_params)

                time_taken = time.time() - start
                print("computed_results generation: " + str(time_taken))

                start = time.time()
                it_agents, agents_epi, vars_util, _, _ = daskutil.handle_futures(MethodType.ItineraryDist, day, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, False, dask_full_array_mapping)

                time_taken = time.time() - start
                print("sync results: " + str(time_taken))
        else:
            dask_params = []
            delayed_computations = []
            futures = []
            
            chunk_start = time.time()
            for hh_insts_partial in util.yield_chunks(hh_insts, dask_chunk_size):
                agents_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict()
                vars_util_partial = vars.Vars()

                # if not dask_full_array_mapping:
                #     vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                # else:
                #     vars_util_partial.agents_seir_state = [] # to be populated hereunder

                vars_util_partial.cells_agents_timesteps = customdict.CustomDict()
                
                for hh_inst in hh_insts_partial:
                    agents_partial, _, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, None, None, is_itinerary=True, is_dask_task=dask_full_array_mapping, agents_epi=agents_epi, agents_epi_partial=agents_epi_partial)                  

                params = day, weekday, weekdaystr, hh_insts_partial, agents_partial, agents_epi_partial, vars_util_partial, dyn_params, False, dask_full_array_mapping, log_file_name
            
                # start = time.time()
                if dask_mode == 0:
                    future = client.submit(localitinerary_worker_res, params)
                    futures.append(future)
                elif dask_mode == 1 or dask_mode == 2:
                    delayed_computation = dask.delayed(localitinerary_worker_res)(params)
                    delayed_computations.append(delayed_computation)
                elif dask_mode == 3:
                    dask_params.append(params)
            
                # time_taken = time.time() - start
                # print("delayed_results generation: " + str(time_taken))
            
            chunk_time_taken = time.time() - chunk_start
            print("chunk stage: " + str(chunk_time_taken))

            start = time.time()

            if dask_mode == 1:
                futures = client.compute(delayed_computations) # client.compute delays computation (and doesn't seem to work with the large dataset)
            elif dask_mode == 2:
                futures = dask.compute(delayed_computations)
            elif dask_mode == 3:
                futures = client.map(localitinerary_worker_res, dask_params)

            time_taken = time.time() - start
            print("computed_results: " + str(time_taken))

            start = time.time()
            it_agents, agents_epi, vars_util, _, _ = daskutil.handle_futures(MethodType.ItineraryDist, day, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, False, dask_full_array_mapping)
            time_taken = time.time() - start
            print("sync results: " + str(time_taken))

        if not keep_workers_open:
            start = time.time()
            client.shutdown()
            time_taken = time.time() - start
            print("client shutdown time taken " + str(time_taken))         
    except:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_distributed_map_batched(client: Client, 
                                    hh_insts, 
                                    day, 
                                    weekday, 
                                    weekdaystr, 
                                    it_agents,
                                    agents_epi, 
                                    vars_util, 
                                    dyn_params,
                                    keep_workers_open=True,
                                    dask_batch_size=1,
                                    dask_full_array_mapping=False,
                                    dask_scatter=False,
                                    dask_submit=False,
                                    dask_map_batched_results=True,
                                    log_file_name="output.txt"): 
    original_log_file_name = log_file_name
    stack_trace_log_file_name = ""
    task_results_stack_trace_log_file_name = ""
    try:
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_dist_stack_trace" + ".txt"
        task_results_stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_res_task_results_stack_trace" + ".txt"

        dask_params = []

        agents_future, agents_epi_future, agents_seir_state_future, agents_infection_type_future, agents_infection_severity_future, dyn_params_future = None, None, None, None, None, None
        if dask_scatter or dask_submit:
            scatter_start_time = time.time()

            if dask_scatter:
                agents_future = client.scatter(it_agents) # broadcast=True, direct=True
            else:
                agents_future = client.submit(load_data_future, it_agents)
           
            if dask_scatter:
                agents_epi_future = client.scatter(agents_epi) # broadcast=True, direct=True
            else:
                agents_epi_future = client.submit(load_data_future, agents_epi)
            
            if dask_scatter:
                agents_seir_state_future = client.scatter(vars_util.agents_seir_state) # broadcast=True, direct=True
            else:
                agents_seir_state_future = client.submit(load_data_future, vars_util.agents_seir_state)

            if len(vars_util.agents_infection_type) > 0:
                if dask_scatter:
                    agents_infection_type_future = client.scatter(vars_util.agents_infection_type) # broadcast=True, direct=True
                else:
                    agents_infection_type_future = client.submit(load_data_future, vars_util.agents_infection_type)

            else:
                agents_infection_type_future = vars_util.agents_infection_type

            if len(vars_util.agents_infection_severity) > 0:
                if dask_scatter:
                    agents_infection_severity_future = client.scatter(vars_util.agents_infection_severity) # broadcast=True, direct=True
                else:
                    agents_infection_severity_future = client.submit(load_data_future, vars_util.agents_infection_severity)
            else:
                agents_infection_severity_future = vars_util.agents_infection_severity

            if dask_scatter:
                dyn_params_future = client.scatter(dyn_params) # broadcast=True, direct=True
            else:
                dyn_params_future = client.submit(load_data_future, dyn_params)

            scatter_time_taken = time.time() - scatter_start_time
            print("scatter/upload time_taken: " + str(scatter_time_taken))

        start = time.time()
        for hh_inst in hh_insts:
            if not dask_scatter and not dask_submit:
                agents_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict()
                vars_util_partial = vars.Vars()

                # if not dask_full_array_mapping:
                #     vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                # else:
                #     vars_util_partial.agents_seir_state = [] # to be populated hereunder
                
                vars_util_partial.cells_agents_timesteps = customdict.CustomDict()

                agents_partial, _, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, None, None, is_itinerary=True, is_dask_task=dask_full_array_mapping, agents_epi=agents_epi, agents_epi_partial=agents_epi_partial)                  
                
                params = day, weekday, weekdaystr, hh_inst, agents_partial, agents_epi_partial, vars_util_partial, dyn_params, False, dask_full_array_mapping, log_file_name  
            else:
                params = day, weekday, weekdaystr, hh_inst, agents_future, agents_epi_future, agents_seir_state_future, agents_infection_type_future, agents_infection_severity_future, dyn_params_future, True, dask_full_array_mapping, log_file_name

            dask_params.append(params)
            
            # time_taken = time.time() - start
            # print("delayed_results generation: " + str(time_taken))
        time_taken = time.time()
        print("generating dask_params, time_taken: " + str(time_taken))

        start = time.time()

        futures = client.map(localitinerary_worker_res, dask_params, batch_size=dask_batch_size)

        time_taken = time.time() - start
        print("mapping with batch size: " + str(dask_batch_size) + ", time_taken: " + str(time_taken))

        start = time.time()
        # time.sleep(600)
        if dask_map_batched_results:
            it_agents, agents_epi, vars_util = daskutil.handle_futures_batches(day, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, False, dask_full_array_mapping)
        else:
            it_agents, agents_epi, vars_util, _, _ = daskutil.handle_futures(MethodType.ItineraryDist, day, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, False, dask_full_array_mapping)

        time_taken = time.time() - start
        print("sync results: " + str(time_taken))

        if not keep_workers_open:
            start = time.time()
            client.shutdown()
            time_taken = time.time() - start
            print("client shutdown time taken " + str(time_taken))         
    except:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_distributed(client: Client,
                            day,
                            weekday,
                            weekdaystr,
                            it_agents,
                            agents_epi,
                            vars_util, 
                            dynparams,
                            hh_insts,
                            keep_processes_open=True,
                            use_mp=False, # in this context use_mp means use single dask worker and mp in each node
                            dask_numtasks=-1,
                            dask_mode=0, # 0 client.submit, 1 dask.delayed (client.compute), 2 dask.delayed (dask.compute), 3 client.map
                            dask_full_array_mapping=False,
                            dask_nodes_n_workers=None,
                            dask_combined_scores_nworkers=None, # frequency distribution based on CPU scores
                            dask_workers_time_taken=None,
                            dask_mp_processes_time_taken=None,
                            dask_dynamic_load_balancing=None,
                            use_mp_innerproc_assignment=False,
                            f=None,
                            actors=None,
                            log_file_name="output.txt"): 
    stack_trace_log_file_name = ""
    task_results_stack_trace_log_file_name = ""
    try:
        folder_name = ""
        if log_file_name != "output.txt":
            folder_name = os.path.dirname(log_file_name)
        else:
            folder_name = os.getcwd()
        
        dask_perf_log_file_name = os.path.join(folder_name, "dask_it_perf_log_" + str(day) + ".html")
        stack_trace_log_file_name = os.path.join(folder_name, "it_main_dist_stack_trace_" + str(day) + ".txt")
        task_results_stack_trace_log_file_name = os.path.join(folder_name, "it_main_res_task_results_stack_trace_" + str(day) + ".txt")

        # dask_perf_log_file_name = log_file_name.replace(".txt", "") + "_daskperflog_" + str(day) + ".html"
        # stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_dist_stack_trace" + ".txt"
        # task_results_stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_res_task_results_stack_trace" + ".txt"

        # process_counter = manager.Value("i", num_processes)

        with performance_report(filename=dask_perf_log_file_name):
            workers = list(client.scheduler_info()["workers"].keys()) # list()
            # worker = client.scheduler_info()["workers"][workers[0]]
            # print("local_directory: " + str(worker["local_directory"]))

            node_worker_index_by_worker_url = {}
        
            w_index = 0
            for n_index, num_workers in enumerate(dask_nodes_n_workers):
                if use_mp:
                    node_worker_index_by_worker_url[workers[n_index]] = (n_index, n_index)
                else:
                    for _ in range(num_workers):
                        node_worker_index_by_worker_url[workers[w_index]] = (n_index, w_index)
                        w_index += 1
            
            if dask_numtasks == -1:
                if not use_mp or not use_mp_innerproc_assignment:
                    dask_numtasks = len(workers)
                else:
                    dask_numtasks = sum(dask_nodes_n_workers)

            start = time.time()

            if not use_mp and dask_combined_scores_nworkers is not None: # load balancing
                mp_hh_inst_indices = util.itinerary_load_balancing(hh_insts, dask_numtasks, dask_nodes_n_workers, dask_combined_scores_nworkers)
            elif not use_mp and dask_dynamic_load_balancing and dask_workers_time_taken is not None:
                inverted_dask_nodes_time_taken = {i: 1/v for i,v in dask_workers_time_taken.items()} # invert to assign less work to slower nodes, and more work to faster nodes
                inverted_dask_nodes_time_taken = dict(sorted(inverted_dask_nodes_time_taken.items(), key=lambda item: item[1])) # sort by slowest first
                workers = [workers[wid[1]] for wid in inverted_dask_nodes_time_taken.keys()] # re-arrange workers to slowest first based on inverted_dask_nodes_time_taken
                inverted_nodes_time_taken = np.array([tt for tt in inverted_dask_nodes_time_taken.values()]) # pass a numpy array of inverted time_takens to util.itinerary_load_balancing
                inverted_nodes_time_taken_sum = sum(inverted_nodes_time_taken) # calculate sum
                inverted_nodes_time_taken = np.array([tt/inverted_nodes_time_taken_sum for tt in inverted_nodes_time_taken]) # normalize to sum to 1
                mp_hh_inst_indices = util.itinerary_load_balancing(hh_insts, dask_numtasks, inverted_nodes_time_taken)
            else:
                mp_hh_inst_indices = util.split_residences_by_weight(hh_insts, dask_numtasks) # to do - for the time being this assumes equal split but we may have more information about the cores of the workers
            
            # just for dask with mp strategy, re-join the workers into a per-node split. will be re-split into separate processes in each dask worker node
            if use_mp and use_mp_innerproc_assignment:
                temp_mp_hh_inst_indices = [[] for _ in range(len(workers))]

                cursor = 0        
                for ni, num_workers in enumerate(dask_nodes_n_workers):
                    for _ in range(num_workers):
                        temp_mp_hh_inst_indices[ni].extend(mp_hh_inst_indices[cursor])
                        cursor += 1

                mp_hh_inst_indices = temp_mp_hh_inst_indices
                dask_numtasks = len(workers)

            time_taken = time.time() - start
            print("split residences by indices (load balancing): " + str(time_taken))
            if f is not None:
                f.flush()
            
            start = time.time()

            dask_params = []
            delayed_computations = []
            futures = []

            # mem troubleshooting
            hh_insts_size = util.asizeof_formatted(hh_insts)
            it_agents_size = util.asizeof_formatted(it_agents)
            agents_epi_size = util.asizeof_formatted(agents_epi)
            vars_util_size = util.asizeof_formatted(vars_util)
            print(f"hh_insts size: {hh_insts_size}, it_agents size: {it_agents_size}, agents_epi size: {agents_epi_size}, vars_util size: {vars_util_size}")

            for worker_index in range(dask_numtasks):
                worker_assign_start = time.time()

                # cells_partial = {}
                hh_insts_partial = []

                worker_url = workers[worker_index]
                remote_worker_index = node_worker_index_by_worker_url[worker_url]

                mp_hh_inst_ids_this_proc = mp_hh_inst_indices[worker_index] # worker_index

                if len(mp_hh_inst_ids_this_proc) > 0:
                    # start_partial = time.time()
                    hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc]      

                    agents_partial, agents_ids_by_ages_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict(), customdict.CustomDict()
                    vars_util_partial = vars.Vars()

                    # if not dask_full_array_mapping:
                    #     vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                    #     # vars_util_partial.agents_seir_state = copy(vars_util.agents_seir_state)
                    # else:
                    #     vars_util_partial.agents_seir_state = [] # to be populated hereunder
                    # vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                    vars_util_partial.cells_agents_timesteps = customdict.CustomDict()
                    
                    for hh_inst in hh_insts_partial:
                        agents_partial, agents_ids_by_ages_partial, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, None, None, True, dask_full_array_mapping, agents_epi, agents_epi_partial)

                    # if not dask_full_array_mapping:
                    #     agent_ids = sorted(list(agents_partial.keys()))

                    #     mask = np.isin(np.arange(len(vars_util_partial.agents_seir_state)), agent_ids, invert=True)

                    #     vars_util_partial.agents_seir_state = ma.masked_array(vars_util_partial.agents_seir_state, mask=mask)
                    
                    # Define parameters
                    if not use_mp:
                        params = (day, 
                                weekday, 
                                weekdaystr, 
                                hh_insts_partial, 
                                agents_partial,
                                agents_epi_partial,
                                vars_util_partial, # deepcopy(vars_util_partial) - without this was causing some nasty issues with multiprocessing (maybe with Dask it is fine)
                                dynparams, 
                                remote_worker_index,
                                log_file_name)  
                    else:
                        params = (day, 
                                weekday, 
                                weekdaystr, 
                                hh_insts_partial, 
                                agents_partial,
                                agents_epi_partial,
                                vars_util_partial, # deepcopy(vars_util_partial) - without this was causing some nasty issues with multiprocessing (maybe with Dask it is fine)
                                dynparams, 
                                log_file_name)  

                    params_size = util.asizeof_formatted(params)
                    print(f"params for {worker_index} size: {params_size}")

                    hh_insts__partial_size = util.asizeof_formatted(hh_insts_partial)
                    it_agents_partial_size = util.asizeof_formatted(agents_partial)
                    agents_epi_partial_size = util.asizeof_formatted(agents_epi_partial)
                    agents_ids_by_ages_partial_size = util.asizeof_formatted(agents_ids_by_ages_partial)
                    vars_util_partial_size = util.asizeof_formatted(vars_util_partial)
                    dyn_params_size = util.asizeof_formatted(dynparams)
                    print(f"hh_insts size: {hh_insts__partial_size}, it_agents size: {it_agents_partial_size}, agents_epi size: {agents_epi_partial_size}, agents_ids_by_ages size: {agents_ids_by_ages_partial_size}, vars_util size: {vars_util_partial_size}, dyn_params size: {dyn_params_size}")
                    # print("starting process index with " + str(len(hh_insts_partial)) + " residences on " + str(remote_worker_index) + " (" + worker_url + ") at " + str(time.time()))
                    # if f is not None:
                    #     f.flush()

                    if not use_mp:
                        if dask_mode == 0:
                            future = client.submit(itinerary_mp.localitinerary_worker, params, workers=worker_url)
                            futures.append(future)
                        elif dask_mode == 1 or dask_mode == 2:
                            delayed_computation = dask.delayed(itinerary_mp.localitinerary_worker)(params)
                            delayed_computations.append(delayed_computation)
                            # delayed_computations = [dask.delayed(itinerary_mp.localitinerary_worker)(dask_params[i]) for i in range(len(dask_params))]
                        elif dask_mode == 3:
                            dask_params.append(params)
                    else:
                        actor = actors[worker_index]
                        future = actor.run_itinerary_parallel(params)
                        futures.append(future)
                        # old impl
                        # delayed_computation = dask.delayed(localitinerary_dist_worker)(params)
                        # delayed_computations.append(delayed_computation)

                worker_assign_time_taken = time.time() - worker_assign_start
                
                if not use_mp:
                    dask_workers_time_taken[remote_worker_index] = [worker_assign_time_taken, None]
                else:
                    dask_workers_time_taken[remote_worker_index[0]] = [worker_assign_time_taken, None]

            time_taken = time.time() - start
            print("futures / delayed results generation: " + str(time_taken))
            if f is not None:
                f.flush()

            if not use_mp:
                start = time.time()
                if dask_mode == 1: # use_mp or dask_mode == 1
                    futures = client.compute(delayed_computations) # client.compute delays computation (and doesn't seem to work with the large dataset)
                elif dask_mode == 2:
                    futures = dask.compute(delayed_computations)
                elif dask_mode == 3:
                    futures = client.map(itinerary_mp.localitinerary_worker, dask_params)

                time_taken = time.time() - start

                print("futures generation: " + str(time_taken))
                if f is not None:
                    f.flush()

            start = time.time()

            if not use_mp:
                method_type = MethodType.ItineraryDist
            else:
                method_type = MethodType.ItineraryDistMP

            it_agents, agents_epi, vars_util, dask_workers_time_taken, dask_mp_processes_time_taken = daskutil.handle_futures(method_type, day, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, True, dask_full_array_mapping, f, dask_workers_time_taken, dask_mp_processes_time_taken)

            time_taken = time.time() - start
            print("sync results: " + str(time_taken))

            if f is not None:
                f.flush()

            if not keep_processes_open:
                start = time.time()
                client.shutdown()
                time_taken = time.time() - start
                print("client shutdown time taken " + str(time_taken))  
                if f is not None:
                    f.flush()
    except:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)
        raise

def handle_futures_recurring_batches(day, client, batch_size, agents_dynamic, agents_epi, vars_util, task_results_stack_trace_log_file_name, map_params, max_recursion_calls, recursion_count):
    if batch_size > len(map_params):
        batch_size = len(map_params)

    current_params = map_params[:batch_size]
    del map_params[:batch_size]

    futures = []

    for params in current_params:
        future = client.submit(localitinerary_worker_res, params)
        futures.append(future)

    for future in as_completed(futures):
        try:
            result = future.result()

            agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result = result

            # was sync_results_res
            agents_dynamic, vars_util = daskutil.sync_results_it(day, agents_dynamic, agents_epi, vars_util, agents_dynamic_partial_result, vars_util_partial_result)
        
            agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result = None, None, None
        except:
            with open(task_results_stack_trace_log_file_name, 'a') as f:
                traceback.print_exc(file=f)
        finally:
            future.release()

    if len(map_params) == 0 or recursion_count == max_recursion_calls:
        return agents_dynamic, agents_epi, vars_util
    else:
        recursion_count += 1
        return handle_futures_recurring_batches(day, client, batch_size, agents_dynamic, vars_util, task_results_stack_trace_log_file_name, map_params, max_recursion_calls, recursion_count)

def handle_futures_non_recurring_batches(day, client, futures, agents_dynamic, agents_epi, vars_util, task_results_stack_trace_log_file_name, map_params):
    temp_futures = []

    if len(futures) == 0 and len(map_params) > 0: # take care of all remaining with this batch
        for params in map_params:
            future = client.submit(localitinerary_worker_res, params)
            futures.append(future)

        map_params = []

    for future in as_completed(futures):
        try:
            result = future.result()

            agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result = result

            # was sync_results_res
            agents_dynamic, vars_util = daskutil.sync_results_it(day, agents_dynamic, agents_epi, vars_util, agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result)
        
            agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result = None, None
        except:
            with open(task_results_stack_trace_log_file_name, 'a') as f:
                traceback.print_exc(file=f)
        finally:
            future.release()

            if len(map_params) > 0:
                params = map_params.pop(0)
                future = client.submit(localitinerary_worker_res, params)
                temp_futures.append(future)

    if len(temp_futures) == 0 and len(map_params) == 0:
        return agents_dynamic, agents_epi, vars_util
    else:
        return handle_futures_non_recurring_batches(day, client, temp_futures, agents_dynamic, vars_util, task_results_stack_trace_log_file_name, map_params)

def load_json_future(jsonstr):
    return json.loads(jsonstr, object_hook=jsonutil.jsonKeys2int)

def load_data_future(data):
    return data

def localitinerary_worker_res(params):
    import os
    import sys
    import shutil

    original_stdout = None
    f = None
    stack_trace_log_file_name = ""

    try:
        agents_dynamic, agents_epi, vars_util_mp = None, None
        if len(params) == 11:
            day, weekday, weekdaystr, hh_inst, agents_dynamic, agents_epi, vars_util_mp, dyn_params, dask_scatter, dask_full_array_mapping, log_file_name = params
        else:
            day, weekday, weekdaystr, hh_inst, agents_dynamic_future, agents_epi_future, agents_seir_state, agents_infection_type, agents_infection_severity, dyn_params, dask_scatter, dask_full_array_mapping, log_file_name = params
            vars_util_mp_future = vars.Vars(agents_seir_state=agents_seir_state, agents_infection_type=agents_infection_type, agents_infection_severity=agents_infection_severity)

        hh_insts = []
        if isinstance(hh_inst, list):
            hh_insts = hh_inst
        else:
            hh_insts.append(hh_inst)
        
        hh_inst = None

        if dask_scatter:
            agents_dynamic_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict()
            vars_util_partial = vars.Vars()

            for hh_inst in hh_insts:
                agents_dynamic_partial, agents_epi_partial, _, vars_util_partial = util.split_dicts_by_agentsids_copy(hh_inst["resident_uids"], agents_dynamic_future, agents_epi_future, vars_util_mp_future, agents_dynamic_partial, agents_epi_partial, vars_util_partial, None, None, is_itinerary=True, is_dask_full_array_mapping=dask_full_array_mapping)

            agents_dynamic = agents_dynamic_partial
            agents_epi = agents_epi_partial
            vars_util_mp = vars_util_partial

        worker = get_worker()
        
        # original_stdout = sys.stdout
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_res_stack_trace_" + worker.id + ".txt"
        # log_file_name = log_file_name.replace(".txt", "") + "_it_res_" + worker.id + "_" + str(hh_inst["id"]) + ".txt" 
        # f = open(log_file_name, "w")
        # sys.stdout = f

        # subfolder_path = os.path.dirname(log_file_name)

        # if not os.path.exists(subfolder_path):
        #     os.makedirs(subfolder_path)
        # else:
        #     shutil.rmtree(subfolder_path)
        #     os.makedirs(subfolder_path)
        
        # agents_static = worker.client.futures["agents_static"]
        # agents_static = worker.plugins["read_only_data"]

        agents_ids_by_ages = worker.data["agents_ids_by_ages"]
        timestepmins = worker.data["timestepmins"]
        n_locals = worker.data["n_locals"]
        n_tourists = worker.data["n_tourists"]
        locals_ratio_to_full_pop = worker.data["locals_ratio_to_full_pop"]

        itineraryparams = worker.data["itineraryparams"]
        epidemiologyparams = worker.data["epidemiologyparams"]
        cells_industries_by_indid_by_wpid = worker.data["cells_industries_by_indid_by_wpid"] 
        cells_restaurants = worker.data["cells_restaurants"] 
        cells_hospital = worker.data["cells_hospital"] 
        cells_testinghub = worker.data["cells_testinghub"] 
        cells_vaccinationhub = worker.data["cells_vaccinationhub"] 
        cells_entertainment_by_activityid = worker.data["cells_entertainment_by_activityid"] 
        cells_religious = worker.data["cells_religious"] 
        cells_households = worker.data["cells_households"] 
        cells_breakfast_by_accomid = worker.data["cells_breakfast_by_accomid"] 
        cells_airport = worker.data["cells_airport"] 
        cells_transport = worker.data["cells_transport"] 
        cells_institutions = worker.data["cells_institutions"] 
        cells_accommodation = worker.data["cells_accommodation"] 
        agents_static = worker.data["agents_static"]

        itinerary_util = itinerary.Itinerary(itineraryparams,
                                            timestepmins, 
                                            n_locals, 
                                            n_tourists, 
                                            locals_ratio_to_full_pop, 
                                            agents_static,
                                            agents_dynamic, 
                                            agents_epi,
                                            agents_ids_by_ages,
                                            vars_util_mp,
                                            cells_industries_by_indid_by_wpid, 
                                            cells_restaurants,
                                            cells_hospital,
                                            cells_testinghub, 
                                            cells_vaccinationhub, 
                                            cells_entertainment_by_activityid,
                                            cells_religious, 
                                            cells_households,
                                            cells_breakfast_by_accomid,
                                            cells_airport, 
                                            cells_transport, 
                                            cells_institutions, 
                                            cells_accommodation,
                                            epidemiologyparams, 
                                            dyn_params)
        
        if day == 1 or weekdaystr == "Monday":
            for hh_inst in hh_insts: # may be single
                # start = time.time()
                itinerary_util.generate_working_days_for_week_residence(hh_inst["resident_uids"], hh_inst["is_hh"])
                # time_taken = time.time() - start
                # print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(process_index))

        for hh_inst in hh_insts: # may be single
            # start = time.time()
            itinerary_util.generate_local_itinerary(day, weekday, hh_inst["resident_uids"])
            # time_taken = time.time() - start
            # print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken))
        
        # return agents_dynamic, vars_util_mp.cells_agents_timesteps, vars_util_mp.agents_seir_state, vars_util_mp.agents_infection_type, vars_util_mp.agents_infection_severity, vars_util_mp.agents_seir_state_transition_for_day, vars_util_mp.contact_tracing_agent_ids
        return agents_dynamic, agents_epi, vars_util_mp
    except:
        with open(stack_trace_log_file_name, 'a+') as f: # it_res_stack_trace.txt
            traceback.print_exc(file=f)
    finally:
        if original_stdout is not None:
            sys.stdout = original_stdout

        if f is not None:
            # Close the file
            f.close()

def localitinerary_dist_worker(params):
    # from shared_mp import agents_static
    # import sys
    # sys.path.insert(0, '~/AppsPy/mtdcovabm/simulator')

    import os

    worker_index = -1
    original_stdout = None
    f = None
    stack_trace_log_file_name = ""

    try:  
        # sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_itinerary, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params
        day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_partial, agents_ids_by_ages, vars_util_partial, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, epidemiologyparams, dyn_params, proc_use_pool, num_processes, worker_index, log_file_name = params
        
        original_stdout = sys.stdout
        original_log_file_name = log_file_name

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_dist_stack_trace_" + str(worker_index) + ".txt"
        log_file_name = log_file_name.replace(".txt", "") + "_it_dist_" + str(worker_index) + ".txt"
        f = open(log_file_name, "w")
        sys.stdout = f

        # implementation for static agents via dask plugins
        print("interpreter: " + os.path.dirname(sys.executable))
        print("current working directory: " + os.getcwd())

        worker = get_worker()
        # agents_static = worker.client.futures["agents_static"]
        # agents_static = worker.plugins["read_only_data"]

        print("worker memory limit: " + str(worker.memory_limit))
        
        agents_static = worker.data["agents_static"]
        # agents_static = agents_static_future.result()

        # print("agents_static: " + str(len(agents_static)))
        # plugin = worker.plugins['read-only-data']
        # persons = plugin.persons

        # persons_len = ""
        # if persons is not None:
        #     persons_len = len(persons)
        # else:
        #     persons_len = "None"

        # print("persons len: " + str(persons_len))
        # print("actual persons len: " + str(plugin.persons_len))

        # manager, pool = worker.plugins["initialize_mp"]

        manager, pool = None, None

        if proc_use_pool == 0 or proc_use_pool == 3:
            print("generating manager and pool")
            manager = mp.Manager()
            pool = mp.Pool(initializer=shared_mp.init_pool_processes, initargs=(agents_static,))

        # pool = mp.Pool(initializer=shared_mp.init_pool_processes, initargs=(agents_static,))

        print("cells_agents_timesteps in process " + str(worker_index) + " is " + str(id(vars_util_partial.cells_agents_timesteps)))
        print(f"Itinerary Worker Child #{worker_index+1} at {str(time.time())}", flush=True)

        agents_partial, vars_util_partial = itinerary_mp.localitinerary_parallel(manager,
                                                                            pool,
                                                                            day,
                                                                            weekday,
                                                                            weekdaystr,
                                                                            itineraryparams,
                                                                            timestepmins, 
                                                                            n_locals, 
                                                                            n_tourists, 
                                                                            locals_ratio_to_full_pop, 
                                                                            agents_partial, 
                                                                            agents_ids_by_ages,
                                                                            vars_util_partial,
                                                                            cells_industries_by_indid_by_wpid, 
                                                                            cells_restaurants,
                                                                            cells_hospital,
                                                                            cells_testinghub, 
                                                                            cells_vaccinationhub, 
                                                                            cells_entertainment_by_activityid,
                                                                            cells_religious, 
                                                                            cells_households,
                                                                            cells_breakfast_by_accomid,
                                                                            cells_airport, 
                                                                            cells_transport, 
                                                                            cells_institutions, 
                                                                            cells_accommodation,
                                                                            epidemiologyparams, 
                                                                            dyn_params,
                                                                            hh_insts,
                                                                            num_processes, # to do - this is to be passed dynamically based on worker/s config/s,                                                                           
                                                                            proc_use_pool=proc_use_pool,
                                                                            use_shm=manager is not None,
                                                                            log_file_name=original_log_file_name,
                                                                            dask=True)

        # to do - somehow to return and combine results
        for i in range(1000):
            if i in agents_partial:
                if agents_partial[i]["itinerary"] is None or len(agents_partial[i]["itinerary"]) == 0:
                    print("itinerary_dist, itinerary is empty " + str(i))

        return worker_index, agents_partial, vars_util_partial
    except:
        with open(stack_trace_log_file_name, 'w') as f: # it_dist_stack_trace.txt
            traceback.print_exc(file=f)
    finally:
        # process_counter.value -= 1

        if original_stdout is not None:
            sys.stdout = original_stdout

            if f is not None:
                # Close the file
                f.close()
