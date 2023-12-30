import sys
import os
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import threading
import numpy as np
import numpy.ma as ma
import traceback
import contactnetwork, util, vars, customdict, daskutil
import time
import dask
from dask.distributed import Client, get_worker, performance_report
from copy import copy
from util import MethodType
from agents_epi import AgentsEpi
import gc

def contactnetwork_distributed(client: Client,
                            day, 
                            weekday, 
                            agents_epi_util,
                            vars_util,
                            dynparams,              
                            dask_mode=0,
                            dask_numtasks=-1,
                            dask_full_array_mapping=False,
                            keep_processes_open=True,
                            use_mp=False,
                            dask_nodes_n_workers=None,
                            dask_workers_time_taken=None,
                            dask_mp_processes_time_taken=None,
                            use_mp_innerproc_assignment=False,
                            f=None,
                            actors=None,
                            log_file_name="output.txt"):
    # process_counter = manager.Value("i", num_processes)
    original_log_file_name = log_file_name
    stack_trace_log_file_name = ""
    task_results_stack_trace_log_file_name = ""

    try:
        folder_name = ""
        if log_file_name != "output.txt":
            folder_name = os.path.dirname(log_file_name)
        else:
            folder_name = os.getcwd()

        dask_perf_log_file_name = os.path.join(folder_name, "dask_cn_perf_log_" + str(day) + ".html")
        stack_trace_log_file_name = os.path.join(folder_name, "cn_main_dist_stack_trace_" + str(day) + ".txt")
        task_results_stack_trace_log_file_name = os.path.join(folder_name, "cn_main_res_task_results_stack_trace_" + str(day) + ".txt")

        # stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_main_dist_stack_trace" + ".txt"
        # task_results_stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_main_res_task_results_stack_trace" + ".txt"

        with performance_report(filename=dask_perf_log_file_name):
            workers = list(client.scheduler_info()["workers"].keys()) 

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
            cells_agents_timesteps_dicts = util.split_cellsagentstimesteps_balanced(vars_util.cells_agents_timesteps, dask_numtasks)
            
            # just for dask with mp strategy, re-join the workers into a per-node split. will be re-split into separate processes in each dask worker node 
            if use_mp and use_mp_innerproc_assignment:
                temp_cells_agents_timesteps_dicts = [customdict.CustomDict() for _ in range(len(workers))]

                cursor = 0
                for ni, num_workers in enumerate(dask_nodes_n_workers):
                    for _ in range(num_workers):
                        temp_cells_agents_timesteps_dicts[ni].update(cells_agents_timesteps_dicts[cursor])
                        cursor += 1

                cells_agents_timesteps_dicts = temp_cells_agents_timesteps_dicts
                dask_numtasks = len(workers)

            time_taken = time.time() - start
            print("split_cellsagentstimesteps_balanced (load balancing): " + str(time_taken))
            if f is not None:
                f.flush()
            
            start = time.time()
            dask_params, futures, delayed_computations = [], [], []

            cat_size = util.asizeof_formatted(vars_util.cells_agents_timesteps)
            agents_epi_util_size = util.asizeof_formatted(agents_epi_util)
            vars_util_size = util.asizeof_formatted(vars_util)
            print(f"cat size: {cat_size}, agents_epi size: {agents_epi_util_size}, vars_util size: {vars_util_size}")

            for worker_index in range(dask_numtasks):
                worker_assign_start = time.time()

                worker_url = workers[worker_index]
                remote_worker_index = node_worker_index_by_worker_url[worker_url]
                # cells_partial = {}

                cells_agents_timesteps_partial = customdict.CustomDict()
                agents_partial = AgentsEpi()
                vars_util_partial = vars.Vars()

                # if not dask_full_array_mapping:
                #     vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                #     # vars_util_partial.agents_seir_state = copy(vars_util.agents_seir_state) # may be optimized by sending only specific day
                # else:
                #     vars_util_partial.agents_seir_state = [] # to be populated hereunder    
                # vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                cells_agents_timesteps_partial = cells_agents_timesteps_dicts[worker_index]

                unique_agent_ids = set()
                for cell_vals in cells_agents_timesteps_partial.values():
                    for cell_agent_timesteps in cell_vals:
                        unique_agent_ids.add(cell_agent_timesteps[0])

                unique_agent_ids = sorted(list(unique_agent_ids))

                agents_partial, _, vars_util_partial, _ = util.split_dicts_by_agentsids(day, unique_agent_ids, agents_epi_util, vars_util, agents_partial, vars_util_partial, is_dask_task=dask_full_array_mapping)

                # if not dask_full_array_mapping:
                #     mask = np.isin(np.arange(len(vars_util_partial.agents_seir_state)), unique_agent_ids, invert=True)
                    
                #     vars_util_partial.agents_seir_state = ma.masked_array(vars_util_partial.agents_seir_state, mask=mask)

                # print("worker index: " + str(worker_index) + ", agents_seir_state count: " + str(len(vars_util_partial.agents_seir_state)) + ", vals: " + str(vars_util_partial.agents_seir_state))
                
                vars_util_partial.cells_agents_timesteps = cells_agents_timesteps_partial

                print("starting process index with " + str(len(vars_util_partial.cells_agents_timesteps)) + " cells on " + str(worker_index) + " at " + str(time.time()))
                if f is not None:
                    f.flush()

                if not use_mp:
                    params = (day, weekday, agents_partial, vars_util_partial, dynparams, remote_worker_index, log_file_name)
                else:
                    params = (day, weekday, agents_partial, vars_util_partial, dynparams, log_file_name)

                params_size = util.asizeof_formatted(params)
                print(f"params for {worker_index} size: {params_size}")

                cat_partial_size = util.asizeof_formatted(vars_util_partial.cells_agents_timesteps)
                vars_util_partial_size = util.asizeof_formatted(vars_util_partial)
                agents_partial_size = util.asizeof_formatted(agents_partial)
                dyn_params_size = util.asizeof_formatted(dynparams)
                print(f"cat size: {cat_partial_size}, vars_util size: {vars_util_partial_size}, agents partial size: {agents_partial_size}, dyn_params size: {dyn_params_size}")

                if not use_mp:
                    if dask_mode == 0:
                        future = client.submit(contactnetwork_worker, params, workers=worker_url)
                        futures.append(future)
                    elif dask_mode == 1 or dask_mode == 2:
                        delayed_computation = dask.delayed(contactnetwork_worker)(params)
                        delayed_computations.append(delayed_computation)
                        # delayed_computations = [dask.delayed(itinerary_mp.localitinerary_worker)(dask_params[i]) for i in range(len(dask_params))]
                    elif dask_mode == 3:
                        dask_params.append(params)
                else:
                    actor = actors[worker_index]
                    future = actor.run_contactnetwork_parallel(params)
                    futures.append(future)

                worker_assign_time_taken = time.time() - worker_assign_start
                
                if not use_mp:
                    dask_workers_time_taken[remote_worker_index] = [worker_assign_time_taken, None]
                else:
                    dask_workers_time_taken[remote_worker_index[0]] = [worker_assign_time_taken, None]

            time_taken = time.time() - start
            print("futures / delayed_results generation: " + str(time_taken))
            if f is not None:
                f.flush()

            if not use_mp:
                start = time.time()

                if dask_mode == 1:
                    futures = client.compute(delayed_computations)
                elif dask_mode == 2:
                    futures = dask.compute(delayed_computations)
                elif dask_mode == 3:
                    futures = client.map(contactnetwork_worker, dask_params)

                time_taken = time.time() - start
                print("futures generation: " + str(time_taken))
                if f is not None:
                    f.flush()

            start = time.time()

            if not use_mp:
                method_type = MethodType.ContactNetworkDist
            else:
                method_type = MethodType.ContactNetworkDistMP

            _, agents_epi_util, vars_util, dask_workers_time_taken, dask_mp_processes_time_taken = daskutil.handle_futures(method_type, day, futures, None, agents_epi_util, vars_util, task_results_stack_trace_log_file_name, False, True, dask_full_array_mapping, f, dask_workers_time_taken, dask_mp_processes_time_taken)
                
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

def contactnetwork_worker(params):
    import os
    import shutil

    f = None
    original_stdout = sys.stdout
    stack_trace_log_file_name = ""

    try:
        main_start = time.time()

        day, weekday, agents_epi_util, vars_util, dyn_params, process_index, log_file_name = params

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_mp_stack_trace_" + str(day) + "_" + str(process_index) + ".txt"
        log_file_name = log_file_name.replace(".txt", "") + "_cn_" + str(day) + "_" + str(process_index) + ".txt"

        # f = open(log_file_name, "w")
        # sys.stdout = f

        worker = get_worker()

        n_locals = worker.data["n_locals"]
        n_tourists = worker.data["n_tourists"]
        locals_ratio_to_full_pop = worker.data["locals_ratio_to_full_pop"]
        contactnetworkparams = worker.data["contactnetworkparams"]
        epidemiologyparams = worker.data["epidemiologyparams"]
        cells_type = worker.data["cells_type"]
        indids_by_cellid = worker.data["indids_by_cellid"]
        cells_households = worker.data["cells_households"] 
        cells_institutions = worker.data["cells_institutions"] 
        cells_accommodation = worker.data["cells_accommodation"] 
        agents_static = worker.data["agents_static"]

        # print("process " + str(process_index) + " started at " + str(start))

        contact_network_util = contactnetwork.ContactNetwork(n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            agents_static,
                                                            agents_epi_util,
                                                            vars_util,
                                                            cells_type,
                                                            indids_by_cellid,
                                                            cells_households, 
                                                            cells_institutions, 
                                                            cells_accommodation, 
                                                            contactnetworkparams,
                                                            epidemiologyparams, 
                                                            dyn_params, 
                                                            process_index=process_index)

        _, updated_agent_ids, agents_epi_util_partial, vars_util = contact_network_util.simulate_contact_network(day, weekday)
        
        # certain data does not have to go back because it would not have been updated in this context
        vars_util.cells_agents_timesteps = customdict.CustomDict()
        vars_util.agents_seir_state_transition_for_day = customdict.CustomDict()
        vars_util.agents_vaccination_doses = customdict.CustomDict()
        
        main_time_taken = time.time() - main_start
        # print("process " + str(process_index) + " ended at " + str(time.time()))

        return process_index, updated_agent_ids, agents_epi_util_partial, vars_util, main_time_taken
    except Exception as e:
        # log on the node where it happened
        actual_stack_trace_log_file_name = stack_trace_log_file_name.replace(".txt", "_actual.txt")

        with open(actual_stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

        return {"exception": e, "traceback": traceback.format_exc(), "logfilename": stack_trace_log_file_name}
    finally:
        gc.collect()

        if f is not None:
            # Close the file
            f.close()

        if original_stdout is not None:
            sys.stdout = original_stdout