import sys
import os
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import threading
import numpy as np
import numpy.ma as ma
import traceback
import contactnetwork, util, vars, customdict, daskutil
from epidemiology import Epidemiology
import time
import dask
from dask.distributed import Client, get_worker, performance_report
from copy import copy

def contacttracing_distributed(client: Client,
                            day, 
                            epidemiologyutil: Epidemiology,
                            agents_epi,
                            vars_util,
                            dynparams,              
                            dask_mode=0,
                            dask_numtasks=-1,
                            dask_full_array_mapping=False,
                            keep_processes_open=True,
                            f=None,
                            log_file_name="output.txt"):
    stack_trace_log_file_name = ""
    task_results_stack_trace_log_file_name = ""

    original_stdout = None
    f = None

    try:
        folder_name = ""
        if log_file_name != "output.txt":
            folder_name = os.path.dirname(log_file_name)
        else:
            folder_name = os.getcwd()

        dask_perf_log_file_name = os.path.join(folder_name, "dask_ct_perf_log_" + str(day) + ".html")
        stack_trace_log_file_name = os.path.join(folder_name, "ct_main_dist_stack_trace_" + str(day) + ".txt")
        task_results_stack_trace_log_file_name = os.path.join(folder_name, "ct_main_res_task_results_stack_trace_" + str(day) + ".txt")

        # stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_main_dist_stack_trace" + ".txt"
        # task_results_stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_cn_main_res_task_results_stack_trace" + ".txt"

        with performance_report(filename=dask_perf_log_file_name):
            workers = client.scheduler_info()["workers"].keys()

            if dask_numtasks == -1:
                dask_numtasks = len(workers)

            directcontacts_size = sum([sys.getsizeof(c) for c in vars_util.directcontacts_by_simcelltype_by_day]) / (1024 * 1024)
            agent1_index_size = sum([sys.getsizeof(i) for i in vars_util.dc_by_sct_by_day_agent1_index]) / (1024 * 1024)
            agent2_index_size = sum([sys.getsizeof(i) for i in vars_util.dc_by_sct_by_day_agent2_index]) / (1024 * 1024)

            print("day: {0}, contact_tracing_agent_ids_len: {1}, directcontacts len: {2}, directcontacts size: {3}, agent1 index size: {4}, agent2 index size: {5}".format(str(day), str(len(vars_util.contact_tracing_agent_ids)), str(len(vars_util.directcontacts_by_simcelltype_by_day)), str(directcontacts_size), str(agent1_index_size), str(agent2_index_size)))
            if f is not None:
                f.flush()

            start = time.time()
            contact_tracing_agent_ids_list = list(vars_util.contact_tracing_agent_ids)
            contacttracingagentids_indices = [i for i, _ in enumerate(contact_tracing_agent_ids_list)]
            np.random.shuffle(contacttracingagentids_indices)
            mp_contacttracingagentids_indices = np.array_split(contacttracingagentids_indices, dask_numtasks)
            time_taken = time.time() - start
            print("splitting contact_tracing_agent_ids, time_taken: " + str(time_taken))
            if f is not None:
                f.flush()
            
            start = time.time()
            dask_params, futures, delayed_computations = [], [], []

            for worker_index in range(dask_numtasks):
                vars_util_partial = vars.Vars()

                worker_contacttracingagentids_indices = mp_contacttracingagentids_indices[worker_index]

                start = time.time()
                vars_util_partial.contact_tracing_agent_ids = set([contact_tracing_agent_ids_list[index] for index in worker_contacttracingagentids_indices]) # simply to retain type
                time_taken = time.time() - start
                print("partialising contact_tracing_agent_ids, process: " + str(worker_index) + ", time taken: " + str(time_taken))
                if f is not None:
                    f.flush()

                if len(vars_util_partial.contact_tracing_agent_ids) > 0:
                    start = time.time()
                    vars_util_partial.directcontacts_by_simcelltype_by_day = vars_util.directcontacts_by_simcelltype_by_day
                    vars_util_partial.directcontacts_by_simcelltype_by_day_start_marker = vars_util.directcontacts_by_simcelltype_by_day_start_marker
                    vars_util_partial.dc_by_sct_by_day_agent1_index = vars_util.dc_by_sct_by_day_agent1_index
                    vars_util_partial.dc_by_sct_by_day_agent2_index = vars_util.dc_by_sct_by_day_agent2_index
                    
                    time_taken = time.time() - start
                    print("partialising directcontacts_by_simcelltype_by_day, process: " + str(worker_index) + ", time taken: " + str(time_taken))

                    print("starting process index " + str(worker_index) + " at " + str(time.time()))
                    if f is not None:
                        f.flush()

                    params = (day, agents_epi, vars_util_partial, dynparams, worker_index, log_file_name)

                    if dask_mode == 0:
                        future = client.submit(contacttracing_worker, params)
                        futures.append(future)
                    elif dask_mode == 1 or dask_mode == 2:
                        delayed_computation = dask.delayed(contacttracing_worker)(params)
                        delayed_computations.append(delayed_computation)
                        # delayed_computations = [dask.delayed(itinerary_mp.localitinerary_worker)(dask_params[i]) for i in range(len(dask_params))]
                    elif dask_mode == 3:
                        dask_params.append(params)

            time_taken = time.time() - start
            print("delayed_results generation: " + str(time_taken))
            if f is not None:
                f.flush()

            start = time.time()

            if dask_mode == 1:
                futures = client.compute(delayed_computations)
            elif dask_mode == 2:
                futures = dask.compute(delayed_computations)
            elif dask_mode == 3:
                futures = client.map(contacttracing_worker, dask_params)

            time_taken = time.time() - start
            print("futures generation: " + str(time_taken))
            if f is not None:
                f.flush()

            start = time.time()
            _, agents_epi, vars_util, _ = daskutil.handle_futures(day, futures, None, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, True, dask_full_array_mapping)
            
            time_taken = time.time() - start
            print("syncing pool imap results back with main process. time taken " + str(time_taken))
            if f is not None:
                f.flush()

            epidemiologyutil.contact_tracing_clean_up(day)
            
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

def contacttracing_worker(params):
    from shared_mp import agents_static

    original_stdout = None
    f = None
    stack_trace_log_file_name = ""
    try:
        main_start = time.time()

        # sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cell_agents_timesteps, tourists_active_ids, cells_mp, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params
        day, agents_epi, vars_util, dyn_params, process_index, log_file_name = params

        original_stdout = sys.stdout
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_ct_mp_stack_trace_" + str(day) + "_" + str(process_index) + ".txt"
        log_file_name = log_file_name.replace(".txt", "") + "_ct_" + str(day) + "_" + str(process_index) + ".txt"
        f = open(log_file_name, "w")
        sys.stdout = f

        worker = get_worker()

        n_locals = worker.data["n_locals"]
        n_tourists = worker.data["n_tourists"]
        locals_ratio_to_full_pop = worker.data["locals_ratio_to_full_pop"]

        epidemiologyparams = worker.data["epidemiologyparams"]
        cells_households = worker.data["cells_households"] 
        cells_institutions = worker.data["cells_institutions"] 
        cells_accommodation = worker.data["cells_accommodation"] 
        agents_static = worker.data["agents_static"]

        epidemiology_util = Epidemiology(epidemiologyparams, 
                                        n_locals,
                                        n_tourists,
                                        locals_ratio_to_full_pop,
                                        agents_static,
                                        agents_epi,
                                        vars_util,
                                        cells_households,
                                        cells_institutions,
                                        cells_accommodation,
                                        dyn_params,
                                        process_index)

        process_index, updated_agents_ids, agents_epi, vars_util = epidemiology_util.contact_tracing(day, True)
        
        main_time_taken = time.time() - main_start

        agents_epi_partial = {agentid:agents_epi[agentid] for agentid in updated_agents_ids}
        
        print("updated_agent_ids len: {0}, agents_epi_partial len: {1}, main_time_taken: {2}".format(str(len(updated_agents_ids)), str(len(agents_epi_partial)), str(main_time_taken)))

        # vars_util.directcontacts_by_simcelltype_by_day = []
        # vars_util.dc_by_sct_by_day_agent1_index = []
        # vars_util.dc_by_sct_by_day_agent2_index = []
        # vars_util.contact_tracing_agent_ids = set()
        # vars_util.agents_seir_state = []
        # vars_util.agents_seir_state_transition_for_day = customdict.CustomDict()
        # vars_util.agents_infection_type = customdict.CustomDict()
        # vars_util.agents_infection_severity = customdict.CustomDict()
        # vars_util.agents_vaccination_doses = []

        return process_index, agents_epi_partial, main_time_taken
    except Exception as e:
        traceback_str = traceback.format_exc()

        exception_info = {"processindex": process_index, 
                          "logfilename": stack_trace_log_file_name, 
                          "type": type(e).__name__, 
                          "message": str(e), 
                          "traceback": traceback_str}
        
        return exception_info
    finally:
        if original_stdout is not None:
            sys.stdout = original_stdout

            if f is not None:
                # Close the file
                f.close()
