import sys
# sys.path.insert(0, '~/AppsPy/mtdcovabm/simulator')
import dask
import asyncio
# from queue import Empty
import threading
import numpy as np
import traceback
import itinerary, itinerary_mp, vars, shared_mp
import time
import util
from copy import copy, deepcopy
from dask.distributed import get_worker, as_completed
import multiprocessing as mp
from memory_profiler import profile

# fp = open("memory_profiler_dist.log", "w+")
# @profile(stream=fp)

def localitinerary_finegrained_distributed(client, 
                                           hh_insts, 
                                           day, 
                                           weekday, 
                                           weekdaystr, 
                                           agents_dynamic, 
                                           vars_util, 
                                           dyn_params,
                                           keep_workers_open=True, 
                                           log_file_name="output.txt"):
    original_log_file_name = log_file_name
    stack_trace_log_file_name = ""
    try:
        start = time.time()
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_res_stack_trace" + ".txt"

        delayed_computations = []
        futures = []
        for hh_inst in hh_insts:
            agents_dynamic_partial = {}
            vars_util_partial = vars.Vars()
            vars_util_partial.agents_seir_state = [] # to be populated hereunder
            vars_util_partial.agents_vaccination_doses = [] # to be populated hereunder
            vars_util_partial.cells_agents_timesteps = {}

            agents_dynamic_partial, _, vars_util_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], agents_dynamic, vars_util, agents_dynamic_partial, vars_util_partial, None, None, is_itinerary=True, is_dask_task=True)                  
            
            params = day, weekday, weekdaystr, hh_inst, agents_dynamic_partial, vars_util_partial, dyn_params, log_file_name
            future = client.submit(localitinerary_worker_res, params)
            futures.append(future)
            # delayed_computations.append(dask.delayed(localitinerary_worker)(params))

        for future in as_completed(futures):
            result = future.result()

            agents_dynamic_partial_result, vars_util_partial_result = result

            agents_dynamic, vars_util = sync_results_res(agents_dynamic, vars_util, agents_dynamic_partial_result, vars_util_partial_result, list(agents_dynamic_partial_result.keys()))

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

def localitinerary_distributed(client,
                            day,
                            weekday,
                            weekdaystr,
                            itineraryparams,
                            timestepmins,
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop,
                            agents_dynamic,
                            agents_ids_by_ages,
                            vars_util, 
                            cells_industries_by_indid_by_wpid, # cells related only being used for mp now
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
                            dynparams,
                            hh_insts,
                            num_processes=10,
                            num_threads=2,
                            proc_use_pool=0, # Pool apply_async 0, Process 1, ProcessPoolExecutor = 2, Pool IMap 3, Dask MP Scheduler = 4
                            sync_use_threads=False,
                            sync_use_queue=False,
                            keep_processes_open=True,
                            use_mp=False, # in this context use_mp means use single dask worker and mp in each node
                            log_file_name="output.txt"): 
    original_log_file_name = log_file_name
    stack_trace_log_file_name = ""
    try:
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_dist_stack_trace" + ".txt"

        # process_counter = manager.Value("i", num_processes)

        vars_util.reset_cells_agents_timesteps()

        workers = client.scheduler_info()["workers"].keys() # list()
        # worker = client.scheduler_info()["workers"][workers[0]]
        # print("local_directory: " + str(worker["local_directory"]))
        
        num_workers = len(workers)

        start = time.time()
        mp_hh_inst_indices = util.split_residences_by_weight(hh_insts, num_workers) # to do - for the time being this assumes equal split but we may have more information about the cores of the workers
        time_taken = time.time() - start
        print("split residences by indices (load balancing): " + str(time_taken))
        
        start = time.time()

        dask_params = []

        for worker_index in range(num_workers):
            # cells_partial = {}
            hh_insts_partial = []

            mp_hh_inst_ids_this_proc = mp_hh_inst_indices[worker_index]

            start_partial = time.time()
            hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc]      

            agents_partial, agents_ids_by_ages_partial = {}, {}
            vars_util_partial = vars.Vars()
            vars_util_partial.agents_seir_state = vars_util.agents_seir_state
            vars_util_partial.agents_vaccination_doses = vars_util.agents_vaccination_doses
            vars_util_partial.cells_agents_timesteps = {}

            for hh_inst in hh_insts_partial:
                agents_partial, agents_ids_by_ages_partial, vars_util_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], agents_dynamic, vars_util, agents_partial, vars_util_partial, agents_ids_by_ages, agents_ids_by_ages_partial, is_itinerary=True)                  

            # agents_partial = {uid:agents[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}
            # agents_ids_by_ages_partial = {uid:agents_ids_by_ages[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}

            time_taken_partial = time.time() - start_partial
            print("creating partial dicts. time taken: " + str(time_taken_partial))

            print("starting process index " + str(worker_index) + " at " + str(time.time()))

            # Define parameters
            if use_mp: # call itinerary_dist.localitinerary_dist_worker
                params = (day, 
                        weekday, 
                        weekdaystr, 
                        hh_insts_partial, 
                        itineraryparams, 
                        timestepmins, 
                        n_locals, 
                        n_tourists, 
                        locals_ratio_to_full_pop,
                        agents_partial, 
                        agents_ids_by_ages_partial, 
                        deepcopy(vars_util_partial), 
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
                        dynparams, 
                        proc_use_pool,
                        num_processes,
                        worker_index,
                        original_log_file_name) 
            else: # call itinerary_mp.localitinerary_worker
                params = (day, 
                        weekday, 
                        weekdaystr, 
                        hh_insts_partial, 
                        agents_partial,
                        deepcopy(vars_util_partial), 
                        dynparams, 
                        worker_index, 
                        log_file_name)  

            dask_params.append(params)

        start = time.time()
        if use_mp:
            delayed_computations = [dask.delayed(localitinerary_dist_worker)(dask_params[i]) for i in range(num_workers)]
        else:
            delayed_computations = [dask.delayed(itinerary_mp.localitinerary_worker)(dask_params[i]) for i in range(num_workers)]
        
        time_taken = time.time() - start
        print("delayed_results generation: " + str(time_taken))

        start = time.time()
        # futures = client.compute(delayed_computations) # this delays computation (and doesn't seem to work with the large dataset)
        results = dask.compute(*delayed_computations) # this blocks (and seems to work)
        time_taken = time.time() - start
        print("computed_results generation: " + str(time_taken))

        time_taken = time.time() - start

        print("started pool/processes. time taken: " + str(time_taken))

        start = time.time()

        # results = client.gather(futures) # another way of collecting the results

        # # for future in as_completed(futures): # to use with futures
        for result in results: # simple traversal of already collected results (block method)
            # result = future.result()

            if use_mp:
                worker_index, agents_partial_results_combined, vars_util_partial_results_combined = result
            else:
                worker_index, agents_partial_results_combined, vars_util_partial_results_combined, _, _, _, _ = result

            print("processing results for worker " + str(worker_index))

            for i in range(1000):
                if i in agents_partial_results_combined:
                    if agents_partial_results_combined[i]["itinerary"] is None or len(agents_partial_results_combined[i]["itinerary"]) == 0:
                        print("itinerary_dist results, itinerary is empty " + str(i))

            agents_dynamic, vars_util = sync_results(worker_index, mp_hh_inst_indices, hh_insts, agents_dynamic, vars_util, agents_partial_results_combined, vars_util_partial_results_combined)

        if not keep_processes_open:
            start = time.time()
            client.shutdown()
            time_taken = time.time() - start
            print("client shutdown time taken " + str(time_taken))         
    except:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

def sync_results(process_index, mp_hh_inst_indices, hh_insts, agents_dynamic, vars_util, agents_partial, vars_util_partial):
    if agents_partial is not None and len(agents_partial) > 0:
        mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

        hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc] 

        for hh_inst in hh_insts_partial:
            agents_dynamic, vars_util = util.sync_state_info_by_agentsids(hh_inst["resident_uids"], agents_dynamic, vars_util, agents_partial, vars_util_partial)

        vars_util = util.sync_state_info_sets(vars_util, vars_util_partial)

        start_cat = time.time()
        
        vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)

        time_taken_cat = time.time() - start_cat
        print("cells_agents_timesteps sync for process {0}, time taken: {1}".format(process_index, str(time_taken_cat)))

    return agents_dynamic, vars_util

def sync_results_res(agents_dynamic, vars_util, agents_partial, vars_util_partial, res_agent_ids):
    agents_dynamic, vars_util = util.sync_state_info_by_agentsids(res_agent_ids, agents_dynamic, vars_util, agents_partial, vars_util_partial, task_agent_ids=res_agent_ids)
    vars_util = util.sync_state_info_sets(vars_util, vars_util_partial)

    # start_cat = time.time()
    vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)
    # time_taken_cat = time.time() - start_cat
    # print("cells_agents_timesteps sync, time taken: {0}".format(str(time_taken_cat)))

    return agents_dynamic, vars_util
    
def localitinerary_worker_res(params):
    import os
    import sys

    original_stdout = None
    f = None
    stack_trace_log_file_name = ""

    try:
        day, weekday, weekdaystr, hh_inst, agents_dynamic, vars_util_mp, dyn_params, log_file_name = params

        worker = get_worker()
        
        # original_stdout = sys.stdout
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_res_stack_trace_" + worker.id + ".txt"
        # log_file_name = log_file_name.replace(".txt", "") + "_it_res_" + worker.id + "_" + str(hh_inst["id"]) + ".txt" 
        # f = open(log_file_name, "w")
        # sys.stdout = f
        
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
                                            dyn_params,
                                            task_agent_ids=list(agents_dynamic.keys()))
        
        if day == 1 or weekdaystr == "Monday":
            # start = time.time()
            itinerary_util.generate_working_days_for_week_residence(hh_inst["resident_uids"], hh_inst["is_hh"])
            # time_taken = time.time() - start
            # print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(process_index))

        # start = time.time()
        itinerary_util.generate_local_itinerary(day, weekday, hh_inst["resident_uids"])
        # time_taken = time.time() - start
        # print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken))
        
        return agents_dynamic, vars_util_mp
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
                                                                            log_file_name=original_log_file_name)

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
