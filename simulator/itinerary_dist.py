import sys
# sys.path.insert(0, '~/AppsPy/mtdcovabm/simulator')
import dask
import asyncio
# from queue import Empty
import threading
import numpy as np
import traceback
import itinerary_mp, vars
import time
import util
from copy import copy, deepcopy
from dask.distributed import get_worker, as_completed
import multiprocessing as mp

def localitinerary_distributed(client,
                            day,
                            weekday,
                            weekdaystr,
                            popsubfolder,
                            itineraryparams,
                            timestepmins,
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop,
                            agents_static,
                            agents_dynamic,
                            agents_ids_by_ages,
                            tourists,
                            vars_util, 
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
                            tourist_entry_infection_probability,
                            epidemiologyparams,
                            dynparams,
                            tourists_active_ids,
                            hh_insts,
                            num_processes=10,
                            num_threads=2,
                            proc_use_pool=0, # Pool apply_async 0, Process 1, ProcessPoolExecutor = 2, Pool IMap 3, Dask MP Scheduler = 4
                            sync_use_threads=False,
                            sync_use_queue=False,
                            keep_processes_open=True,
                            log_file_name="output.txt"): 
    original_log_file_name = log_file_name
    stack_trace_log_file_name = ""
    try:
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_dist_stack_trace" + ".txt"

        # process_counter = manager.Value("i", num_processes)

        vars_util.reset_cells_agents_timesteps()

        workers = client.scheduler_info()["workers"].keys()
        
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
                agents_partial, agents_ids_by_ages_partial, vars_util_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], agents_dynamic, agents_ids_by_ages, vars_util, agents_partial, agents_ids_by_ages_partial, vars_util_partial, is_itinerary=True)                  

            tourists_active_ids = []
            tourists = None # to do - to handle

            # agents_partial = {uid:agents[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}
            # agents_ids_by_ages_partial = {uid:agents_ids_by_ages[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}

            time_taken_partial = time.time() - start_partial
            print("creating partial dicts. time taken: " + str(time_taken_partial))

            print("starting process index " + str(worker_index) + " at " + str(time.time()))

            # Define parameters
            params = (day, 
                    weekday, 
                    weekdaystr, 
                    popsubfolder,
                    hh_insts_partial, 
                    itineraryparams, 
                    timestepmins, 
                    n_locals, 
                    n_tourists, 
                    locals_ratio_to_full_pop,
                    agents_static,
                    agents_partial, 
                    agents_ids_by_ages_partial, 
                    deepcopy(vars_util), 
                    tourists, 
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
                    tourist_entry_infection_probability, 
                    epidemiologyparams, 
                    dynparams, 
                    tourists_active_ids, 
                    proc_use_pool,
                    worker_index,
                    original_log_file_name)             

            dask_params.append(params)

        start = time.time()
        delayed_computations = [dask.delayed(localitinerary_worker)(dask_params[i]) for i in range(num_workers)]
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

            worker_index, agents_partial, vars_util_partial = result

            print("processing results for worker " + str(worker_index))

            mp_hh_inst_ids_this_proc = mp_hh_inst_indices[worker_index]

            hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc] 

            for hh_inst in hh_insts_partial:
                agents_dynamic, vars_util = util.sync_state_info_by_agentsids(hh_inst["resident_uids"], agents_dynamic, vars_util, agents_partial, vars_util_partial)

            vars_util = util.sync_state_info_sets(vars_util, vars_util_partial)

            start_cat = time.time()
            
            vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)

            time_taken_cat = time.time() - start_cat
            print("cells_agents_timesteps sync for process {0}, time taken: {1}".format(worker_index, str(time_taken_cat)))

            time_taken = time.time() - start
            print("syncing results back with main process. time taken " + str(time_taken))

        if not keep_processes_open:
            start = time.time()
            client.shutdown()
            time_taken = time.time() - start
            print("client shutdown time taken " + str(time_taken))         
    except:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_worker(params):
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
        day, weekday, weekdaystr, popsubfolder, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_static, agents_dynamic, agents_ids_by_ages, vars_util_mp, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, proc_use_pool, worker_index, log_file_name = params
        
        original_stdout = sys.stdout
        original_log_file_name = log_file_name

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_dist_stack_trace_" + str(worker_index) + ".txt"
        log_file_name = log_file_name.replace(".txt", "") + "_it_" + str(worker_index) + ".txt"
        f = open(log_file_name, "w")
        sys.stdout = f

        # implementation for static agents via dask plugins
        print("interpreter: " + os.path.dirname(sys.executable))
        print("current working directory: " + os.getcwd())

        # worker = get_worker()

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
            manager = mp.Manager()
            pool = mp.Pool()

        # pool = mp.Pool(initializer=shared_mp.init_pool_processes, initargs=(agents_static,))

        print("cells_agents_timesteps in process " + str(worker_index) + " is " + str(id(vars_util_mp.cells_agents_timesteps)))
        print(f"Itinerary Worker Child #{worker_index+1} at {str(time.time())}", flush=True)

        agents_dynamic, vars_util_mp = itinerary_mp.localitinerary_parallel(manager,
                                                                            pool,
                                                                            day,
                                                                            weekday,
                                                                            weekdaystr,
                                                                            popsubfolder,
                                                                            itineraryparams,
                                                                            timestepmins, 
                                                                            n_locals, 
                                                                            n_tourists, 
                                                                            locals_ratio_to_full_pop, 
                                                                            agents_dynamic, 
                                                                            agents_ids_by_ages,
                                                                            tourists,
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
                                                                            tourist_entry_infection_probability, 
                                                                            epidemiologyparams, 
                                                                            dyn_params,
                                                                            tourists_active_ids,
                                                                            hh_insts,
                                                                            1, # to do - this is to be passed dynamically based on worker/s config/s,
                                                                            proc_use_pool=True,
                                                                            log_file_name=original_log_file_name)

        # to do - somehow to return and combine results

        return worker_index, agents_dynamic, vars_util_mp
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
