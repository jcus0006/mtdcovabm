import sys
import os
import multiprocessing as mp
# import multiprocessing.shared_memory as shm
import concurrent.futures
# from queue import Empty
import threading
import numpy as np
import traceback
import itinerary, vars, static, customdict
import time
import util, daskutil
from util import MethodType
from copy import copy, deepcopy
from dask import compute, delayed
from dask.distributed import as_completed, get_worker
import gc
import psutil

def localitinerary_parallel(manager,
                            pool,
                            day,
                            weekday,
                            weekdaystr,
                            itineraryparams,
                            timestepmins,
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop,
                            it_agents,
                            agents_epi,
                            agents_ids_by_ages,
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
                            epidemiologyparams,
                            dynparams,
                            hh_insts,
                            num_processes=10,
                            num_threads=2,
                            proc_use_pool=0, # Pool apply_async 0, Process 1, ProcessPoolExecutor = 2, Pool IMap 3, Dask MP Scheduler = 4
                            use_shm=True,
                            sync_use_threads=False,
                            sync_use_queue=False,
                            keep_processes_open=True,
                            log_file_name="output.txt",
                            dask=False,
                            agents_static=None,
                            static_agents_dict=None): 
    stack_trace_log_file_name = ""

    try:
        # p = psutil.Process()
        # print(f"main process #{0}: {p}, affinity {p.cpu_affinity()}", flush=True)
        # time.sleep(0.0001)
        # p.cpu_affinity([0])

        # global agents_main
        # global vars_util_main

        # agents_main = agents
        # vars_util_main = vars_util

        # manager = mp.Manager()
        # sync_queue = manager.Queue()

        folder_name = ""
        if log_file_name != "output.txt":
            folder_name = os.path.dirname(log_file_name)
        else:
            folder_name = os.getcwd()
        
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_mp_stack_trace_" + str(day) + ".txt"
        task_results_stack_trace_log_file_name = os.path.join(folder_name, "it_main_res_task_results_stack_trace_" + str(day) + ".txt")

        process_counter = None
        if manager is not None:
            process_counter = manager.Value("i", num_processes)

        interventions_totals = [0, 0, 0, 0]

        agents_partial_results_combined, vars_util_partial_results_combined = [], []

        print("proc_use_pool: " + str(proc_use_pool))
        print("num_processes: " + str(num_processes))
        
        if num_processes > 1:
        #     sync_threads = []
        #     proc_processes = []

        #     if num_threads > 1:
        #         if sync_use_queue:
        #             # option 1 - single thread
        #             if sync_use_threads:
        #                 # option 2 - multiple threads
                        
        #                 for ti in range(num_threads-1):
        #                     cpu_affinity = 1 + ti + num_processes
        #                     # cpu_affinity = 0

        #                     # if ti > 0:
        #                     #     cpu_affinity = ti + num_processes

        #                     t = threading.Thread(target=sync_state_info, args=(sync_queue, process_counter, cpu_affinity))
        #                     t.start()
        #                     sync_threads.append(t)
        #             else:
        #                 # option 3 - multiple processes
                        
        #                 for pi in range(num_threads-1):
        #                     cpu_affinity = 1 + pi + num_processes

        #                     process = mp.Process(target=sync_state_info, args=(sync_queue, process_counter, cpu_affinity, agents, vars_util))
        #                     process.start()
        #                     proc_processes.append(process)

            # pool = mp.Pool(initializer=init_worker, initargs=(process_counter,))
            # pool = mp.Pool(processes=num_processes)

            # mp_hh_inst_indices = []

            # hh_inst_indices = [i for i, _ in enumerate(hh_insts)]
            # np.random.shuffle(hh_inst_indices)
            # mp_hh_inst_indices = np.array_split(hh_inst_indices, num_processes)

            start = time.time()
            mp_hh_inst_indices = util.split_residences_by_weight(hh_insts, num_processes)
            time_taken = time.time() - start
            print("split residences by indices (load balancing): " + str(time_taken))
            
            start = time.time()

            proc_processes = []
            proc_futures = []
            imap_params = []
            imap_results = []

            # hh_insts_size = util.asizeof_formatted(hh_insts)
            # it_agents_size = util.asizeof_formatted(it_agents)
            # agents_epi_size = util.asizeof_formatted(agents_epi)
            # agents_ids_by_ages_size = util.asizeof_formatted(agents_ids_by_ages)
            # vars_util_size = util.asizeof_formatted(vars_util)
            # print(f"hh_insts size: {hh_insts_size}, it_agents size: {it_agents_size}, agents_epi size: {agents_epi_size}, agents_ids_by_ages size: {agents_ids_by_ages_size}, vars_util size: {vars_util_size}")
            
            for process_index in range(num_processes):
                # cells_partial = {}
                hh_insts_partial = []

                mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

                # agents_partial = []
                # for index in mp_hh_inst_ids_this_proc:
                #     # cell = cells[cell_key]
                #     hh_inst = hh_insts[index]

                #     # cells_partial[cell_key] = cell
                #     hh_insts_partial.append(hh_inst)

                start_partial = time.time()
                hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc]      

                agents_partial, agents_ids_by_ages_partial, agents_epi_partial = {}, {}, {}
                vars_util_partial = vars.Vars()
                # vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                vars_util_partial.cells_agents_timesteps = customdict.CustomDict()

                for hh_inst in hh_insts_partial:
                    agents_partial, agents_ids_by_ages_partial, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, agents_ids_by_ages, agents_ids_by_ages_partial, True, False, agents_epi, agents_epi_partial)                  

                # agents_partial = {uid:agents[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}
                # agents_ids_by_ages_partial = {uid:agents_ids_by_ages[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}

                time_taken_partial = time.time() - start_partial
                print("creating partial dicts. time taken: " + str(time_taken_partial))

                print("starting process index " + str(process_index) + " at " + str(time.time()) + " with " + str(len(hh_insts_partial)) + " residences")

                # Define parameters
                # params = (day, 
                #         weekday, 
                #         weekdaystr,
                #         hh_insts_partial, 
                #         itineraryparams, 
                #         timestepmins, 
                #         n_locals, 
                #         n_tourists, 
                #         locals_ratio_to_full_pop,
                #         agents_partial, 
                #         agents_epi_partial,
                #         agents_ids_by_ages_partial, 
                #         vars_util_partial, # deepcopy(vars_util_partial)
                #         cells_industries_by_indid_by_wpid, 
                #         cells_restaurants, 
                #         cells_hospital,
                #         cells_testinghub, 
                #         cells_vaccinationhub, 
                #         cells_entertainment_by_activityid, 
                #         cells_religious, 
                #         cells_households, 
                #         cells_breakfast_by_accomid, 
                #         cells_airport, 
                #         cells_transport, 
                #         cells_institutions, 
                #         cells_accommodation, 
                #         epidemiologyparams, 
                #         dynparams, 
                #         use_shm, # use_shm
                #         process_index, 
                #         process_counter,
                #         log_file_name,
                #         static_agents_dict)

                params = (day, 
                        weekday, 
                        weekdaystr, 
                        hh_insts_partial, 
                        agents_partial,
                        agents_epi_partial,
                        vars_util_partial, # deepcopy(vars_util_partial) - without this was causing some nasty issues with multiprocessing (maybe with Dask it is fine)
                        dynparams, 
                        use_shm,
                        process_index,
                        process_counter,
                        log_file_name,
                        static_agents_dict)  
                
                # params_size = util.asizeof_formatted(params)
                # print(f"params for {process_index} size: {params_size}")

                # hh_insts__partial_size = util.asizeof_formatted(hh_insts_partial)
                # it_agents_partial_size = util.asizeof_formatted(agents_partial)
                # agents_epi_partial_size = util.asizeof_formatted(agents_epi_partial)
                # agents_ids_by_ages_partial_size = util.asizeof_formatted(agents_ids_by_ages_partial)
                # vars_util_partial_size = util.asizeof_formatted(vars_util_partial)
                # dyn_params_size = util.asizeof_formatted(dynparams)
                # static_agents_dict_size = util.asizeof_formatted(static_agents_dict)
                # print(f"hh_insts size: {hh_insts__partial_size}, it_agents size: {it_agents_partial_size}, agents_epi size: {agents_epi_partial_size}, agents_ids_by_ages size: {agents_ids_by_ages_partial_size}, vars_util size: {vars_util_partial_size}, dyn_params size: {dyn_params_size}, static agents dict size: {static_agents_dict_size}")

                # pool.apply_async(localitinerary_worker, args=((sync_queue, day, weekday, weekdaystr, hh_insts_partial, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, process_index, process_counter),))
                if proc_use_pool == 0:
                    pool.apply_async(localitinerary_worker, args=(params,))
                elif proc_use_pool == 1:
                    process = mp.Process(target=localitinerary_worker, args=(params,))
                    process.start()
                    proc_processes.append(process)
                elif proc_use_pool == 2:
                    # Create a ProcessPoolExecutor
                    executor = concurrent.futures.ProcessPoolExecutor()

                    # Submit tasks to the executor with parameters
                    proc_futures.append(executor.submit(localitinerary_worker, params))
                else:
                    imap_params.append(params)

                if proc_use_pool == 0 or proc_use_pool == 2:
                    # clean up local memory
                    hh_insts_partial = None
                    agents_partial = None
                    agents_ids_by_ages_partial = None

            if proc_use_pool == 3:
                imap_results = pool.imap(localitinerary_worker, imap_params)
           
            delayed_values = None
            if proc_use_pool == 4:
                delayed_values = [delayed(localitinerary_worker)(param) for param in params]

            time_taken = time.time() - start

            print("started pool/processes. time taken: " + str(time_taken))

            # start = time.time()
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
            # sync_state_info(sync_queue, process_counter, 0, agents, vars_util) # blocking
        
            # if num_threads > 1:
            #     if sync_use_threads:
            #         # Wait for all threads to complete
            #         for t in sync_threads:
            #             t.join()
            #     else:
            #         # Wait for all processes to complete
            #         for process in proc_processes:
            #             process.join()

            # sync_time_end = time.time()
            # time_taken = sync_time_end - start
            # print("itinerary state info sync (combined). time taken " + str(time_taken) + ", ended at " + str(sync_time_end))

            futures = None
            if proc_use_pool == 4:
                futures = compute(delayed_values, scheduler="processes", num_processes=num_processes)
            
            if proc_use_pool == 3 or proc_use_pool == 4:
                start = time.time()

                # util.log_memory_usage(prepend_text="Before syncing itinerary results ")
                if proc_use_pool == 3:
                    it_agents, agents_epi, vars_util, interventions_totals, _, _ = daskutil.handle_futures(MethodType.ItineraryMP, day, imap_results, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, True, True, False, None)
                elif proc_use_pool == 4:
                    for future in as_completed(futures):
                        result = future.result()

                        process_index, agents_partial_results, vars_util_partial_results, interventions_totals_partial = result

                        agents_dynamic, vars_util, interventions_totals = sync_results(day, process_index, mp_hh_inst_indices, hh_insts, agents_dynamic, vars_util, interventions_totals, agents_partial_results, vars_util_partial_results, interventions_totals_partial) # TODO - to base on it_agents and agents_epi
                        
                        # print("processing results for process " + str(process_index) + ". num agents ws: " + str(num_agents_ws) + ", num agents it: " + str(num_agents_it))
                        # print(working_schedule_times_by_resid_ordered)
                        # print(itinerary_times_by_resid_ordered)            

                # util.log_memory_usage(prepend_text= "After syncing itinerary results ") 
                            
                time_taken = time.time() - start
                print("syncing pool imap results back with main process. time taken " + str(time_taken))

            if not keep_processes_open:
                start = time.time()

                if proc_use_pool == 0 or proc_use_pool == 3: # multiprocessing.pool -> apply_async or imap
                    pool.close()
                    pool.join()
                elif proc_use_pool == 1:
                    for process in proc_processes:
                        process.join()
                elif proc_use_pool == 2:
                    # Collect and print the results (incomplete)
                    for future in concurrent.futures.as_completed(proc_futures):
                        result = future.result() 

                        if result is not None:
                            print("result: " + str(result))

                manager.shutdown()
                time_taken = time.time() - start
                print("pool/processes close/join time taken " + str(time_taken))       
        else:
            # params = sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            params = day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, it_agents, agents_epi, agents_ids_by_ages, vars_util, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, epidemiologyparams, dynparams, use_shm, -1, process_counter, log_file_name, agents_static, True
            result = localitinerary_worker(params, True)

            if not type(result) is dict:
                process_index, it_agents, agents_epi, vars_util, interventions_totals, _, _, _, _, _ = result
            else:
                exception_info = result

                with open(exception_info["logfilename"], "a") as fi:
                    fi.write(f"Exception Type: {exception_info['type']}\n")
                    fi.write(f"Exception Message: {exception_info['message']}\n")
                    fi.write(f"Traceback: {exception_info['traceback']}\n")

        # persist new intervention counts
        dynparams.statistics.new_tests, dynparams.statistics.new_vaccinations, dynparams.statistics.new_quarantined, dynparams.statistics.new_hospitalised = interventions_totals[0], interventions_totals[1], interventions_totals[2], interventions_totals[3]
    except:
        with open(stack_trace_log_file_name, 'w') as fi2:
            traceback.print_exc(file=fi2)
        raise
    finally:
        gc.collect()

def sync_results(day, process_index, mp_hh_inst_indices, hh_insts, agents_dynamic, vars_util, interventions_totals, agents_partial, vars_util_partial, interventions_totals_partial):
    index_start_time = time.time()
    mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

    hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc] 
    index_time_taken = time.time() - index_start_time
    print("index in process " + str(process_index) + ", time_taken: " + str(index_time_taken))

    agentsids_start_time = time.time()
    resident_uids = []
    for hh_inst in hh_insts_partial:
        resident_uids.extend(hh_inst["resident_uids"])

    agents_dynamic, vars_util = util.sync_state_info_by_agentsids(resident_uids, agents_dynamic, vars_util, agents_partial, vars_util_partial)

    agentsids_time_taken = time.time() - agentsids_start_time
    print("agentsids in process " + str(process_index) + ", time_taken: " + str(agentsids_time_taken))

    sets_start_time = time.time()
    vars_util = util.sync_state_info_sets(day, vars_util, vars_util_partial)
    sets_time_taken = time.time() - sets_start_time

    start_cat = time.time()
    vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)
    time_taken_cat = time.time() - start_cat
    print("cells_agents_timesteps sync for process {0}, time taken: {1}".format(process_index, str(time_taken_cat)))

    interventions_totals = util.sync_interventions_totals(interventions_totals, interventions_totals_partial)

    return agents_dynamic, vars_util, interventions_totals

log_file = open("output.log", "a")

def custom_print(*args, **kwargs):
    print(*args, **kwargs)

    with open("ouput.log", "a") as log_file:
        print(*args, **kwargs, file=log_file)

def localitinerary_worker(params, single_proc=False):
    import os
    import sys

    process_counter = None
    node_worker_index = -1
    f = None
    stack_trace_log_file_name = ""
    original_stdout = None

    try:
        main_start = time.time()
        pre_mem_info = psutil.virtual_memory()

        use_mp = False
        agents_static = None
        static_agents_dict = None
        
        # sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_itinerary, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params
        if len(params) > 32:
            use_mp = True # could likely be in else of line 409

            if len(params) > 33: # 33 params (sp)
                day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, it_agents, agents_epi, agents_ids_by_ages, vars_util_mp, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, epidemiologyparams, dyn_params, use_shm, node_worker_index, process_counter, log_file_name, agents_static, is_single_proc = params
            else: # 32 params (mp not optimized)
                is_single_proc = False
                day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, it_agents, agents_epi, agents_ids_by_ages, vars_util_mp, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, epidemiologyparams, dyn_params, use_shm, node_worker_index, process_counter, log_file_name, static_agents_dict = params
        else:
            is_single_proc = False
            if len(params) > 10: # mp (opt)
                use_mp = True
                day, weekday, weekdaystr, hh_insts, it_agents, agents_epi, vars_util_mp, dyn_params, use_shm, node_worker_index, process_counter, log_file_name, static_agents_dict = params
            else: # dask strat 1
                day, weekday, weekdaystr, hh_insts, it_agents, agents_epi, vars_util_mp, dyn_params, node_worker_index, log_file_name = params

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_mp_stack_trace_" + str(day) + "_" + str(node_worker_index) + ".txt"
        
        if use_mp and not single_proc:
            original_stdout = sys.stdout
            log_file_name = log_file_name.replace(".txt", "") + "_it_" + str(day) + "_" + str(node_worker_index) + ".txt"
            f = open(log_file_name, "w")
            sys.stdout = f

        # util.log_memory_usage(f, "Start: ", pre_mem_info)

        worker = None
        if use_mp:
            if agents_static is None:
                from shared_mp import agents_ids_by_ages
                from shared_mp import timestepmins
                from shared_mp import n_locals
                from shared_mp import n_tourists
                from shared_mp import locals_ratio_to_full_pop

                from shared_mp import itineraryparams
                from shared_mp import epidemiologyparams
                from shared_mp import cells_industries_by_indid_by_wpid
                from shared_mp import cells_restaurants
                from shared_mp import cells_hospital
                from shared_mp import cells_testinghub
                from shared_mp import cells_vaccinationhub
                from shared_mp import cells_entertainment_by_activityid
                from shared_mp import cells_religious
                from shared_mp import cells_households
                from shared_mp import cells_breakfast_by_accomid
                from shared_mp import cells_airport
                from shared_mp import cells_transport
                from shared_mp import cells_institutions
                from shared_mp import cells_accommodation
                from shared_mp import agents_static
            
            if static_agents_dict is not None:
                agents_static.static_agents_dict = static_agents_dict
        else:
            worker = get_worker()
            # agents_static = worker.client.futures["agents_static"]
            # agents_static = worker.plugins["read_only_data"]

            if worker is None:
                raise TypeError("Worker is none")
            
            if worker.data is None or len(worker.data) == 0:
                raise TypeError("Worker.data is None or empty")
            else:
                if worker.data["itineraryparams"] is None or len(worker.data["itineraryparams"]) == 0:
                    raise TypeError("Worker.data['itineraryparams'] is None or empty")
            
            # new_tourists = worker.data["new_tourists"]
            # print("new tourists {0}".format(str(new_tourists)))

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
            
        # print("cells_agents_timesteps id in process " + str(process_index) + " is " + str(id(vars_util_mp.cells_agents_timesteps)))
        # print(f"Itinerary Worker Child #{node_worker_index[1]+1} at {str(time.time())}", flush=True)
        if isinstance(node_worker_index, tuple):
            print("Itinerary Worker Child #{0} at {1}".format(str(node_worker_index[1]+1), {str(time.time())}))
        else:
            print("Itinerary Worker Child #{0} at {1}".format(str(node_worker_index+1), {str(time.time())}))

        if worker is not None:
            print("worker memory limit: " + str(worker.memory_limit))
            print("worker cwd: " + os.getcwd())
            print("worker interpreter: " + os.path.dirname(sys.executable))
            # print("worker.data cwd: " + str(worker.data["cwd"]))
            # print("worker.data interpreter: " + str(worker.data["interpreter"]))

        # print(f"Itinerary Worker Child #{process_index+1}: Set my affinity to {process_index+1}, affinity now {p.cpu_affinity()}", flush=True)
        
        # current_directory = os.getcwd()
        # print("current directory: " + str(current_directory))

        # interpreter_exe = os.path.dirname(sys.executable)
        # print("interpreter: " + str(interpreter_exe))
        # os.chdir(interpreter_exe.replace("/bin", ""))

        # agents = None
        # with open(os.path.join(current_directory, "population", popsubfolder, "agents_updated.json"), "r") as read_file:          
        #     agents = json.load(read_file)

        # print("agents: " + str(len(agents)))

        # agents_static = static.Static()
        # agents_static.populate(agents, n_locals, n_tourists, is_shm=False) # for now trying without multiprocessing.RawArray

        itinerary_util = itinerary.Itinerary(itineraryparams,
                                            timestepmins, 
                                            n_locals, 
                                            n_tourists, 
                                            locals_ratio_to_full_pop, 
                                            agents_static,
                                            it_agents, 
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
                                            dyn_params,
                                            process_index=node_worker_index)

        num_agents_working_schedule = 0
        working_schedule_times_by_resid = {}
        if day == 1 or weekdaystr == "Monday":
            # print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday))
            start = time.time()
            for hh_inst in hh_insts:
                start = time.time()
                # print("day " + str(day) + ", res id: " + str(hh_inst["id"]) + ", is_hh: " + str(hh_inst["is_hh"]))
                itinerary_util.generate_working_days_for_week_residence(hh_inst["resident_uids"], hh_inst["is_hh"])
                time_taken = time.time() - start
                working_schedule_times_by_resid[hh_inst["id"]] = time_taken
                num_agents_working_schedule += len(hh_inst["resident_uids"])

            time_taken = time.time() - start
            print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(node_worker_index))

        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday))
        start = time.time()                    

        num_agents_itinerary = 0
        itinerary_times_by_resid = {}
        for hh_inst in hh_insts:
            res_start = time.time()
            itinerary_util.generate_local_itinerary(day, weekday, hh_inst["resident_uids"])
            res_timetaken = time.time() - res_start
            itinerary_times_by_resid[hh_inst["id"]] = res_timetaken
            num_agents_itinerary += len(hh_inst["resident_uids"])

        # if itinerary_util.epi_util.contact_tracing_agent_ids is not None and len(itinerary_util.epi_util.contact_tracing_agent_ids) > 0:
        #     itinerary_util.sync_queue.put(["v", None, "contact_tracing_agent_ids", itinerary_util.epi_util.contact_tracing_agent_ids])

        time_taken = time.time() - start
        # itinerary_sum_time_taken += time_taken
        # avg_time_taken = itinerary_sum_time_taken / day
        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", ws_interprocess_time: " + str(itinerary_util.working_schedule_interprocess_communication_aggregated_time) + ", itin_interprocess_time: " + str(itinerary_util.itinerary_interprocess_communication_aggregated_time) + ", proc index: " + str(node_worker_index))
        
        print("process " + str(node_worker_index) + ", ended at " + str(time.time()))

        working_schedule_times_by_resid_ordered_keys = sorted(working_schedule_times_by_resid, key=working_schedule_times_by_resid.get, reverse=True)
        itinerary_times_by_resid_ordered_keys = sorted(itinerary_times_by_resid, key=itinerary_times_by_resid.get, reverse=True)

        working_schedule_times_by_resid_ordered = {key: working_schedule_times_by_resid[key] for key in working_schedule_times_by_resid_ordered_keys}
        itinerary_times_by_resid_ordered = {key: itinerary_times_by_resid[key] for key in itinerary_times_by_resid_ordered_keys}
        
        interventions_totals = [itinerary_util.new_tests, itinerary_util.new_vaccinations, itinerary_util.new_quarantined, itinerary_util.new_hospitalised]
        # for i in range(1000):
        #     if i in agents_dynamic:
        #         if agents_dynamic[i]["itinerary"] is None or len(agents_dynamic[i]["itinerary"]) == 0:
        #             print("itinerary_mp, itinerary is empty " + str(i))
        main_time_taken = time.time() - main_start

        return node_worker_index, it_agents, agents_epi, vars_util_mp, interventions_totals, working_schedule_times_by_resid_ordered, itinerary_times_by_resid_ordered, num_agents_working_schedule, num_agents_itinerary, main_time_taken
    except Exception as e:
       # log on the node where it happened
        actual_stack_trace_log_file_name = stack_trace_log_file_name.replace(".txt", "_actual.txt")

        with open(actual_stack_trace_log_file_name, 'w') as fi:
            traceback.print_exc(file=fi)

        return {"exception": e, "traceback": traceback.format_exc(), "logfilename": stack_trace_log_file_name}
    finally:
        gc.collect()       

        if process_counter is not None:
            process_counter.value -= 1

        if f is not None:
            # util.log_memory_usage(f, "End: ")

            # Close the file
            f.close()

        if original_stdout is not None:
            sys.stdout = original_stdout

# def sync_state_info(sync_queue, process_counter, cpu_affinity=None, agents_main_temp=None, vars_util_main_temp=None):
#     from multiprocessing import queues

#     cpu_affinity = None # temporarily disabled
#     if cpu_affinity is not None:
#         import psutil

#         p = psutil.Process()
#         print(f"Sync State Thread/Process Child #{cpu_affinity}: {p}, affinity {p.cpu_affinity()}", flush=True)
#         time.sleep(0.0001)
#         p.cpu_affinity([cpu_affinity])

#         print(f"Sync State Thread/Process Child #{cpu_affinity}: Set my affinity to {cpu_affinity}, affinity now {p.cpu_affinity()}", flush=True)

#     if agents_main_temp is None:
#         global agents_main
#         global vars_util_main
#     else:
#         agents_main = agents_main_temp
#         vars_util_main = vars_util_main_temp

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
#                     if attr_name != "vars":
#                         vars_util_main.update(attr_name, value)
#                     else:
#                         agent_seir_state_transition_for_day, agent_seir_state, agent_infection_type, agent_infection_severity = value
#                         vars_util_main.update("agents_seir_state_transition_for_day", agent_seir_state_transition_for_day)
#                         vars_util_main.update("agents_seir_state", agent_seir_state)
#                         vars_util_main.update("agents_infection_type", agent_infection_type)
#                         vars_util_main.update("agents_infection_severity", agent_infection_severity)

#         except queues.Empty:
#             continue  # Queue is empty, continue polling
    
#     sync_time_end = time.time()
#     time_taken = sync_time_end - start
#     print("itinerary state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))

# def sync_state_info_mpi(comm, process_counter, cpu_affinity=None, agents_main_temp=None, vars_util_main_temp=None):
#     cpu_affinity = None # temporarily disabled
#     if cpu_affinity is not None:
#         import psutil

#         p = psutil.Process()
#         print(f"Sync State Thread/Process Child #{cpu_affinity}: {p}, affinity {p.cpu_affinity()}", flush=True)
#         time.sleep(0.0001)
#         p.cpu_affinity([cpu_affinity])

#         print(f"Sync State Thread/Process Child #{cpu_affinity}: Set my affinity to {cpu_affinity}, affinity now {p.cpu_affinity()}", flush=True)

#     if agents_main_temp is None:
#         global agents_main
#         global vars_util_main
#     else:
#         agents_main = agents_main_temp
#         vars_util_main = vars_util_main_temp

#     start = time.time()
    
#     # Receive results from worker processes
#     while process_counter.value > 0 or comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
#         probe = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
#         if probe:
#             message = comm.recv()

#             if message is None:
#                 print("none message")
#             else:
#                 print("valid message")
        
#         time.sleep(0.001)

#     sync_time_end = time.time()
#     time_taken = sync_time_end - start
#     print("itinerary state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))
