import multiprocessing as mp
import multiprocessing.shared_memory as shm
import concurrent.futures
# from queue import Empty
import threading
import numpy as np
import traceback
from simulator import itinerary, vars
import time
import sys
import copy
import psutil
from mpi4py import MPI

agents_main = None
vars_util_main = None
def localitinerary_parallel(day,
                            weekday,
                            weekdaystr,
                            itineraryparams,
                            timestepmins,
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop,
                            agents,
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
                            proc_use_pool=0,
                            sync_use_threads=False,
                            sync_use_queue=False): 
    try:
        # p = psutil.Process()
        # print(f"main process #{0}: {p}, affinity {p.cpu_affinity()}", flush=True)
        # time.sleep(0.0001)
        # p.cpu_affinity([0])

        global agents_main
        global vars_util_main

        agents_main = agents
        vars_util_main = vars_util

        manager = mp.Manager()
        # sync_queue = manager.Queue()
        process_counter = manager.Value("i", num_processes)

        vars_util.reset_cells_agents_timesteps()

        # Initialize MPI
        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        # size = comm.Get_size()
        # print("MPI rank {0}, size {1}".format(rank, size))
        
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
            pool = mp.Pool()

            mp_hh_inst_indices = []

            hh_inst_indices = [i for i, _ in enumerate(hh_insts)]
            np.random.shuffle(hh_inst_indices)
            mp_hh_inst_indices = np.array_split(hh_inst_indices, num_processes)

            start = time.time()

            proc_processes = []
            proc_futures = []
            imap_params = []
            imap_results = []

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

                agents_partial, agents_ids_by_ages_partial = {}, {}
                vars_util_partial = vars.Vars()
                vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                vars_util_partial.agents_vaccination_doses = vars_util.agents_vaccination_doses

                for hh_inst in hh_insts_partial:
                    for uid in hh_inst["resident_uids"]:
                        agents_partial[uid] = agents[uid]
                        agents_ids_by_ages_partial[uid] = agents_ids_by_ages[uid]

                        if uid in vars_util.agents_seir_state_transition_for_day:
                            vars_util_partial.agents_seir_state_transition_for_day[uid] = vars_util.agents_seir_state_transition_for_day[uid]

                        if uid in vars_util.agents_infection_type:
                            vars_util_partial.agents_infection_type[uid] = vars_util.agents_infection_type[uid]

                        if uid in vars_util.agents_infection_severity:
                            vars_util_partial.agents_infection_severity[uid] = vars_util.agents_infection_severity[uid]

                tourists_active_ids = []
                tourists = None # to do - to handle

                # agents_partial = {uid:agents[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}
                # agents_ids_by_ages_partial = {uid:agents_ids_by_ages[uid] for hh_inst in hh_insts_partial for uid in hh_inst["resident_uids"]}

                time_taken_partial = time.time() - start_partial
                print("creating partial dicts. time taken: " + str(time_taken_partial))

                print("starting process index " + str(process_index) + " at " + str(time.time()))

                # Define parameters
                params = (day, weekday, weekdaystr, hh_insts_partial, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_partial, agents_ids_by_ages_partial, vars_util, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, process_index, process_counter)
                
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
                elif proc_use_pool == 3:
                    imap_params.append(params)

                if proc_use_pool == 0 or proc_use_pool == 2:
                    # clean up local memory
                    hh_insts_partial = None
                    agents_partial = None
                    agents_ids_by_ages_partial = None

            if proc_use_pool == 3:
                imap_results = pool.imap(localitinerary_worker, imap_params)

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

            if proc_use_pool == 3:
                start = time.time()

                for result in imap_results:
                    process_index, agents_partial, vars_util_partial = result

                    print("processing results for process " + str(process_index))

                    mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

                    hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc] 

                    for hh_inst in hh_insts_partial:
                        for uid in hh_inst["resident_uids"]:
                            agents[uid] = agents_partial[uid]

                            if uid in vars_util_partial.agents_seir_state_transition_for_day:
                                vars_util.agents_seir_state[uid] = vars_util_partial.agents_seir_state[uid] # not partial
                                vars_util.agents_seir_state_transition_for_day[uid] = vars_util_partial.agents_seir_state_transition_for_day[uid]
                                vars_util.agents_infection_type = vars_util_partial.agents_infection_type[uid]
                                vars_util.agents_infection_severity = vars_util_partial.agents_infection_severity[uid]

                    if len(vars_util_partial.contact_tracing_agent_ids) > 0:
                        vars_util.contact_tracing_agent_ids.update(vars_util_partial.contact_tracing_agent_ids)

                    if len(vars_util_partial.directcontacts_by_simcelltype_by_day) > 0:
                        vars_util.directcontacts_by_simcelltype_by_day.update(vars_util_partial.directcontacts_by_simcelltype_by_day)

                    start_cat = time.time()
                    for cellid, agents_timesteps in vars_util_partial.cells_agents_timesteps.items():
                        if cellid not in vars_util.cells_agents_timesteps:
                            vars_util.cells_agents_timesteps[cellid] = []

                        vars_util.cells_agents_timesteps[cellid] += agents_timesteps

                    time_taken_cat = time.time() - start_cat
                    print("cells_agents_timesteps sync for process {0}, time taken: {1}".format(process_index, str(time_taken_cat)))

                time_taken = time.time() - start
                print("syncing pool imap results back with main process. time taken " + str(time_taken))

            start = time.time()

            if proc_use_pool == 0 or proc_use_pool == 3: # multiprocessing.pool -> apply_async or imap
                pool.close()
                pool.join()
            elif proc_use_pool == 1:
                for process in proc_processes:
                    process.join()
            elif proc_use_pool == 2:
                # Collect and print the results
                for future in concurrent.futures.as_completed(proc_futures):
                    result = future.result()

                    if result is not None:
                        print("result: " + str(result))

            manager.shutdown()
            time_taken = time.time() - start
            print("pool/processes close/join time taken " + str(time_taken))
                
        else:
            # params = sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            params = day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_ids_by_ages, vars_util, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            localitinerary_worker(params)
    except:
        print("it_main crash")

        with open('it_main_mp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_worker(params):
    process_index = -1
    try:  
        # sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_itinerary, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params
        day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_ids_by_ages, vars_util, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params

        import psutil
        p = psutil.Process()
        print(f"Itinerary Worker Child #{process_index+1}: {p}, affinity {p.cpu_affinity()}", flush=True)
        time.sleep(0.0001)
        p.cpu_affinity([process_index+1])

        # while True:
        #     time.sleep(0.001)

        print(f"Itinerary Worker Child #{process_index+1}: Set my affinity to {process_index+1}, affinity now {p.cpu_affinity()}", flush=True)

        # print("process started " + str(time.time()))
        # print("process " + str(process_index))

        # agents_mp_itinerary.convert_from_shared_memory_readonly(itinerary=True)
        # agents_mp_itinerary.convert_from_shared_memory_dynamic(itinerary=True)

        itinerary_util = itinerary.Itinerary(itineraryparams,
                                            timestepmins, 
                                            n_locals, 
                                            n_tourists, 
                                            locals_ratio_to_full_pop, 
                                            agents, 
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
                                            dyn_params,
                                            tourists_active_ids)

        if day == 1 or weekdaystr == "Monday":
            # print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday))
            start = time.time()
            for hh_inst in hh_insts:
                # print("day " + str(day) + ", res id: " + str(hh_inst["id"]) + ", is_hh: " + str(hh_inst["is_hh"]))
                itinerary_util.generate_working_days_for_week_residence(hh_inst["resident_uids"], hh_inst["is_hh"])

            time_taken = time.time() - start
            print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(process_index))

        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday))
        start = time.time()                    

        for hh_inst in hh_insts:
            itinerary_util.generate_local_itinerary(day, weekday, hh_inst["resident_uids"])

        # if itinerary_util.epi_util.contact_tracing_agent_ids is not None and len(itinerary_util.epi_util.contact_tracing_agent_ids) > 0:
        #     itinerary_util.sync_queue.put(["v", None, "contact_tracing_agent_ids", itinerary_util.epi_util.contact_tracing_agent_ids])

        time_taken = time.time() - start
        # itinerary_sum_time_taken += time_taken
        # avg_time_taken = itinerary_sum_time_taken / day
        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", ws_interprocess_time: " + str(itinerary_util.working_schedule_interprocess_communication_aggregated_time) + ", itin_interprocess_time: " + str(itinerary_util.itinerary_interprocess_communication_aggregated_time) + ", proc index: " + str(process_index))
        
        print("process " + str(process_index) + ", ended at " + str(time.time()))

        return process_index, agents, vars_util
    except:
        print("it_process crash in process {proc_index}".format(process_index))

        filename = "it_mp_stack_trace_{proc_index}.txt".format(process_index)

        with open(filename, 'w') as f:
            traceback.print_exc(file=f)
    finally:
        process_counter.value -= 1

def sync_state_info(sync_queue, process_counter, cpu_affinity=None, agents_main_temp=None, vars_util_main_temp=None):
    from multiprocessing import queues

    cpu_affinity = None # temporarily disabled
    if cpu_affinity is not None:
        import psutil

        p = psutil.Process()
        print(f"Sync State Thread/Process Child #{cpu_affinity}: {p}, affinity {p.cpu_affinity()}", flush=True)
        time.sleep(0.0001)
        p.cpu_affinity([cpu_affinity])

        print(f"Sync State Thread/Process Child #{cpu_affinity}: Set my affinity to {cpu_affinity}, affinity now {p.cpu_affinity()}", flush=True)

    if agents_main_temp is None:
        global agents_main
        global vars_util_main
    else:
        agents_main = agents_main_temp
        vars_util_main = vars_util_main_temp

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
                    if attr_name != "vars":
                        vars_util_main.update(attr_name, value)
                    else:
                        agent_seir_state_transition_for_day, agent_seir_state, agent_infection_type, agent_infection_severity = value
                        vars_util_main.update("agents_seir_state_transition_for_day", agent_seir_state_transition_for_day)
                        vars_util_main.update("agents_seir_state", agent_seir_state)
                        vars_util_main.update("agents_infection_type", agent_infection_type)
                        vars_util_main.update("agents_infection_severity", agent_infection_severity)

        except queues.Empty:
            continue  # Queue is empty, continue polling
    
    sync_time_end = time.time()
    time_taken = sync_time_end - start
    print("itinerary state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))

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
