import sys
sys.path.insert(0, '~/AppsPy/mtdcovabm/simulator')

import dask
import asyncio
# from queue import Empty
import threading
import numpy as np
import traceback
import itinerary, vars
import time
import util
from copy import copy, deepcopy
from dask.distributed import get_worker, as_completed

def localitinerary_parallel(client,
                            day,
                            weekday,
                            weekdaystr,
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
                            proc_use_pool=0,
                            sync_use_threads=False,
                            sync_use_queue=False,
                            keep_processes_open=True,
                            log_file_name="output.txt"): 
    stack_trace_log_file_name = ""
    try:
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_dask_stack_trace" + ".txt"

        # process_counter = manager.Value("i", num_processes)

        vars_util.reset_cells_agents_timesteps()
        
        if num_processes > 1:
            start = time.time()
            mp_hh_inst_indices = util.split_residences_by_weight(hh_insts, num_processes)
            time_taken = time.time() - start
            print("split residences by indices (load balancing): " + str(time_taken))
            
            start = time.time()

            dask_params = []

            for process_index in range(num_processes):
                # cells_partial = {}
                hh_insts_partial = []

                mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

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

                print("starting process index " + str(process_index) + " at " + str(time.time()))

                # Define parameters
                params = (day, 
                        weekday, 
                        weekdaystr, 
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
                        process_index,
                        log_file_name)             

                dask_params.append(params)

            start = time.time()
            delayed_computations = [dask.delayed(localitinerary_worker)(dask_params[i]) for i in range(num_processes)]
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

                process_index, agents_partial, vars_util_partial, working_schedule_times_by_resid_ordered, itinerary_times_by_resid_ordered, num_agents_ws, num_agents_it = result

                print("processing results for worker " + str(process_index) + ". num agents ws: " + str(num_agents_ws) + ", num agents it: " + str(num_agents_it))

                mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

                hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc] 

                for hh_inst in hh_insts_partial:
                    agents_dynamic, vars_util = util.sync_state_info_by_agentsids(hh_inst["resident_uids"], agents_dynamic, vars_util, agents_partial, vars_util_partial)

                vars_util = util.sync_state_info_sets(vars_util, vars_util_partial)

                start_cat = time.time()
                
                vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)

                time_taken_cat = time.time() - start_cat
                print("cells_agents_timesteps sync for process {0}, time taken: {1}".format(process_index, str(time_taken_cat)))

                time_taken = time.time() - start
                print("syncing results back with main process. time taken " + str(time_taken))

            if not keep_processes_open:
                start = time.time()
                client.shutdown()
                time_taken = time.time() - start
                print("client shutdown time taken " + str(time_taken))         
        else:
            # params = sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            params = day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_dynamic, agents_ids_by_ages, vars_util, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, log_file_name
            localitinerary_worker(params)
    except:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_worker(params):
    # from shared_mp import agents_static
    # import sys
    # sys.path.insert(0, '~/AppsPy/mtdcovabm/simulator')

    import os

    process_index = -1
    original_stdout = None
    f = None
    stack_trace_log_file_name = ""

    try:  
        # sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_itinerary, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params
        day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_static, agents_dynamic, agents_ids_by_ages, vars_util_mp, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, log_file_name = params
        
        original_stdout = sys.stdout
        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_mp_stack_trace_" + str(process_index) + ".txt"
        log_file_name = log_file_name.replace(".txt", "") + "_it_" + str(process_index) + ".txt"
        f = open(log_file_name, "w")
        sys.stdout = f

        # implementation for static agents via dask plugins
        print("interpreter: " + os.path.dirname(sys.executable))
        print("current working directory: " + os.getcwd())

        worker = get_worker()

        plugin = worker.plugins['read-only-data']
        persons = plugin.persons

        persons_len = ""
        if persons is not None:
            persons_len = len(persons)
        else:
            persons_len = "None"

        print("persons len: " + str(persons_len))
        print("actual persons len: " + str(plugin.persons_len))

        print("cells_agents_timesteps in process " + str(process_index) + " is " + str(id(vars_util_mp.cells_agents_timesteps)))
        print(f"Itinerary Worker Child #{process_index+1} at {str(time.time())}", flush=True)

        itinerary_util = itinerary.Itinerary(itineraryparams,
                                            timestepmins, 
                                            n_locals, 
                                            n_tourists, 
                                            locals_ratio_to_full_pop, 
                                            agents_static,
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
                                            process_index)

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
            print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(process_index))

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
        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", ws_interprocess_time: " + str(itinerary_util.working_schedule_interprocess_communication_aggregated_time) + ", itin_interprocess_time: " + str(itinerary_util.itinerary_interprocess_communication_aggregated_time) + ", proc index: " + str(process_index))
        
        print("process " + str(process_index) + ", ended at " + str(time.time()))

        working_schedule_times_by_resid_ordered_keys = sorted(working_schedule_times_by_resid, key=working_schedule_times_by_resid.get, reverse=True)
        itinerary_times_by_resid_ordered_keys = sorted(itinerary_times_by_resid, key=itinerary_times_by_resid.get, reverse=True)

        working_schedule_times_by_resid_ordered = {key: working_schedule_times_by_resid[key] for key in working_schedule_times_by_resid_ordered_keys}
        itinerary_times_by_resid_ordered = {key: itinerary_times_by_resid[key] for key in itinerary_times_by_resid_ordered_keys}

        return process_index, agents_dynamic, vars_util_mp, working_schedule_times_by_resid_ordered, itinerary_times_by_resid_ordered, num_agents_working_schedule, num_agents_itinerary
    except:
        with open(stack_trace_log_file_name, 'w') as f: # it_mp_stack_trace.txt
            traceback.print_exc(file=f)
    finally:
        # process_counter.value -= 1

        if original_stdout is not None:
            sys.stdout = original_stdout

            if f is not None:
                # Close the file
                f.close()

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
