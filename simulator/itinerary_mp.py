import multiprocessing as mp
import multiprocessing.shared_memory as shm
import threading
import numpy as np
import traceback
from simulator import itinerary
import time
import sys
import copy

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
                            num_processes=4):
    try:
        global agents_main
        global vars_util_main

        agents_main = agents
        vars_util_main = vars_util

        manager = mp.Manager()
        sync_queue = manager.Queue()
        process_counter = manager.Value("i", num_processes)

        vars_util.reset_cells_agents_timesteps()

        if num_processes > 1:
            # pool = mp.Pool(initializer=init_worker, initargs=(process_counter,))
            pool = mp.Pool()

            mp_hh_inst_indices = []

            hh_inst_indices = [i for i, _ in enumerate(hh_insts)]
            np.random.shuffle(hh_inst_indices)
            mp_hh_inst_indices = np.array_split(hh_inst_indices, num_processes)

            for process_index in range(num_processes):
                # cells_partial = {}
                hh_insts_partial = []

                mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

                for index in mp_hh_inst_ids_this_proc:
                    # cell = cells[cell_key]
                    hh_inst = hh_insts[index]

                    # cells_partial[cell_key] = cell
                    hh_insts_partial.append(hh_inst)

                print("starting process index " + str(process_index) + " at " + str(time.time()))
                # pool.apply_async(localitinerary_worker, args=((sync_queue, day, weekday, weekdaystr, hh_insts_partial, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, process_index, process_counter),))
                pool.apply_async(localitinerary_worker, args=((sync_queue, day, weekday, weekdaystr, hh_insts_partial, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_ids_by_ages, vars_util, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, process_index, process_counter),))
            
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
            
            # sync_time_end = time.time()
            # time_taken = sync_time_end - start
            # print("itinerary state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))
        

            # option 1 - single thread
            # lock = threading.Lock()
            # sync_state_info(sync_queue, process_counter, None)

            # option 2 - multiple threads
            # # Create multiple threads to process items from the queue
            # lock = threading.Lock()
            # num_threads = 6
            # threads = []
            # for _ in range(num_threads):
            #     t = threading.Thread(target=sync_state_info, args=(sync_queue, process_counter, lock))
            #     t.start()
            #     threads.append(t)

            # # Wait for all threads to complete
            # for t in threads:
            #     t.join()

            # option 3 - multiple processes
            lock = mp.Lock()
            processes = []
            for process_index in range(6):
                process = mp.Process(target=sync_state_info, args=(sync_queue, process_counter, lock))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            start = time.time()
            pool.close()
            time_taken = time.time() - start
            print("pool close time taken " + str(time_taken))

            start = time.time()
            pool.join()
            time_taken = time.time() - start
            print("pool join time taken " + str(time_taken))
        else:
            # params = sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            params = sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_ids_by_ages, vars_util, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            localitinerary_worker(params)
    except:
        with open('it_main_mp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_worker(params):
    try:
        print("process started " + str(time.time()))
    
        # sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_itinerary, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params
        sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_ids_by_ages, vars_util, tourists, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params

        print("process " + str(process_index))

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
                                            tourists_active_ids, 
                                            sync_queue)

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

        # agents_mp.convert_from_shared_memory_dynamic() # this will have to be called from multiple processes when parallelised
        for hh_inst in hh_insts:
            itinerary_util.generate_local_itinerary(day, weekday, hh_inst["resident_uids"])

        if itinerary_util.epi_util.contact_tracing_agent_ids is not None and len(itinerary_util.epi_util.contact_tracing_agent_ids) > 0:
            itinerary_util.sync_queue.put(["v", None, "contact_tracing_agent_ids", itinerary_util.epi_util.contact_tracing_agent_ids])

        time_taken = time.time() - start
        # itinerary_sum_time_taken += time_taken
        # avg_time_taken = itinerary_sum_time_taken / day
        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(process_index))
        
        print("process " + str(process_index) + ", ended at " + str(time.time()))
    except:
        with open('it_mp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)
    finally:
        process_counter.value -= 1

def sync_state_info(sync_queue, process_counter, lock):
    global agents_main
    global vars_util_main

    start = time.time()
    while process_counter.value > 0 or not sync_queue.empty(): # True
        try:
            type, index, attr_name, value = None, None, None, None
            with lock:
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
                    vars_util_main.update(attr_name, value)
        except sync_queue.Empty:
            continue  # Queue is empty, continue polling
    
    sync_time_end = time.time()
    time_taken = sync_time_end - start
    print("itinerary state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))
