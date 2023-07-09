import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import traceback
from simulator import itinerary
from simulator.agents_mp import Agents
from simulator.cells_mp import Cells, CellType, CellSubType
import time
import sys
import copy

def localitinerary_parallel(day,
                            weekday,
                            weekdaystr,
                            itineraryparams,
                            timestepmins,
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop,
                            agents_mp,
                            agents_mp_it,
                            vars_mp,
                            cat_util,
                            tourists, 
                            industries,
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
        manager = mp.Manager()
        sync_queue = manager.Queue()
        process_counter = manager.Value("i", num_processes)

        cat_util.reset_cells_agents_timesteps()

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
                pool.apply_async(localitinerary_worker, args=((sync_queue, day, weekday, weekdaystr, hh_insts_partial, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, process_index, process_counter),))
            
            start = time.time()
            while process_counter.value > 0 or not sync_queue.empty(): # True
                try:
                    type, index, attr_name, value = sync_queue.get(timeout=0.001)  # Poll the queue with a timeout (0.01 / 0 might cause problems)

                    if type == "a":
                        agents_mp.set(index, attr_name, value)
                    elif type == "c":
                        cat_util.update_cells_agents_timesteps(index, value)
                    elif type == "v":
                        vars_mp.update(attr_name, value)
                except mp.queues.Empty:
                    continue  # Queue is empty, continue polling
            
            sync_time_end = time.time()
            time_taken = sync_time_end - start
            print("itinerary state info sync. time taken " + str(time_taken) + ", ended at " + str(sync_time_end))
            
            start = time.time()
            pool.close()
            time_taken = time.time() - start
            print("pool close time taken " + str(time_taken))

            start = time.time()
            pool.join()
            time_taken = time.time() - start
            print("pool join time taken " + str(time_taken))

            start = time.time()
            agents_mp_it.cleanup_shared_memory_dynamic(itinerary=True)
            time_taken = time.time() - start
            print("agents_mp_it clean up time taken " + str(time_taken))
        else:
            # params = sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            params = sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_it, tourists, industries, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dynparams, tourists_active_ids, -1, process_counter
            localitinerary_worker(params)
    except:
        with open('it_main_mp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)

def localitinerary_worker(params):
    try:
        print("process started " + str(time.time()))
    
        # sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_itinerary, tourists, industries, cells_breakfast_by_accomid, cells_entertainment, cells_mp, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params
        sync_queue, day, weekday, weekdaystr, hh_insts, itineraryparams, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_itinerary, tourists, industries, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, tourist_entry_infection_probability, epidemiologyparams, dyn_params, tourists_active_ids, process_index, process_counter = params

        print("process " + str(process_index))

        agents_mp_itinerary.convert_from_shared_memory_readonly(itinerary=True)
        agents_mp_itinerary.convert_from_shared_memory_dynamic(itinerary=True)

        # start = time.time()
        # cells_mp.convert_from_shared_memory_readonly()
        # time_taken = time.time() - start
        # print("cells_mp convert_to_shared_memory_readonly time taken " + str(time_taken))

        # start = time.time()
        # cells_restaurants = cells_mp.get_keys(sub_type=CellSubType.Restaurant)
        # cells_hospital = cells_mp.get_keys(type=CellType.Hospital)
        # cells_testinghub = cells_mp.get_keys(sub_type=CellSubType.TestingHub)
        # cells_vaccinationhub = cells_mp.get_keys(sub_type=CellSubType.VaccinationHub)
        # cells_religious = cells_mp.get_keys(type=CellType.Religion)
        # cells_households = cells_mp.get_keys(type=CellType.Household)
        # cells_airport = cells_mp.get_keys(type=CellType.Airport)
        # cells_transport = cells_mp.get_keys(type=CellType.Transport)
        # cells_institutions = cells_mp.get_keys(type=CellType.Institution)
        # cells_accommodation = cells_mp.get_keys(type=CellType.Accommodation)

        # cress = sys.getsizeof(cells_restaurants)
        # chs = sys.getsizeof(cells_hospital)
        # cths = sys.getsizeof(cells_testinghub)
        # cvhs = sys.getsizeof(cells_vaccinationhub)
        # crels = sys.getsizeof(cells_religious)
        # chhs = sys.getsizeof(cells_households)
        # cas = sys.getsizeof(cells_airport)
        # cts = sys.getsizeof(cells_transport)
        # cis = sys.getsizeof(cells_institutions)
        # caccs = sys.getsizeof(cells_accommodation)

        # alls = cress + chs + cths + cvhs + crels + chhs + cas + cts + cis + caccs

        # print(f"Total size {alls}, Rest: {cress}, Hosp: {chs}, TestHub: {cths}, Rel: {crels}, House: {chhs}, Airport: {cas}, Transport: {cts}, Institutions: {cis}, Accom: {caccs}")

        # time_taken = time.time() - start
        # print("cells types generation (itinerary) time taken " + str(time_taken))

        itinerary_util = itinerary.Itinerary(itineraryparams, 
                                            timestepmins, 
                                            n_locals, 
                                            n_tourists, 
                                            locals_ratio_to_full_pop, 
                                            agents_mp_itinerary, 
                                            tourists, 
                                            industries, 
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