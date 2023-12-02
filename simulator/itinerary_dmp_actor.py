import sys
import os
import time
import traceback
from dask.distributed import get_worker
import multiprocessing as mp
from copy import copy, deepcopy
import util, daskutil, shared_mp, customdict, vars, itinerary

class ItineraryDMPActor:
    def __init__(self, params):
        worker = get_worker()

        numprocesses, workerindex = params

        self.num_processes = numprocesses
        self.worker_index = workerindex
        self.manager = mp.Manager()
        self.pool = mp.Pool(processes=numprocesses, initializer=shared_mp.init_pool_processes_dask_mp, initargs=(worker,))

    def run_parallel(self, params):
        main_start = time.time()

        day, weekday, weekdaystr, hh_insts, it_agents, agents_epi, vars_util, dynparams, log_file_name = params

        folder_name = ""
        if log_file_name != "output.txt":
            folder_name = os.path.dirname(log_file_name)
        else:
            folder_name = os.getcwd()

        worker = get_worker()
        agents_ids_by_ages = worker.data["agents_ids_by_ages"]

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_main_mp_stack_trace_" + str(day) + ".txt"
        task_results_stack_trace_log_file_name = os.path.join(folder_name, "it_main_res_task_results_stack_trace_" + str(day) + ".txt")

        mp_hh_inst_indices = util.split_residences_by_weight(hh_insts, self.num_processes) # to do - for the time being this assumes equal split but we may have more information about the cores of the workers

        imap_params = []
        for process_index in range(self.num_processes):
            # cells_partial = {}
            hh_insts_partial = []

            mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

            start_partial = time.time()
            hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc]      

            agents_partial, agents_ids_by_ages_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict(), customdict.CustomDict() # {}, {}, {}
            vars_util_partial = vars.Vars()
            vars_util_partial.agents_seir_state = vars_util.agents_seir_state
            vars_util_partial.cells_agents_timesteps = customdict.CustomDict()

            for hh_inst in hh_insts_partial:
                agents_partial, agents_ids_by_ages_partial, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, agents_ids_by_ages, agents_ids_by_ages_partial, True, False, agents_epi, agents_epi_partial)                  

            time_taken_partial = time.time() - start_partial

            remote_index = (self.worker_index, process_index)
            params = (day, 
                    weekday, 
                    weekdaystr, 
                    hh_insts_partial, 
                    agents_partial,
                    agents_epi_partial,
                    deepcopy(vars_util_partial), # deepcopy(vars_util_partial) - without this was causing some nasty issues with multiprocessing (maybe with Dask it is fine)
                    dynparams, 
                    remote_index, 
                    log_file_name)  
            
            imap_params.append(params)

        imap_results = self.pool.imap(run_itinerary, imap_params)

        workers_remote_time_taken = {}
        workers_remote_time_taken[-1] = 0

        it_agents, agents_epi, vars_util, workers_remote_time_taken, _ = daskutil.handle_futures(day, imap_results, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, True, True, False, None, True, workers_remote_time_taken)
            
        main_time_taken = time.time() - main_start
        workers_remote_time_taken[-1] = main_time_taken

        return self.worker_index, it_agents, agents_epi, vars_util, workers_remote_time_taken
    
def run_itinerary(params):
    f = None
    stack_trace_log_file_name = ""
    # original_stdout = sys.stdout

    try:
        main_start = time.time()

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

        day, weekday, weekdaystr, hh_insts, agents_dynamic, agents_epi, vars_util_mp, dyn_params, node_worker_index, log_file_name = params

        stack_trace_log_file_name = log_file_name.replace(".txt", "") + "_it_mp_stack_trace_" + str(day) + "_" + str(node_worker_index) + ".txt"
        log_file_name = log_file_name.replace(".txt", "") + "_it_" + str(day) + "_" + str(node_worker_index) + ".txt"

        # f = open(log_file_name, "w")
        # sys.stdout = f

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

        start = time.time()                    

        num_agents_itinerary = 0
        itinerary_times_by_resid = {}
        for hh_inst in hh_insts:
            res_start = time.time()
            itinerary_util.generate_local_itinerary(day, weekday, hh_inst["resident_uids"])
            res_timetaken = time.time() - res_start
            itinerary_times_by_resid[hh_inst["id"]] = res_timetaken
            num_agents_itinerary += len(hh_inst["resident_uids"])

        time_taken = time.time() - start
        # itinerary_sum_time_taken += time_taken
        # avg_time_taken = itinerary_sum_time_taken / day
        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", ws_interprocess_time: " + str(itinerary_util.working_schedule_interprocess_communication_aggregated_time) + ", itin_interprocess_time: " + str(itinerary_util.itinerary_interprocess_communication_aggregated_time) + ", proc index: " + str(node_worker_index))
        
        print("process " + str(node_worker_index) + ", ended at " + str(time.time()))

        main_time_taken = time.time() - main_start

        return node_worker_index, agents_dynamic, agents_epi, vars_util_mp, None, None, num_agents_working_schedule, num_agents_itinerary, main_time_taken
    except Exception as e:
        traceback_str = traceback.format_exc()

        exception_info = {"processindex": node_worker_index, 
                          "logfilename": stack_trace_log_file_name, 
                          "type": type(e).__name__, 
                          "message": str(e), 
                          "traceback": traceback_str}
        
        return exception_info
    finally:
        if f is not None:
            # Close the file
            f.close()

        # sys.stdout = original_stdout