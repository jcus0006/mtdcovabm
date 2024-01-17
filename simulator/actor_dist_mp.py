import sys
import os
import time
import platform
import traceback
import numpy as np
import numpy.ma as ma
from dask.distributed import get_worker
import multiprocessing as mp
from copy import copy, deepcopy
import util, daskutil, shared_mp, customdict, vars, itinerary, contactnetwork, tourism_dist
from util import MethodType
import psutil
import gc

class ActorDistMP:
    def __init__(self, params):
        worker = get_worker()

        numprocesses, workerindex, logsubfoldername, logfilename = params

        current_directory = os.getcwd()
        subfolder_name = logfilename.replace(".txt", "")
        log_subfolder_path = os.path.join(current_directory, logsubfoldername, subfolder_name)

        # # Create the subfolder if it doesn't exist
        # if not os.path.exists(log_subfolder_path):
        #     os.makedirs(log_subfolder_path)
        # else:
        #     if workerindex > 0: # don't delete the sub folder from the client machine (this assumes worker 0 is the client)
        #         shutil.rmtree(log_subfolder_path)
        #         os.makedirs(log_subfolder_path)

        self.folder_name = log_subfolder_path

        self.num_processes = numprocesses
        self.worker_index = workerindex
        self.manager = mp.Manager()
        self.pool = mp.Pool(processes=numprocesses, initializer=shared_mp.init_pool_processes_dask_mp, initargs=(worker,))

    def close_pool(self):
        self.pool.close()
        self.pool.join()
        self.manager.shutdown()

    def run_itinerary_parallel(self, params):
        stack_trace_log_file_name = ""
        f = None
        original_stdout = sys.stdout

        try:
            main_start = time.time()

            day, weekday, weekdaystr, hh_insts, it_agents, agents_epi, vars_util, dynparams, log_file_name = params

            worker = get_worker()
            agents_ids_by_ages = worker.data["agents_ids_by_ages"]

            stack_trace_log_file_name = os.path.join(self.folder_name, "it_main_mp_stack_trace_" + str(day) + ".txt")
            task_results_stack_trace_log_file_name = os.path.join(self.folder_name, "it_main_res_task_results_stack_trace_" + str(day) + ".txt")
            log_file_name = os.path.join(self.folder_name, "it_main_mp_" + str(day) + ".txt")

            f = open(log_file_name, "w")
            sys.stdout = f

            # util.log_memory_usage(f, "Before assigning work. ")

            mp_hh_inst_indices = util.split_residences_by_weight(hh_insts, self.num_processes) # to do - for the time being this assumes equal split but we may have more information about the cores of the workers

            workers_remote_time_taken = {}
            workers_remote_time_taken[-1] = 0

            imap_params = []
            for process_index in range(self.num_processes):
                start = time.time()

                # cells_partial = {}
                hh_insts_partial = []

                mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

                start_partial = time.time()
                hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc]      

                agents_partial, agents_ids_by_ages_partial, agents_epi_partial = customdict.CustomDict(), customdict.CustomDict(), customdict.CustomDict() # {}, {}, {}
                vars_util_partial = vars.Vars()
                # vars_util_partial.agents_seir_state = vars_util.agents_seir_state
                vars_util_partial.cells_agents_timesteps = customdict.CustomDict()

                for hh_inst in hh_insts_partial:
                    agents_partial, agents_ids_by_ages_partial, vars_util_partial, agents_epi_partial = util.split_dicts_by_agentsids(hh_inst["resident_uids"], it_agents, vars_util, agents_partial, vars_util_partial, agents_ids_by_ages, agents_ids_by_ages_partial, True, False, agents_epi, agents_epi_partial)                  

                # remote_index = (self.worker_index, process_index)
                params = (day, 
                        weekday, 
                        weekdaystr, 
                        hh_insts_partial, 
                        agents_partial,
                        agents_epi_partial,
                        deepcopy(vars_util_partial), # deepcopy(vars_util_partial) - without this was causing some nasty issues with multiprocessing (maybe with Dask it is fine)
                        dynparams, 
                        process_index, 
                        self.folder_name,
                        log_file_name)  
                
                imap_params.append(params)

                time_taken = time.time() - start

                workers_remote_time_taken[process_index] = [time_taken, None]

            imap_results = self.pool.imap(run_itinerary_single, imap_params)

            # util.log_memory_usage(f, "After assigning work. ")

            it_agents, agents_epi, vars_util, workers_remote_time_taken, _ = daskutil.handle_futures(MethodType.ItineraryMP, day, imap_results, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, True, True, False, None, workers_remote_time_taken)
                
            # util.log_memory_usage(f, "After syncing results. ")

            main_time_taken = time.time() - main_start
            workers_remote_time_taken[-1] = main_time_taken

            return self.worker_index, it_agents, agents_epi, vars_util, workers_remote_time_taken
        except Exception as e:
            # log on the node where it happened
            actual_stack_trace_log_file_name = stack_trace_log_file_name.replace(".txt", "_actual.txt")

            with open(actual_stack_trace_log_file_name, 'w') as fi:
                traceback.print_exc(file=fi)

            return {"exception": e, "traceback": traceback.format_exc(), "logfilename": stack_trace_log_file_name}
        finally:
            gc.collect()

            if f is not None:
                f.close()

            sys.stdout = original_stdout
    
    def run_contactnetwork_parallel(self, params):
        stack_trace_log_file_name = ""
        f = None
        original_stdout = sys.stdout

        try:
            main_start = time.time()

            day, weekday, agents_epi, vars_util, dynparams, log_file_name = params

            stack_trace_log_file_name = os.path.join(self.folder_name, "cn_main_mp_stack_trace_" + str(day) + ".txt")
            task_results_stack_trace_log_file_name = os.path.join(self.folder_name, "cn_main_mp_task_results_stack_trace_" + str(day) + ".txt")
            log_file_name = os.path.join(self.folder_name, "cn_main_mp_" + str(day) + ".txt")

            f = open(log_file_name, "w")
            sys.stdout = f

            # util.log_memory_usage(f, "Before assigning work. ")

            cells_agents_timesteps_dicts = util.split_cellsagentstimesteps_balanced(vars_util.cells_agents_timesteps, self.num_processes)

            workers_remote_time_taken = {}
            workers_remote_time_taken[-1] = 0

            imap_params = []
            for process_index in range(self.num_processes):
                start = time.time()

                cells_agents_timesteps_partial = customdict.CustomDict()
                agents_partial = customdict.CustomDict()
                vars_util_partial = vars.Vars()

                # vars_util_partial.agents_seir_state = vars_util.agents_seir_state  

                cells_agents_timesteps_partial = cells_agents_timesteps_dicts[process_index]

                unique_agent_ids = set()
                for cell_vals in cells_agents_timesteps_partial.values():
                    for cell_agent_timesteps in cell_vals:
                        unique_agent_ids.add(cell_agent_timesteps[0])

                unique_agent_ids = sorted(list(unique_agent_ids))

                agents_partial, _, vars_util_partial, _ = util.split_dicts_by_agentsids(unique_agent_ids, agents_epi, vars_util, agents_partial, vars_util_partial, is_dask_task=False)

                # mask = np.isin(np.arange(len(vars_util_partial.agents_seir_state)), unique_agent_ids, invert=True)        
                # vars_util_partial.agents_seir_state = ma.masked_array(vars_util_partial.agents_seir_state, mask=mask)
        
                vars_util_partial.cells_agents_timesteps = cells_agents_timesteps_partial

                params = (day, weekday, agents_partial, vars_util_partial, dynparams, process_index, self.folder_name, log_file_name)
                
                imap_params.append(params)

                time_taken = time.time() - start

                workers_remote_time_taken[process_index] = [time_taken, None]

            imap_results = self.pool.imap(run_contactnetwork_single, imap_params)
            
            # util.log_memory_usage(f, "After assigning work. ")

            _, agents_epi, vars_util, workers_remote_time_taken, _ = daskutil.handle_futures(MethodType.ContactNetworkMP, day, imap_results, None, agents_epi, vars_util, task_results_stack_trace_log_file_name, False, True, False, None, workers_remote_time_taken)
                
            # util.log_memory_usage(f, "After syncing results. ")

            # certain data does not have to go back because it would not have been updated in this context
            vars_util.cells_agents_timesteps = customdict.CustomDict()
            # vars_util.agents_seir_state_transition_for_day = customdict.CustomDict()
            vars_util.agents_vaccination_doses = customdict.CustomDict()

            # util.log_memory_usage(f, "Before returning. ")

            main_time_taken = time.time() - main_start
            workers_remote_time_taken[-1] = main_time_taken

            return self.worker_index, agents_epi, vars_util, workers_remote_time_taken
        except Exception as e:
            # log on the node where it happened
            actual_stack_trace_log_file_name = stack_trace_log_file_name.replace(".txt", "_actual.txt")

            with open(actual_stack_trace_log_file_name, 'w') as fi:
                traceback.print_exc(file=fi)

            return {"exception": e, "traceback": traceback.format_exc(), "logfilename": stack_trace_log_file_name}
        finally:
            gc.collect()

            if f is not None:
                f.close()

            sys.stdout = original_stdout
    
    def run_update_tourist_data_remote(self, params):
        return tourism_dist.update_tourist_data_remote(params, self.folder_name)

def run_itinerary_single(params):
    f = None
    stack_trace_log_file_name = ""
    worker_index = None
    original_stdout = sys.stdout

    try:
        pre_mem_info = psutil.virtual_memory()

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

        day, weekday, weekdaystr, hh_insts, agents_dynamic, agents_epi, vars_util_mp, dyn_params, worker_index, folder_name, log_file_name = params

        stack_trace_log_file_name = os.path.join(folder_name, "it_mp_stack_trace_" + str(day) + "_" + str(worker_index) + ".txt")
        log_file_name = os.path.join(folder_name, "it_mp_" + str(day) + "_" + str(worker_index) + ".txt")

        f = open(log_file_name, "w")
        sys.stdout = f

        # util.log_memory_usage(f, "Pre global memory important. ", pre_mem_info)
        # util.log_memory_usage(f, "Before processing itinerary. ")

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
                                            process_index=worker_index)
        
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
            print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(worker_index))

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
        print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", ws_interprocess_time: " + str(itinerary_util.working_schedule_interprocess_communication_aggregated_time) + ", itin_interprocess_time: " + str(itinerary_util.itinerary_interprocess_communication_aggregated_time) + ", proc index: " + str(worker_index))
        
        print("process " + str(worker_index) + ", ended at " + str(time.time()))

        main_time_taken = time.time() - main_start

        # util.log_memory_usage(f, "After processing itinerary. ")

        return worker_index, agents_dynamic, agents_epi, vars_util_mp, None, None, num_agents_working_schedule, num_agents_itinerary, main_time_taken
    except Exception as e:
        raise
    finally:
        gc.collect()

        if f is not None:
            # Close the file
            f.close()

        sys.stdout = original_stdout

def run_contactnetwork_single(params):
    f = None
    stack_trace_log_file_name = ""
    worker_index = None
    original_stdout = sys.stdout

    try:
        pre_mem_info = psutil.virtual_memory()

        main_start = time.time()

        from shared_mp import n_locals
        from shared_mp import n_tourists
        from shared_mp import locals_ratio_to_full_pop
        from shared_mp import contactnetworkparams
        from shared_mp import epidemiologyparams
        from shared_mp import cells_type
        from shared_mp import indids_by_cellid
        from shared_mp import cells_households
        from shared_mp import cells_institutions
        from shared_mp import cells_accommodation
        from shared_mp import agents_static

        day, weekday, agents_epi, vars_util, dyn_params, worker_index, folder_name, log_file_name = params

        stack_trace_log_file_name = os.path.join(folder_name, "cn_mp_stack_trace_" + str(day) + "_" + str(worker_index) + ".txt")
        log_file_name = os.path.join(folder_name, "cn_mp_" + str(day) + "_" + str(worker_index) + ".txt")

        f = open(log_file_name, "w")
        sys.stdout = f

        # util.log_memory_usage(f, "Pre global memory important. ", pre_mem_info)
        # util.log_memory_usage(f, "Before processing contact network. ")

        contact_network_util = contactnetwork.ContactNetwork(n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            agents_static,
                                                            agents_epi,
                                                            vars_util,
                                                            cells_type,
                                                            indids_by_cellid,
                                                            cells_households, 
                                                            cells_institutions, 
                                                            cells_accommodation, 
                                                            contactnetworkparams,
                                                            epidemiologyparams, 
                                                            dyn_params, 
                                                            process_index=worker_index)

        _, _, agents_epi, vars_util = contact_network_util.simulate_contact_network(day, weekday)
        
        # util.log_memory_usage(f, "After processing contact network. ") 

        # certain data does not have to go back because it would not have been updated in this context
        vars_util.cells_agents_timesteps = customdict.CustomDict()
        # vars_util.agents_seir_state_transition_for_day = customdict.CustomDict()
        vars_util.agents_vaccination_doses = customdict.CustomDict()

        main_time_taken = time.time() - main_start

        # util.log_memory_usage(f, "After cleaning data structures. ")

        return worker_index, agents_epi, vars_util, main_time_taken
    except Exception as e:
        raise
    finally:
        gc.collect()
        
        if f is not None:
            # Close the file
            f.close()

        sys.stdout = original_stdout