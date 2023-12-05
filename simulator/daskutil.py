import time
import traceback
import util
from dask.distributed import as_completed

def handle_futures(day, futures, it_agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, extra_params=False, log_timings=False, dask_full_array_mapping=False, f=None, multiprocessing=False, workers_remote_time_taken=None, processes_remote_time_taken=None):
    future_count = 0

    if not multiprocessing:
        results = as_completed(futures)
    else:
        results = futures
    
    if workers_remote_time_taken is None:
        workers_remote_time_taken = {}

    for future in results:
        itinerary, itinerary_daskmp, contactnetwork, contactnetwork_daskmp, contacttracing = False, False, False, False, False
        try:
            if not multiprocessing:
                result = future.result()
            else:
                result = future

            remote_worker_index = -1
            remote_time_taken = None

            if type(result) is not dict: # success
                if not extra_params:
                    if len(result) == 2:
                        it_agents_partial_result, vars_util_partial_result = result
                    elif len(result) == 3:
                        if type(result[2]) is not float:
                            remote_worker_index, it_agents_partial_result, vars_util_partial_result = result
                        else:
                            remote_worker_index, agents_epi_partial_result, remote_time_taken = result
                            contacttracing = True  
                    elif len(result) == 4:
                        remote_worker_index, agents_epi_partial_result, vars_util_partial_result, remote_time_taken = result

                        if type(remote_time_taken) is not dict:
                            contactnetwork = True      
                        else:
                            contactnetwork_daskmp = True
                    else:
                        remote_worker_index, it_agents_partial_result, agents_epi_partial_result, vars_util_partial_result, remote_time_taken = result
                        itinerary_daskmp = True
                else:
                    remote_worker_index, it_agents_partial_result, agents_epi_partial_result, vars_util_partial_result, _, _, _, _, remote_time_taken = result

                if remote_worker_index == -1 and log_timings:
                    remote_worker_index = future_count

                if remote_time_taken is not None and workers_remote_time_taken is not None:
                    if not itinerary_daskmp and not contactnetwork_daskmp:
                        workers_remote_time_taken[remote_worker_index] = remote_time_taken
                    else:
                        if processes_remote_time_taken is not None:
                            processes_remote_time_taken = remote_time_taken

                        workers_remote_time_taken[remote_worker_index] = processes_remote_time_taken[-1]
                        del processes_remote_time_taken[-1]

                itinerary = not contactnetwork and not contacttracing and not itinerary_daskmp and not contactnetwork_daskmp
                    
                if itinerary or itinerary_daskmp: 
                    it_agents, agents_epi, vars_util = sync_results_it(day, it_agents, agents_epi, vars_util, it_agents_partial_result, agents_epi_partial_result, vars_util_partial_result, remote_worker_index, remote_time_taken, f)
                elif contactnetwork or contactnetwork_daskmp:
                    agents_epi, vars_util = sync_results_cn(day, agents_epi, vars_util, agents_epi_partial_result, vars_util_partial_result, remote_worker_index, remote_time_taken, f)
                else:
                    agents_epi = sync_results_ct(day, agents_epi, agents_epi_partial_result, remote_worker_index, remote_time_taken, f)

                it_agents_partial_result, agents_epi_partial_result, vars_util_partial_result = None, None, None
            else:
                exception_info = result

                with open(exception_info["logfilename"], "a") as f:
                    f.write(f"Exception Type: {exception_info['type']}\n")
                    f.write(f"Exception Message: {exception_info['message']}\n")
                    f.write(f"Traceback: {exception_info['traceback']}\n")

                ex_type = type(exception_info['type'], (Exception,), {})
                new_ex = ex_type(exception_info['message'])
                setattr(new_ex, '__traceback__', exception_info['traceback'])
                raise new_ex
        except:
            with open(task_results_stack_trace_log_file_name, 'a') as f:
                traceback.print_exc(file=f)

            raise
        finally:
            if not multiprocessing and not itinerary_daskmp and not contactnetwork_daskmp:
                future.release()
            
            future_count += 1

    return it_agents, agents_epi, vars_util, workers_remote_time_taken, processes_remote_time_taken, exception_info

def handle_futures_batches(day, futures, agents, agents_epi, vars_util, task_results_stack_trace_log_file_name, extra_params=False, log_timings=False, dask_full_array_mapping=False):
    future_count = 0

    main_log_file_name = task_results_stack_trace_log_file_name.replace(".txt", "_as_completed.txt")
    
    for batch in as_completed(futures, with_results=True).batches():
        try:
            for future, result in batch:
                try:
                    vars_util_partial_result = vars.Vars()

                    if not extra_params:
                        agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result = result
                        # agents_dynamic_partial_result, vars_util_partial_result.cells_agents_timesteps, vars_util_partial_result.agents_seir_state, vars_util_partial_result.agents_infection_type, vars_util_partial_result.agents_infection_severity, vars_util_partial_result.agents_seir_state_transition_for_day, vars_util_partial_result.contact_tracing_agent_ids = result
                    else:
                        _, agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result, _, _, _, _ = result

                    worker_process_index = -1
                    if log_timings:
                        worker_process_index = future_count
                        
                    agents, agents_epi, vars_util = sync_results_it(day, agents, agents_epi, vars_util, agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result, worker_process_index)
            
                    agents_dynamic_partial_result, agents_epi_partial_result, vars_util_partial_result = None, None, None
                except:
                    with open(task_results_stack_trace_log_file_name, 'a') as f:
                        traceback.print_exc(file=f)
                finally:
                    future.release()
                    future_count += 1
        except:
            with open(main_log_file_name, 'a') as f:
                    traceback.print_exc(file=f)

    return agents, agents_epi, vars_util
    
def sync_results(day, process_index, mp_hh_inst_indices, hh_insts, agents, agents_epi, vars_util, agents_partial, agents_epi_partial, vars_util_partial):
    if agents_partial is not None and len(agents_partial) > 0:
        mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

        hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc] 

        resident_uids = []
        for hh_inst in hh_insts_partial:
            resident_uids.extend(hh_inst["resident_uids"])

        agents, agents_epi, vars_util = util.sync_state_info_by_agentsids(resident_uids, agents, agents_epi, vars_util, agents_partial, agents_epi_partial, vars_util_partial)

        vars_util = util.sync_state_info_sets(day, vars_util, vars_util_partial)

        start_cat = time.time()
        
        vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)

        time_taken_cat = time.time() - start_cat
        print("cells_agents_timesteps sync for process {0}, time taken: {1}".format(process_index, str(time_taken_cat)))

    return agents, agents_epi, vars_util

def sync_results_it(day, agents, agents_epi, vars_util, agents_partial, agents_epi_partial, vars_util_partial, worker_process_index=-1, remote_time_taken=None, f=None):
    agentsids_start = time.time()
    agents, agents_epi, vars_util = util.sync_state_info_by_agentsids(list(agents_epi_partial.keys()), agents, agents_epi, vars_util, agents_partial, agents_epi_partial, vars_util_partial)
    agentsids_time_taken = time.time() - agentsids_start

    sets_start = time.time()
    vars_util = util.sync_state_info_sets(day, vars_util, vars_util_partial)
    sets_time_taken = time.time() - sets_start

    start_cat = time.time()
    vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)
    cat_time_taken = time.time() - start_cat

    rtt_str = ""
    if remote_time_taken is not None:
        rtt_str = str(remote_time_taken)

    if worker_process_index != -1:
        print("worker {0}: remote time_taken {1}, agents sync {2}, sets sync {3}, cat sync {4}".format(str(worker_process_index), rtt_str, str(agentsids_time_taken), str(sets_time_taken), str(cat_time_taken)))

        if f is not None:
            f.flush()

    return agents, agents_epi, vars_util

def sync_results_cn(day, agents_epi, vars_util, agents_epi_partial, vars_util_partial, worker_process_index=-1, remote_time_taken=None, f=None):
    agentsids_start = time.time()
    agents_epi, vars_util = util.sync_state_info_by_agentsids_cn(list(agents_epi_partial.keys()), agents_epi, vars_util, agents_epi_partial, vars_util_partial)
    agentsids_time_taken = time.time() - agentsids_start

    sets_start = time.time()
    vars_util = util.sync_state_info_sets(day, vars_util, vars_util_partial)
    sets_time_taken = time.time() - sets_start

    start_cat = time.time()
    vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)
    cat_time_taken = time.time() - start_cat

    rtt_str = ""
    if remote_time_taken is not None:
        rtt_str = str(remote_time_taken)

    if worker_process_index != -1:
        print("worker {0}: remote time_taken {1}, agents sync {2}, sets sync {3}, cat sync {4}".format(str(worker_process_index), rtt_str, str(agentsids_time_taken), str(sets_time_taken), str(cat_time_taken)))
        
        if f is not None:
            f.flush()

    return agents_epi, vars_util

def sync_results_ct(day, agents_epi, agents_epi_partial, worker_process_index=-1, remote_time_taken=None, f=None):
    agentsids_start = time.time()
    agents_epi = util.sync_state_info_by_agentsids_ct(list(agents_epi_partial.keys()), agents_epi, agents_epi_partial)
    agentsids_time_taken = time.time() - agentsids_start

    rtt_str = ""
    if remote_time_taken is not None:
        rtt_str = str(remote_time_taken)

    if worker_process_index != -1:
        # print("worker {0}: remote time_taken {1}, agents sync {2}, sets sync {3}, cat sync {4}".format(str(worker_process_index), rtt_str, str(agentsids_time_taken), str(sets_time_taken), str(cat_time_taken)))
        print("worker {0}, day {1}: remote time_taken {2}, agents sync {3}".format(str(worker_process_index), str(day), rtt_str, str(agentsids_time_taken)))
        
        if f is not None:
            f.flush()

    return agents_epi