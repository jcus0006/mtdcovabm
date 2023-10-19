import time
import traceback
import util
from dask.distributed import as_completed

def handle_futures(futures, agents_dynamic, vars_util, task_results_stack_trace_log_file_name, extra_params=False, log_timings=False, dask_full_array_mapping=False):
    future_count = 0
    for future in as_completed(futures):
        try:
            result = future.result()

            if not extra_params:
                agents_dynamic_partial_result, vars_util_partial_result = result
            else:
                _, agents_dynamic_partial_result, vars_util_partial_result, _, _, _, _ = result

            worker_process_index = -1
            if log_timings:
                worker_process_index = future_count
                
            agents_dynamic, vars_util = sync_results_res(agents_dynamic, vars_util, agents_dynamic_partial_result, vars_util_partial_result, worker_process_index)
        
            agents_dynamic_partial_result, vars_util_partial_result = None, None
        except:
            with open(task_results_stack_trace_log_file_name, 'a') as f:
                traceback.print_exc(file=f)
        finally:
            future.release()
            future_count += 1

    return agents_dynamic, vars_util

def handle_futures_batches(futures, agents_dynamic, vars_util, task_results_stack_trace_log_file_name, extra_params=False, log_timings=False, dask_full_array_mapping=False):
    future_count = 0

    main_log_file_name = task_results_stack_trace_log_file_name.replace(".txt", "_as_completed.txt")
    
    for batch in as_completed(futures, with_results=True).batches():
        try:
            for future, result in batch:
                try:
                    vars_util_partial_result = vars.Vars()

                    if not extra_params:
                        agents_dynamic_partial_result, vars_util_partial_result = result
                        # agents_dynamic_partial_result, vars_util_partial_result.cells_agents_timesteps, vars_util_partial_result.agents_seir_state, vars_util_partial_result.agents_infection_type, vars_util_partial_result.agents_infection_severity, vars_util_partial_result.agents_seir_state_transition_for_day, vars_util_partial_result.contact_tracing_agent_ids = result
                    else:
                        _, agents_dynamic_partial_result, vars_util_partial_result, _, _, _, _ = result

                    worker_process_index = -1
                    if log_timings:
                        worker_process_index = future_count
                        
                    agents_dynamic, vars_util = sync_results_res(agents_dynamic, vars_util, agents_dynamic_partial_result, vars_util_partial_result, worker_process_index)
            
                    agents_dynamic_partial_result, vars_util_partial_result = None, None
                except:
                    with open(task_results_stack_trace_log_file_name, 'a') as f:
                        traceback.print_exc(file=f)
                finally:
                    future.release()
                    future_count += 1
        except:
            with open(main_log_file_name, 'a') as f:
                    traceback.print_exc(file=f)

    return agents_dynamic, vars_util
    
def sync_results(process_index, mp_hh_inst_indices, hh_insts, agents_dynamic, vars_util, agents_partial, vars_util_partial):
    if agents_partial is not None and len(agents_partial) > 0:
        mp_hh_inst_ids_this_proc = mp_hh_inst_indices[process_index]

        hh_insts_partial = [hh_insts[index] for index in mp_hh_inst_ids_this_proc] 

        resident_uids = []
        for hh_inst in hh_insts_partial:
            resident_uids.extend(hh_inst["resident_uids"])

        agents_dynamic, vars_util = util.sync_state_info_by_agentsids(resident_uids, agents_dynamic, vars_util, agents_partial, vars_util_partial)

        vars_util = util.sync_state_info_sets(vars_util, vars_util_partial)

        start_cat = time.time()
        
        vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)

        time_taken_cat = time.time() - start_cat
        print("cells_agents_timesteps sync for process {0}, time taken: {1}".format(process_index, str(time_taken_cat)))

    return agents_dynamic, vars_util

def sync_results_res(agents_dynamic, vars_util, agents_partial, vars_util_partial, worker_process_index=-1):
    agentsids_start = time.time()
    agents_dynamic, vars_util = util.sync_state_info_by_agentsids(list(agents_partial.keys()), agents_dynamic, vars_util, agents_partial, vars_util_partial)
    agentsids_time_taken = time.time() - agentsids_start

    if worker_process_index > -1:
        print("agentsids sync in process " + str(worker_process_index) + ", time_taken: " + str(agentsids_time_taken))

    sets_start = time.time()
    vars_util = util.sync_state_info_sets(vars_util, vars_util_partial)
    sets_time_taken = time.time() - sets_start

    if worker_process_index > -1:
        print("sets sync in process " + str(worker_process_index) + ", time_taken: " + str(sets_time_taken))

    start_cat = time.time()
    vars_util = util.sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial)
    time_taken_cat = time.time() - start_cat

    if worker_process_index > -1:
        print("cat sync in process " + str(worker_process_index) + ", time_taken: " + str(sets_time_taken))

    return agents_dynamic, vars_util