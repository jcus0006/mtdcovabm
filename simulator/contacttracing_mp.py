import numpy as np
import multiprocessing as mp
import time
import traceback
from simulator import util, vars
from simulator.epidemiology import Epidemiology
from simulator import agents as agents_util

def contacttracing_parallel(manager,
                            pool,
                            day,
                            epidemiologyparams, 
                            n_locals, 
                            n_tourists, 
                            locals_ratio_to_full_pop, 
                            agents, 
                            vars_util,
                            cells_households, 
                            cells_institutions, 
                            cells_accommodation, 
                            dyn_params, 
                            num_processes=10,
                            num_threads=2,
                            keep_processes_open=True):
    process_counter = manager.Value("i", num_processes)
    
    start = time.time()
    ct_agents = agents_util.initialize_agents_dict_ct(manager, agents)
    time_taken = time.time() - start
    print("initialize_agents_dict_ct, time_taken: " + str(time_taken))

    if num_processes > 1:
        start = time.time()
        contact_tracing_agent_ids_list = list(vars_util.contact_tracing_agent_ids)
        contacttracingagentids_indices = [i for i, _ in enumerate(contact_tracing_agent_ids_list)]
        np.random.shuffle(contacttracingagentids_indices)
        mp_contacttracingagentids_indices = np.array_split(contacttracingagentids_indices, num_processes)
        time_taken = time.time() - start
        print("splitting contact_tracing_agent_ids, time_taken: " + str(time_taken))

        start = time.time()
        directcontacts_by_simcelltype_by_day_list = list(vars_util.directcontacts_by_simcelltype_by_day)
        directcontacts_by_simcelltype_by_day_indices = [i for i, _ in enumerate(directcontacts_by_simcelltype_by_day_list)]
        np.random.shuffle(directcontacts_by_simcelltype_by_day_indices)
        mp_directcontacts_by_simcelltype_by_day_indices = np.array_split(directcontacts_by_simcelltype_by_day_indices, num_processes)
        time_taken = time.time() - start
        print("splitting directcontacts_by_simcelltype_by_day, time_taken: " + str(time_taken))

        imap_params, imap_results = [], []

        for process_index in range(num_processes):
            agents_partial = {}
            vars_util_partial = vars.Vars()
            
            proc_contacttracingagentids_indices = mp_contacttracingagentids_indices[process_index]
            proc_directcontact_by_simcelltype_by_day_indices = mp_directcontacts_by_simcelltype_by_day_indices[process_index]

            start = time.time()
            vars_util_partial.contact_tracing_agent_ids = set([contact_tracing_agent_ids_list[index] for index in proc_contacttracingagentids_indices]) # simply to retain type
            time_taken = time.time() - start
            print("partialising contact_tracing_agent_ids, process: " + str(process_index) + ", time taken: " + str(time_taken))

            if len(vars_util_partial.contact_tracing_agent_ids) > 0:
                start = time.time()
                vars_util_partial.directcontacts_by_simcelltype_by_day = set([directcontacts_by_simcelltype_by_day_list[index] for index in proc_directcontact_by_simcelltype_by_day_indices]) # simply to retain type
                time_taken = time.time() - start
                print("partialising directcontacts_by_simcelltype_by_day, process: " + str(process_index) + ", time taken: " + str(time_taken))

                agents_partial = ct_agents

                # unique_agent_ids = set()
                # for cell_vals in cells_agents_timesteps_partial.values():
                #     for cell_agent_timesteps in cell_vals:
                #         unique_agent_ids.add(cell_agent_timesteps[0])

                # unique_agent_ids = list(unique_agent_ids)

                # agents_partial, _, vars_util_partial = util.split_dicts_by_agentsids(unique_agent_ids, agents, None, vars_util, agents_partial, agents_ids_by_ages_partial, vars_util_partial)

                print("starting process index " + str(process_index) + " at " + str(time.time()))

                params = (day,
                        epidemiologyparams, 
                        n_locals, 
                        n_tourists, 
                        locals_ratio_to_full_pop, 
                        agents_partial, 
                        vars_util_partial,
                        cells_households, 
                        cells_institutions, 
                        cells_accommodation, 
                        dyn_params,
                        process_index,
                        process_counter)
                
                imap_params.append(params)
                # pool.apply_async(contactnetwork_worker, args=(params,))

        imap_results = pool.imap(contacttracing_worker, imap_params)

        start = time.time()

        for result in imap_results:
            process_index, updated_agent_ids, agents_partial, vars_util_partial = result

            print("processing results for process " + str(process_index))

            agents, vars_util = util.sync_state_info_by_agentsids(updated_agent_ids, agents, vars_util, agents_partial, vars_util_partial, contact_tracing=True)

        time_taken = time.time() - start
        print("syncing pool imap results back with main process. time taken " + str(time_taken))
        
        if not keep_processes_open:
            start = time.time()
            pool.close()
            manager.shutdown()
            time_taken = time.time() - start
            print("pool close time taken " + str(time_taken))

            start = time.time()
            pool.join()
            time_taken = time.time() - start
            print("pool join time taken " + str(time_taken))
    else:
        params = day, epidemiologyparams, n_locals, n_tourists, locals_ratio_to_full_pop, ct_agents, vars_util, cells_households, cells_institutions, cells_accommodation, dyn_params, -1, process_counter

        contacttracing_worker(params)

def contacttracing_worker(params):
    try:
        start = time.time()
        print("process started " + str(start))

        # sync_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, cell_agents_timesteps, tourists_active_ids, cells_mp, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params
        day, epidemiologyparams, n_locals, n_tourists, locals_ratio_to_full_pop, agents, vars_util, cells_households, cells_institutions, cells_accommodation, dyn_params, process_index, process_counter = params

        print("process " + str(process_index))

        epidemiology_util = Epidemiology(epidemiologyparams, 
                                            n_locals,
                                            n_tourists,
                                            locals_ratio_to_full_pop,
                                            agents,
                                            vars_util,
                                            cells_households,
                                            cells_institutions,
                                            cells_accommodation,
                                            dyn_params,
                                            process_index)

        process_index, updated_agent_ids, agents_partial, vars_util = epidemiology_util.contact_tracing(day)
        
        ended = time.time()
        time_taken = ended - start
        print("process " + str(process_index) + ", ended at " + str(ended) + ", time_taken: " + str(time_taken))

        return process_index, updated_agent_ids, agents_partial, vars_util
    except:
        with open('ct_mp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)
    finally:
        process_counter.value -= 1