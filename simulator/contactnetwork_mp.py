import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import traceback
from simulator import contactnetwork
from simulator.agents import Agents
import time
# proc_counter = None

# def init_worker(process_counter):
#     global proc_counter

#     proc_counter = process_counter

def contactnetwork_parallel(day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp, agents_mp_cn, agents_directcontacts_by_simcelltype_by_day, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, tourists_active_ids, cells, cells_households, cells_institutions, cells_accommodation, cells_agents_timesteps, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, num_processes=4):
    manager = mp.Manager()
    result_queue = manager.Queue()
    process_counter = manager.Value("i", num_processes)

    if num_processes > 1:
        # pool = mp.Pool(initializer=init_worker, initargs=(process_counter,))
        pool = mp.Pool()

        mp_cells_keys = []

        cells_agents_timesteps_keys = list(cells_agents_timesteps.keys())
        np.random.shuffle(cells_agents_timesteps_keys)
        mp_cells_keys = np.array_split(cells_agents_timesteps_keys, num_processes)

        for process_index in range(num_processes):
            # cells_partial = {}
            cells_agents_timesteps_partial = {}

            cells_keys = mp_cells_keys[process_index]

            for cell_key in cells_keys:
                # cell = cells[cell_key]
                cell_agents_timesteps = cells_agents_timesteps[cell_key]

                # cells_partial[cell_key] = cell
                cells_agents_timesteps_partial[cell_key] = cell_agents_timesteps

            pool.apply_async(contactnetwork_worker, args=((result_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, agents_directcontacts_by_simcelltype_by_day, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, cells, cells_agents_timesteps_partial, tourists_active_ids, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter),))

        # update memory from multiprocessing.queue
        
        start = time.time()
        while process_counter.value > 0 or not result_queue.empty(): # True
            try:
                agent_index, attr_name, value = result_queue.get(timeout=1)  # Poll the queue with a timeout (0 for now - might cause problems)

                agents_mp.set(agent_index, attr_name, value)
            except mp.queues.Empty:
                continue  # Queue is empty, continue polling

        time_taken = time.time() - start
        print("cn/epi state info sync. time taken " + str(time_taken))
        
        start = time.time()
        # Close the pool of processes

        # close shm?
        # agents_mp_cn = None

        pool.close()
        pool.join()

        time_taken = time.time() - start
        print("closing pool. time taken " + str(time_taken))
    else:
        params = result_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, agents_directcontacts_by_simcelltype_by_day, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, cells, cells_agents_timesteps, tourists_active_ids, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, -1, process_counter

        contactnetwork_worker(params)

def contactnetwork_worker(params):
    # still to see how to handle (whether mp.Array or local in process and then sync with main memory)
    # in the case of Agents, these are purely read only however, 
    #  - using mp.dict will introduce speed degradation from locking, 
    #  - passing large coll as param creates overhead from ser/deserialisation, and
    #  - splitting the dict into smaller pieces may be impossible
    # agents, agents_seir_state, agent_seir_state_transition_for_day, agents_infection_type, agents_infection_severity
    # and to check more params from Epidemiology ctor (especially optional params)

    try:
        result_queue, day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, agents_directcontacts_by_simcelltype_by_day, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, cells, cell_agents_timesteps, tourists_active_ids, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index, process_counter = params

        # agents_mp_cn = None
        # if process_index == -1:
        #     agents_mp_cn = agents_temp
        # else:
        #     shared_memory_names = agents_temp

        #     agents_mp_cn = Agents()
        #     agents_mp_cn.convert_from_shared_memory(shared_memory_names)

        agents_mp_cn.convert_from_shared_memory_readonly(contactnetwork=True)
        agents_mp_cn.convert_from_shared_memory_dynamic(contactnetwork=True)

        contact_network_util = contactnetwork.ContactNetwork(n_locals, n_tourists, locals_ratio_to_full_pop, agents_mp_cn, agents_directcontacts_by_simcelltype_by_day, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, cells, cell_agents_timesteps, tourists_active_ids, cells_households, cells_institutions, cells_accommodation, contactnetworkparams, epidemiologyparams, dynparams, contact_network_sum_time_taken, process_index=process_index, result_queue=result_queue)

        contact_network_util.simulate_contact_network(day, weekday)
        
        # agents_mp_cn = None
        # global proc_counter
        
    except:
        with open('cn_mp_stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)
    finally:
        process_counter.value -= 1