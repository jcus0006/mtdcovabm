import os
import sys
import traceback
from dask.distributed import get_worker
import gc

# agents_static_to_sync are tourists who are arriving today
# departed_tourist_agent_ids are tourists who left the previous day
# sync_dynamic_agents is never True; handling was retained but not used
def update_tourist_data_remote(params, folder_name=None, dask_worker=None, it_agents=None, agents_epi=None, vars_util=None, f=None, self_actor_index=None, tourists=None, touristsgroups=None, n_locals=None):
    f = None
    stack_trace_log_file_name = ""
    original_stdout = sys.stdout

    try:
        sync_dynamic_agents = it_agents is not None
        departed_tourist_group_ids = []

        if tourists is not None: # dask_full_stateful case
            if folder_name is not None:
                day, agents_static_to_sync, departed_tourist_agent_ids, departed_tourist_group_ids, process_index = params
            else:
                day, agents_static_to_sync, departed_tourist_agent_ids, departed_tourist_group_ids, logfoldername, logfilename, process_index = params
        else:
            if not sync_dynamic_agents:
                if folder_name is not None:
                    day, agents_static_to_sync, departed_tourist_agent_ids, process_index = params
                else:
                    day, agents_static_to_sync, departed_tourist_agent_ids, logfoldername, logfilename, process_index = params
            else:
                if folder_name is not None:
                    day, agents_static_to_sync, it_agents_to_sync, agents_epi_to_sync, vars_util_to_sync, departed_tourist_agent_ids, process_index = params
                else:
                    day, agents_static_to_sync, it_agents_to_sync, agents_epi_to_sync, vars_util_to_sync, departed_tourist_agent_ids, logfoldername, logfilename, process_index = params

        if folder_name is None:
            current_directory = os.getcwd()
            subfolder_name = logfilename.replace(".txt", "")
            folder_name = os.path.join(current_directory, logfoldername, subfolder_name)
        
        if self_actor_index is None:
            stack_trace_log_file_name = os.path.join(folder_name, "utd_dist_stack_trace_" + str(day) + "_" + str(process_index) + ".txt")
            log_file_name = os.path.join(folder_name, "utd_dist_" + str(day) + "_" + str(process_index) + ".txt")
        else:
            stack_trace_log_file_name = os.path.join(folder_name, "utd_dist_stack_trace_" + str(day) + "_" + str(self_actor_index) + "_" + str(process_index) + ".txt")
            log_file_name = os.path.join(folder_name, "utd_dist_" + str(day) + "_" + str(self_actor_index) + "_" + str(process_index) + ".txt")

        f = open(log_file_name, "w")
        sys.stdout = f

        if dask_worker is None:
            dask_worker = get_worker()
        
        agents_static = dask_worker.data["agents_static"]

        print("asts {0}, dep {1}".format(str(len(agents_static_to_sync)), str(len(departed_tourist_agent_ids))))
        if f is not None:
            f.flush()

        for agentid, staticinfo in agents_static_to_sync.items():
            age, res_cellid, age_bracket_index, epi_age_bracket_index, pub_transp_reg, soc_rate = staticinfo

            # print("saving soc_rate {0} for agentid {1}".format(str(soc_rate), str(agentid)))

            if not agents_static.use_tourists_dict:
                agents_static.set(agentid, "age", age)
                agents_static.set(agentid, "res_cellid", res_cellid)
                agents_static.set(agentid, "age_bracket_index", age_bracket_index)
                agents_static.set(agentid, "epi_age_bracket_index", epi_age_bracket_index)
                agents_static.set(agentid, "pub_transp_reg", pub_transp_reg)
                agents_static.set(agentid, "soc_rate", soc_rate)

                # print(f"adding to shm array. agentid {agentid}, age {age}, res_cellid: {res_cellid}, abi: {age_bracket_index}, epi_abi: {epi_age_bracket_index}, pub_transp_reg: {pub_transp_reg}, soc_rate: {soc_rate}")
            else:
                props = {"age": age, "res_cellid": res_cellid, "age_bracket_index": age_bracket_index, "epi_age_bracket_index": epi_age_bracket_index, "pub_transp_reg": pub_transp_reg, "soc_rate": soc_rate}
                agents_static.set_props(agentid, props)
                # print(f"adding to dict. age {age}, agentid {agentid}, res_cellid: {res_cellid}, abi: {age_bracket_index}, epi_abi: {epi_age_bracket_index}, pub_transp_reg: {pub_transp_reg}, soc_rate: {soc_rate}")

            if sync_dynamic_agents:
                it_agents[agentid] = it_agents_to_sync[agentid]
                agents_epi[agentid] = agents_epi_to_sync[agentid]
                vars_util.agents_seir_state[agentid] = vars_util_to_sync.agents_seir_state[agentid]

                if agentid in vars_util_to_sync.agents_infection_type:
                    vars_util.agents_infection_type[agentid] = vars_util_to_sync.agents_infection_type[agentid]
                
                if agentid in vars_util_to_sync.agents_infection_severity:
                    vars_util.agents_infection_severity[agentid] = vars_util_to_sync.agents_infection_severity[agentid]

                print(f"synced it_agents and agents_epi {agentid}")
                if f is not None:
                    f.flush()

            # print("saved soc_rate {0}".format(str(dask_worker.data["agents_static"].get(agentid, "soc_rate"))))

        print("worker {0}, staticagents len {1}, departing_tourist_agent_ids {2}".format(str(dask_worker.id), str(len(agents_static.shm_age) + len(agents_static.static_agents_dict)), str(departed_tourist_agent_ids)))
        if f is not None:
            f.flush()
        
        for agentid in departed_tourist_agent_ids:
            agents_static.delete(agentid)
            # agents_static.set(agentid, "age", None)
            # agents_static.set(agentid, "res_cellid", None)
            # agents_static.set(agentid, "age_bracket_index", None)
            # agents_static.set(agentid, "epi_age_bracket_index", None)
            # agents_static.set(agentid, "soc_rate", None)

            # this cannot work because the tourists dict is not synced upon arrival in the first place
            # if tourists is not None:
            #     del tourists[agentid - n_locals]
            
            if sync_dynamic_agents:
                del it_agents[agentid]
                del agents_epi[agentid]
                del vars_util.agents_seir_state[agentid]

                if agentid in vars_util.agents_infection_type:
                    del vars_util.agents_infection_type[agentid]

                if agentid in vars_util.agents_infection_severity:
                    del vars_util.agents_infection_severity[agentid]
        
        # this cannot work because the touristsgroups dict is not synced upon arrival in the first place
        # if touristsgroups is not None:
        #     for groupid in departed_tourist_group_ids:
        #         del touristsgroups[groupid]

        return process_index, True, it_agents, agents_epi, vars_util
    except Exception as e:
        if stack_trace_log_file_name == "":
            stack_trace_log_file_name = os.path.join(os.getcwd(), "dask_error.txt")

        with open(stack_trace_log_file_name, 'w') as fi:
            traceback.print_exc(file=fi)

        raise
    finally:
        gc.collect()

        if f is not None:
            f.flush()
            # Close the file
            f.close()

        sys.stdout = original_stdout