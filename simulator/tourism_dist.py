import os
import sys
import traceback
from dask.distributed import get_worker
import shutil

# agents_static_to_sync are tourists who are arriving today
# departed_tourist_agent_ids are tourists who left the previous day
def update_tourist_data_remote(params, folder_name=None):
    f = None
    stack_trace_log_file_name = ""
    original_stdout = sys.stdout

    try:
        if len(params) == 5:
            day, agents_static_to_sync, departed_tourist_agent_ids, log_file_name, process_index = params
        else:
            day, agents_static_to_sync, departed_tourist_agent_ids, process_index = params

        if folder_name is None:
            folder_name = ""
            if log_file_name != "output.txt":
                folder_name = os.path.dirname(log_file_name)
            else:
                folder_name = os.getcwd()
        
        stack_trace_log_file_name = os.path.join(folder_name, "utd_dist_stack_trace_" + str(day) + "_" + str(process_index) + ".txt")

        log_file_name = os.path.join(folder_name, "utd_dist_" + str(day) + "_" + str(process_index) + ".txt")
        f = open(log_file_name, "w")
        sys.stdout = f

        dask_worker = get_worker()
        agents_static = dask_worker.data["agents_static"]

        print("asts {0}, dep {1}".format(str(len(agents_static_to_sync)), str(len(departed_tourist_agent_ids))))

        for agentid, staticinfo in agents_static_to_sync.items():
            age, res_cellid, age_bracket_index, epi_age_bracket_index, pub_transp_reg, soc_rate = staticinfo

            print("saving soc_rate {0} for agentid {1}".format(str(soc_rate), str(agentid)))

            agents_static.set(agentid, "age", age)
            agents_static.set(agentid, "res_cellid", res_cellid)
            agents_static.set(agentid, "age_bracket_index", age_bracket_index)
            agents_static.set(agentid, "epi_age_bracket_index", epi_age_bracket_index)
            agents_static.set(agentid, "pub_transp_reg", pub_transp_reg)
            agents_static.set(agentid, "soc_rate", soc_rate)

            print("saved soc_rate {0}".format(str(dask_worker.data["agents_static"].get(agentid, "soc_rate"))))

        # print("worker {0}, staticagents len {1}, departing_tourist_agent_ids {2}".format(str(dask_worker.id), str(len(agents_static.shm_age)), str(departed_tourist_agent_ids)))

        # for agentid in departed_tourist_agent_ids:
        #     agents_static.set(agentid, "age", None)
        #     agents_static.set(agentid, "res_cellid", None)
        #     agents_static.set(agentid, "age_bracket_index", None)
        #     agents_static.set(agentid, "epi_age_bracket_index", None)
        #     agents_static.set(agentid, "soc_rate", None)

        return process_index, True
    except Exception as e:
        with open(stack_trace_log_file_name, 'w') as f:
            traceback.print_exc(file=f)

        raise
    finally:
        if f is not None:
            f.flush()
            # Close the file
            f.close()

        sys.stdout = original_stdout