import os
from dask.distributed import get_worker, get_client
import customdict, vars

def receive_results(params):
    sender_url, is_itinerary_result, data = params

    # sync results after itinerary or contact network (is_itinerary_result is included just in case for time being - might not be needed)
    # when is_itinerary_result is True and receiving state information from a remote node, keep track, so as not to send again, for current day onlys
    agents_epi_partial, vars_util_partial = data

# the client will start each stage on each actor
# it will also maintain a flag to monitor progress (for each stage: itinerary, contact network)
# certain methods like contact tracing still happen locally (and direct contacts are still synced with the main node because of this)
class ActorDist:
    def __init__(self, params):
        self.worker = get_worker()
        self.client = get_client()

        workers_keys, workerindex, res_ids_by_worker_lookup, agent_ids_by_worker_lookup, cell_ids_by_worker_lookup, worker_by_res_ids_lookup, worker_by_agent_ids_lookup, worker_by_cell_ids_lookup, logsubfoldername, logfilename = params

        current_directory = os.getcwd()
        subfolder_name = logfilename.replace(".txt", "")
        log_subfolder_path = os.path.join(current_directory, logsubfoldername, subfolder_name)

        self.folder_name = log_subfolder_path

        self.workers_keys = workers_keys
        self.worker_index = workerindex
        self.worker_key = self.workers_keys[self.worker_index]
        self.res_ids_by_process_lookup = res_ids_by_worker_lookup # {worker_id: [resid0, resid1, ..]}
        self.agent_ids_by_process_lookup = agent_ids_by_worker_lookup
        self.cell_ids_by_process_lookup = cell_ids_by_worker_lookup
        self.worker_by_res_ids_lookup = worker_by_res_ids_lookup # {id: worker_id}
        self.worker_by_agent_ids_lookup = worker_by_agent_ids_lookup
        self.worker_by_cell_ids_lookup = worker_by_cell_ids_lookup
        self.res_ids = set(self.res_ids_by_process_lookup[self.worker_index]) # [resid0, resid1, ..]
        self.agent_ids = set(self.agent_ids_by_process_lookup[self.worker_index])
        self.cell_ids = set(self.cell_ids_by_process_lookup[self.worker_index])

        self.it_agents = customdict.CustomDict()
        self.agents_epi = customdict.CustomDict()
        self.vars_util = vars.Vars()

    def itinerary(self):
        # load worker data
        # working schedule on res_ids
        # local_itinerary on res_ids
        # send results
        # return only time-logging information to client
        
        pass

    def contact_network(self):
        # load worker data
        # simulate_contact_network
        # send results
        # return direct contacts info to client and time-logging information

        pass

    def send_results(self, worker_key, is_itinerary_result):
        # if itinerary, get list of cell ids from cells_agents_timesteps, then get list of cell ids that are not in self.cells_ids (intersection with sets)
        # these are the cases for which the contact tracing will be computed on another remote node
        # iterate on the workers, skipping this one, build a dict of the following structure: {worker_index: (agents_epi, vars_util)} 
        # in vars_util include cells_agents_timesteps that are relevant to each worker and all agent information for agents that show up in those cells_agents_timesteps
        # note any properties that are not relevant for the itinerary and don't include them
        
        # if not is_itinerary_result, would be contact network. get list of contacts that include agents that are not in self.agents_ids (consider BST Search)
        # the same applies, in terms of the structure: {worker_index: (agents_epi, vars_util)}
        # do not include direct contacts, these will be returned directly to the client and synced accordingly (contact tracing happens on the client anyway)
        data = None

        self.client.submit(receive_results, (self.worker.address, is_itinerary_result, data), workers=worker_key)

    # as is would be called at the end of the simulation day
    # but if certain dynamic data would already be statefully available, it would be better to use it (and not receive it again)
    # in that case, the workers would need to keep a temporary store of additional dynamic data available on remote nodes (for current day only)
    def clean_up(self):
        self.vars_util.reset_daily_structures()

        for id in list(self.it_agents.keys()): # can try BST search and compare times
            if id not in self.agent_ids:
                del self.it_agents[id]

                try:
                    del self.agents_epi[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_seir_state[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_infection_type[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_infection_severity[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_vaccination_doses[id]
                except:
                    pass