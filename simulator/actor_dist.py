import os
import time
from dask.distributed import get_worker, get_client, as_completed
import customdict, vars, tourism_dist, itinerary, util
from cellsclasses import CellType
from enum import Enum

# the client will start each stage on each actor
# it will also maintain a flag to monitor progress (for each stage: itinerary, contact network)
# certain methods like contact tracing still happen locally (and direct contacts are still synced with the main node because of this)
class ActorDist:
    def __init__(self, params):
        self.worker = get_worker()
        self.client = get_client()

        self.remote_actors = []

        workers_keys, workerindex, hh_insts, it_agents, agents_epi, vars_util, dyn_params, agent_ids_by_worker_lookup, cell_ids_by_worker_lookup, worker_by_res_ids_lookup, worker_by_agent_ids_lookup, worker_by_cell_ids_lookup, logsubfoldername, logfilename = params

        current_directory = os.getcwd()
        subfolder_name = logfilename.replace(".txt", "")
        log_subfolder_path = os.path.join(current_directory, logsubfoldername, subfolder_name)

        self.folder_name = log_subfolder_path

        self.workers_keys = workers_keys
        self.worker_index = workerindex
        self.worker_key = self.workers_keys[self.worker_index]
        self.agent_ids_by_process_lookup = agent_ids_by_worker_lookup
        self.cell_ids_by_process_lookup = cell_ids_by_worker_lookup
        self.worker_by_res_ids_lookup = worker_by_res_ids_lookup # {id: worker_id}
        self.worker_by_agent_ids_lookup = worker_by_agent_ids_lookup
        self.worker_by_cell_ids_lookup = worker_by_cell_ids_lookup
        self.agent_ids = set(self.agent_ids_by_process_lookup[self.worker_index])
        self.cell_ids = set(self.cell_ids_by_process_lookup[self.worker_index])

        self.day = None # 1 - 365
        self.weekday = None
        self.weekday_str = None
        self.simstage = None # 0 itinerary
        # self.workers_sync_flag = None
        # self.init_worker_flags()
        self.dyn_params = dyn_params
        self.hh_insts = hh_insts
        self.it_agents = it_agents # this does not have to be shared with the remote nodes or returned to the client!
        self.agents_epi = agents_epi
        self.vars_util = vars_util # direct contacts do not have to be shared with the remote nodes, it can be simply returned to the client

        self.it_time_taken_sum = 0
        self.it_time_taken_avg = 0
        self.cn_time_taken_sum = 0
        self.cn_time_taken_avg = 0

    def set_remote_actors(self, actors):
        self.remote_actors = actors

    def tourists_sync(self, params):
        self.simstage = SimStage.TouristSync

        return tourism_dist.update_tourist_data_remote(params, self.folder_name)
    
    def itinerary(self):
        self.simstage = SimStage.Itinerary

        # load worker data
        if self.worker is None:
            raise TypeError("Worker is none")
        
        if self.worker.data is None or len(self.worker.data) == 0:
            raise TypeError("Worker.data is None or empty")
        else:
            if self.worker.data["itineraryparams"] is None or len(self.worker.data["itineraryparams"]) == 0:
                raise TypeError("Worker.data['itineraryparams'] is None or empty")
        
        # new_tourists = worker.data["new_tourists"]
        # print("new tourists {0}".format(str(new_tourists)))

        agents_ids_by_ages = self.worker.data["agents_ids_by_ages"]
        timestepmins = self.worker.data["timestepmins"]
        n_locals = self.worker.data["n_locals"]
        n_tourists = self.worker.data["n_tourists"]
        locals_ratio_to_full_pop = self.worker.data["locals_ratio_to_full_pop"]

        itineraryparams = self.worker.data["itineraryparams"]
        epidemiologyparams = self.worker.data["epidemiologyparams"]
        cells_industries_by_indid_by_wpid = self.worker.data["cells_industries_by_indid_by_wpid"] 
        cells_restaurants = self.worker.data["cells_restaurants"] 
        cells_hospital = self.worker.data["cells_hospital"] 
        cells_testinghub = self.worker.data["cells_testinghub"] 
        cells_vaccinationhub = self.worker.data["cells_vaccinationhub"] 
        cells_entertainment_by_activityid = self.worker.data["cells_entertainment_by_activityid"] 
        cells_religious = self.worker.data["cells_religious"] 
        cells_households = self.worker.data["cells_households"] 
        cells_breakfast_by_accomid = self.worker.data["cells_breakfast_by_accomid"] 
        cells_airport = self.worker.data["cells_airport"] 
        cells_transport = self.worker.data["cells_transport"] 
        cells_institutions = self.worker.data["cells_institutions"] 
        cells_accommodation = self.worker.data["cells_accommodation"] 
        agents_static = self.worker.data["agents_static"]

        itinerary_util = itinerary.Itinerary(itineraryparams,
                                            timestepmins, 
                                            n_locals, 
                                            n_tourists, 
                                            locals_ratio_to_full_pop, 
                                            agents_static,
                                            self.it_agents, 
                                            self.agents_epi,
                                            agents_ids_by_ages,
                                            self.vars_util,
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
                                            self.dyn_params,
                                            process_index=self.worker_index)

        num_agents_working_schedule = 0
        working_schedule_times_by_resid = {}

        if self.day == 1 or self.weekday_str == "Monday":
            # print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday))
            start = time.time()
            for hh_inst in self.hh_insts:
                start = time.time()
                # print("day " + str(day) + ", res id: " + str(hh_inst["id"]) + ", is_hh: " + str(hh_inst["is_hh"]))
                itinerary_util.generate_working_days_for_week_residence(hh_inst["resident_uids"], hh_inst["is_hh"])
                time_taken = time.time() - start
                working_schedule_times_by_resid[hh_inst["id"]] = time_taken
                num_agents_working_schedule += len(hh_inst["resident_uids"])

            time_taken = time.time() - start
            print("generate_working_days_for_week_residence for simday " + str(self.day) + ", weekday " + str(self.weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(self.worker_index))

        print("generate_itinerary_hh for simday " + str(self.day) + ", weekday " + str(self.weekday))
        start = time.time()                    

        num_agents_itinerary = 0
        itinerary_times_by_resid = {}
        for hh_inst in self.hh_insts:
            res_start = time.time()
            itinerary_util.generate_local_itinerary(self.day, self.weekday, hh_inst["resident_uids"])
            res_timetaken = time.time() - res_start
            itinerary_times_by_resid[hh_inst["id"]] = res_timetaken
            num_agents_itinerary += len(hh_inst["resident_uids"])
        
        send_results_start_time = time.time()
        # send results
        self.send_results()
        send_results_time_taken = time.time() - send_results_start_time
        print("send results time taken: " + str(send_results_time_taken))

        time_taken = time.time() - start
        self.it_time_taken_sum += time_taken
        self.it_time_taken_avg = self.it_time_taken_sum / self.day
        print("generate_itinerary_hh for simday " + str(self.day) + ", weekday " + str(self.weekday) + ", time taken: " + str(time_taken) + ", proc index: " + str(self.worker_index))
        
        print("process " + str(self.worker_index) + ", ended at " + str(time.time()))
        # return contact_tracing_agent_ids (that are only added to in this context) and time-logging information to client
        
        return self.worker_index, self.vars_util.contact_tracing_agent_ids, time_taken, send_results_time_taken, self.it_time_taken_avg

    def contact_network(self):
        self.simstage = SimStage.ContactNetwork

        # load worker data
        # simulate_contact_network
        # send results
        # return direct contacts info to client and time-logging information

        pass

    def send_results(self):
        # if itinerary, get list of cell ids from cells_agents_timesteps, then get list of cell ids that are not in self.cells_ids (difference with sets)
        # these are the cases for which the contact network will be computed on another remote node
        # iterate on the workers, skipping this one, build a dict of the following structure: {worker_index: (agents_epi, vars_util)} 
        # in vars_util include cells_agents_timesteps that are relevant to each worker and all agent information for agents that show up in those cells_agents_timesteps
        # note any properties that are not relevant for the itinerary and don't include them
        
        # if not is_itinerary_result, would be contact network. get list of contacts that include agents that are not in self.agents_ids (consider BST Search)
        # the same applies, in terms of the structure: {worker_index: (agents_epi, vars_util)}
        # do not include direct contacts, these will be returned directly to the client and synced accordingly (contact tracing happens on the client anyway)
        
        cells_ids = set(self.vars_util.cells_agents_timesteps.keys())
        cells_ids = cells_ids.difference(self.cell_ids) 

        send_results_by_worker_id = customdict.CustomDict()
        
        for wi, w_cell_ids in self.cell_ids_by_process_lookup.items():
            if wi != self.worker_index:
                cells_ids_to_send = cells_ids.intersection(w_cell_ids)

                agents_epi_to_send = customdict.CustomDict()
                vars_util_to_send = vars.Vars()

                agents_ids_to_send = set()

                for cell_id in cells_ids_to_send:
                    cats_by_cell_id = self.vars_util.cells_agents_timesteps[cell_id]
                    vars_util_to_send.cells_agents_timesteps[cell_id] = cats_by_cell_id

                    for agent_id, _, _ in cats_by_cell_id:
                        agents_ids_to_send.add(agent_id)

                for agent_id in agents_ids_to_send:
                    agents_epi_to_send[agent_id] = self.agents_epi[agent_id]
                    vars_util_to_send.agents_seir_state[agent_id] = self.vars_util.agents_seir_state[agent_id]

                    if agent_id in self.vars_util.agents_seir_state_transition_for_day:
                        vars_util_to_send.agents_seir_state_transition_for_day[agent_id] = self.vars_util.agents_seir_state_transition_for_day[agent_id]

                    if agent_id in self.vars_util.agents_infection_type:
                        vars_util_to_send.agents_infection_type[agent_id] = self.vars_util.agents_infection_type[agent_id]

                    if agent_id in self.vars_util.agents_infection_severity:
                        vars_util_to_send.agents_infection_severity[agent_id] = self.vars_util.agents_infection_severity[agent_id]

                    if agent_id in self.vars_util.agents_vaccination_doses:
                        vars_util_to_send.agents_vaccination_doses[agent_id] = self.vars_util.agents_vaccination_doses[agent_id]

                send_results_by_worker_id[wi] = [agents_epi_to_send, vars_util_to_send]
        
        results = []
        for wi in range(len(self.workers_keys)):
            if wi != self.worker_index:
                data = send_results_by_worker_id[wi]
                params = (self.worker_index, self.simstage, data)
                result = self.remote_actors[wi].receive_results(params)
                results.append(result)
                # results.append(self.remote_actors[wi].receive_results(params))
                # self.client.submit(receive_results, (self.worker.address, self.simstage, data), workers=worker_key)

        result_index = 0
        for result in results:
            # result = result.result()
            print("Message Result {0}: {1}".format(str(result_index), str(result)))
            result_index += 1

    def receive_results(self, params):
        sender_worker_index, simstage, data = params

        # sync results after itinerary or contact network
        # TO DO - when is_itinerary_result is True and receiving state information from a remote node, keep track, so as not to send again, for current day only (i.e. that node already has this data)

        agents_epi_partial, vars_util_partial = data
        
        if simstage == SimStage.Itinerary:
            util.sync_state_info_cells_agents_timesteps(self.vars_util, vars_util_partial)

        self.agents_epi, self.vars_util = util.sync_state_info_by_agentsids_agents_epi(agents_epi_partial.keys(), self.agents_epi, self.vars_util, agents_epi_partial, vars_util_partial)

        # self.workers_sync_flag[sender_worker_index] = True

        return True

    # def init_worker_flags(self):
    #     self.workers_sync_flag = customdict.CustomDict({
    #         i:False for i in range(len(self.workers_keys)) if i != self.worker_index
    #     })

    def reset_day(self, new_day, new_weekday, new_weekday_str, new_dyn_params):
        self.day = new_day
        self.weekday = new_weekday
        self.weekday_str = new_weekday_str
        self.dyn_params = new_dyn_params

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

class SimStage(Enum):
    TouristSync = 0
    Itinerary = 1
    ContactNetwork = 2