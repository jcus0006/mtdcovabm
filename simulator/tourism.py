import numpy as np
from copy import copy
import time
import util, seirstateutil, tourism_dist, vars
from epidemiologyclasses import SEIRState
from dask.distributed import Client, SSHCluster, as_completed
import customdict
import gc

class Tourism:
    def __init__(self, tourismparams, cells, n_locals, tourists, agents_static, it_agents, agents_epi, vars_util, touristsgroupsdays, touristsgroups, rooms_by_accomid_by_accomtype, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids, tourists_active_ids, age_brackets, powerlaw_distribution_parameters, visualise, sociability_rate_min, sociability_rate_max, figure_count, initial_seir_state_distribution, dask_full_stateful):
        self.rng = np.random.default_rng(seed=6)

        self.tourists_arrivals_departures_for_day = tourists_arrivals_departures_for_day
        self.tourists_arrivals_departures_for_nextday = tourists_arrivals_departures_for_nextday
        self.tourists_active_groupids = tourists_active_groupids
        self.tourists_active_ids = tourists_active_ids
        self.age_brackets = age_brackets
        self.visualise = visualise
        self.powerlaw_distribution_parameters, self.sociability_rate_min, self.sociability_rate_max, self.figure_count = powerlaw_distribution_parameters, sociability_rate_min, sociability_rate_max, figure_count
        self.initial_seir_state_distribution = initial_seir_state_distribution

        incoming_airport_duration_dist_params = tourismparams["incoming_airport_duration_distribution_params"]
        outgoing_airport_duration_dist_params = tourismparams["outgoing_airport_duration_distribution_params"]
        
        self.incoming_duration_shape_param, self.incoming_duration_min, self.incoming_duration_max = incoming_airport_duration_dist_params[0], incoming_airport_duration_dist_params[1], incoming_airport_duration_dist_params[2]
        self.outgoing_duration_shape_param, self.outgoing_duration_min, self.outgoing_duration_max = outgoing_airport_duration_dist_params[0], outgoing_airport_duration_dist_params[1], outgoing_airport_duration_dist_params[2]

        self.cells = cells
        self.n_locals = n_locals
        self.tourists = tourists
        self.agents_static = agents_static
        self.it_agents = it_agents
        self.agents_epi = agents_epi
        self.vars_util = vars_util
        self.touristsgroupsdays = touristsgroupsdays
        self.touristsgroups = touristsgroups
        self.rooms_by_accomid_by_accomtype = rooms_by_accomid_by_accomtype
        self.day_timesteps = np.arange(144)
        self.afternoon_timesteps = np.arange(72, 144)

        self.agents_static_to_sync = customdict.CustomDict()

        self.dask_full_stateful = dask_full_stateful

        self.arriving_tourists_agents_ids = [] # [id1, id2] - reset everyday
        self.arriving_tourists_next_day_agents_ids = [] # see above
        self.departing_tourists_agents_ids = {} # {day: []} - cleaned the day after

    def initialize_foreign_arrivals_departures_for_day(self, day, f=None):
        tourist_groupids_by_day = set(self.touristsgroupsdays[day])

        tourist_groupids_by_nextday = []
        if day+1 in self.touristsgroupsdays:
            tourist_groupids_by_nextday = set(self.touristsgroupsdays[day+1])

        self.tourists_arrivals_departures_for_day = {}
        if len(tourist_groupids_by_day) > 0:
            if day == 1:
                self.sample_arrival_departure_timesteps(day, tourist_groupids_by_day, self.tourists_arrivals_departures_for_day, False, f)
            else:
                self.tourists_arrivals_departures_for_day = copy(self.tourists_arrivals_departures_for_nextday)
          
        self.tourists_arrivals_departures_for_nextday = {}
        if len(tourist_groupids_by_nextday) > 0:
            self.sample_arrival_departure_timesteps(day+1, tourist_groupids_by_nextday, self.tourists_arrivals_departures_for_nextday, True, f)
        
        return self.it_agents, self.agents_epi, self.tourists, self.cells, self.tourists_arrivals_departures_for_day, self.tourists_arrivals_departures_for_nextday, self.tourists_active_groupids
    
    def sample_arrival_departure_timesteps(self, day, tourist_groupids, tourists_arrivals_departures, is_next_day, f):
        if not is_next_day:
            self.arriving_tourists_agents_ids = []
        else:
            self.arriving_tourists_next_day_agents_ids = []

        for tour_group_id in tourist_groupids:
            tourists_group = self.touristsgroups[tour_group_id]

            accomtype = tourists_group["accomtype"]
            accominfo = tourists_group["accominfo"]
            arrivalday = tourists_group["arr"]
            departureday = tourists_group["dep"]
            purpose = tourists_group["purpose"]
            subgroupsmemberids = tourists_group["subgroupsmemberids"] # rooms in accom

            if arrivalday == day or departureday == day:
                if arrivalday == day:
                    # sample an arrival timestep
                    arrival_ts = self.rng.choice(self.day_timesteps, size=1)[0]
                    arrival_airport_duration = util.sample_gamma_reject_out_of_range(self.incoming_duration_shape_param, self.incoming_duration_min, self.incoming_duration_max, 1, True, True)

                    tourists_arrivals_departures[tour_group_id] = {"ts": arrival_ts, "airport_duration": arrival_airport_duration, "arrival": True}
                
                    self.tourists_active_groupids.append(tour_group_id) 
                else:
                    if departureday - arrivalday == 1: # if only 1 day trip force departure to be in the afternoon
                        timesteps = self.afternoon_timesteps
                    else:
                        timesteps = self.day_timesteps

                     # sample a departure timestep
                    departure_ts = self.rng.choice(timesteps, size=1)[0]
                    departure_airport_duration = util.sample_gamma_reject_out_of_range(self.outgoing_duration_shape_param, self.outgoing_duration_min, self.outgoing_duration_max, 1, True, True)

                    tourists_arrivals_departures[tour_group_id] = {"ts": departure_ts, "airport_duration": departure_airport_duration, "arrival": False}

                if arrivalday == day:
                    ages = []
                    new_agent_ids = []
                    res_cell_ids = []
                    num_tourists_in_group = 0
                    group_accom_id = None
                    under_age_agent = False

                    agents_ids_by_ages = {}
                    agents_ids_by_agebrackets = {i:[] for i in range(len(self.age_brackets))}

                    for accinfoindex, accinfo in enumerate(accominfo):
                        accomid, roomid, _ = accinfo[0], accinfo[1], accinfo[2]

                        group_accom_id = accomid

                        room_members = subgroupsmemberids[accinfoindex] # this room

                        cellindex = self.rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                        self.cells[cellindex]["place"]["member_uids"] = np.array(room_members) + self.n_locals 

                        num_tourists_in_group += len(room_members)

                        # handle as part of "agents" dict, to avoid having to do checks against 2 dicts
                        for tourist_id in room_members: # tourists ids in room
                            tourist = self.tourists[tourist_id]
                            # new_agent_id = self.get_next_available_agent_id()
                            new_agent_id = tourist_id + self.n_locals
                            new_agent_ids.append(new_agent_id)
                            res_cell_ids.append(cellindex)
                            tourist["agentid"] = new_agent_id
                            tourist["initial_tourist"] = True

                            age_bracket_index = -1
                            for i, ab in enumerate(self.age_brackets):
                                if tourist["age"] >= ab[0] and tourist["age"] <= ab[1]:
                                    age_bracket_index = i
                                    break

                            ages.append(tourist["age"])

                            if tourist["age"] < 16:
                                under_age_agent = True

                            epi_age_bracket_index = util.get_sus_mort_prog_age_bracket_index(tourist["age"])

                            self.it_agents[new_agent_id] = None
                            self.agents_epi[new_agent_id] = None

                            new_it_agent = { "touristid": tourist_id, "itinerary": {}, "itinerary_nextday": {}}
                            new_agent_epi = {"touristid": tourist_id, "state_transition_by_day": None, "test_day": None, "test_result_day": None, "quarantine_days": None, "hospitalisation_days": None}

                            age_bracket_index, agents_ids_by_ages, agents_ids_by_agebrackets = util.set_age_brackets_tourists(tourist["age"], agents_ids_by_ages, new_agent_id, self.age_brackets, agents_ids_by_agebrackets)

                            self.it_agents[new_agent_id] = new_it_agent
                            self.agents_epi[new_agent_id] = new_agent_epi          

                            # stbd_exists = "state_transition_by_day" in self.agents_epi[new_agent_id]
                            # print(f"new agent id {new_agent_id} tourist id {tourist_id} state_transition_by_day exists {str(stbd_exists)}")
                            # if f is not None:
                            #     f.flush()

                            if not self.agents_static.use_tourists_dict:
                                self.agents_static.set(new_agent_id, "age", tourist["age"])
                                self.agents_static.set(new_agent_id, "res_cellid", cellindex)
                                self.agents_static.set(new_agent_id, "age_bracket_index", age_bracket_index)
                                self.agents_static.set(new_agent_id, "epi_age_bracket_index", epi_age_bracket_index)
                                self.agents_static.set(new_agent_id, "pub_transp_reg", True)
                            else:
                                props = {"age": tourist["age"], "res_cellid": cellindex, "age_bracket_index": age_bracket_index, "epi_age_bracket_index": epi_age_bracket_index, "pub_transp_reg": True, "soc_rate": 0}
                                self.agents_static.set_props(new_agent_id, props)

                            self.agents_static_to_sync[new_agent_id] = [tourist["age"], cellindex, age_bracket_index, epi_age_bracket_index, True, 0]

                            self.tourists_active_ids.append(tourist_id)
                            if not is_next_day:
                                self.arriving_tourists_agents_ids.append(new_agent_id)
                            else:
                                self.arriving_tourists_next_day_agents_ids.append(new_agent_id)

                        tourists_group["under_age_agent"] = under_age_agent
                        tourists_group["group_accom_id"] = group_accom_id
                        tourists_group["agent_ids"] = new_agent_ids
                        tourists_group["res_cell_ids"] = res_cell_ids
                        tourists_group["pub_transp_reg"] = True
                        tourists_group["initial_tourist"] = True

                        avg_age = round(sum(ages) / num_tourists_in_group)

                        for i, ab in enumerate(self.age_brackets):
                            if avg_age >= ab[0] and avg_age <= ab[1]:
                                tourists_group["age_bracket_index"] = i
                                break
                    
                    new_soc_rates = {agentid:{} for agentid in new_agent_ids}
                    new_soc_rates = util.generate_sociability_rate_powerlaw_dist(new_soc_rates, agents_ids_by_agebrackets, self.powerlaw_distribution_parameters, self.visualise, self.sociability_rate_min, self.sociability_rate_max, self.figure_count)

                    for agentid, prop in new_soc_rates.items():
                        self.agents_static.set(agentid, "soc_rate", prop["soc_rate"])
                        self.agents_static_to_sync[agentid][5] = prop["soc_rate"] # index 5 is soc_rate

                    # agents_seir_state_tourists_subset = self.agents_seir_state[new_agent_ids] # subset from agents_seir_state with new_agent_ids as indices
                    # agents_seir_state_tourists_subset = seirstateutil.initialize_agent_states(len(agents_seir_state_tourists_subset), self.initial_seir_state_distribution, agents_seir_state_tourists_subset)

                    for id in new_agent_ids:
                        self.vars_util.agents_seir_state[id] = SEIRState.Undefined

                    # self.num_active_tourists += num_tourists_in_group

    def sample_initial_tourists(self, tourist_groupids, f):
        for tour_group_id in tourist_groupids:
            tourists_group = self.touristsgroups[tour_group_id]

            accomtype = tourists_group["accomtype"]
            accominfo = tourists_group["accominfo"]
            subgroupsmemberids = tourists_group["subgroupsmemberids"] # rooms in accom

            self.tourists_active_groupids.append(tour_group_id) 

            ages = []
            new_agent_ids = []
            res_cell_ids = []
            num_tourists_in_group = 0
            group_accom_id = None
            under_age_agent = False

            agents_ids_by_ages = {}
            agents_ids_by_agebrackets = {i:[] for i in range(len(self.age_brackets))}

            for accinfoindex, accinfo in enumerate(accominfo):
                accomid, roomid, _ = accinfo[0], accinfo[1], accinfo[2]

                group_accom_id = accomid

                room_members = subgroupsmemberids[accinfoindex] # this room

                cellindex = self.rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                self.cells[cellindex]["place"]["member_uids"] = np.array(room_members) + self.n_locals 

                num_tourists_in_group += len(room_members)

                # handle as part of "agents" dict, to avoid having to do checks against 2 dicts
                for tourist_id in room_members: # tourists ids in room
                    tourist = self.tourists[tourist_id]
                    # new_agent_id = self.get_next_available_agent_id()
                    new_agent_id = tourist_id + self.n_locals
                    new_agent_ids.append(new_agent_id)
                    res_cell_ids.append(cellindex)
                    tourist["agentid"] = new_agent_id

                    age_bracket_index = -1
                    for i, ab in enumerate(self.age_brackets):
                        if tourist["age"] >= ab[0] and tourist["age"] <= ab[1]:
                            age_bracket_index = i
                            break

                    ages.append(tourist["age"])

                    if tourist["age"] < 16:
                        under_age_agent = True

                    epi_age_bracket_index = util.get_sus_mort_prog_age_bracket_index(tourist["age"])

                    self.it_agents[new_agent_id] = None
                    self.agents_epi[new_agent_id] = None

                    new_it_agent = { "touristid": tourist_id, "initial_tourist": True, "itinerary": {}, "itinerary_nextday": {}}
                    new_agent_epi = {"touristid": tourist_id, "state_transition_by_day": None, "test_day": None, "test_result_day": None, "quarantine_days": None, "hospitalisation_days": None}

                    age_bracket_index, agents_ids_by_ages, agents_ids_by_agebrackets = util.set_age_brackets_tourists(tourist["age"], agents_ids_by_ages, new_agent_id, self.age_brackets, agents_ids_by_agebrackets)

                    self.it_agents[new_agent_id] = new_it_agent
                    self.agents_epi[new_agent_id] = new_agent_epi          

                    if not self.agents_static.use_tourists_dict:
                        self.agents_static.set(new_agent_id, "age", tourist["age"])
                        self.agents_static.set(new_agent_id, "res_cellid", cellindex)
                        self.agents_static.set(new_agent_id, "age_bracket_index", age_bracket_index)
                        self.agents_static.set(new_agent_id, "epi_age_bracket_index", epi_age_bracket_index)
                        self.agents_static.set(new_agent_id, "pub_transp_reg", True)
                    else:
                        props = {"age": tourist["age"], "res_cellid": cellindex, "age_bracket_index": age_bracket_index, "epi_age_bracket_index": epi_age_bracket_index, "pub_transp_reg": True, "soc_rate": 0}
                        self.agents_static.set_props(new_agent_id, props)

                    self.agents_static_to_sync[new_agent_id] = [tourist["age"], cellindex, age_bracket_index, epi_age_bracket_index, True, 0]

                    self.tourists_active_ids.append(tourist_id)

                tourists_group["under_age_agent"] = under_age_agent
                tourists_group["group_accom_id"] = group_accom_id
                tourists_group["agent_ids"] = new_agent_ids
                tourists_group["res_cell_ids"] = res_cell_ids
                tourists_group["pub_transp_reg"] = True
                tourists_group["initial_tourist"] = True
                tourists_group["itinerary"] = None
                tourists_group["itinerary_nextday"] = None

                avg_age = round(sum(ages) / num_tourists_in_group)

                for i, ab in enumerate(self.age_brackets):
                    if avg_age >= ab[0] and avg_age <= ab[1]:
                        tourists_group["age_bracket_index"] = i
                        break
            
            new_soc_rates = {agentid:{} for agentid in new_agent_ids}
            new_soc_rates = util.generate_sociability_rate_powerlaw_dist(new_soc_rates, agents_ids_by_agebrackets, self.powerlaw_distribution_parameters, self.visualise, self.sociability_rate_min, self.sociability_rate_max, self.figure_count)

            for agentid, prop in new_soc_rates.items():
                self.agents_static.set(agentid, "soc_rate", prop["soc_rate"])
                self.agents_static_to_sync[agentid][5] = prop["soc_rate"] # index 5 is soc_rate

            # agents_seir_state_tourists_subset = self.agents_seir_state[new_agent_ids] # subset from agents_seir_state with new_agent_ids as indices
            # agents_seir_state_tourists_subset = seirstateutil.initialize_agent_states(len(agents_seir_state_tourists_subset), self.initial_seir_state_distribution, agents_seir_state_tourists_subset)

            for id in new_agent_ids:
                self.vars_util.agents_seir_state[id] = SEIRState.Undefined

                    # self.num_active_tourists += num_tourists_in_group
    def get_next_available_agent_id(self):
        if not self.agents_static:
            return 0
        else:
            return max(self.agents_static.keys()) + 1

    def sync_and_clean_tourist_data(self, day, client: Client, actors, remote_log_subfolder_name, log_file_name, dask_full_stateful, f=None):
        departing_tourists_agent_ids = []

        start = time.time()
        for tour_grp_id, grp_arr_dep_info in self.tourists_arrivals_departures_for_day.items():
            if not grp_arr_dep_info["arrival"]:
                tourists_group = self.touristsgroups[tour_grp_id]

                accomtype = tourists_group["accomtype"]
                accominfo = tourists_group["accominfo"]
                subgroupsmemberids = tourists_group["subgroupsmemberids"] # rooms in accom

                # num_tourists_in_group = 0

                for accinfoindex, accinfo in enumerate(accominfo):
                    accomid, roomid, _ = accinfo[0], accinfo[1], accinfo[2]

                    room_members = subgroupsmemberids[accinfoindex] # this room

                    # num_tourists_in_group += len(room_members)
                    for tourist_id in room_members:
                        self.tourists_active_ids.remove(tourist_id)

                        tourist = self.tourists[tourist_id]
                        agentid = tourist["agentid"]

                        departing_tourists_agent_ids.append(agentid)

                        # self.agents_static.delete(agentid)
                        # self.agents_static.set(agentid, "age", None)
                        # self.agents_static.set(agentid, "res_cellid", None)
                        # self.agents_static.set(agentid, "age_bracket_index", None)
                        # self.agents_static.set(agentid, "epi_age_bracket_index", None)
                        # self.agents_static.set(agentid, "pub_transp_reg", None)

                    # self.tourists_active_ids.extend(room_members)

                    cellindex = self.rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                    self.cells[cellindex]["place"]["member_uids"] = []

                self.tourists_active_groupids.remove(tour_grp_id)

        self.departing_tourists_agents_ids[day] = departing_tourists_agent_ids
        time_taken = time.time() - start
        print("sync_and_clean_tourist_data on client: " + str(time_taken))
        if f is not None:
            f.flush()

        # sync new tourists with remote workers and remove tourists who have left on the previous day
        prev_day_departing_tourists_agents_ids = []
            
        if day-1 in self.departing_tourists_agents_ids:
            prev_day_departing_tourists_agents_ids = self.departing_tourists_agents_ids[day-1]

        if client is not None:
            start = time.time()

            futures = []
            workers = list(client.scheduler_info()["workers"].keys()) # list()

            for worker_index, worker in enumerate(workers):
                if len(actors) == 0:
                    params = (day, self.agents_static_to_sync, prev_day_departing_tourists_agents_ids, remote_log_subfolder_name, log_file_name, worker_index)
                    future = client.submit(tourism_dist.update_tourist_data_remote, params, workers=worker)
                    futures.append(future)
                else:
                    if not dask_full_stateful:
                        params = (day, self.agents_static_to_sync, prev_day_departing_tourists_agents_ids, worker_index)
                    else:
                        it_agents_to_sync, agents_epi_to_sync = customdict.CustomDict(), customdict.CustomDict()
                        vars_util_to_sync = vars.Vars()

                        for agentid in self.agents_static_to_sync.keys():
                            it_agents_to_sync[agentid] = self.it_agents[agentid]
                            agents_epi_to_sync[agentid] = self.agents_epi[agentid]
                            vars_util_to_sync.agents_seir_state[agentid] = self.vars_util.agents_seir_state[agentid]

                        params = (day, self.agents_static_to_sync, it_agents_to_sync, agents_epi_to_sync, vars_util_to_sync, prev_day_departing_tourists_agents_ids, worker_index)

                    actor = actors[worker_index]
                    
                    if not dask_full_stateful:
                        future = actor.run_update_tourist_data_remote(params)
                    else:
                        future = actor.tourists_sync(params)

                    futures.append(future)

            self.agents_static_to_sync = customdict.CustomDict()
            self.it_agents_to_sync = customdict.CustomDict()
            self.agents_epi_to_sync = customdict.CustomDict()
            
            success = False
            for future in as_completed(futures):
                if len(actors) == 0:
                    process_index, success, _, _, _ = future.result()

                    future.release()
                else:
                    process_index, success = future.result()                   

                print("process_index {0}, success {1}".format(str(process_index), str(success)))
                if f is not None:
                    f.flush()

            time_taken = time.time() - start
            print("sync_and_clean_tourist_data remotely, success {0}, time_taken {1}".format(str(success), str(time_taken)))
            if f is not None:
                f.flush()

        if len(prev_day_departing_tourists_agents_ids) > 0:
            start_prev_day_del = time.time()
            for agentid in prev_day_departing_tourists_agents_ids:
                del self.it_agents[agentid]
                del self.agents_epi[agentid] #       
                del self.vars_util.agents_seir_state[agentid]

                if agentid in self.vars_util.agents_infection_type:
                    del self.vars_util.agents_infection_type[agentid]

                if agentid in self.vars_util.agents_infection_severity:
                    del self.vars_util.agents_infection_severity[agentid]

                self.agents_static.delete(agentid)
                # print(f"deleted agent {agentid} from agents_static")
    
            del self.departing_tourists_agents_ids[day-1]
            time_taken_prev_day_del = time.time() - start_prev_day_del
            print(f"deleting departuring tourists from main node, time_taken: {time_taken_prev_day_del}")

            gc.collect()