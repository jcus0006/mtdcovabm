import numpy as np
from copy import copy
import util, seirstateutil, tourism_dist
from dask.distributed import Client, SSHCluster, as_completed

class Tourism:
    def __init__(self, tourismparams, cells, n_locals, tourists, agents_static, it_agents, agents_epi, agents_seir_state, touristsgroupsdays, touristsgroups, rooms_by_accomid_by_accomtype, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids, tourists_active_ids, age_brackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count, initial_seir_state_distribution):
        self.rng = np.random.default_rng(seed=6)

        self.tourists_arrivals_departures_for_day = tourists_arrivals_departures_for_day
        self.tourists_arrivals_departures_for_nextday = tourists_arrivals_departures_for_nextday
        self.tourists_active_groupids = tourists_active_groupids
        self.tourists_active_ids = tourists_active_ids
        self.age_brackets = age_brackets
        self.powerlaw_distribution_parameters, self.params, self.sociability_rate_min, self.sociability_rate_max, self.figure_count = powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count
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
        self.agents_seir_state = agents_seir_state
        self.touristsgroupsdays = touristsgroupsdays
        self.touristsgroups = touristsgroups
        self.rooms_by_accomid_by_accomtype = rooms_by_accomid_by_accomtype
        self.day_timesteps = np.arange(144)
        self.afternoon_timesteps = np.arange(72, 144)

        self.agents_static_to_sync = {}

    def initialize_foreign_arrivals_departures_for_day(self, day):
        tourist_groupids_by_day = set(self.touristsgroupsdays[day])

        tourist_groupids_by_nextday = []
        if day+1 in self.touristsgroupsdays:
            tourist_groupids_by_nextday = set(self.touristsgroupsdays[day+1])

        self.tourists_arrivals_departures_for_day = {}
        if len(tourist_groupids_by_day) > 0:
            if day == 1:
                self.sample_arrival_departure_timesteps(day, tourist_groupids_by_day, self.tourists_arrivals_departures_for_day)
            else:
                self.tourists_arrivals_departures_for_day = copy(self.tourists_arrivals_departures_for_nextday)
            
        self.tourists_arrivals_departures_for_nextday = {}
        if len(tourist_groupids_by_nextday):
            self.sample_arrival_departure_timesteps(day+1, tourist_groupids_by_nextday, self.tourists_arrivals_departures_for_nextday)
        
        return self.it_agents, self.agents_epi, self.tourists, self.cells, self.tourists_arrivals_departures_for_day, self.tourists_arrivals_departures_for_nextday, self.tourists_active_groupids
    
    def sample_arrival_departure_timesteps(self, day, tourist_groupids, tourists_arrivals_departures):
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

                            age_bracket_index = -1
                            for i, ab in enumerate(self.age_brackets):
                                if tourist["age"] >= ab[0] and tourist["age"] <= ab[1]:
                                    age_bracket_index = i
                                    break

                            ages.append(tourist["age"])

                            if tourist["age"] < 16:
                                under_age_agent = True

                            epi_age_bracket_index = util.get_sus_mort_prog_age_bracket_index(tourist["age"])

                            new_it_agent = self.it_agents[new_agent_id]
                            new_agent_epi = self.agents_epi[new_agent_id]

                            new_it_agent = { "touristid": tourist_id, "itinerary": {}, "itinerary_nextday": {}}
                            new_agent_epi = {"touristid": tourist_id, "state_transition_by_day": [], "test_day": [], "test_result_day": [], "quarantine_days": [], "hospitalisation_days": []}

                            age_bracket_index, agents_ids_by_ages, agents_ids_by_agebrackets = util.set_age_brackets_tourists(tourist["age"], agents_ids_by_ages, new_agent_id, self.age_brackets, agents_ids_by_agebrackets)

                            self.it_agents[new_agent_id] = new_it_agent
                            self.agents_epi[new_agent_id] = new_agent_epi

                            self.agents_static.set(new_agent_id, "age", tourist["age"])
                            self.agents_static.set(new_agent_id, "res_cellid", cellindex)
                            self.agents_static.set(new_agent_id, "age_bracket_index", age_bracket_index)
                            self.agents_static.set(new_agent_id, "epi_age_bracket_index", epi_age_bracket_index)
                            self.agents_static.set(new_agent_id, "pub_transp_reg", True)

                            self.agents_static_to_sync[new_agent_id] = [tourist["age"], cellindex, age_bracket_index, epi_age_bracket_index, True, 0]

                            self.tourists_active_ids.append(tourist_id)

                        tourists_group["under_age_agent"] = under_age_agent
                        tourists_group["group_accom_id"] = group_accom_id
                        tourists_group["agent_ids"] = new_agent_ids
                        tourists_group["res_cell_ids"] = res_cell_ids
                        tourists_group["pub_transp_reg"] = True

                        avg_age = round(sum(ages) / num_tourists_in_group)

                        for i, ab in enumerate(self.age_brackets):
                            if avg_age >= ab[0] and avg_age <= ab[1]:
                                tourists_group["age_bracket_index"] = i
                                break
                    
                    new_soc_rates = {agentid:{} for agentid in new_agent_ids}
                    new_soc_rates = util.generate_sociability_rate_powerlaw_dist(new_soc_rates, agents_ids_by_agebrackets, self.powerlaw_distribution_parameters, self.params, self.sociability_rate_min, self.sociability_rate_max, self.figure_count)

                    for agentid, prop in new_soc_rates.items():
                        self.agents_static_to_sync[agentid][5] = prop["soc_rate"] # index 5 is soc_rate

                    agents_seir_state_tourists_subset = self.agents_seir_state[new_agent_ids] # subset from agents_seir_state with new_agent_ids as indices
                    agents_seir_state_tourists_subset = seirstateutil.initialize_agent_states(len(agents_seir_state_tourists_subset), self.initial_seir_state_distribution, agents_seir_state_tourists_subset)

                    for index, agent_idx in enumerate(new_agent_ids):
                        self.agents_seir_state[agent_idx] = agents_seir_state_tourists_subset[index]

                    # self.num_active_tourists += num_tourists_in_group

    def get_next_available_agent_id(self):
        if not self.agents_static:
            return 0
        else:
            return max(self.agents_static.keys()) + 1

    def sync_and_clean_tourist_data(self, day, client: Client, log_file_name):
        departing_tourist_agent_ids = []

        for tour_grp_id, grp_arr_dep_info in self.tourists_arrivals_departures_for_day.items(): # this represents prev day when called
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

                        departing_tourist_agent_ids.append(agentid)

                        self.agents_static.set(agentid, "age", None)
                        self.agents_static.set(agentid, "res_cellid", None)
                        self.agents_static.set(agentid, "age_bracket_index", None)
                        self.agents_static.set(agentid, "epi_age_bracket_index", None)
                        self.agents_static.set(agentid, "pub_transp_reg", None)

                    # self.tourists_active_ids.extend(room_members)

                    cellindex = self.rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                    self.cells[cellindex]["place"]["member_uids"] = []

                self.tourists_active_groupids.remove(tour_grp_id)

        if client is not None:
            futures = []
            workers = list(client.scheduler_info()["workers"].keys()) # list()

            for workerindex, worker in enumerate(workers):
                future = client.submit(tourism_dist.update_tourist_data_remote, (day, self.agents_static_to_sync, departing_tourist_agent_ids, log_file_name, workerindex), workers=worker)
                futures.append(future)
            
            for future in as_completed(futures):
                success = future.result()
                print("success {0}".format(str(success)))
                future.release()

