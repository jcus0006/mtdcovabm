import numpy as np
from copy import copy
from simulator import util

class Tourism:
    def __init__(self, tourismparams, cells, tourists, agents, touristsgroupsdays, touristsgroups, rooms_by_accomid_by_accomtype, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids, age_brackets, epi_util):
        self.rng = np.random.default_rng(seed=6)

        self.tourists_arrivals_departures_for_day = tourists_arrivals_departures_for_day
        self.tourists_arrivals_departures_for_nextday = tourists_arrivals_departures_for_nextday
        self.tourists_active_groupids = tourists_active_groupids
        self.age_brackets = age_brackets
        self.epi_util = epi_util

        incoming_airport_duration_dist_params = tourismparams["incoming_airport_duration_distribution_params"]
        outgoing_airport_duration_dist_params = tourismparams["outgoing_airport_duration_distribution_params"]
        
        self.incoming_duration_shape_param, self.incoming_duration_min, self.incoming_duration_max = incoming_airport_duration_dist_params[0], incoming_airport_duration_dist_params[1], incoming_airport_duration_dist_params[2]
        self.outgoing_duration_shape_param, self.outgoing_duration_min, self.outgoing_duration_max = outgoing_airport_duration_dist_params[0], outgoing_airport_duration_dist_params[1], outgoing_airport_duration_dist_params[2]

        self.cells = cells
        self.tourists = tourists
        self.agents = agents
        self.touristsgroupsdays = touristsgroupsdays
        self.touristsgroups = touristsgroups
        self.rooms_by_accomid_by_accomtype = rooms_by_accomid_by_accomtype
        self.day_timesteps = np.arange(144)
        self.afternoon_timesteps = np.arange(72, 144)

    def initialize_foreign_arrivals_departures_for_day(self, day):
        if day > 1: # clean up previous day
            for tour_grp_id, grp_arr_dep_info in self.tourists_arrivals_departures_for_day.items(): # this representes prev day when called
                if not grp_arr_dep_info["arrival"]:
                    tourists_group = self.touristsgroups[tour_grp_id]

                    accomtype = tourists_group["accomtype"]
                    accominfo = tourists_group["accominfo"]
                    subgroupsmemberids = tourists_group["subgroupsmemberids"] # rooms in accom

                    for accinfoindex, accinfo in enumerate(accominfo):
                        accomid, roomid, _ = accinfo[0], accinfo[1], accinfo[2]

                        subgroupmmembers = subgroupsmemberids[accinfoindex] # this room

                        cellindex = self.rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                        self.cells[cellindex]["place"]["member_uids"] = []

                        # remove from "agents" dict
                        # this means that tourists arriving on later days may be assigned the same agent id. 
                        # tourists still have to be uniquely identified by tourist_id

                        for tourist_id in subgroupmmembers: # tourists ids in room
                            tourist = self.tourists[tourist_id]
                            agent_id = tourist["agentid"]

                            del self.agents[agent_id]

                    self.tourists_active_groupids.remove(tour_grp_id)

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
        
        return self.agents, self.tourists, self.cells, self.tourists_arrivals_departures_for_day, self.tourists_arrivals_departures_for_nextday, self.tourists_active_groupids
    
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
                    num_tourists = 0
                    group_accom_id = None
                    under_age_agent = False
                    for accinfoindex, accinfo in enumerate(accominfo):
                        accomid, roomid, _ = accinfo[0], accinfo[1], accinfo[2]

                        group_accom_id = accomid

                        room_members = subgroupsmemberids[accinfoindex] # this room

                        cellindex = self.rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                        self.cells[cellindex]["place"]["member_uids"] = room_members

                        num_tourists += len(room_members)

                        # handle as part of "agents" dict, to avoid having to do checks against 2 dicts
                        for tourist_id in room_members: # tourists ids in room
                            tourist = self.tourists[tourist_id]
                            new_agent_id = self.get_next_available_agent_id()
                            tourist["agentid"] = new_agent_id

                            age_bracket_index = -1
                            for i, ab in enumerate(self.age_brackets):
                                if tourist["age"] >= ab[0] and tourist["age"] <= ab[1]:
                                    age_bracket_index = i
                                    break

                            ages.append(tourist["age"])

                            if tourist["age"] < 16:
                                under_age_agent = True

                            epi_age_bracket_index = self.epi_util.get_sus_mort_prog_age_bracket_index(tourist["age"])

                            self.agents[new_agent_id] = { "touristid": tourist_id, "curr_cellid": cellindex, "res_cellid": cellindex, "state_transition_by_day": {}, "age_bracket_index": age_bracket_index, "epi_age_bracket_index": epi_age_bracket_index}

                        tourists_group["under_age_agent"] = under_age_agent
                        tourists_group["group_accom_id"] = group_accom_id

                        avg_age = round(sum(ages) / num_tourists)

                        for i, ab in enumerate(self.age_brackets):
                            if avg_age >= ab[0] and avg_age <= ab[1]:
                                tourists_group["age_bracket_index"] = i
                                break

    def get_next_available_agent_id(self):
        if not self.agents:
            return 0
        else:
            return max(self.agents.keys()) + 1
