import numpy as np
from simulator import util

class Tourism:
    def __init__(self, tourismparams, cells, tourists, agents, touristsgroupsdays, touristsgroups, rooms_by_accomid_by_accomtype, tourists_arrivals_departures_for_day, tourists_active_groupids, age_brackets, epi_util):
        self.tourists_arrivals_departures_for_day = tourists_arrivals_departures_for_day
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
        self.day_timesteps = np.arange(145)

    def initialize_foreign_arrivals_departures_for_day(self, day):
        tourist_groupids_by_day = self.touristsgroupsdays[day]

        largest_agent_id = np.max(np.array(sorted(list(self.agents.keys()))))

        last_agent_id = largest_agent_id + 1

        for tour_group_id in tourist_groupids_by_day:
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
                    arrival_ts = np.random.choice(self.day_timesteps, size=1)[0]
                    arrival_airport_duration = util.sample_gamma_reject_out_of_range(self.incoming_duration_shape_param, self.incoming_duration_min, self.incoming_duration_max, 1, True, True)

                    self.tourists_arrivals_departures_for_day[tour_group_id] = {"ts": arrival_ts, "airport_duration": arrival_airport_duration, "arrival": True}
                    self.tourists_active_groupids.append(tour_group_id) 
                else:
                     # sample a departure timestep
                    departure_ts = np.random.choice(self.day_timesteps, size=1)[0]
                    departure_airport_duration = util.sample_gamma_reject_out_of_range(self.outgoing_duration_shape_param, self.outgoing_duration_min, self.outgoing_duration_max, 1, True, True)

                    self.tourists_arrivals_departures_for_day[tour_group_id] = {"ts": departure_ts, "airport_duration": departure_airport_duration, "arrival": False}
                    # self.tourists_active_groupids.remove(tour_group_id)

                for accinfoindex, accinfo in enumerate(accominfo):
                    accomid, roomid, roomsize = accinfo[0], accinfo[1], accinfo[2]

                    subgroupmmembers = subgroupsmemberids[accinfoindex] # this room

                    cellindex = self.rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                    if arrivalday == day:
                        self.cells[cellindex]["place"]["member_uids"] = subgroupmmembers

                        # handle as part of "agents" dict, to avoid having to do checks against 2 dicts
                        for tourist_id in subgroupmmembers: # tourists ids in room
                            tourist = self.tourists[tourist_id]
                            tourist["agentid"] = last_agent_id

                            age_bracket_index = -1
                            for i, ab in enumerate(self.age_brackets):
                                if tourist["age"] >= ab[0] and tourist["age"] <= ab[1]:
                                    age_bracket_index = i
                                    break

                            epi_age_bracket_index = self.epi_util.get_sus_mort_prog_age_bracket_index(tourist["age"])

                            self.agents[last_agent_id] = { "touristid": tourist_id, "curr_cellid": cellindex, "res_cellid": cellindex, "state_transition_by_day": {}, "age_bracket_index": age_bracket_index, "epi_age_bracket_index": epi_age_bracket_index}

                            last_agent_id += 1
                    else:
                        self.cells[cellindex]["place"]["member_uids"] = []

                        # remove from "agents" dict
                        # this means that tourists arriving on later days may be assigned the same agent id. 
                        # tourists still have to be uniquely identified by tourist_id

                        for tourist_id in subgroupmmembers: # tourists ids in room
                            tourist = self.tourists[tourist_id]
                            agent_id = tourist["agentid"]

                            del self.agents[agent_id]

        return self.agents, self.tourists, self.cells, self.tourists_arrivals_departures_for_day, self.tourists_active_groupids