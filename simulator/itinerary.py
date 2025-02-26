import numpy as np
import random
from enum import Enum
from enum import IntEnum
from copy import deepcopy
from copy import copy
import util, seirstateutil
from epidemiology import Epidemiology
from epidemiologyclasses import SEIRState, InfectionType, Severity, QuarantineType
import os
import traceback

class Itinerary:
    def __init__(self, 
                params, 
                timestepmins, 
                n_locals, 
                n_tourists, 
                locals_ratio_to_full_pop,
                agents_static,
                it_agents, 
                agents_epi,
                agents_ids_by_ages,          
                vars_util,
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
                dynparams,
                tourists=None,  
                process_index=-1):
        
        self.rng = np.random.default_rng() # seed=6

        self.one_to_two_hours = np.arange(6, 13)

        self.vars_util = vars_util
        # self.cells_agents_timesteps = cells_agents_timesteps # to be filled in during itinerary generation. key is cellid, value is (agentid, starttimestep, endtimestep)
        self.epi_util = Epidemiology(epidemiologyparams, n_locals, n_tourists, locals_ratio_to_full_pop, agents_static, agents_epi, self.vars_util, cells_households, cells_institutions, cells_accommodation, dynparams, process_index)

        # self.sync_queue = sync_queue
        # self.comm = comm
        # self.rank = rank
        self.params = params
        self.timestepmins = timestepmins
        self.timesteps_in_hour = round(60 / self.timestepmins)
        self.n_locals = n_locals
        self.n_tourists = n_tourists
        self.agents_static = agents_static
        self.it_agents = it_agents
        self.agents_epi = agents_epi
        self.agents_ids_by_ages = agents_ids_by_ages
        self.tourists = tourists
        self.industries = cells_industries_by_indid_by_wpid
        # print("industries keys: " + str(self.industries.keys()))
        self.cells_restaurants = cells_restaurants
        self.cells_hospital = cells_hospital
        self.cells_testinghub = cells_testinghub
        self.cells_vaccinationhub = cells_vaccinationhub
        self.cells_entertainment = cells_entertainment_by_activityid
        self.cells_religious = cells_religious
        self.cells_households = cells_households
        self.cells_breakfast_by_accomid = cells_breakfast_by_accomid
        self.cells_airport = cells_airport
        self.cells_transport = cells_transport

        # tourism
        self.tourism_group_activity_probability_by_purpose = self.params["tourism_group_activity_probability_by_purpose"]
        self.tourism_accom_breakfast_probability = self.params["tourism_accom_breakfast_probability"]

        # local
        self.guardian_age_threshold = self.params["guardian_age_threshold"]
        self.non_daily_activities_employed_distribution = self.params["non_daily_activities_employed_distribution"] # [minage, maxage, normalworkday, vacation_local, vacation_travel, sick_leave]
        self.non_daily_activities_schools_distribution = self.params["non_daily_activities_schools_distribution"]
        self.non_daily_activities_nonworkingday_distribution = self.params["non_daily_activities_nonworkingday_distribution"] # [minage, maxage, local, travel, sick]
        self.non_daily_activities_num_days = self.params["non_daily_activities_num_days"]

        if self.epi_util.dyn_params.airport_lockdown:
            nda_employed_vacation_travel_col_index = 2 # without age range (i.e. 4-2)
            nda_nonworkday_travel_col_index = 1 # without age range (i.e. 3-2)

            # split age_ranges from frequency distribution matrix
            nda_age_ranges = [sublst[:2] for sublst in self.non_daily_activities_employed_distribution]
            nda_empl_sliced_arr = [sublst[2:] for sublst in self.non_daily_activities_employed_distribution]
            nda_nonworkday_sliced_arr = [sublst[2:] for sublst in self.non_daily_activities_nonworkingday_distribution]

            # convert to numpy array
            nda_age_ranges = np.array(nda_age_ranges)
            nda_empl_sliced_arr = np.array(nda_empl_sliced_arr)
            nda_nonworkday_sliced_arr = np.array(nda_nonworkday_sliced_arr)

            # change the second element of every row to 0
            nda_empl_sliced_arr[:, nda_employed_vacation_travel_col_index] = 0
            nda_nonworkday_sliced_arr[:, nda_nonworkday_travel_col_index] = 0

            # normalize the array such that each row sums to 0
            # calculate row sums and reshape
            nda_empl_sliced_arr_row_sums = nda_empl_sliced_arr.sum(axis=1).reshape(-1, 1) # sums to 1 horizontallys
            nda_nonworkday_sliced_arr_row_sums = nda_nonworkday_sliced_arr.sum(axis=1).reshape(-1, 1)

            nda_empl_normalized_arr = nda_empl_sliced_arr / nda_empl_sliced_arr_row_sums # normalize
            nda_nonworkday_normalized_arr = nda_nonworkday_sliced_arr / nda_nonworkday_sliced_arr_row_sums 

            self.non_daily_activities_employed_distribution = []
            self.non_daily_activities_nonworkingday_distribution = []
            for index, age_range in enumerate(nda_age_ranges):
                self.non_daily_activities_employed_distribution.append(np.concatenate((age_range, nda_empl_normalized_arr[index])).tolist())
                self.non_daily_activities_nonworkingday_distribution.append(np.concatenate((age_range, nda_nonworkday_normalized_arr[index])).tolist())

        self.industries_working_hours = self.params["industries_working_hours"]
        self.activities_working_hours = self.params["activities_workplaces_working_hours_overrides"]
        self.working_categories_mindays = self.params["working_categories_mindays"]
        self.shift_working_hours = self.params["shift_working_hours"]
        self.sleeping_hours_by_age_groups = self.params["sleeping_hours_by_age_groups"]
        self.religious_ceremonies_hours = self.params["religious_ceremonies_hours"]
        sleeping_hours_range = self.params["sleeping_hours_range"]
        self.min_sleep_hours, self.max_sleep_hours = sleeping_hours_range[0], sleeping_hours_range[1]
        self.activities_by_week_days_distribution = self.params["activities_by_week_days_distribution"]
        self.activities_by_agerange_distribution = self.params["activities_by_agerange_distribution"]
        self.activities_duration_hours = self.params["activities_duration_hours"]
        self.tourism_airport_duration_min_max = self.params["tourism_airport_duration_min_max"]
        self.public_transport_usage_non_reg_daily_probability = self.params["public_transport_usage_probability"][1]

        self.age_brackets = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in self.sleeping_hours_by_age_groups] # [[0, 4], [5, 9], ...]
        self.age_brackets_workingages = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in self.non_daily_activities_employed_distribution] # [[15, 19], [20, 24], ...]

        self.potential_timesteps = np.arange(144)

        self.tourist_entry_infection_probability = epidemiologyparams["tourist_entry_infection_probability"] 
        self.pre_exposure_days = epidemiologyparams["pre_exposure_days"]

        self.working_schedule_interprocess_communication_aggregated_time = 0
        self.itinerary_interprocess_communication_aggregated_time = 0
        
        self.process_index = process_index

        # these always start at 0 for a specific day
        self.new_tests = 0
        self.new_vaccinations = 0
        self.new_quarantined = 0 # this is incremented/decremented at the start/end of quarantine, respectively, then simply added to the main stat
        self.new_hospitalised = 0 # this is incremented/decremented at the start/end of hospitalisation, respectively, then simply added to the main stat

        # Calculate probability of each activity for agent
        self.activities_by_week_days_by_age_groups = {}

        # 1 Bars/nightclubs, 2 food, 3 entertainment (indoor), 4 entertainment (outdoor), 5 gym, 6 sport, 7 shopping, 8 religious, 9 stay home, 10 other residence visit
        self.entertainment_activity_ids = [1, 3, 4, 5]
        self.non_entertainment_activity_ids = [2, 6, 7, 8, 10]

        for age_bracket_index, activity_probs_by_agerange_dist in enumerate(self.activities_by_agerange_distribution):
            activity_probs_for_agerange = activity_probs_by_agerange_dist[2:]

            activities_by_week_days_for_agerange = []

            for activity_index, activity_prob in enumerate(activity_probs_for_agerange):
                activity_id = activity_index + 1
                activity_by_week_days_for_age_range = [activity_id]

                for day in range(1, 8):
                    if self.epi_util.dyn_params.entertainment_lockdown and activity_id in self.entertainment_activity_ids:
                        prob_product = 0
                    elif self.epi_util.dyn_params.activities_lockdown and activity_id in self.non_entertainment_activity_ids:
                        prob_product = 0
                    else:
                        prob_product = self.activities_by_week_days_distribution[activity_index][day] * activity_prob

                    activity_by_week_days_for_age_range.append(prob_product)

                activities_by_week_days_for_agerange.append(activity_by_week_days_for_age_range)

            # remove activity id from 2d matrix
            sliced_arr = [sublst[1:] for sublst in activities_by_week_days_for_agerange]

            sliced_arr = np.array(sliced_arr)

            # Divide each element by the sum of the column (maintain sum to 1), i.e. distribution across days of the week
            normalized_arr = sliced_arr / sliced_arr.sum(axis=0) # sums to 1 vertically

            self.activities_by_week_days_by_age_groups[age_bracket_index] = normalized_arr

    # to be called at the beginning of a new week
    def generate_working_days_for_week_residence(self, resident_uids, is_hh):
        for agentindex, agentid in enumerate(resident_uids):
            agent_dynamic = self.it_agents[agentid]

            if seirstateutil.agents_seir_state_get(self.vars_util.agents_seir_state, agentid) != SEIRState.Deceased: #agentindex
                # if "working_schedule" not in agent:
                #     agent["working_schedule"] = {}
                
                if self.agents_static.get(agentid, "empstatus") == 0: # 0: employed, 1: unemployed, 2: inactive
                    prev_working_schedule = agent_dynamic["working_schedule"]

                    if prev_working_schedule is None:
                        prev_working_schedule = {}

                    # employed
                    agent_dynamic["working_schedule"] = {} # {workingday:(start,end)}

                    working_schedule = agent_dynamic["working_schedule"]
                    
                    agent_industry = Industry(self.agents_static.get(agentid, "empind"))

                    industry_working_hours_by_ind = self.industries_working_hours[agent_industry - 1]
                    industry_working_week_start_day, industry_working_week_end_day, industry_working_days = industry_working_hours_by_ind[1], industry_working_hours_by_ind[2], industry_working_hours_by_ind[3]

                    ent_activity = self.agents_static.get(agentid, "ent_activity")
                    if agent_industry == Industry.ArtEntertainmentRecreation and ent_activity is None:
                        print("problem in agents, agent " + str(agentid) + " None ent_activity")
                        
                    if agent_industry == Industry.ArtEntertainmentRecreation and ent_activity > -1:
                        activity_working_hours_overrides = self.activities_working_hours[ent_activity - 1]
                        industry_start_work_hour, industry_end_work_hour, industry_working_hours = activity_working_hours_overrides[2], activity_working_hours_overrides[3], activity_working_hours_overrides[4]
                    else:
                        industry_start_work_hour, industry_end_work_hour, industry_working_hours = industry_working_hours_by_ind[4], industry_working_hours_by_ind[5], industry_working_hours_by_ind[6]
                    
                    is_shift_based = self.agents_static.get(agentid, "isshiftbased")

                    working_days = []

                    if is_shift_based:
                        min_working_days = self.working_categories_mindays[1][1]

                        max_working_days = min_working_days + 1

                        num_working_days = self.rng.choice(np.arange(min_working_days, max_working_days + 1), size=1)[0]

                        working_days_range = np.arange(industry_working_week_start_day, industry_working_week_end_day + 1)

                        working_days = self.rng.choice(working_days_range, size=num_working_days, replace=False)
                        
                        working_days = sorted(working_days)

                        for index, day in enumerate(working_days):
                            if index == 0 and 7 in prev_working_schedule or (working_days[index - 1] == day-1):
                                if index == 0 and 7 in prev_working_schedule:
                                    previous_day_schedule = prev_working_schedule[7]
                                else:
                                    previous_day_schedule = working_schedule[working_days[index - 1]]

                                previous_day_start_hour = previous_day_schedule[0]

                                for shift_working_hour_option in self.shift_working_hours:
                                    if shift_working_hour_option[1] == previous_day_start_hour: # if starting hour is the same, it ensures 24 hours would have passed
                                        working_schedule[day] = (shift_working_hour_option[1], shift_working_hour_option[2])
                                        break
                            else:
                                working_hours_options = np.arange(len(self.shift_working_hours))
                                
                                sampled_working_hours_option = self.rng.choice(working_hours_options, size=1)[0]

                                working_schedule[day] = (self.shift_working_hours[sampled_working_hours_option][1], self.shift_working_hours[sampled_working_hours_option][2])
                    else:
                        empftpt = self.agents_static.get(agentid, "empftpt")
                        is_full_time = empftpt == 0 or empftpt == 1

                        if is_full_time:
                            min_working_days = self.working_categories_mindays[0][1]
                        else:
                            min_working_days = self.working_categories_mindays[2][1]
                    
                        max_working_days = industry_working_days

                        num_working_days = self.rng.choice(np.arange(min_working_days, max_working_days + 1), size=1)[0]

                        working_days_range = np.arange(industry_working_week_start_day, industry_working_week_end_day + 1)

                        working_days = self.rng.choice(working_days_range, size=num_working_days, replace=False)

                        working_days = sorted(working_days)
                        
                        ind_start_work_hour = industry_start_work_hour
                        ind_end_work_hour = industry_end_work_hour

                        if industry_end_work_hour < industry_start_work_hour: # indicates working overnight
                            ind_end_work_hour = 24 + industry_end_work_hour

                        working_hours_range = np.arange(ind_start_work_hour, ind_end_work_hour + 1)
                        
                        for day in working_days:
                            if is_full_time: # assumed 8 hours (8 hours does not work overnight)
                                if industry_working_hours > 8: # 2 options, start from beginning or half
                                    options = np.arange(2)
                                    sampled_option = self.rng.choice(options, size=1)[0]

                                    if sampled_option == 0:
                                        start_hour = industry_start_work_hour
                                    else:
                                        if industry_working_hours == 16:
                                            start_hour = industry_start_work_hour + 8
                                        else:
                                            start_hour = industry_start_work_hour + 4

                                    if start_hour + 8 <= industry_end_work_hour:
                                        end_hour = start_hour + 8
                                    else:
                                        end_hour = start_hour + 4

                                    working_schedule[day] = (start_hour, end_hour)
                                else:
                                    working_schedule[day] = (industry_start_work_hour, industry_end_work_hour)
                            else: # part time
                                possible_slots = int(industry_working_hours / 4)
                                options = np.arange(possible_slots)
                                sampled_option = self.rng.choice(options, size=1)[0]

                                start_hour = sampled_option * 4

                                # if start_hour > 0:
                                #     start_hour += 1

                                end_hour = start_hour + 4

                                actual_start_hour = working_hours_range[start_hour]
                                actual_end_hour = working_hours_range[end_hour]

                                if actual_end_hour > 24:
                                    actual_end_hour = actual_end_hour - 24

                                working_schedule[day] = (actual_start_hour, actual_end_hour)
                    
                    # in this scenario, the whole agent object is being passed over to the main process to be synced (because working_schedule might not be a key in main memory)
                    # however, a better approach would be to pass an extra param to indicate creation of a key in main memory
                    # having said that, this only happens on the first day of every week, for working agents only (i.e. it may not affect considerably)
                    # self.sync_queue.put(["a", agentid, "", agent]) 

                    # start = time.time()
                    # self.sync_queue.put(["a", agentid, "working_schedule", working_schedule])
                    # # Share the result with the main process
                    # # self.comm.send(["a", agentid, "working_schedule", working_schedule], dest=0, tag=self.rank)
                    # time_taken = time.time() - start
                    # self.working_schedule_interprocess_communication_aggregated_time += time_taken

    def generate_local_itinerary(self, simday, weekday, resident_uids):
        guardian_id, guardian_hospitalisation_days, guardian_quarantine_days, guardian_non_daily_activity_recurring, guardian_prevday_non_daily_activity_recurring, guardian_itinerary, guardian_itinerary_nextday = None, None, None, None, None, None, None
        res_cellid = None

        cohab_agents_ids_by_ages = {}
        for agentid in resident_uids:
            cohab_agents_ids_by_ages[agentid] = self.agents_ids_by_ages[agentid]

            if guardian_id is None:
                agent = self.it_agents[agentid]

                temp_guardian_id = self.agents_static.get(agentid, "guardian_id")
                if temp_guardian_id != -1:
                    guardian_id = temp_guardian_id

        cohab_agents_ids_by_ages = sorted(cohab_agents_ids_by_ages.items(), key=lambda y: y[1], reverse=True)

        for agentindex, (agentid, age) in enumerate(cohab_agents_ids_by_ages):
            agent = self.it_agents[agentid]
            agent_epi = self.agents_epi[agentid]

            agent_seir_state = seirstateutil.agents_seir_state_get(self.vars_util.agents_seir_state, agentid) #agentindex

            if agent_seir_state != SEIRState.Deceased:  
                agent_is_guardian = guardian_id is not None and agentid == guardian_id

                new_seir_state, new_infection_type, new_infection_severity = None, None, None

                if res_cellid is None:
                    res_cellid = self.agents_static.get(agentid, "res_cellid")

                if simday == 1 and agent_seir_state == SEIRState.Infectious:
                    # assume infection happened in the previous 14 days, sample an "exposed" start day in the last 14 days to simulate the seir state transitioning trajectory
                    potential_days = np.arange(simday - self.pre_exposure_days, simday) # agent would have been exposed on a particular day/timestamp during the previous 14 days (until day 0, i.e. last day before sim start)

                    sampled_day = self.rng.choice(potential_days, size=1, replace=False)[0]

                    agent_state_transition_by_day = agent_epi["state_transition_by_day"]
                    agent_epi_age_bracket_index = self.agents_static.get(agentid, "epi_age_bracket_index")
                    agent_quarantine_days = agent_epi["quarantine_days"]

                    if agent_state_transition_by_day is None:
                        agent_state_transition_by_day = []
                    
                    agent_state_transition_by_day, seir_state, inf_type, inf_sev, _ = self.epi_util.simulate_seir_state_transition(simday, agent_epi, agentid, sampled_day, self.potential_timesteps, agent_state_transition_by_day, agent_epi_age_bracket_index, agent_quarantine_days)

                    agent_epi["state_transition_by_day"] = agent_state_transition_by_day
                    agent_epi["quarantine_days"] = agent_quarantine_days
                    
                    self.vars_util.agents_seir_state = seirstateutil.agents_seir_state_update(self.vars_util.agents_seir_state, seir_state, agentid)
                    self.vars_util.agents_infection_type[agentid] = inf_type
                    self.vars_util.agents_infection_severity[agentid] = inf_sev

                    # print(f"updating agent (simday1 and infectious case) {agentid}, state {self.vars_util.agents_seir_state[agentid]}, inf_type {self.vars_util.agents_infection_type[agentid]}, inf_sev {self.vars_util.agents_infection_severity[agentid]}")

                 # this updates the state, infection type and severity, if relevant for the current day! (such that the itinenary may also handle public health interventions)
                new_states = seirstateutil.update_agent_state(self.vars_util.agents_seir_state, self.vars_util.agents_infection_type, self.vars_util.agents_infection_severity, agentid, agent_epi, agentindex, simday)

                # if new_states is not None:
                #     new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep = new_states[0], new_states[1], new_states[2], new_states[3], new_states[4], new_states[5]

                #     self.vars_util.agents_seir_state_transition_for_day[agentid] = [new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep]

                age_bracket_index = self.agents_static.get(agentid, "age_bracket_index")
                working_age_bracket_index = self.agents_static.get(agentid, "working_age_bracket_index")

                is_departureday, is_arrivalday, is_work_or_school_day = False, False, False
                sampled_non_daily_activity = None
                prevday_non_daily_end_activity = None
                wakeup_timestep, sleep_timestep, min_start_sleep_hour = None, None, None
                sampled_departure_timestep = None
                next_day_arrive_home_timestep, next_day_sleep_timestep = None, None
                non_daily_activity_recurring = None
                
                start_work_timestep_with_leeway, end_work_timestep_with_leeway = None, None
                start_school_timestep, end_school_timestep = None, None
                
                prev_day_itinerary, prev_day_last_event = None, None
                if simday > 1:
                    prev_day_itinerary = agent["itinerary"]
                    # prev_day_itinerary_nextday = agent["itinerary_nextday"]

                    prev_day_itinerary_timesteps = list(prev_day_itinerary.keys()) # should only be the last timestep of the previous day
                    # prev_day_itinerary_timesteps = sorted(prev_day_itinerary.keys(), reverse=True)

                    if len(prev_day_itinerary_timesteps) > 0:
                        prev_day_last_ts = prev_day_itinerary_timesteps[0]
                        prev_day_last_action, prev_day_last_cell = prev_day_itinerary[prev_day_last_ts]
                        prev_day_last_event = prev_day_last_ts, prev_day_last_action, prev_day_last_cell
                
                hosp_start_day_ts, hosp_end_day_ts, hosp_start_day, hosp_end_day = None, None, None, None
                quar_start_day_ts, quar_end_day_ts, quar_start_day, quar_end_day = None, None, None, None

                guardian = None
                is_guardian_quarantined, is_guardian_hospitalised = False, False
                if age < self.guardian_age_threshold and guardian_id is not None:
                    if guardian_hospitalisation_days is not None and len(guardian_hospitalisation_days) > 0:
                        hosp_start_day, hosp_end_day = guardian_hospitalisation_days[0], guardian_hospitalisation_days[2]

                        if simday >= hosp_start_day and simday <= hosp_end_day:
                            is_guardian_hospitalised = True

                    if guardian_quarantine_days is not None and len(guardian_quarantine_days) > 0:
                        quar_start_day, quar_end_day = guardian_quarantine_days[0], guardian_quarantine_days[2]

                        if simday >= quar_start_day and simday <= quar_end_day:
                            is_guardian_quarantined = True

                is_quarantined, is_hospitalised = False, False
                is_quarantined_or_hospitalised, is_quarantine_hospital_start_day, is_quarantine_hospital_end_day, is_prevday_quar_hosp_end_day, is_nextday_quar_hosp_start_day = False, False, False, False, False
                hosp_start_day_ts, hosp_end_day_ts, hosp_start_day, hosp_end_day = None, None, None, None
                quar_start_day_ts, quar_end_day_ts, quar_start_day, quar_end_day = None, None, None, None

                if agent_epi["hospitalisation_days"] is not None and len(agent_epi["hospitalisation_days"]) > 0:
                    hosp_start_day, hosp_end_day = agent_epi["hospitalisation_days"][0], agent_epi["hospitalisation_days"][2]

                    # hosp_start_day = hosp_start_day_ts[0]
                    # hosp_end_day = hosp_end_day_ts[0]

                    if simday >= hosp_start_day and simday <= hosp_end_day:
                        # print("is hospitalised")
                        is_hospitalised = True
                        is_quarantined_or_hospitalised = True

                    if simday == hosp_start_day:
                        is_quarantine_hospital_start_day = True

                    if simday == hosp_end_day:
                        is_quarantine_hospital_end_day = True

                    # if simday == hosp_end_day + 1:
                    #     is_prevday_quar_hosp_end_day = True

                    # if simday+1 == hosp_start_day:
                    #     is_nextday_quar_hosp_start_day = True

                if agent_epi["quarantine_days"] is not None and len(agent_epi["quarantine_days"]) > 0:
                    quar_start_day, quar_end_day = agent_epi["quarantine_days"][0], agent_epi["quarantine_days"][2]

                    # quar_start_day = quar_start_day_ts[0]
                    # quar_end_day = quar_end_day_ts[0]

                    if simday >= quar_start_day and simday <= quar_end_day:
                        # print("is quarantined")
                        is_quarantined = True
                        is_quarantined_or_hospitalised = True

                    if simday == quar_start_day:
                        is_quarantine_hospital_start_day = True

                    if simday == quar_end_day:
                        is_quarantine_hospital_end_day = True

                    # if simday == quar_end_day + 1:
                    #     is_prevday_quar_hosp_end_day = True

                    # if simday+1 == quar_start_day:
                    #     is_nextday_quar_hosp_start_day = True

                    # if is_quarantined_start_day or is_quarantined_end_day still to handle as per usual
                    # to see how to handle the else of this, and structure accordingly

                if "non_daily_activity_recurring" not in agent:
                    agent["non_daily_activity_recurring"] = None

                # always reset on the next day (this is only to be referred to by the guardian and cleared on the next day)
                if "prevday_non_daily_activity_recurring" not in agent or (agent["prevday_non_daily_activity_recurring"] is not None and len(agent["prevday_non_daily_activity_recurring"]) > 0):
                    agent["prevday_non_daily_activity_recurring"] = None

                if "non_daily_activity_recurring" in agent and agent["non_daily_activity_recurring"] is not None: # and (not is_quarantined or is_quarantine_start_day or is_quarantine_end_day)
                    non_daily_activity_recurring = copy(agent["non_daily_activity_recurring"])

                    if simday in non_daily_activity_recurring:
                        sampled_non_daily_activity = non_daily_activity_recurring[simday]

                    if sampled_non_daily_activity is None:
                        if simday-1 in non_daily_activity_recurring:
                            prevday_non_daily_end_activity = copy(non_daily_activity_recurring[simday-1])
                            agent["prevday_non_daily_activity_recurring"] = non_daily_activity_recurring
                            agent["non_daily_activity_recurring"] = None
                            
                            if prevday_non_daily_end_activity == NonDailyActivity.Travel: # clear as at this point, values here would be stale
                                agent["itinerary"] = None
                                agent["itinerary_nextday"] = None

                # first include overnight entries or sample wakeup timestep, and sample work/school start/end timesteps (if applicable)
                # then, sample_non_daily_activity, unless recurring sampled_non_daily_activity already exists
                # dict for this day (itinerary) potentially up onto next day (itinerary_nextday)
                # only skip if Travel, as fine-grained detail is not useful in outbound travel scenario
                if sampled_non_daily_activity is None or sampled_non_daily_activity != NonDailyActivity.Travel:                
                    # get previous night sleep time
                    prev_night_sleep_hour, prev_night_sleep_timestep, same_day_sleep_hour, same_day_sleep_timestep = None, None, None, None
                    overnight_end_work_ts, overnight_end_activity_ts, activity_overnight_cellid = None, None, None
                    
                    if simday == 1 or is_quarantine_hospital_end_day: # sample previous night sleeptimestep for first simday or if previous day was quarantined or hospitalised
                        prev_weekday = weekday - 1

                        if prev_weekday < 0:
                            prev_weekday = 7

                        sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                        min_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7]
                        
                        alpha, beta = alpha_weekday, beta_weekday
                        if prev_weekday == 6 or prev_weekday == 7: # weekend
                            alpha, beta = alpha_weekend, beta_weekend

                        sampled_sleep_hour_from_range = round(self.rng.beta(alpha, beta, 1)[0] * start_hour_range + 1)

                        prev_night_sleep_hour = min_start_sleep_hour + (sampled_sleep_hour_from_range - 1) # this is 1 based; if sampled_sleep_hour_from_range is 1, sleep_hour should be min_start_sleep_hour

                        if prev_night_sleep_hour >= 24:
                            if prev_night_sleep_hour == 24:
                                prev_night_sleep_hour = 0
                            else:
                                prev_night_sleep_hour -= 24

                            same_day_sleep_hour = prev_night_sleep_hour
                            prev_night_sleep_hour = None

                        if prev_night_sleep_hour is not None:
                            prev_night_sleep_timestep = self.get_timestep_by_hour(prev_night_sleep_hour)
                        else:
                            same_day_sleep_timestep = self.get_timestep_by_hour(same_day_sleep_hour)
                    elif prevday_non_daily_end_activity == NonDailyActivity.Travel:
                        is_arrivalday = True
                        departure_day = None
                        prevday_non_daily_activity_recurring_keys = list(agent["prevday_non_daily_activity_recurring"].keys()) # should always have keys cause prev day is travel
                        departure_day = prevday_non_daily_activity_recurring_keys[0]

                        if (guardian is None or 
                            agent["prevday_non_daily_activity_recurring"] != guardian_prevday_non_daily_activity_recurring):
                            agent["itinerary"] = None # {timestep: cellindex}
                            agent["itinerary_nextday"] = None
                            
                            # potential_arrival_timesteps = np.arange(0, 144)

                            sampled_arrival_timestep = self.rng.choice(self.potential_timesteps, size=1)[0]

                            potential_timesteps_until_arrive_home = np.arange(12, 19) # 2 - 3 hours until arrive home

                            sampled_timesteps_until_arrive_home = self.rng.choice(potential_timesteps_until_arrive_home, size=1)[0]

                            arrive_home_timestep = sampled_arrival_timestep + sampled_timesteps_until_arrive_home
                            
                            self.sample_airport_activities(agent, res_cellid, range(sampled_arrival_timestep, arrive_home_timestep+1), True, False, 0)

                            if arrive_home_timestep > 143:
                                next_day_arrive_home_timestep = arrive_home_timestep - 143

                            if next_day_arrive_home_timestep is not None: # assume sleep in the next hour
                                sampled_timesteps_until_sleep =  self.rng.choice(np.arange(1, 7), size=1, replace=False)[0]
                                
                                next_day_sleep_timestep = next_day_arrive_home_timestep + sampled_timesteps_until_sleep
                            else:
                                wakeup_timestep = arrive_home_timestep
                        else:
                            agent["itinerary"] = deepcopy(guardian_itinerary) # create a copy by value, so it can be extended without affecting guardian and vice versa
                            agent["itinerary_nextday"] = deepcopy(guardian_itinerary_nextday)

                            if agent["itinerary"] is not None:
                                for timestep, (action, _) in agent["itinerary"].items():
                                    if action == Action.Home:
                                        wakeup_timestep = timestep
                                        break
                                        
                            if agent["itinerary_nextday"] is not None:
                                for timestep, (action, _) in agent["itinerary_nextday"].items():
                                    if action == Action.Sleep:
                                        next_day_sleep_timestep = timestep
                                        
                                    if action == Action.Home:
                                        next_day_arrive_home_timestep = timestep

                                    if next_day_sleep_timestep is not None and next_day_arrive_home_timestep is not None:
                                        break
                            
                        # immunity_multiplier, asymptomatic_multiplier = 1, 1
                        immunity_multiplier, asymptomatic_multiplier, _ = util.calculate_vaccination_multipliers(self.vars_util.agents_vaccination_doses, agentid, simday, self.epi_util.vaccination_immunity_multiplier, self.epi_util.vaccination_asymptomatic_multiplier, self.epi_util.vaccination_exp_decay_interval)

                        # sample local re-entry infection probability
                        exposed_rand = random.random()

                        entry_infection_probability = self.tourist_entry_infection_probability * immunity_multiplier

                        if exposed_rand < entry_infection_probability:                 
                            potential_days = np.arange(departure_day, simday+1) # agent would have been exposed on a particular day/timestamp during the travel. 

                            sampled_day = self.rng.choice(potential_days, size=1, replace=False)[0] # sample a day at random from the travel duration.

                            agent_state_transition_by_day = agent_epi["state_transition_by_day"]
                            agent_epi_age_bracket_index = self.agents_static.get(agentid, "epi_age_bracket_index")
                            agent_quarantine_days = agent_epi["quarantine_days"]

                            if agent_state_transition_by_day is None:
                                agent_state_transition_by_day = []

                            agent_state_transition_by_day, seir_state, inf_type, inf_sev, _ = self.epi_util.simulate_seir_state_transition(simday, agent_epi, agentid, sampled_day, self.potential_timesteps, agent_state_transition_by_day, agent_epi_age_bracket_index, agent_quarantine_days, asymptomatic_multiplier)
                            
                            agent_epi["state_transition_by_day"] = agent_state_transition_by_day
                            agent_epi["quarantine_days"] = agent_quarantine_days

                            self.vars_util.agents_seir_state = seirstateutil.agents_seir_state_update(self.vars_util.agents_seir_state, seir_state, agentid)
                            self.vars_util.agents_infection_type[agentid] = inf_type
                            self.vars_util.agents_infection_severity[agentid] = inf_sev

                            # print(f"setting infection type {inf_type} and infection severity {inf_sev} for agent {agentid}, new infection_type_len {str(len(self.vars_util.agents_infection_type))}")
                        
                            # immediately clear any events that happened prior to today as they would not be relevant anymore
                            # delete_indices = []
                            # for day_entry_index, day_entry in enumerate(agent_state_transition_by_day): # iterate on the indices to be able to delete directly
                            #     if day_entry[0] < simday: # day is first index
                            #         delete_indices.append(day_entry_index)

                            # if len(delete_indices) > 0:
                            #     agent_state_transition_by_day = agent_state_transition_by_day[delete_indices[-1]+1:]

                            # agent_epi["state_transition_by_day"] = agent_state_transition_by_day
                                    
                            # this updates the state, infection type and severity, if relevant for the current day! (such that the itineray may also handle public health interventions)
                            # itinerary would have already tried to update agent state in the beginning for normal cases. this needs to be done again for returning agents
                            new_states = seirstateutil.update_agent_state(self.vars_util.agents_seir_state, self.vars_util.agents_infection_type, self.vars_util.agents_infection_severity, agentid, agent_epi, agentindex, simday)

                            # if new_states is not None:
                            #     new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep = new_states[0], new_states[1], new_states[2], new_states[3], new_states[4], new_states[5]

                            #     self.vars_util.agents_seir_state_transition_for_day[agentid] = [new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep]
                    else:
                        if agent["itinerary_nextday"] is not None and len(agent["itinerary_nextday"]) > 0: # overnight itinerary (scheduled from previous day; to include into "itinerary" dict)
                            # get morning sleeptimestep
                            worked_overnight = False
                            activity_overnight = False
                            return_home_ts = None
                            for timestep, (action, cellid) in agent["itinerary_nextday"].items():
                                if action == Action.Work:
                                    worked_overnight = True

                                if action == Action.LocalActivity:
                                    activity_overnight = True
                                    activity_overnight_cellid = cellid

                                if action == Action.Home:
                                    return_home_ts = timestep

                                if action == Action.Sleep:
                                    same_day_sleep_timestep = timestep
                                    same_day_sleep_hour = self.get_hour_by_timestep(same_day_sleep_timestep)

                            if same_day_sleep_hour is None: # both may be filled in or one of them, if one of them, assume they are the same and simply convert
                                same_day_sleep_hour = self.get_hour_by_timestep(return_home_ts)
                            elif return_home_ts is None:
                                return_home_ts = self.get_timestep_by_hour(same_day_sleep_hour)

                            if worked_overnight:
                                overnight_end_work_ts = return_home_ts
                            elif activity_overnight:
                                overnight_end_activity_ts = return_home_ts
                        else:
                            # get previous night sleeptimestep
                            if agent["itinerary"] is not None:
                                for timestep, (action, _) in agent["itinerary"].items():
                                    if action == Action.Sleep:
                                        prev_night_sleep_timestep = timestep
                                        prev_night_sleep_hour = self.get_hour_by_timestep(prev_night_sleep_timestep)
                                        break

                    if not is_arrivalday: # or (is_quarantine_hospital_start_day or is_quarantine_hospital_end_day)
                        # initialise "itinerary" and "itinerary_nextday" for today/overnight, accordingly
                        agent["itinerary"] = None # {timestep: cellindex}
                        agent["itinerary_nextday"] = None

                    # add overnight values into "itinerary" dict
                    if overnight_end_work_ts is not None:
                        self.add_to_itinerary(agent, 0, Action.Work, self.agents_static.get(agentid, "work_cellid"))
                        self.add_to_itinerary(agent, overnight_end_work_ts, Action.Home, res_cellid)
                    elif overnight_end_activity_ts is not None:
                        self.add_to_itinerary(agent, 0, Action.LocalActivity, activity_overnight_cellid)
                        self.add_to_itinerary(agent, overnight_end_activity_ts, Action.Home, res_cellid)

                    # set wake up hour
                    start_work_school_hour = None
                    wakeup_hour = None
                    working_schedule = None

                    if prevday_non_daily_end_activity == NonDailyActivity.Travel and next_day_sleep_timestep is not None and guardian is None:
                        self.add_to_itinerary(agent, next_day_sleep_timestep, Action.Sleep, res_cellid, next_day=True)                        

                    if "working_schedule" in agent:
                        working_schedule = agent["working_schedule"]

                    if next_day_sleep_timestep is None and next_day_arrive_home_timestep is None and (not is_quarantined_or_hospitalised or ((is_quarantine_hospital_start_day or is_quarantine_hospital_end_day) and not (is_quarantined and is_hospitalised))):
                        if wakeup_timestep is None:
                            empstatus = self.agents_static.get(agentid, "empstatus")
                            if empstatus == 0 or self.agents_static.get(agentid, "sc_student") == 1:
                                if empstatus == 0:
                                    if weekday in working_schedule: # will not be possible 2 days after each other for shift
                                        start_work_school_hour = working_schedule[weekday][0]

                                        if prev_night_sleep_hour is not None:
                                            latest_wake_up_hour = prev_night_sleep_hour + self.max_sleep_hours

                                            if latest_wake_up_hour >= 24:
                                                if latest_wake_up_hour == 24:
                                                    latest_wake_up_hour = 0
                                                else:
                                                    latest_wake_up_hour -= 24 
                                        elif same_day_sleep_hour is not None:
                                            latest_wake_up_hour = same_day_sleep_hour + self.max_sleep_hours
                                        else: # this would be previous day hospital / quarantined (assume sleep hour as would be irrelevant and not relative to other activities)
                                            latest_wake_up_hour = 0 + self.max_sleep_hours

                                        if latest_wake_up_hour <= 24 and latest_wake_up_hour >= start_work_school_hour - 1:
                                            wakeup_hour = start_work_school_hour - 1

                                            if wakeup_hour < 0:
                                                wakeup_hour = 0

                                            wakeup_timestep = self.get_timestep_by_hour(wakeup_hour) # force wake up before work
                                        
                                else: # student
                                    if weekday >= 1 and weekday <= 5: # weekday
                                        start_work_school_hour = 8
                                        
                                        if prev_night_sleep_hour is not None:
                                            latest_wake_up_hour = prev_night_sleep_hour + self.max_sleep_hours

                                            if latest_wake_up_hour >= 24:
                                                if latest_wake_up_hour == 24:
                                                    latest_wake_up_hour = 0
                                                else:
                                                    latest_wake_up_hour -= 24 
                                        elif same_day_sleep_hour is not None:
                                            latest_wake_up_hour = same_day_sleep_hour + self.max_sleep_hours
                                        else: # this would be previous day hospital/ quarantined (assume sleep hour as would be irrelevant and not relative to other activities)
                                            sleep_hours_range = np.arange(self.min_sleep_hours, self.max_sleep_hours + 1)

                                            # Calculate the middle index of the array
                                            mid = len(sleep_hours_range) // 2

                                            sigma = 1.0
                                            probs = np.exp(-(np.arange(len(sleep_hours_range)) - mid)**2 / (2*sigma**2))
                                            probs /= probs.sum()

                                            # Sample from the array with probabilities favouring the middle range (normal dist)
                                            sampled_sleep_hours_duration = self.rng.choice(sleep_hours_range, size=1, replace=False, p=probs)[0]

                                            latest_wake_up_hour = 0 + sampled_sleep_hours_duration

                                        if latest_wake_up_hour <= 24 and latest_wake_up_hour >= start_work_school_hour - 1:
                                            wakeup_hour = start_work_school_hour - 1
                                            wakeup_timestep = self.get_timestep_by_hour(wakeup_hour) # force wake up before school

                        if wakeup_timestep is None:
                            sleep_hours_range = np.arange(self.min_sleep_hours, self.max_sleep_hours + 1)

                            # Calculate the middle index of the array
                            mid = len(sleep_hours_range) // 2

                            sigma = 1.0
                            probs = np.exp(-(np.arange(len(sleep_hours_range)) - mid)**2 / (2*sigma**2))
                            probs /= probs.sum()

                            # Sample from the array with probabilities favouring the middle range (normal dist)
                            sampled_sleep_hours_duration = self.rng.choice(sleep_hours_range, size=1, replace=False, p=probs)[0]

                            # if prev_night_sleep_hour is None and wakeup_hour is None:
                            #     # this could be a case where agent shifts from quarantine to hospitalisation or vice versa
                            #     wakeup_hour = 7 # assume 7am
                            # else:
                            if prev_night_sleep_hour is not None:
                                wakeup_hour = prev_night_sleep_hour + sampled_sleep_hours_duration
                            elif same_day_sleep_hour is not None:
                                wakeup_hour = same_day_sleep_hour + sampled_sleep_hours_duration
                            else: # this would be previous day hospita/ quarantined (assume sleep hour as would be irrelevant and not relative to other activities)
                                wakeup_hour = 0 + sampled_sleep_hours_duration

                            if wakeup_hour > 24:
                                wakeup_hour = wakeup_hour - 24

                            wakeup_timestep = self.get_timestep_by_hour(wakeup_hour)

                        if is_quarantine_hospital_start_day or not is_arrivalday: # quarantine hospitalisation might cancel vacation
                            self.add_to_itinerary(agent, wakeup_timestep, Action.WakeUp, res_cellid)

                        # sample non daily activity (if not recurring)
                        end_work_next_day = False

                        # travel with guardian overrides
                        if guardian_id is not None and guardian_non_daily_activity_recurring is not None: 
                            # ensure travel recurring activity (skip if non travel)
                            if (guardian_non_daily_activity_recurring is not None and len(guardian_non_daily_activity_recurring) > 0 and 
                                guardian_non_daily_activity_recurring[list(guardian_non_daily_activity_recurring.keys())[0]] == NonDailyActivity.Travel):
                                # if kid within non daily activity of his/her own, skip
                                # if "non_daily_activity_recurring" not in agent or agent["non_daily_activity_recurring"] is None or agent["non_daily_activity_recurring"] == guardian["non_daily_activity_recurring"]:
                                guardian_sampled_non_daily_activity, guardian_sampled_non_daily_activity_departure_day = None, None

                                if guardian_non_daily_activity_recurring is not None:           
                                    if simday in guardian_non_daily_activity_recurring:
                                        if simday-1 not in guardian_non_daily_activity_recurring: # if first day of non_daily_activity_recurring
                                            guardian_sampled_non_daily_activity_departure_day = guardian_non_daily_activity_recurring[simday]
                                        else:
                                            guardian_sampled_non_daily_activity = guardian_non_daily_activity_recurring[simday]

                                    if guardian_sampled_non_daily_activity_departure_day is not None:
                                        is_departureday = True
                                        agent["non_daily_activity_recurring"] = guardian_non_daily_activity_recurring
                                        sampled_non_daily_activity = guardian_sampled_non_daily_activity_departure_day
                                    elif guardian_sampled_non_daily_activity is not None:
                                        if agent["non_daily_activity_recurring"] == guardian_non_daily_activity_recurring:
                                            sampled_non_daily_activity = guardian_sampled_non_daily_activity     

                        if sampled_non_daily_activity is None: # sample non daily activity (normal case)
                            # set the working / school hours
                            if self.agents_static.get(agentid, "empstatus") == 0 and not self.epi_util.dyn_params.workplaces_lockdown: # 0: employed, 1: unemployed, 2: inactive
                                # employed. consider workingday/ vacationlocal/ vacationtravel/ sickleave
                                
                                if weekday in working_schedule: # working day
                                    is_work_or_school_day = True

                                    working_hours = working_schedule[weekday]
                                    start_work_hour = working_hours[0]
                                    end_work_hour = working_hours[1]

                                    start_work_timestep_with_leeway, end_work_timestep_with_leeway = self.get_timestep_by_hour(start_work_hour, 2), self.get_timestep_by_hour(end_work_hour, 3)

                                    if prevday_non_daily_end_activity is not None and wakeup_timestep < start_work_timestep_with_leeway:
                                        sampled_non_daily_activity = NonDailyActivityEmployed.NormalWorkingDay
                                    else:
                                        non_daily_activities_dist_by_ab = self.non_daily_activities_employed_distribution[working_age_bracket_index]
                                        non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                                        non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                                        sampled_non_daily_activity_index = self.rng.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                                        sampled_non_daily_activity = NonDailyActivityEmployed(sampled_non_daily_activity_index + 1)
                                        
                                        if sampled_non_daily_activity == NonDailyActivityEmployed.NormalWorkingDay and wakeup_timestep > start_work_timestep_with_leeway:
                                            is_work_or_school_day = False # to avoid recurring local vacation
                                            sampled_non_daily_activity = NonDailyActivityEmployed.VacationLocal

                                        if sampled_non_daily_activity == NonDailyActivityEmployed.VacationTravel and prevday_non_daily_end_activity == NonDailyActivity.Travel:
                                            if wakeup_timestep < start_work_timestep_with_leeway:
                                                sampled_non_daily_activity = NonDailyActivityEmployed.NormalWorkingDay
                                            else:
                                                is_work_or_school_day = False # to avoid recurring local vacation
                                                sampled_non_daily_activity = NonDailyActivityEmployed.VacationLocal                            

                                    if sampled_non_daily_activity == NonDailyActivityEmployed.NormalWorkingDay: # sampled normal working day                             
                                        if start_work_timestep_with_leeway <= 143:
                                            self.add_to_itinerary(agent, start_work_timestep_with_leeway, Action.Work, self.agents_static.get(agentid, "work_cellid"))
                                        else:
                                            self.add_to_itinerary(agent, start_work_timestep_with_leeway - 143, Action.Work, self.agents_static.get(agentid, "work_cellid"), next_day=True)

                                        if end_work_timestep_with_leeway <= 143 and end_work_timestep_with_leeway > start_work_timestep_with_leeway:
                                            self.add_to_itinerary(agent, end_work_timestep_with_leeway, Action.Home, res_cellid)
                                        else:
                                            end_work_next_day = True

                                            if end_work_timestep_with_leeway > 143:
                                                end_work_ts_with_leeway = end_work_timestep_with_leeway - 143
                                            else:
                                                end_work_ts_with_leeway = end_work_timestep_with_leeway

                                            self.add_to_itinerary(agent, end_work_ts_with_leeway, Action.Home, res_cellid, next_day=True) 
                                    elif sampled_non_daily_activity == NonDailyActivityEmployed.VacationTravel:
                                        is_departureday = True                           

                                    sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                                else: 
                                    # non working day
                                    # unemployed/inactive. only consider local / travel / sick
                                    non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[working_age_bracket_index]
                                    non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                                    non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                                    sampled_non_daily_activity_index = self.rng.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                                    sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                                    if sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Travel:
                                        if guardian is None: # should never be not None here
                                            is_departureday = True
                                        else:
                                            sampled_non_daily_activity = NonDailyActivityNonWorkingDay.Local

                                    sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                            elif self.agents_static.get(agentid, "sc_student") == 1 and not self.epi_util.dyn_params.schools_lockdown:
                                if weekday <= 5: # monday to friday
                                    is_work_or_school_day = True

                                    start_school_hour = 8
                                    end_school_hour = 15 # students end 1 hour before teachers

                                    start_school_timestep, end_school_timestep = self.get_timestep_by_hour(start_school_hour), self.get_timestep_by_hour(end_school_hour)
                                
                                    # students. only consider schoolday / sick
                                    if prevday_non_daily_end_activity is not None and wakeup_timestep < start_school_timestep:
                                        sampled_non_daily_activity = NonDailyActivityStudent.NormalSchoolDay
                                    else:
                                        non_daily_activities_dist_by_ab = self.non_daily_activities_schools_distribution[age_bracket_index]
                                        non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                                        non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                                        sampled_non_daily_activity_index = self.rng.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                                        sampled_non_daily_activity = NonDailyActivityStudent(sampled_non_daily_activity_index + 1)
                                        
                                        if sampled_non_daily_activity == NonDailyActivityStudent.NormalSchoolDay and wakeup_timestep > start_school_timestep:
                                            is_work_or_school_day = False # to aovid recurring sick
                                            sampled_non_daily_activity = NonDailyActivityStudent.Sick

                                    if sampled_non_daily_activity == NonDailyActivityStudent.NormalSchoolDay: # sampled normal working day
                                        self.add_to_itinerary(agent, start_school_timestep, Action.School, self.agents_static.get(agentid, "school_cellid"))

                                        self.add_to_itinerary(agent, end_school_timestep, Action.Home, res_cellid)                          
                                    # elif sampled_non_daily_activity == NonDailyActivityStudent.Sick:
                                    #     print("sick school day - stay home")    

                                    sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                                else:
                                    non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[working_age_bracket_index]
                                    non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                                    non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                                    sampled_non_daily_activity_index = self.rng.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                                    sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                                    if sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Travel:
                                        if guardian is None:
                                            is_departureday = True
                                        else:
                                            sampled_non_daily_activity = NonDailyActivityNonWorkingDay.Local

                                    sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                            else:
                                # unemployed/inactive. only consider local / travel / sick
                                non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[working_age_bracket_index]
                                non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                                non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                                sampled_non_daily_activity_index = self.rng.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                                sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                                if sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Travel:
                                    if guardian is None:
                                        is_departureday = True
                                    else:
                                        sampled_non_daily_activity = NonDailyActivityNonWorkingDay.Local

                                # if sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Local: # sampled normal working day
                                #     print("unemployed/inactive local")
                                # elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Travel:
                                #     print("unemployed/inactive travel")                           
                                # elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Sick:
                                #     print("unemployed/inactive sick")      

                                sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                                    
                            # generate number of days for non daily activities i.e. 
                            if ((is_work_or_school_day and sampled_non_daily_activity != NonDailyActivity.NormalWorkOrSchoolDay) or
                                sampled_non_daily_activity == NonDailyActivity.Travel):
                                non_daily_activities_means = self.non_daily_activities_num_days[age_bracket_index]

                                mean_num_days = 0

                                if sampled_non_daily_activity == NonDailyActivity.Local:
                                    mean_num_days = non_daily_activities_means[2]
                                if sampled_non_daily_activity == NonDailyActivity.Travel:
                                    mean_num_days = non_daily_activities_means[3]
                                elif sampled_non_daily_activity == NonDailyActivity.Sick:
                                    mean_num_days = non_daily_activities_means[4]

                                sampled_num_days = self.rng.poisson(mean_num_days)

                                agent["non_daily_activity_recurring"] = {}
                                non_daily_activity_recurring = agent["non_daily_activity_recurring"]
                                for day in range(simday, simday+sampled_num_days+1):
                                    non_daily_activity_recurring[day] = sampled_non_daily_activity
                            
                        # if not sick and not travelling (local or normal work/school day), sample sleep timestep
                        # fill in activity_timestep_ranges, representing "free time"
                        # sample activities to fill in the activity_timestep_ranges
                        # if kid with guardian, "copy" itinerary of guardian, where applicable
                        # if sick stay at home all day
                        # if travelling, yet to be done
                        # finally, if not travelling, fill in cells_agents_timesteps (to be used in contact network)
                        # cells_agents_timesteps -> {cellid: (agentid, starttimestep, endtimestep)}
                        if (sampled_non_daily_activity == NonDailyActivity.Local or 
                            sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay):
                            # schedule sleeping hours
                            sleep_hour = None

                            if self.agents_static.get(agentid, "empstatus") == 0 and sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay and self.agents_static.get(agentid, "isshiftbased"):
                                set_sleeping_hour = False
                                # sample a timestep from 30 mins to 2 hours randomly
                                timesteps_options = np.arange(round(self.timesteps_in_hour / 2), round((self.timesteps_in_hour * 2) + 1))
                                sampled_timestep = self.rng.choice(timesteps_options, size=1)[0]

                                sleep_timestep = end_work_timestep_with_leeway + sampled_timestep
                                sleep_hour = sleep_timestep / self.timesteps_in_hour

                                if not end_work_next_day and sleep_timestep <= 143:
                                    self.add_to_itinerary(agent, sleep_timestep, Action.Sleep, res_cellid) 
                                else:
                                    if sleep_timestep > 143: # might have skipped midnight or work might have ended overnight
                                        sleep_timestep -= 143

                                    self.add_to_itinerary(agent, sleep_timestep, Action.Sleep, res_cellid, next_day=True)

                            if sleep_timestep is None:
                                # set sleeping hours by age brackets

                                sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                                min_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7]
                                
                                alpha, beta = alpha_weekday, beta_weekday
                                if weekday == 6 or weekday == 7: # weekend
                                    alpha, beta = alpha_weekend, beta_weekend

                                if wakeup_hour is None:
                                    wakeup_hour = self.get_hour_by_timestep(wakeup_timestep)

                                if wakeup_hour > min_start_sleep_hour:
                                    min_start_sleep_hour = wakeup_hour

                                sampled_sleep_hour_from_range = round(self.rng.beta(alpha, beta, 1)[0] * start_hour_range + 1)

                                sleep_hour = min_start_sleep_hour + (sampled_sleep_hour_from_range - 1) # this is 1 based; if sampled_sleep_hour_from_range is 1, sleep_hour should be min_start_sleep_hour

                                sleep_timestep = self.get_timestep_by_hour(sleep_hour)

                                if sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay:
                                    if self.agents_static.get(agentid, "empstatus") == 0:
                                        end_work_school_ts = end_work_timestep_with_leeway
                                    elif self.agents_static.get(agentid, "sc_student") == 1:
                                        end_work_school_ts = end_school_timestep

                                    if sleep_timestep <= end_work_school_ts: # if sampled a time which is earlier or same time as end work school, schedule sleep for 30 mins from work/school end
                                        sleep_timestep = end_work_school_ts + round(self.timesteps_in_hour / 2) # sleep 30 mins after work/school end
                                            
                                if sleep_timestep <= 143: # am
                                    self.add_to_itinerary(agent, sleep_timestep, Action.Sleep, res_cellid) 
                                else:
                                    sleep_timestep -= 143
                                    self.add_to_itinerary(agent, sleep_timestep, Action.Sleep, res_cellid, next_day=True) 

                            # find out the activity_timestep_ranges (free time - to be filled in by actual activities further below)
                            wakeup_ts, work_ts, end_work_ts, sleep_ts, overnight_end_work_ts, overnight_sleep_ts = None, None, None, None, None, None
                            activity_timestep_ranges = []

                            if sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay:
                                # find activity timestep ranges i.e. time between wake up and work/school, and time between end work/school and sleep
                                if agent["itinerary"] is not None:
                                    for timestep, (action, cellid) in agent["itinerary"].items():
                                        if action == Action.WakeUp:
                                            wakeup_ts = timestep
                                        elif action == Action.Work or action == Action.School:
                                            work_ts = timestep
                                        elif action == Action.Home:
                                            end_work_ts = timestep
                                        elif action == Action.Sleep:
                                            sleep_ts = timestep

                                        if wakeup_ts is not None and work_ts is not None and end_work_ts is not None and sleep_ts is not None:
                                            break
                                
                                if agent["itinerary_nextday"] is not None:
                                    for timestep, (action, cellid) in agent["itinerary_nextday"].items():
                                        if action == Action.Home:
                                            overnight_end_work_ts = timestep
                                        elif action == Action.Sleep:
                                            sleep_ts = 143 + timestep

                                        if overnight_end_work_ts is not None and sleep_ts is not None:
                                            break
                                
                                if wakeup_ts is None and wakeup_timestep is not None:
                                    wakeup_ts = wakeup_timestep

                                # if work is scheduled for this day and more than 2 hours between wakeup hour and work, sample activities for the range
                                if work_ts is not None:
                                    wakeup_until_work_ts = work_ts - wakeup_ts

                                    wakeup_until_work_hours = self.get_hour_by_timestep(wakeup_until_work_ts)

                                    if wakeup_until_work_hours >= 2: # take activities if 2 hours or more
                                        activity_timestep_ranges.append(range(wakeup_ts+1, work_ts+1))

                                    if overnight_end_work_ts is None: # non-overnight (normal case) - if overnight, from work till midnight is already taken up
                                        # if more than 2 hours between end work hour and sleep
                                        endwork_until_sleep_ts = sleep_ts - end_work_ts
                                        endwork_until_sleep_hours = self.get_hour_by_timestep(endwork_until_sleep_ts)

                                        if endwork_until_sleep_hours >= 2: # take activities if 2 hours or more
                                            activity_timestep_ranges.append(range(end_work_ts+1, sleep_ts+1))
                                    # else: # ends work after midnight (will be handled next day), check whether activities for normal day are possible
                                    #     if sleep_ts is not None and wakeup_ts is not None:
                                    #         wakeup_until_sleep_ts = sleep_ts - wakeup_ts

                                    #         activity_timestep_ranges.append(range(wakeup_ts+1, sleep_ts+1))
                                else:
                                    if sleep_ts+1 - wakeup_ts+1 >= 12: # if range is 2 hours or more
                                        activity_timestep_ranges.append(range(wakeup_ts+1, sleep_ts+1))
                            else:
                                # find activity timestep ranges for non-workers or non-work-day
                                if agent["itinerary"] is not None:
                                    for timestep, (action, cellid) in agent["itinerary"].items():
                                        if action == Action.WakeUp:
                                            wakeup_ts = timestep
                                        elif action == Action.Sleep:
                                            sleep_ts = timestep

                                        if wakeup_ts is not None and sleep_ts is not None:
                                            break
                                
                                if agent["itinerary_nextday"] is not None:
                                    for timestep, (action, cellid) in agent["itinerary_nextday"].items():
                                        if action == Action.Sleep:
                                            sleep_ts = 143
                                            overnight_sleep_ts = timestep
                                            sleep_ts += overnight_sleep_ts
                                            break

                                if wakeup_ts is None and wakeup_timestep is not None:
                                    wakeup_ts = wakeup_timestep

                                if sleep_ts+1 - wakeup_ts+1 >= 12: # if range is 2 hours or more
                                    activity_timestep_ranges.append(range(wakeup_ts+1, sleep_ts+1))

                            if (guardian is None or 
                                (("non_daily_activity_recurring" in guardian) and
                                (agent["non_daily_activity_recurring"] != guardian_non_daily_activity_recurring) and
                                not is_guardian_hospitalised and not is_guardian_quarantined)):
                                # sample activities
                                self.sample_activities(weekday, agent, res_cellid, sleep_timestep, activity_timestep_ranges, age_bracket_index)

                        elif sampled_non_daily_activity == NonDailyActivity.Sick: # stay home all day, simply sample sleep - to refer to on the next day itinerary
                            sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                            min_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7]
                            
                            alpha, beta = alpha_weekday, beta_weekday
                            if weekday == 6 or weekday == 7: # weekend
                                alpha, beta = alpha_weekend, beta_weekend

                            sampled_sleep_hour_from_range = round(self.rng.beta(alpha, beta, 1)[0] * start_hour_range + 1)

                            sleep_hour = min_start_sleep_hour + (sampled_sleep_hour_from_range - 1) # this is 1 based; if sampled_sleep_hour_from_range is 1, sleep_hour should be min_start_sleep_hour

                            sleep_timestep = self.get_timestep_by_hour(sleep_hour)

                            if sleep_timestep <= 143: # am
                                self.add_to_itinerary(agent, sleep_timestep, Action.Sleep, res_cellid)
                            else:
                                sleep_timestep -= 143
                                self.add_to_itinerary(agent, sleep_timestep, Action.Sleep, res_cellid, next_day=True)
                        elif (sampled_non_daily_activity == NonDailyActivity.Travel and 
                            (guardian is None or 
                            ("non_daily_activity_recurring" in guardian and agent["non_daily_activity_recurring"] != guardian_non_daily_activity_recurring))): # same day departure (kids will be copied from guardian)
                            # initialise Itinerary
                            remaining_timesteps_for_day = np.arange(wakeup_timestep+1, 144)

                            if len(remaining_timesteps_for_day) > 0:
                                # choose a random timestep in remaining timesteps for day
                                sampled_departure_timestep = self.rng.choice(remaining_timesteps_for_day, size=1)[0]

                                self.sample_airport_activities(agent, res_cellid, range(wakeup_timestep, sampled_departure_timestep + 1), False, False, 0)

                        if guardian is not None: # kids with a guardian
                            if (sampled_non_daily_activity == NonDailyActivity.Travel and 
                                ("non_daily_activity_recurring" in guardian and
                                agent["non_daily_activity_recurring"] == guardian_non_daily_activity_recurring) and 
                                is_departureday):
                                agent["itinerary"] = guardian_itinerary # create a copy by reference such that transport is also included accordingly
                                agent["itinerary_nextday"] = guardian_itinerary_nextday
                            else:
                                if sampled_non_daily_activity != NonDailyActivity.Sick and sampled_non_daily_activity != NonDailyActivity.Travel:
                                    wakeup_ts = None
                                    home_ts = None
                                    sleep_ts = None
                                    # get earliest home_ts and sleep_ts indicating the range of timesteps to replace with guardian activities
                                    # wakeup_ts replaces home_ts, if the latter is not found
                                    if agent["itinerary"] is not None:
                                        for timestep, (action, cellid) in agent["itinerary"].items():
                                            if action == Action.Home:
                                                if home_ts is None:
                                                    home_ts = timestep
                                                else:
                                                    if timestep > home_ts: # always consider latest home timestep, to then fill in activities from that point onwards (covers school case)
                                                        home_ts = timestep

                                            if action == Action.WakeUp:
                                                wakeup_ts = timestep

                                            if action == Action.Sleep:
                                                sleep_ts = timestep

                                            if home_ts is not None and wakeup_ts is not None and sleep_ts is not None:
                                                break

                                    if home_ts is None:
                                        home_ts = wakeup_ts

                                    if sleep_ts is None:
                                        sleep_ts = 143 # in the context of kids, only sleep is possible after midnight i.e. itinerary_nextday
                                    
                                    if home_ts is not None and sleep_ts is not None:
                                        # fill in kid itinerary by guardian activities
                                        for timestep, (action, cellid) in guardian_itinerary.items():
                                            if action == Action.LocalActivity:
                                                if timestep >= home_ts and timestep < sleep_ts:
                                                    self.add_to_itinerary(agent, timestep, action, cellid)
                                            elif action == Action.Transport:
                                                self.add_to_itinerary(agent, timestep, action, cellid)

                if sampled_non_daily_activity != NonDailyActivity.Travel and is_quarantined_or_hospitalised and not is_quarantine_hospital_start_day:
                    self.add_to_itinerary(agent, 0, Action.WakeUp, res_cellid)

                start_work_school_ts, end_work_school_ts, work_school_cellid, work_school_action = None, None, None, None
                if sampled_non_daily_activity != NonDailyActivity.Travel or (is_departureday or is_arrivalday):
                    if start_work_timestep_with_leeway is not None:
                        start_work_school_ts = start_work_timestep_with_leeway
                        end_work_school_ts = end_work_timestep_with_leeway
                        work_school_cellid = self.agents_static.get(agentid, "work_cellid")
                        work_school_action = Action.Work
                    
                    if start_school_timestep is not None:
                        start_work_school_ts = start_school_timestep
                        end_work_school_ts = end_school_timestep
                        work_school_cellid = self.agents_static.get(agentid, "school_cellid")
                        work_school_action = Action.School
                    
                currently_on_travel_vacation = False

                if sampled_non_daily_activity == NonDailyActivity.Travel and not (is_departureday or is_arrivalday):
                    currently_on_travel_vacation = True

                arr_dep_ts = None

                if is_arrivalday:
                    arr_dep_ts = wakeup_timestep
                elif is_departureday:
                    arr_dep_ts = sampled_departure_timestep

                min_start_sleep_ts = None
                if min_start_sleep_hour is not None:
                    min_start_sleep_ts = self.get_timestep_by_hour(min_start_sleep_hour)

                agent, agent_epi, temp_is_hospitalised, temp_is_quarantined = self.sample_intervention_activities(agentid,
                                                                                                                agent, 
                                                                                                                agent_epi,
                                                                                                                agentindex, 
                                                                                                                res_cellid,
                                                                                                                simday, 
                                                                                                                wakeup_timestep, 
                                                                                                                sleep_timestep, 
                                                                                                                start_work_school_ts, 
                                                                                                                end_work_school_ts, 
                                                                                                                work_school_cellid, 
                                                                                                                work_school_action, 
                                                                                                                is_departureday, 
                                                                                                                is_arrivalday, 
                                                                                                                arr_dep_ts, 
                                                                                                                currently_on_travel_vacation, 
                                                                                                                False,
                                                                                                                age_bracket_index,
                                                                                                                min_start_sleep_ts)

                if temp_is_hospitalised or temp_is_quarantined:
                    is_quarantined_or_hospitalised = True
                    is_quarantine_hospital_start_day = True

                    if temp_is_hospitalised and not is_hospitalised:
                        is_hospitalised = True

                    if temp_is_quarantined and not is_quarantined:
                        is_quarantined = True

                if agent_epi["quarantine_days"] is not None and len(agent_epi["quarantine_days"]) > 0:
                    agent_epi["prev_day_quarantine_days"] = copy(agent_epi["quarantine_days"])
                    
                if sampled_non_daily_activity != NonDailyActivity.Travel or is_departureday or is_arrivalday: # is_arrivalday
                    if sampled_non_daily_activity != NonDailyActivity.Sick and guardian is None:
                        self.sample_transport_cells(agent, agentid, prev_day_last_event)

                    self.update_cell_agents_timesteps(simday, agent["itinerary"], [agentid], [res_cellid])

                # agent["prev_day_history"] = [simday, is_arrivalday, is_departureday, is_quarantined, is_hospitalised, is_quarantine_hospital_start_day, is_quarantine_hospital_end_day, temp_is_quarantined, temp_is_hospitalised]

                # something_wrong_sleep = True

                # for ts, (act, cellid) in agent["itinerary"].items():
                #     if act == Action.Sleep:
                #         something_wrong_sleep = False
                #         break

                # if something_wrong_sleep:
                #     for ts, (act, cellid) in agent["itinerary_nextday"].items():
                #         if act == Action.Sleep:
                #             something_wrong_sleep = False
                #             break

                # if (something_wrong_sleep and
                #     ((not is_quarantined_or_hospitalised or is_quarantine_hospital_start_day ) and
                #     (sampled_non_daily_activity != NonDailyActivity.Travel and not is_arrivalday and not is_departureday))):
                #     print("something wrong")

                if len(agent["itinerary"]) == 0: 
                    print("something wrong, itinerary empty")

                if agent_is_guardian:
                    guardian_hospitalisation_days = copy(agent_epi["hospitalisation_days"])
                    guardian_quarantine_days = copy(agent_epi["quarantine_days"])
                    guardian_non_daily_activity_recurring = copy(agent["non_daily_activity_recurring"])
                    guardian_prevday_non_daily_activity_recurring = copy(agent["prevday_non_daily_activity_recurring"])
                    guardian_itinerary = copy(agent["itinerary"])
                    guardian_itinerary_nextday = copy(agent["itinerary_nextday"])

                agent_itinerary_timesteps = sorted(agent["itinerary"].keys())
                last_timestep = agent_itinerary_timesteps[-1]
                for timestep in agent_itinerary_timesteps:
                    # if timestep < 0 or timestep > 143:
                    #     temp_it = agent["itinerary"][timestep]
                    #     raise Exception(f"Error: Out of range timesteps in itinerary. Day {simday}, Agent {agentid}, Timestep {timestep}, Itinerary {temp_it}")                   
                    
                    if timestep != last_timestep:
                        del agent["itinerary"][timestep]

                if agent["itinerary_nextday"] is not None:
                    agent_itinerary_nextday_timesteps = sorted(agent["itinerary_nextday"].keys())
                    for timestep in agent_itinerary_nextday_timesteps:
                        if timestep < 0 or timestep > 143:
                            temp_it = agent["itinerary_nextday"][timestep]
                            print(f"Error: Out of range timesteps in itinerary nextday. Day {simday}, Agent {agentid}, Timestep {timestep}, Itinerary {temp_it}")
                            # raise Exception(f"Error: Out of range timesteps in itinerary nextday. Day {simday}, Agent {agentid}, Timestep {timestep}, Itinerary {temp_it}")                   

                # old code
                # start = time.time()
                # self.sync_queue.put(["a", agentid, "", agent])
                    

                # old logs
                # if agent["itinerary"] is None or len(agent["itinerary"]) == 0:
                #     print("itinerary: itinerary empty for agent " + str(agentid))
                # else:
                #     if agentid < 10:
                #         first_key = list(agent["itinerary"])[0]
                #         print("first action: " + str(agent["itinerary"][first_key]))

                # time_taken = time.time() - start
                # self.itinerary_interprocess_communication_aggregated_time += time_taken

                # test_itinerary_keys = sorted(list(agent["itinerary"].keys()), reverse=True)
                # test_itinerarynextday_keys = sorted(list(agent["itinerary_nextday"].keys()), reverse=True)

                # if (not is_arrivalday and not is_departureday and "non_daily_activity_recurring" not in agent or agent["non_daily_activity_recurring"] is None or len(agent["non_daily_activity_recurring"]) == 0) and not is_work_or_school_day:
                #     if ((len(test_itinerary_keys) == 0 or agent["itinerary"][test_itinerary_keys[0]][0] != Action.Sleep) and
                #         (len(test_itinerarynextday_keys) == 0 or agent["itinerary_nextday"][test_itinerarynextday_keys[0]][0] != Action.Sleep)):
                #         if ((len(test_itinerary_keys) > 2 and agent["itinerary"][test_itinerary_keys[0]][0] == Action.LocalActivity and agent["itinerary"][test_itinerary_keys[1]][0] == Action.Sleep) or
                #             (len(test_itinerarynextday_keys) > 2 and agent["itinerary_nextday"][test_itinerarynextday_keys[0]][0] == Action.LocalActivity and agent["itinerary_nextday"][test_itinerarynextday_keys[1]][0] == Action.Sleep)):
                #                 print("to check")

        guardian_hospitalisation_days = None
        guardian_quarantine_days = None
        guardian_non_daily_activity_recurring = None
        guardian_prevday_non_daily_activity_recurring = None
        guardian_itinerary = None
        guardian_itinerary_nextday = None

    def update_cell_agents_timesteps(self, simday, itinerary, agentids, rescellids):
        # fill in cells_agents_timesteps
        start_timesteps = sorted(list(itinerary.keys()))
        
        itinerary = self.combine_same_cell_itinerary_entries(start_timesteps, itinerary)

        start_timesteps = sorted(list(itinerary.keys()))

        # prev_cell_id = -1
        for index, curr_ts in enumerate(start_timesteps):
            curr_itinerary = itinerary[curr_ts]
            start_ts = curr_ts
            curr_action, curr_cell_id = curr_itinerary[0], curr_itinerary[1]

            if index == 0: # first: guarantees at least 1 iteration
                start_ts = 0
                end_ts = 143

                if len(start_timesteps) > 1:
                    next_ts = start_timesteps[index+1] # assume at least 2 start_timesteps in a day - might cause problems
                    end_ts = next_ts
            elif index == len(start_timesteps) - 1: # last: guarantees at least 2 iterations
                # prev_itinerary = start_timesteps[index-1]
                # start_ts = prev_itinerary
                end_ts = 143
            else: # mid: guarantees at least 3 iterations
                # prev_itinerary = start_timesteps[index-1]
                next_ts = start_timesteps[index+1]

                # start_ts = prev_itinerary
                end_ts = next_ts

            if start_ts != end_ts:
                agent_cell_timestep_ranges = []
                
                for index, agentid in enumerate(agentids):
                    if curr_cell_id == -1: # must be residence for group case, overwrite accordingly
                        curr_cell_id = rescellids[index]
                    
                    if start_ts < 0 or start_ts > 143 or end_ts < 0 or end_ts > 143 or (start_ts > end_ts):
                        print(f"Error: Out of range timesteps would be added in cells_agents_timesteps. Day {simday}, Agent {agentid}, Start Timestep {start_ts}, End Timestep {end_ts}")
                        # raise Exception(f"Error: Out of range timesteps would be added in cells_agents_timesteps. Day {simday}, Agent {agentid}, Start Timestep {start_ts}, End Timestep {end_ts}")   
                
                    agent_cell_timestep_ranges.append((agentid, max(0, start_ts), min(143, end_ts)))

                if curr_cell_id not in self.vars_util.cells_agents_timesteps:
                    self.vars_util.cells_agents_timesteps[curr_cell_id] = []
                    
                # start = time.time()
                self.vars_util.cells_agents_timesteps[curr_cell_id] += agent_cell_timestep_ranges
                # self.sync_queue.put(["c", curr_cell_id, "cells_agents_timesteps", agent_cell_timestep_ranges])

                # time_taken = time.time() - start
                # self.itinerary_interprocess_communication_aggregated_time += time_taken
            # prev_cell_id = curr_cell_id

    def combine_same_cell_itinerary_entries(self, start_timesteps, itinerary):
        updated_itinerary = {}
        skip_ts = []

        for index, curr_ts in enumerate(start_timesteps):
            if curr_ts not in skip_ts:
                curr_itinerary = itinerary[curr_ts]
                curr_action, curr_cell_id = curr_itinerary[0], curr_itinerary[1]

                if index < len(start_timesteps) - 1:
                    skip_ts = self.skip_same_cells(start_timesteps, itinerary, curr_cell_id, skip_ts, index)

                updated_itinerary[curr_ts] = curr_action, curr_cell_id

        return updated_itinerary

    def skip_same_cells(self, start_timesteps, itinerary, curr_cell_id, skip_ts, index):
        if index < len(start_timesteps) - 1:
            next_timestep = start_timesteps[index+1]

            next_itinerary = itinerary[next_timestep]

            _, next_cell_id = next_itinerary[0], next_itinerary[1]

            if curr_cell_id == next_cell_id:
                skip_ts.append(next_timestep)

                self.skip_same_cells(start_timesteps, itinerary, curr_cell_id, skip_ts, index+1)

        return skip_ts

    def generate_tourist_itinerary(self, simday, weekday, touristsgroups, tourists_active_groupids, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, log_file_name, f=None):
        stack_trace_log_file_name = ""
        try:
            folder_name = ""
            if log_file_name != "output.txt":
                folder_name = os.path.dirname(log_file_name)
            else:
                folder_name = os.getcwd()
            
            stack_trace_log_file_name = os.path.join(folder_name, "tourism_stack_trace_" + str(simday) + ".txt")

            agents_quar_hosp = {}

            for groupid in tourists_active_groupids:
                tourists_group = touristsgroups[groupid]

                accomtype = tourists_group["accomtype"] # 1 = Collective, 2 = Other rented, 3 = Non rented
                accominfo = tourists_group["accominfo"]
                arrivalday = tourists_group["arr"]
                departureday = tourists_group["dep"]
                purpose = tourists_group["purpose"] # 1 = Holiday, 2 = Business, 3 = Visiting Family, 4 = Other
                subgroupsmemberids = tourists_group["subgroupsmemberids"] # rooms in accom
                agentids = tourists_group["agent_ids"] # if len(subgroupsmemberids) > 0:
                # rescellids = tourists_group["res_cell_ids"]

                is_group_activity_for_day = False
                is_arrivalday = arrivalday == simday
                is_departureday = departureday == simday
                is_arrivalnextday = arrivalday == simday + 1
                is_departurenextday = departureday == simday + 1

                arr_dep_for_day, arr_dep_ts, arr_dep_time, airport_duration = None, None, None, None
                arr_dep_for_nextday, dep_nextday_ts, dep_nextday_time = None, None, None

                if is_arrivalday or is_departureday:
                    is_group_activity_for_day = True
                    arr_dep_for_day = tourists_arrivals_departures_for_day[groupid]
                    arr_dep_ts = arr_dep_for_day["ts"]
                    airport_duration = arr_dep_for_day["airport_duration"]
                    arr_dep_time = self.get_hour_by_timestep(arr_dep_ts)
                else:
                    if tourists_group["under_age_agent"]:
                        is_group_activity_for_day = True
                    else:
                        tour_group_activity_rand = random.random()
                        tour_group_activity_prob = self.tourism_group_activity_probability_by_purpose[purpose-1]

                        is_group_activity_for_day = tour_group_activity_rand < tour_group_activity_prob

                if is_departurenextday:
                    if simday < 365:
                        arr_dep_for_nextday = tourists_arrivals_departures_for_nextday[groupid]
                        dep_nextday_ts = arr_dep_for_nextday["ts"]
                        airport_duration = arr_dep_for_nextday["airport_duration"]
                        dep_nextday_time = self.get_hour_by_timestep(dep_nextday_ts)
                
                for agentid in agentids:
                    is_quarantined, is_hospitalised = False, False
                    is_quarantined_or_hospitalised, is_quarantine_hospital_start_day, is_quarantine_hospital_end_day = False, False, False
                    hosp_start_day_ts, hosp_end_day_ts, hosp_start_day, hosp_end_day = None, None, None, None
                    quar_start_day_ts, quar_end_day_ts, quar_start_day, quar_end_day = None, None, None, None

                    if not is_arrivalnextday:
                        agent = self.it_agents[agentid]
                        agent_epi = self.agents_epi[agentid]

                        if is_arrivalday or (simday == 1 and agent["initial_tourist"]):
                            # sample tourist entry infection probability
                            exposed_rand = random.random()

                            if exposed_rand < self.tourist_entry_infection_probability:                 
                                potential_days = np.arange(arrivalday-self.pre_exposure_days, arrivalday+1) # assume tourist might have been exposed in the previous X days from arriving

                                sampled_day = self.rng.choice(potential_days, size=1, replace=False)[0] # sample a day at random from the previous X days

                                agent_state_transition_by_day = agent_epi["state_transition_by_day"]
                                agent_epi_age_bracket_index = self.agents_static.get(agentid, "epi_age_bracket_index")
                                agent_quarantine_days = agent_epi["quarantine_days"]

                                if agent_state_transition_by_day is None:
                                    agent_state_transition_by_day = []

                                # agent_state_transition_by_day, seir_state, infection_type, infection_severity, recovered
                                agent_state_transition_by_day, seir_state, inf_type, inf_sev, _  = self.epi_util.simulate_seir_state_transition(simday, agent_epi, agentid, sampled_day, self.potential_timesteps, agent_state_transition_by_day, agent_epi_age_bracket_index, agent_quarantine_days)

                                agent_epi["state_transition_by_day"] = agent_state_transition_by_day
                                agent_epi["quarantine_days"] = agent_quarantine_days

                                self.vars_util.agents_seir_state = seirstateutil.agents_seir_state_update(self.vars_util.agents_seir_state, seir_state, agentid)
                                self.vars_util.agents_infection_type[agentid] = inf_type
                                self.vars_util.agents_infection_severity[agentid] = inf_sev

                                # print(f"updating agent (airport entry case) {agentid}, state {self.vars_util.agents_seir_state[agentid]}, inf_type {self.vars_util.agents_infection_type[agentid]}, inf_sev {self.vars_util.agents_infection_severity[agentid]}")
                            else:
                                self.vars_util.agents_seir_state = seirstateutil.agents_seir_state_update(self.vars_util.agents_seir_state, SEIRState.Susceptible, agentid)                       

                        # this updates the state, infection type and severity, if relevant for the current day! (such that the itinerary may also handle public health interventions)
                        new_states = seirstateutil.update_agent_state(self.vars_util.agents_seir_state, self.vars_util.agents_infection_type, self.vars_util.agents_infection_severity, agentid, agent_epi, agentid, simday) # agentid is guaranteed to be agentindex here

                        # if new_states is not None:
                        #     new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep = new_states[0], new_states[1], new_states[2], new_states[3], new_states[4], new_states[5]

                        #     self.vars_util.agents_seir_state_transition_for_day[agentid] = [new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep]

                        if agent_epi["hospitalisation_days"] is not None and len(agent_epi["hospitalisation_days"]) > 0:
                            hosp_start_day, hosp_end_day = agent_epi["hospitalisation_days"][0], agent_epi["hospitalisation_days"][2]

                            # hosp_start_day = hosp_start_day_ts[0]
                            # hosp_end_day = hosp_end_day_ts[0]

                            if simday >= hosp_start_day and simday <= hosp_end_day:
                                # print("is hospitalised")
                                is_group_activity_for_day = False
                                is_hospitalised = True
                                is_quarantined_or_hospitalised = True

                            if simday == hosp_start_day:
                                is_quarantine_hospital_start_day = True

                            if simday == hosp_end_day:
                                is_quarantine_hospital_end_day = True

                        if agent_epi["quarantine_days"] is not None and len(agent_epi["quarantine_days"]) > 0:
                            quar_start_day, quar_end_day = agent_epi["quarantine_days"][0], agent_epi["quarantine_days"][2]

                            # quar_start_day = quar_start_day_ts[0]
                            # quar_end_day = quar_end_day_ts[0]

                            if simday >= quar_start_day and simday <= quar_end_day:
                                # print("is quarantined")
                                is_group_activity_for_day = False
                                is_quarantined = True
                                is_quarantined_or_hospitalised = True

                            if simday == quar_start_day:
                                is_quarantine_hospital_start_day = True

                            if simday == quar_end_day:
                                is_quarantine_hospital_end_day = True

                    agents_quar_hosp[agentid] = [is_quarantined_or_hospitalised, is_quarantined, is_hospitalised, is_quarantine_hospital_start_day, is_quarantine_hospital_end_day]

                if not is_arrivalnextday:
                    if is_group_activity_for_day:
                        agent_quar_hosp = agents_quar_hosp[tourists_group["reftourid"] + self.n_locals] # convert tourist id to agent id
                        # sample wake up time, breakfast time / place, sleep time, fill in activities in between
                        tourists_group, wakeup_timestep, sleep_timestep = self.handle_tourism_itinerary(simday, weekday, tourists_group, accomtype, tourists_group["group_accom_id"], is_arrivalday, is_departureday, is_departurenextday, arr_dep_ts, arr_dep_time, dep_nextday_time, airport_duration, groupid, is_group_activity_for_day, agent_quar_hosp)

                        for agentid in agentids:
                            agent = self.it_agents[agentid]
                            res_cellid = self.agents_static.get(agentid, "res_cellid")
                            age_bracket_index = self.agents_static.get(agentid, "age_bracket_index")
                            agent["itinerary"] = copy(tourists_group["itinerary"])
                            agent["itinerary_nextday"] = copy(tourists_group["itinerary_nextday"])

                            agent, agent_epi, temp_is_hospitalised, temp_is_quarantined = self.sample_intervention_activities(agentid, 
                                                                                                                            agent,
                                                                                                                            agent_epi, 
                                                                                                                            agentid, # agentid guaranteed to be agentindex here
                                                                                                                            res_cellid,
                                                                                                                            simday, 
                                                                                                                            wakeup_timestep, 
                                                                                                                            sleep_timestep, 
                                                                                                                            is_departure_day_today= is_departureday, 
                                                                                                                            is_arrival_day_today= is_arrivalday, 
                                                                                                                            arr_dep_ts= arr_dep_ts, 
                                                                                                                            currently_on_travel_vacation= False, 
                                                                                                                            is_tourist= True,
                                                                                                                            age_bracket_index= age_bracket_index)
                            
                            # just for debugging purposes
                            if temp_is_hospitalised or temp_is_quarantined:
                                if temp_is_hospitalised:
                                    is_hospitalised = True

                                if temp_is_quarantined:
                                    is_quarantined = True

                            if agent_epi["quarantine_days"] is not None and len(agent_epi["quarantine_days"]) > 0:
                                agent_epi["prev_day_quarantine_days"] = copy(agent_epi["quarantine_days"])

                            self.update_cell_agents_timesteps(simday, agent["itinerary"], [agentid], [res_cellid])

                            # clear all itinerary entries apart from last 1 for following day
                            agent_itinerary_timesteps = sorted(agent["itinerary"].keys())
                            last_timestep = agent_itinerary_timesteps[-1]
                            for timestep in agent_itinerary_timesteps:
                                if timestep < 0 or timestep > 143:
                                    temp_it = agent["itinerary"][timestep]
                                    print(f"Error: Out of range timesteps in itinerary. Day {simday}, Agent {agentid}, Timestep {timestep}, Itinerary {temp_it}")
                                    # raise Exception(f"Error: Out of range timesteps in itinerary. Day {simday}, Agent {agentid}, Timestep {timestep}, Itinerary {temp_it}")   
                    
                                if timestep != last_timestep:
                                    del agent["itinerary"][timestep]
                    else:
                        for accinfoindex, accinfo in enumerate(accominfo):
                            accomid, roomid, _ = accinfo[0], accinfo[1], accinfo[2]

                            room_members = subgroupsmemberids[accinfoindex] # this room

                            for tourist_id in room_members: # tourists ids in room
                                tourist = self.tourists[tourist_id]
                                agent = self.it_agents[tourist["agentid"]]
                                res_cellid = self.agents_static.get(agentid, "res_cellid")
                                age_bracket_index = self.agents_static.get(agentid, "age_bracket_index")
                                agent_quar_hosp = agents_quar_hosp[tourist["agentid"]]

                                agent, wakeup_timestep, sleep_timestep = self.handle_tourism_itinerary(simday, weekday, agent, accomid, accomtype, is_arrivalday, is_departureday, is_departurenextday, arr_dep_ts, arr_dep_time, dep_nextday_time, airport_duration, groupid, is_group_activity_for_day, agent_quar_hosp, tourist["agentid"])

                                agent, agent_epi, temp_is_hospitalised, temp_is_quarantined = self.sample_intervention_activities(agentid, 
                                                                                                                    agent,
                                                                                                                    agent_epi, 
                                                                                                                    agentid, # agentid guaranteed to be agentindex here
                                                                                                                    res_cellid,
                                                                                                                    simday, 
                                                                                                                    wakeup_timestep, 
                                                                                                                    sleep_timestep, 
                                                                                                                    is_departure_day_today= is_departureday, 
                                                                                                                    is_arrival_day_today= is_arrivalday, 
                                                                                                                    arr_dep_ts= arr_dep_ts, 
                                                                                                                    currently_on_travel_vacation= False, 
                                                                                                                    is_tourist= True,
                                                                                                                    age_bracket_index= age_bracket_index)
                                
                                # just for debugging purposes
                                if temp_is_hospitalised or temp_is_quarantined:
                                    if temp_is_hospitalised:
                                        is_hospitalised = True

                                    if temp_is_quarantined:
                                        is_quarantined = True

                                self.update_cell_agents_timesteps(simday, agent["itinerary"], [tourist["agentid"]], [res_cellid])

                                if tourists_group["reftourid"] == tourist_id: # if reference tourist id is being handled as a single tourist, copy his itineraries to the tourists_group
                                    tourists_group["itinerary"] = copy(agent["itinerary"])
                                    tourists_group["itinerary_nextday"] = copy(agent["itinerary_nextday"])

                                # clear all itinerary entries apart from last 1 for following day
                                agent_itinerary_timesteps = sorted(agent["itinerary"].keys())
                                last_timestep = agent_itinerary_timesteps[-1]
                                for timestep in agent_itinerary_timesteps:
                                    if timestep < 0 or timestep > 143:
                                        temp_it = agent["itinerary"][timestep]
                                        print(f"Error: Out of range timesteps in itinerary. Day {simday}, Agent {agentid}, Timestep {timestep}, Itinerary {temp_it}")
                                        # raise Exception(f"Error: Out of range timesteps in itinerary. Day {simday}, Agent {agentid}, Timestep {timestep}, Itinerary {temp_it}")   
                                    
                                    if timestep != last_timestep:
                                        del agent["itinerary"][timestep]
        except Exception as e:
            with open(stack_trace_log_file_name, 'w') as fi:
                traceback.print_exc(file=fi)

            raise

    def handle_tourism_itinerary(self, day, weekday, agent_group, accomid, accomtype, is_arrivalday, is_departureday, is_departurenextday, arr_dep_ts, arr_dep_time, dep_nextday_time, airport_duration, groupid, is_group_activity_for_day, agent_quar_hosp, agentid = -1):
        res_cell_id = -1 # this will be updated later when calling self.update_cell_agents_timesteps
        # if "res_cellid" in agent_group:
        #     res_cell_id = agent_group["res_cellid"]

        if is_group_activity_for_day:
            age_bracket_index = agent_group["age_bracket_index"]
        else:
            age_bracket_index = self.agents_static.get(agentid, "age_bracket_index")

        checkin_timestep, next_day_checkin_timestep, wakeup_timestep, sleep_timestep  = None, None, None, None
        start_ts, end_ts, breakfast_ts, wakeup_ts, sleep_ts, overnight_sleep_ts = None, None, None, None, None, None
        prev_night_sleep_hour, prev_night_sleep_timestep, same_day_sleep_hour, same_day_sleep_timestep = None, None, None, None
        overnight_end_activity_ts, activity_overnight_cellid = None, None
        airport_overnight_cellids_by_ts  = {}
        prev_night_airport_cellid = None

        if is_arrivalday:
            agent_group["itinerary"] = None # {timestep: cellindex}
            agent_group["itinerary_nextday"] = None
            
            checkin_timestep = arr_dep_ts + airport_duration

            airport_timestep_range = range(arr_dep_ts, checkin_timestep + 1)

            self.sample_airport_activities(agent_group, res_cell_id, airport_timestep_range, True, False, groupid)
        
        prev_day_itinerary, prev_day_last_event = None, None
        if "initial_tourist" not in agent_group and not is_arrivalday:
            prev_day_itinerary = agent_group["itinerary"]
            # prev_day_itinerary_nextday = agent["itinerary_nextday"]

            prev_day_itinerary_timesteps = list(prev_day_itinerary.keys()) # should only be the last timestep of the previous day
            # prev_day_itinerary_timesteps = sorted(prev_day_itinerary.keys(), reverse=True)

            if len(prev_day_itinerary_timesteps) > 0:
                prev_day_last_ts = prev_day_itinerary_timesteps[0]
                prev_day_last_action, prev_day_last_cell = prev_day_itinerary[prev_day_last_ts]
                prev_day_last_event = prev_day_last_ts, prev_day_last_action, prev_day_last_cell

        is_quarantined_or_hospitalised, is_quarantined, is_hospitalised, is_quarantine_hospital_start_day, is_quarantine_hospital_end_day = agent_quar_hosp

        if (is_arrivalday or is_departureday) or not is_quarantined_or_hospitalised or is_quarantine_hospital_start_day or is_quarantine_hospital_end_day:
            max_leave_for_airport_time, max_leave_for_airport_ts = None, None # 3 hours before departure flight
            if is_arrivalday:
                next_day_checkin_timestep = None
                if checkin_timestep > 143:
                    next_day_checkin_timestep = checkin_timestep - 143

                if next_day_checkin_timestep is not None: # assume sleep in the next hour
                    sampled_ts =  self.rng.choice(np.arange(1, 7), size=1, replace=False)[0]
                    
                    sleep_ts = next_day_checkin_timestep + sampled_ts
                else:
                    wakeup_timestep = checkin_timestep
            else:
                if agent_group["itinerary_nextday"] is not None and len(agent_group["itinerary_nextday"]) > 0: # overnight itinerary (scheduled from previous day; to include into "itinerary" dict)
                    # get morning sleeptimestep
                    activity_overnight = False
                    return_home_ts = None
                    for timestep, (action, cellid) in agent_group["itinerary_nextday"].items():
                        if action == Action.LocalActivity:
                            activity_overnight = True
                            activity_overnight_cellid = cellid

                        if action == Action.Home:
                            return_home_ts = timestep

                        if action == Action.Sleep:
                            same_day_sleep_timestep = timestep
                            same_day_sleep_hour = self.get_hour_by_timestep(same_day_sleep_timestep)

                        if action == Action.Airport:
                            airport_overnight_cellids_by_ts[timestep] = cellid

                    if same_day_sleep_hour is None and return_home_ts is not None: # both may be filled in or one of them, if one of them, assume they are the same and simply convert
                        same_day_sleep_hour = self.get_hour_by_timestep(return_home_ts)
                    elif return_home_ts is None and same_day_sleep_hour is not None:
                        return_home_ts = self.get_timestep_by_hour(same_day_sleep_hour)

                    if same_day_sleep_hour is None and return_home_ts is None:
                        # get previous night sleeptimestep
                        if agent_group["itinerary"] is not None:
                            for timestep, (action, _) in agent_group["itinerary"].items():
                                if action == Action.Sleep:
                                    prev_night_sleep_timestep = timestep
                                    prev_night_sleep_hour = self.get_hour_by_timestep(prev_night_sleep_timestep)
                                    break

                    if activity_overnight and return_home_ts is not None:
                        overnight_end_activity_ts = return_home_ts
                else:
                    # get previous night sleeptimestep
                    if agent_group["itinerary"] is not None:
                        for timestep, (action, _) in agent_group["itinerary"].items():
                            if action == Action.Sleep:
                                prev_night_sleep_timestep = timestep
                                prev_night_sleep_hour = self.get_hour_by_timestep(prev_night_sleep_timestep)
                                break

                if not is_arrivalday:
                    agent_group["itinerary"] = None # {timestep: cellindex}
                    agent_group["itinerary_nextday"] = None

                # add overnight values into "itinerary" dict
                if overnight_end_activity_ts is not None:
                    self.add_to_itinerary(agent_group, 0, Action.LocalActivity, activity_overnight_cellid)
                    self.add_to_itinerary(agent_group, overnight_end_activity_ts, Action.Home, res_cell_id)

                # if prev_night_airport_cellid is not None:
                #     self.add_to_itinerary(agent_group, 0, Action.Airport, prev_night_airport_cellid)

                if len(airport_overnight_cellids_by_ts) > 0: # sampled from previous day, simply add previous day airport activities to today's itinerary and skip the rest
                    for ts, cellid in airport_overnight_cellids_by_ts.items():
                        self.add_to_itinerary(agent_group, ts, Action.Airport, cellid)

            #if len(airport_overnight_cellids_by_ts) == 0 or is_arrivalday: # if airport_overnight_cellids_by_ts contains data for departure case skip going to the airport, waking up and sleeping
            if is_departureday and len(airport_overnight_cellids_by_ts) == 0:
                # if prev_night_airport_cellid is None: # if not already at the airport from previous day
                max_leave_for_airport_time = arr_dep_time - min(arr_dep_time, 3)
                max_leave_for_airport_ts = self.get_timestep_by_hour(max_leave_for_airport_time)

            if (not is_departureday or len(airport_overnight_cellids_by_ts) == 0) and next_day_checkin_timestep is None and wakeup_timestep is None: # would already be set for arrivals
                sleep_hours_range = np.arange(self.min_sleep_hours, self.max_sleep_hours + 1)

                # Calculate the middle index of the array
                mid = len(sleep_hours_range) // 2

                sigma = 1.0
                probs = np.exp(-(np.arange(len(sleep_hours_range)) - mid)**2 / (2*sigma**2))
                probs /= probs.sum()

                # Sample from the array with probabilities favouring the middle range (normal dist)
                sampled_sleep_hours_duration = self.rng.choice(sleep_hours_range, size=1, replace=False, p=probs)[0]

                if prev_night_sleep_hour is not None:
                    wakeup_hour = prev_night_sleep_hour + sampled_sleep_hours_duration
                elif same_day_sleep_hour is not None:
                    wakeup_hour = same_day_sleep_hour + sampled_sleep_hours_duration
                else: # this would be quarantined case (assume sleep hour as would be irrelevant and not relative to other activities)
                    wakeup_hour = 0 + sampled_sleep_hours_duration

                if wakeup_hour > 24:
                    wakeup_hour = wakeup_hour - 24

                if max_leave_for_airport_time is not None:
                    if wakeup_hour > max_leave_for_airport_time:
                        wakeup_hour = max_leave_for_airport_time

                wakeup_timestep = self.get_timestep_by_hour(wakeup_hour)

            if not is_arrivalday and wakeup_timestep is not None:
                if wakeup_timestep <= 143: # am
                    self.add_to_itinerary(agent_group, wakeup_timestep, Action.WakeUp, res_cell_id)
                else:
                    wakeup_timestep -= 143
                    self.add_to_itinerary(agent_group, wakeup_timestep, Action.WakeUp, res_cell_id, next_day=True)

            if not is_departureday: # don't schedule sleep time if departure day (or if need to be at the airport today - simplification)
                max_leave_for_airport_nextday_time = None
                if is_departurenextday and day < 365:
                    max_leave_for_airport_nextday_time = dep_nextday_time - 3

                if max_leave_for_airport_nextday_time is not None and max_leave_for_airport_nextday_time < 0: # go to the airport instead of sleeping
                    # end the day at the airport, and start the next day at the airport too (here max_leave_for_airport_nextday_time will always be negative, so max_leave_for_airport_time will always be less than 24)
                    max_leave_for_airport_time = 24 + max_leave_for_airport_nextday_time
                    max_leave_for_airport_ts = self.get_timestep_by_hour(max_leave_for_airport_time)

                    self.sample_airport_activities(agent_group, res_cell_id, range(max_leave_for_airport_ts, max_leave_for_airport_ts + airport_duration + 1), False, True, groupid)
                else:
                    if next_day_checkin_timestep is None:
                        # set sleeping hours by age brackets
                        sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                        min_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7]
                        
                        alpha, beta = alpha_weekday, beta_weekday
                        if weekday == 6 or weekday == 7: # weekend
                            alpha, beta = alpha_weekend, beta_weekend

                        sampled_sleep_hour_from_range = round(self.rng.beta(alpha, beta, 1)[0] * start_hour_range + 1)

                        sleep_hour = min_start_sleep_hour + (sampled_sleep_hour_from_range - 1) # this is 1 based; if sampled_sleep_hour_from_range is 1, sleep_hour should be min_start_sleep_hour

                        sleep_timestep = self.get_timestep_by_hour(sleep_hour)

                        # if is_arrivalday and checkin_timestep + 1 > sleep_timestep:
                        #     sleep_timestep = checkin_timestep + 1

                        if sleep_timestep <= 143: # am
                            self.add_to_itinerary(agent_group, sleep_timestep, Action.Sleep, res_cell_id) 
                        else:
                            sleep_timestep -= 143
                            self.add_to_itinerary(agent_group, sleep_timestep, Action.Sleep, res_cell_id, next_day=True)   
                    else:
                        self.add_to_itinerary(agent_group, sleep_ts, Action.Sleep, res_cell_id, next_day=True) # next day sleep hour         

            if ((not is_arrivalday and not is_departureday) or
                (is_arrivalday and next_day_checkin_timestep is None and arr_dep_time <= 8) or 
                (is_departureday and max_leave_for_airport_time is not None and max_leave_for_airport_time >= 8)): # if not arrivalday, or arrival day and arrival time before 8am
                # eat breakfast (see if hotel or external based on some conditions & probs)
                eat_at_accom = False

                if accomtype == 1: # collective
                    tourism_accom_breakfast_rand = random.random()

                    eat_at_accom = tourism_accom_breakfast_rand < self.tourism_accom_breakfast_probability

                wakeup_until_breakfast_timesteps = np.arange(1, 7)

                wakeup_until_breakfast_ts = self.rng.choice(wakeup_until_breakfast_timesteps, size=1)[0]

                breakfast_ts = wakeup_timestep + wakeup_until_breakfast_ts

                if eat_at_accom:
                    breakfast_cells = self.cells_breakfast_by_accomid[accomid]

                    breakfast_cells_options = np.array(list(breakfast_cells.keys()))

                    sampled_cell_id = self.rng.choice(breakfast_cells_options, size=1)[0]
                else:
                    cells_restaurants_options = np.array(list(self.cells_restaurants.keys()))
                    sampled_cell_id = self.rng.choice(cells_restaurants_options, size=1)[0]

                if breakfast_ts <= 143:
                    self.add_to_itinerary(agent_group, breakfast_ts, Action.Breakfast, sampled_cell_id)
                else:
                    self.add_to_itinerary(agent_group, breakfast_ts, Action.Breakfast, sampled_cell_id, next_day=True)

            if next_day_checkin_timestep is None:
                # sample activities from breakfast until sleep timestep or airport (if dep) - force dinner as last activity
                activity_timestep_ranges = []
                
                for timestep, (action, cellid) in agent_group["itinerary"].items():
                    if action == Action.Breakfast:
                        start_ts = timestep
                    elif action == Action.WakeUp:
                        wakeup_ts = timestep
                    elif action == Action.Sleep:
                        end_ts = timestep

                    if start_ts is not None and wakeup_ts is not None and end_ts is not None:
                        break
                
                if agent_group["itinerary_nextday"] is not None:
                    for timestep, (action, cellid) in agent_group["itinerary_nextday"].items():
                        if action == Action.Sleep:
                            end_ts = 143
                            overnight_sleep_ts = timestep
                            end_ts += overnight_sleep_ts
                            break

                if start_ts is None and wakeup_ts is not None:
                    start_ts = wakeup_ts

                if end_ts is None and max_leave_for_airport_ts is not None:
                    end_ts = max_leave_for_airport_ts

                if start_ts is not None and end_ts is not None and start_ts < end_ts and (end_ts - start_ts) >= 6: # start_ts before end_ts and at least 1 hour timeframe for activities to take place
                    activity_timestep_ranges.append(range(start_ts+1, end_ts+1))

                    self.sample_activities(weekday, agent_group, res_cell_id, end_ts, activity_timestep_ranges, age_bracket_index, is_tourist_group=True, airport_timestep=max_leave_for_airport_ts)

                if is_departureday and len(airport_overnight_cellids_by_ts) == 0:
                    trip_to_airport_time = self.rng.choice(np.arange(3, 10), size=1)[0]

                    # max_leave_for_airport_ts here will be 0 at most; which means time must have to be moved accordingly to account for this
                    # trip to airport time here is purely random, so it does not add any epidemiological value whatsoever (timings can be played with)

                    if max_leave_for_airport_ts + trip_to_airport_time >= arr_dep_ts + 1:
                        max_arrive_at_airport_ts = arr_dep_ts - 3
                    else:
                        max_arrive_at_airport_ts = max_leave_for_airport_ts + trip_to_airport_time 

                    if max_arrive_at_airport_ts < 0:
                        max_arrive_at_airport_ts = 0

                    if max_arrive_at_airport_ts == arr_dep_ts or (arr_dep_ts - max_arrive_at_airport_ts) < 3:
                        diff_between_arr_dep_ts_and_max_arrive = arr_dep_ts - max_arrive_at_airport_ts

                        to_adjust = 3 - diff_between_arr_dep_ts_and_max_arrive

                        if max_arrive_at_airport_ts - to_adjust > 0:
                            max_arrive_at_airport_ts -= to_adjust
                        else:
                            arr_dep_ts += to_adjust # make sure there's a suitable time between arrival at airport and departure

                    # go to airport, spend the duration as imposed by the tourists_arrivals_departures_for_day dict, in different airport cells at random"
                    # delete tourists_arrivals_departures_for_day and tourists_active_groupids by tourist group id (hence left the country) (this will be after spending time at the airport)"
                    self.sample_airport_activities(agent_group, res_cell_id, range(max_arrive_at_airport_ts, arr_dep_ts + 1), inbound=False, is_departurenextday=False, groupid=groupid)
                    # del tourists_arrivals_departures_for_day[groupid] # to see how "nextday" affect this
                    # tourists_active_groupids.remove(groupid)

                agent_group = self.sample_transport_cells(agent_group, None, prev_day_last_event)
        else:
            self.add_to_itinerary(agent_group, 0, Action.WakeUp, res_cell_id)

            # something_wrong_sleep = True

            # if is_departureday or (is_departurenextday and day < 365):
            #     something_wrong_sleep = False
            # else:
            #     for ts, (act, cellid) in agent_group["itinerary"].items():
            #         if act == Action.Sleep:
            #             something_wrong_sleep = False
            #             break

            #     if something_wrong_sleep:
            #         for ts, (act, cellid) in agent_group["itinerary_nextday"].items():
            #             if act == Action.Sleep:
            #                 something_wrong_sleep = False
            #                 break

            #     if something_wrong_sleep:
            #         print("something wrong")

            # something_wrong_airport = True

            # if is_departurenextday and day < 365 and dep_nextday_time < 3:
            #     for ts, (act, cellid) in agent_group["itinerary"].items():
            #         if act == Action.Airport:
            #             something_wrong_airport = False
            #             break

            #     if something_wrong_airport:
            #         for ts, (act, cellid) in agent_group["itinerary_nextday"].items():
            #             if act == Action.Airport:
            #                 something_wrong_airport = False
            #                 break

            #     if something_wrong_airport:
            #         print("something wrong")

        return agent_group, wakeup_timestep, sleep_timestep

    def sample_activities(self, weekday, agent_group, res_cell_id, sleep_timestep, activity_timestep_ranges, age_bracket_index, is_tourist_group=False, airport_timestep=None):
        # fill in activity_timestep_ranges with actual activities
        for timestep_range in activity_timestep_ranges:
            activities_slot_ts = sum([1 for i in timestep_range])

            activities_slot_hours = self.get_hour_by_timestep(activities_slot_ts)

            next_timestep = timestep_range[0]
            # repeat until no more hours to fill
            while activities_slot_hours > 0:
                # sample activity from activities_by_week_days_distribution X activities_by_agerange_distribution (pre-compute in constructor)

                activities_probs_for_agegroup_and_day = self.activities_by_week_days_by_age_groups[age_bracket_index][:, weekday-1]
                activities_indices = np.arange(len(activities_probs_for_agegroup_and_day))

                sampled_activity_index = self.rng.choice(activities_indices, 1, False, p=activities_probs_for_agegroup_and_day)[0]
                sampled_activity_id = sampled_activity_index + 1

                activity_working_hours_overrides = self.activities_working_hours[sampled_activity_index]

                # sample numhours from activities_duration_hours, 
                # where if sampled_num_hours is > then activities_slot_hours, sampled_num_hours = activities_slot_hours
                # and if activities_slot_hours - sampled_num_hours < 1 hour, sampled_num_hours = sampled_num_hours + (activities_slot_hours - sampled_num_hours)

                min_hours, max_hours = self.activities_duration_hours[sampled_activity_index][1], self.activities_duration_hours[sampled_activity_index][2]

                hours_range = np.arange(min_hours, max_hours+1)

                sampled_num_hours = self.rng.choice(hours_range, size=1)[0]

                last_activity = False

                if sampled_num_hours == activities_slot_hours:
                    last_activity = True
                elif sampled_num_hours > activities_slot_hours:
                    last_activity = True
                    sampled_num_hours = activities_slot_hours # sampled a larger value than the remaining hours available, hence, go for remaining hours
                elif activities_slot_hours - sampled_num_hours < 1:
                    last_activity = True
                    sampled_num_hours = sampled_num_hours + (activities_slot_hours - sampled_num_hours) # less than an hour would be left after adding activity, hence add it to this activity

                action_type = Action.LocalActivity

                if last_activity and is_tourist_group: # force last activity for tourists as restaurant
                    potential_cells = list(self.cells_restaurants.keys())
                else:
                    if sampled_activity_id in list(self.cells_entertainment.keys()): # if this is an entertainment activity
                        potential_cells = list(self.cells_entertainment[sampled_activity_id].keys())
                    else:  # non entertainment activities
                        industry_id = activity_working_hours_overrides[1]

                        if industry_id != 0: # workplaces as venues, e.g. food / shopping
                            if industry_id != 9:
                                potential_venues_by_industry = list(self.industries[industry_id].keys())

                                sampled_wp_id = self.rng.choice(potential_venues_by_industry)

                                potential_cells = list(self.industries[industry_id][sampled_wp_id].keys())
                            else:
                                potential_cells = list(self.cells_restaurants.keys())
                        elif sampled_activity_id == 8: # religious
                            potential_cells = list(self.cells_religious.keys())
                        elif sampled_activity_id == 9: # stay home
                            potential_cells = [res_cell_id]
                            action_type = Action.Home
                        elif sampled_activity_id == 10: # other residence visit
                            potential_cells = list(self.cells_households.keys())

                sampled_cell_id = self.rng.choice(potential_cells)

                if next_timestep <= 143:
                    self.add_to_itinerary(agent_group, next_timestep, action_type, sampled_cell_id) 
                else:
                    self.add_to_itinerary(agent_group, next_timestep - 143, action_type, sampled_cell_id, next_day=True) 

                next_timestep += self.get_timestep_by_hour(sampled_num_hours)

                activities_slot_hours -= sampled_num_hours
            
            if next_timestep <= 143: # always return back home after last sampled activity (airport_timestep here represents time that agent leaves for airport i.e. departure case)
                if next_timestep < sleep_timestep and (airport_timestep is None or next_timestep < airport_timestep): # next_timestep could be slightly larger than sleep_timestep due to rounding, in that case ignore going back Home
                    self.add_to_itinerary(agent_group, next_timestep, Action.Home, res_cell_id)
            else: # next_timestep below was next_timestep - 143 (- 143 has been removed because sleep_timestep will be 143 + x)
                if next_timestep < sleep_timestep and (airport_timestep is None): # next_timestep could be slightly larger than sleep_timestep due to rounding, in that case ignore going back Home
                    self.add_to_itinerary(agent_group, next_timestep - 143, Action.Home, res_cell_id, next_day=True) # in this case airport_timestep indicates that agent ends the day at the airport; meaning the presence of airport_timestep is enough to indicate that agent will not return back to home/accom (this has been removed: or next_timestep - 143 < airport_timestep)

    def sample_airport_activities(self, agent_group, res_cell_id, timestep_range_in_airport, inbound, is_departurenextday, groupid):
        activity_id = 3 # entertainment (indoor)
        activity_index = 1 # zero based

        min_hours, max_hours = self.tourism_airport_duration_min_max[0], self.tourism_airport_duration_min_max[1]

        hours_range = np.arange(min_hours, max_hours+1, min_hours)

        activities_slot_ts = sum([1 for i in timestep_range_in_airport])

        activities_slot_hours = self.get_hour_by_timestep(activities_slot_ts)

        next_timestep = timestep_range_in_airport[0]

        potential_cells = list(self.cells_airport.keys())

        itinerary_nextday_inserted = False
        # repeat until no more hours to fill
        while activities_slot_hours > 0:
            sampled_num_hours = self.rng.choice(hours_range, size=1)[0]

            if sampled_num_hours > activities_slot_hours:
                sampled_num_hours = activities_slot_hours # sampled a larger value than the remaining hours available, hence, go for remaining hours
            elif activities_slot_hours - sampled_num_hours < 1:
                sampled_num_hours = sampled_num_hours + (activities_slot_hours - sampled_num_hours) # less than an hour would be left after adding activity, hence add it to this activity

            sampled_cell_id = self.rng.choice(potential_cells, size=1)[0]

            if next_timestep <= 143:
                self.add_to_itinerary(agent_group, next_timestep, Action.Airport, sampled_cell_id) 
            else:
                itinerary_nextday_inserted = True
                self.add_to_itinerary(agent_group, next_timestep - 143, Action.Airport, sampled_cell_id, next_day=True) 

            next_timestep += self.get_timestep_by_hour(sampled_num_hours)

            activities_slot_hours -= sampled_num_hours
        
        if inbound:
            if next_timestep <= 143: # always return back home after last sampled activity
                self.add_to_itinerary(agent_group, next_timestep, Action.Home, res_cell_id)
            else:
                self.add_to_itinerary(agent_group, next_timestep - 143, Action.Home, res_cell_id, next_day=True)
        else:
            if is_departurenextday and not itinerary_nextday_inserted: # ensure itinerary_nextday is updated such that on departure day agent starts at the same cell
                self.add_to_itinerary(agent_group, 0, Action.Airport, sampled_cell_id, next_day=True)

    def sample_transport_cells(self, agent, agentid, last_prev_day_event):
        if agentid is not None:
            pub_transp_reg = self.agents_static.get(agentid, "pub_transp_reg")
        else:
            pub_transp_reg = True # tourism

        if not pub_transp_reg:
            daily_transport_rand = random.random()

            if daily_transport_rand < self.public_transport_usage_non_reg_daily_probability:
                pub_transp_reg = True

        if pub_transp_reg:
            timestep_keys = sorted(agent["itinerary"].keys())

            pre_transport_itinerary = copy(agent["itinerary"])

            for index, ts in enumerate(timestep_keys):
                action, cellid = pre_transport_itinerary[ts]
                prev_ts, prev_action, prev_cellid = None, None, None

                sample_transport_cell = True
                if index == 0:
                    if last_prev_day_event is not None:
                        prev_ts, prev_action, prev_cellid = last_prev_day_event            
                else:
                    prev_ts = timestep_keys[index-1]
                
                    prev_action, prev_cellid = pre_transport_itinerary[prev_ts]

                if (prev_cellid is None or cellid == prev_cellid or (action == prev_action and action != Action.LocalActivity) or 
                    (action == Action.WakeUp) or
                    ((prev_action == Action.Home or prev_action == Action.WakeUp or prev_action == Action.Breakfast or prev_action == Action.Sleep)
                    and (action == Action.Home or action == Action.WakeUp or action == Action.Breakfast or action == Action.Sleep))):
                    sample_transport_cell = False
                
                if sample_transport_cell:
                    if index == 0 and prev_action == Action.Transport:
                        self.add_to_itinerary(agent, 0, Action.Transport, prev_cellid)
                    else:
                        transp_min_start_ts = prev_ts + 3
                        trans_end_ts = ts - 1

                        sampled_num_timesteps = self.rng.choice(np.arange(1, 7), size=1)[0]

                        transp_start_ts = trans_end_ts - sampled_num_timesteps

                        if index != 0 and transp_start_ts < transp_min_start_ts:
                            transp_start_ts = transp_min_start_ts

                        if transp_start_ts < 0:
                            transp_start_ts = 0

                        potential_cells = list(self.cells_transport.keys())

                        sampled_cell_id = self.rng.choice(potential_cells, size=1)[0]

                        if transp_start_ts < sorted(agent["itinerary"].keys(), reverse=True)[0] and transp_start_ts < trans_end_ts: # no space to fit in transport, hence, skip
                            if transp_start_ts <= 143:
                                self.add_to_itinerary(agent, transp_start_ts, Action.Transport, sampled_cell_id)
                            else:
                                transp_start_ts -= 143
                                self.add_to_itinerary(agent, transp_start_ts, Action.Transport, sampled_cell_id, next_day=True)

        return agent
     
    # covers quarantine, tests and vaccines and is called at the end of the itinerary generation such that intervention activities supercede other activities
    # for quarantine, clear any activities that are already scheduled beyond the start ts, and schedule home from 0 to 143 for subsequent days until quarantine ends
    # in test/vaccine cases, introduce the vaccine activity and sample a test/vaccine cell randomly
    # if test/vaccine start ts coincides with work or school, and if after start_ts + 6 ts (1 hour) work/school would not have ended, 
    # agent goes back to work/school accordingly, otherwise, goes back home, then activites continue as per usual
    def sample_intervention_activities(self, agentid, agent, agent_epi, agentindex, res_cellid, day, wakeup_ts, sleep_ts=None, start_work_school_ts=None, end_work_school_ts=None, work_school_cellid=None, work_school_action=None, is_departure_day_today=False, is_arrival_day_today=False, arr_dep_ts=None, currently_on_travel_vacation=False, is_tourist=False, age_bracket_index=None, min_start_sleep_ts=None):
        # test_result_day: [day,timestep]
        # quarantine_days: [[startday,timestep], [endday, timestep]]
        # test_day: [day,timestep]
        # vaccination_days: [[day,timestep]]

        is_quarantine_startday, is_hospital_startday = False, False

        if (is_arrival_day_today or is_departure_day_today) and arr_dep_ts is None:
            itinerary_timesteps = sorted(list(agent["itinerary"].keys()), reverse=True)

            for ts in itinerary_timesteps:
                action, _ = agent["itinerary"][ts]

                if action == Action.Airport:
                    arr_dep_ts = ts
                    break

        if agent_epi["test_result_day"] is not None and len(agent_epi["test_result_day"]) > 0:
            test_result_day = agent_epi["test_result_day"][0]

            if day == test_result_day:
                start_ts = agent_epi["test_result_day"][1]

                seir_state = seirstateutil.agents_seir_state_get(self.vars_util.agents_seir_state, agentid) #agentindex
                
                if seir_state == SEIRState.Exposed or seir_state == SEIRState.Infectious: # this mostly handles asymptomatic cases
                    false_negative_rand = random.random()

                    is_false_negative = False
                    if false_negative_rand < self.epi_util.testing_false_negative_rate:
                        is_false_negative = True

                    if not is_false_negative:
                        self.vars_util.contact_tracing_agent_ids.add(agentid) 

                        is_quarantine_startday, _ = self.epi_util.schedule_quarantine(agentid, day, start_ts, QuarantineType.Positive, agent=agent_epi)
                else:
                    false_positive_rand = random.random()

                    is_false_positive = False
                    if false_positive_rand < self.epi_util.testing_false_positive_rate:
                        is_false_positive = True

                    if is_false_positive:
                        self.vars_util.contact_tracing_agent_ids.add(agentid)

                        is_quarantine_startday, _ = self.epi_util.schedule_quarantine(agentid, day, start_ts, QuarantineType.Positive, agent=agent_epi)
            elif day > test_result_day:
                agent_epi["test_result_day"] = None

        hospitalisation_ts = None
        hospitalisation_end_day = False
        if agent_epi["hospitalisation_days"] is not None and len(agent_epi["hospitalisation_days"]) > 0:
            start_day, start_ts, end_day = agent_epi["hospitalisation_days"][0], agent_epi["hospitalisation_days"][1], agent_epi["hospitalisation_days"][2]
            # start_day, start_ts = start_day_ts[0], start_day_ts[1]
            # end_day, end_ts = end_day_ts[0], end_day_ts[1]
            
            if start_day == day:
                self.new_hospitalised += 1

                cancel_itinerary_beyond_hospitalisation_ts = True

                hospitalisation_ts = start_ts
                
                cancel_vacation = False
                if not is_tourist and is_departure_day_today and hospitalisation_ts < arr_dep_ts:
                    cancel_vacation = True

                if is_arrival_day_today:
                    if arr_dep_ts is not None:
                        if hospitalisation_ts < arr_dep_ts:
                            hospitalisation_ts = arr_dep_ts
                    else:
                        # this would be the case where arrival is beyond midnight quarantine starts from previous day at 6am
                        hospitalisation_days = [start_day+1, 36, end_day+1]
                        agent_epi = self.epi_util.schedule_hospitalisation(agent_epi, hospitalisation_days)
                        cancel_itinerary_beyond_hospitalisation_ts = False

                if cancel_vacation:
                    agent["non_daily_activity_recurring"] = None

                if cancel_itinerary_beyond_hospitalisation_ts:
                    if not is_departure_day_today or cancel_vacation:
                        is_hospital_startday = True

                        timesteps_to_delete = []
                        if agent["itinerary"] is not None:
                            for timestep,  (action, cellid) in agent["itinerary"].items():
                                if timestep >= hospitalisation_ts:
                                    timesteps_to_delete.append(timestep)
                                # else:
                                #     if action == Action.Sleep:
                                #         print("problem?")

                        for timestep in timesteps_to_delete:
                            del agent["itinerary"][timestep]

                        cells_hospitals_cellids = list(self.cells_hospital.keys())

                        cells_hospital_indices = np.arange(len(cells_hospitals_cellids))

                        sampled_hospital_index = self.rng.choice(cells_hospital_indices, size=1, replace=False)[0]
                        
                        sampled_hospital_cellid = cells_hospitals_cellids[sampled_hospital_index]

                        self.add_to_itinerary(agent, hospitalisation_ts, Action.Hospital, sampled_hospital_cellid)

                        agent["itinerary_nextday"] = None
            elif end_day == day:
                self.new_hospitalised -= 1

                hospitalisation_end_day = True
                agent_epi["hospitalisation_days"] = None
                # agent["deleted_hospital_on"] = day
        
        quarantine_ts = None
        quarantine_end_day = False
        if agent_epi["quarantine_days"] is not None and len(agent_epi["quarantine_days"]) > 0:
            start_day, start_ts, end_day = agent_epi["quarantine_days"][0], agent_epi["quarantine_days"][1], agent_epi["quarantine_days"][2]
            # start_day, start_ts = start_day_ts[0], start_day_ts[1]
            # end_day, end_ts = end_day_ts[0], end_day_ts[1]
            
            if start_day == day or (hospitalisation_end_day and day <= end_day):
                if start_day == day:
                    self.new_quarantined += 1

                cancel_itinerary_beyond_quarantine_ts = True

                quarantine_ts = start_ts
                
                cancel_vacation = False
                if not is_tourist and is_departure_day_today and quarantine_ts < arr_dep_ts:
                    cancel_vacation = True

                if is_arrival_day_today:
                    if arr_dep_ts is not None:
                        if quarantine_ts < arr_dep_ts:
                            quarantine_ts = arr_dep_ts
                    else:
                        agent_epi["updated_quarantine"] = day
                        # this would be the case where arrival is beyond midnight quarantine starts from previous day at 6am
                        agent_epi = self.epi_util.update_quarantine(agent_epi, start_day+1, 36, end_day+1, 36)
                        cancel_itinerary_beyond_quarantine_ts = False

                if cancel_itinerary_beyond_quarantine_ts and hospitalisation_ts is not None:
                    cancel_itinerary_beyond_quarantine_ts = False # would have already been done for hospitalisation

                if cancel_vacation:
                    agent["non_daily_activity_recurring"] = None

                if cancel_itinerary_beyond_quarantine_ts:
                    if not is_departure_day_today or cancel_vacation:
                        is_quarantine_startday = True

                        timesteps_to_delete = []
                        if agent["itinerary"] is not None:
                            for timestep,  (action, cellid) in agent["itinerary"].items():
                                if timestep >= quarantine_ts:
                                    timesteps_to_delete.append(timestep)
                                # else:
                                #     if action == Action.Sleep:
                                #         print("problem?")

                        for timestep in timesteps_to_delete:
                            del agent["itinerary"][timestep]

                        self.add_to_itinerary(agent, quarantine_ts, Action.Home, res_cellid)

                        agent["itinerary_nextday"] = None
            elif end_day == day:
                self.new_quarantined -= 1
                quarantine_end_day = True
                agent_epi["quarantine_days"] = None
                # agent["deleted_quarantine_on"] = day

        if agent_epi["test_day"] is not None and len(agent_epi["test_day"]) > 0:
            test_day = agent_epi["test_day"][0]

            if day == test_day:
                self.new_tests += 1

                reschedule_test = False
                start_ts = agent_epi["test_day"][1]
                
                if currently_on_travel_vacation or (is_departure_day_today and start_ts > arr_dep_ts) or (is_arrival_day_today and start_ts < arr_dep_ts):
                    reschedule_test = True

                if not reschedule_test:       
                    cells_testinghub_cellids = list(self.cells_testinghub.keys())

                    cells_testinghub_indices = np.arange(len(cells_testinghub_cellids))

                    sampled_testinghub_index = self.rng.choice(cells_testinghub_indices, size=1, replace=False)[0]
                    
                    sampled_testinghub_cellid = cells_testinghub_cellids[sampled_testinghub_index]

                    self.add_to_itinerary(agent, start_ts, Action.Test, sampled_testinghub_cellid)

                    self.modify_wakeup_sleep_work_school_for_interventions(agent, res_cellid, start_ts, wakeup_ts, sleep_ts, start_work_school_ts, end_work_school_ts, work_school_cellid, work_school_action, quarantine_ts, age_bracket_index, min_start_sleep_ts)
                else:
                    if not is_tourist or (is_arrival_day_today and start_ts < arr_dep_ts): # re-scheduling not applicable for tourists leaving Malta (out of scope)
                        agent_epi["test_day"] = [day + 1, start_ts]
            elif (day - test_day) == 6: # if 6 days have passed, clear test day, to eventually clear-up memory (5 days are required because if agent has been tested in the last 5 days, another test won't be re-scheduled so quickly)
                agent_epi["test_day"] = None

        if not is_tourist and "vaccination_days" in agent_epi and agent_epi["vaccination_days"] is not None and len(agent_epi["vaccination_days"]) > 0: # does not apply for tourists
            vaccination_day_ts = agent_epi["vaccination_days"][len(agent_epi["vaccination_days"]) - 1] # get last date, in case of more than 1 dose (right now only 1 dose)
            vaccination_day, vaccination_ts = vaccination_day_ts[0], vaccination_day_ts[1]
            
            if day == vaccination_day:
                self.new_vaccinations += 1

                reschedule_test = False
                start_ts = vaccination_ts

                if currently_on_travel_vacation or (is_departure_day_today and start_ts > arr_dep_ts) or (is_arrival_day_today and start_ts < arr_dep_ts):
                    reschedule_test = True
                
                if not reschedule_test:
                    cells_vaccinationhub_cellids = list(self.cells_vaccinationhub.keys())

                    cells_vaccinationhub_indices = np.arange(len(cells_vaccinationhub_cellids))

                    sampled_vaccinationhub_index = self.rng.choice(cells_vaccinationhub_indices, size=1, replace=False)[0]
                    
                    sampled_vaccinationhub_cellid = cells_vaccinationhub_cellids[sampled_vaccinationhub_index]

                    self.add_to_itinerary(agent, start_ts, Action.Vaccine, sampled_vaccinationhub_cellid)

                    agent = self.modify_wakeup_sleep_work_school_for_interventions(agent, res_cellid, start_ts, wakeup_ts, sleep_ts, start_work_school_ts, end_work_school_ts, work_school_cellid, work_school_action, quarantine_ts, age_bracket_index, min_start_sleep_ts)

                    # increase vaccination doses
                    if agentid not in self.vars_util.agents_vaccination_doses:
                        self.vars_util.agents_vaccination_doses[agentid] = []

                    agent_vaccination_doses = self.vars_util.agents_vaccination_doses[agentid]
                    agent_vaccination_doses.append(day)
                    self.vars_util.agents_vaccination_doses[agentid] = agent_vaccination_doses
                else:
                    if is_arrival_day_today and start_ts < arr_dep_ts: # re-scheduling not applicable for tourists leaving Malta (out of scope)
                        agent_epi["vaccination_days"][len(agent_epi["vaccination_days"]) - 1] = [day + 1, start_ts]
            elif day > vaccination_day:
                agent_epi["vaccination_days"] = None # for now this is acceptable because there cannot be double vaccinations dated scheduled. but would be a problem if it overwrites multiple dates

        return agent, agent_epi, is_hospital_startday, is_quarantine_startday

    def modify_wakeup_sleep_work_school_for_interventions(self, agent, res_cellid, start_ts, wakeup_ts, sleep_ts, start_work_school_ts, end_work_school_ts, work_school_cellid, work_school_action, quarantine_ts, age_bracket_index, min_start_sleep_ts):
        if wakeup_ts is not None and start_ts < wakeup_ts: # if vaccination/test start ts before wakeup, adjust to wake-up in a random timestep in the previous 2 hours
            wakeup_ts_before_test_or_vaccine = self.rng.choice(self.one_to_two_hours, size=1)[0]
            new_wakeup_ts = start_ts - wakeup_ts_before_test_or_vaccine

            timesteps_to_delete = []
            if agent["itinerary"] is not None:
                for timestep, (action, cellid) in agent["itinerary"].items():
                    if action == Action.WakeUp:
                        timesteps_to_delete.append(timestep)
                        break
            
            if len(timesteps_to_delete) > 0:
                for timestep in timesteps_to_delete:
                    del agent["itinerary"][timestep]

                self.add_to_itinerary(agent, new_wakeup_ts, Action.WakeUp, res_cellid)

        if sleep_ts is not None and start_ts > sleep_ts: # if vaccination/test start ts after sleep, adjust sleep in a random timestep in the next 2 hours
            sleep_ts_after_test_or_vaccine = self.rng.choice(self.one_to_two_hours, size=1)[0]
            new_sleep_ts = start_ts + sleep_ts_after_test_or_vaccine

            timesteps_to_delete = []
            if agent["itinerary"] is not None:
                for timestep, (action, cellid) in agent["itinerary"].items():
                    if action == Action.Sleep:
                        timesteps_to_delete.append(timestep)
                        break
            
            if len(timesteps_to_delete) > 0:
                for timestep in timesteps_to_delete:
                    del agent["itinerary"][timestep]

                if new_sleep_ts <= 143:
                    self.add_to_itinerary(agent, new_sleep_ts, Action.Sleep, res_cellid)
                else:
                    self.add_to_itinerary(agent, new_sleep_ts - 143, Action.Sleep, res_cellid, next_day=True)

        if start_work_school_ts is not None and end_work_school_ts is not None: # if work or school already scheduled
            duration = self.rng.choice(self.one_to_two_hours, size=1)[0]
            if start_ts + duration < end_work_school_ts - 6: # and finishes 1 hour before work ends, go back to work otherwise go back home (home would already be scheduled so don't do anything)
                self.add_to_itinerary(agent, start_ts + duration, work_school_action, work_school_cellid)

        if quarantine_ts is not None and start_ts > quarantine_ts: # if vaccination/test start ts after start quarantine ts, schedule sleep in a random timestep in the next 2 hours
            back_home_ts_after_test_or_vaccine = self.rng.choice(self.one_to_two_hours, size=1)[0]
            new_sleep_ts = start_ts + back_home_ts_after_test_or_vaccine

            if min_start_sleep_ts is None:
                sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                min_start_sleep_ts = sleeping_hours_by_age_group[2]

            if new_sleep_ts < min_start_sleep_ts:
                new_sleep_ts = min_start_sleep_ts
                print("out of range analysis. new_sleep_ts (min_start_sleep_ts): " + str(new_sleep_ts))

            if new_sleep_ts <= 143:
                print("out of range analysis. new_sleep_ts: " + str(new_sleep_ts))
                self.add_to_itinerary(agent, new_sleep_ts, Action.Sleep, res_cellid)
            else:
                print("out of range analysis. new_sleep_ts (next_day): " + str(new_sleep_ts - 143))
                self.add_to_itinerary(agent, new_sleep_ts - 143, Action.Sleep, res_cellid, next_day=True)

        return agent
    
    # updates the itinerary and itinerary_nextday dictionaries with start_timesteps
    # the dictionary is used at the end of the itinerary generation process to update the cells_agents_timesteps dict (used in contact network)
    # method does not allow replacing of key, in which case the timestep is incremented, and the method is called recursively until finding a freeslot
    def add_to_itinerary(self, agent, timestep, action, cellid, next_day=False):
        cellid = int(cellid)
        timestep = int(timestep)

        if agent["itinerary"] is None:
            agent["itinerary"] = {}
        
        if not next_day:
            if timestep not in agent["itinerary"]:
                agent["itinerary"][timestep] = (action, cellid)
            else:
                if action == Action.Transport:
                    curr_action, curr_cellid = agent["itinerary"][timestep]
                    agent["itinerary"][timestep] = (action, cellid) # replace with transport and move curr_action to next free timestep

                    timestep += 1

                    if timestep <= 143:
                        self.add_to_itinerary(agent, timestep, curr_action, curr_cellid, next_day)
                    else:
                        self.add_to_itinerary(agent, timestep, curr_action, curr_cellid, True)
                else:
                    timestep += 1

                    if timestep <= 143:
                        self.add_to_itinerary(agent, timestep, action, cellid, next_day)
                    else:
                        timestep -= 143
                        self.add_to_itinerary(agent, timestep, action, cellid, True)
        else:
            if agent["itinerary_nextday"] is None:
                agent["itinerary_nextday"] = {}

            if timestep not in agent["itinerary_nextday"]:
                agent["itinerary_nextday"][timestep] = (action, cellid)
            else:
                if action == Action.Transport:
                    curr_action, curr_cellid = agent["itinerary_nextday"][timestep]
                    agent["itinerary_nextday"][timestep] = (action, cellid) # replace with transport and move curr_action to next free timestep

                    timestep += 1
                    self.add_to_itinerary(agent, timestep, curr_action, curr_cellid, next_day)
                else:
                    timestep += 1
                    self.add_to_itinerary(agent, timestep, action, cellid, next_day)

    def get_timestep_by_hour(self, hr, leeway_ts=-1, min_ts=None, max_ts=None, cushion_ts=1):
        actual_timestep = round((hr * self.timesteps_in_hour)) #+ 1
        timestep_with_leeway = actual_timestep

        if leeway_ts > -1:
            leeway_range = np.arange(-leeway_ts, leeway_ts + 1)

            # Calculate the middle index of the array
            mid = len(leeway_range) // 2

            sigma = 1.0
            probs = np.exp(-(np.arange(len(leeway_range)) - mid)**2 / (2*sigma**2))
            probs /= probs.sum()

            # Sample from the array with probabilities favouring the middle range (normal dist)
            sampled_leeway = self.rng.choice(leeway_range, size=1, replace=False, p=probs)[0]

            timestep_with_leeway = actual_timestep + sampled_leeway

            if min_ts is not None and timestep_with_leeway <= min_ts:
                timestep_with_leeway = min_ts + cushion_ts
            
            if max_ts is not None and timestep_with_leeway >= max_ts:
                timestep_with_leeway = max_ts - cushion_ts

        return timestep_with_leeway
    
    def get_hour_by_timestep(self, timestep):       
        return timestep / self.timesteps_in_hour
    
    def convert_to_generic_non_daily_activity(self, nda):
        if type(nda) == NonDailyActivityEmployed:
            if nda == NonDailyActivityEmployed.NormalWorkingDay:
                return NonDailyActivity.NormalWorkOrSchoolDay
            elif nda == NonDailyActivityEmployed.VacationLocal:
                return NonDailyActivity.Local
            elif nda == NonDailyActivityEmployed.VacationTravel:
                return NonDailyActivity.Travel
            elif nda == NonDailyActivityEmployed.SickLeave:
                return NonDailyActivity.Sick
        elif type(nda) == NonDailyActivityStudent:
            if nda == NonDailyActivityStudent.NormalSchoolDay:
                return NonDailyActivity.NormalWorkOrSchoolDay
            elif nda == NonDailyActivityStudent.Sick:
                return NonDailyActivity.Sick
        elif type(nda) == NonDailyActivityNonWorkingDay:
            if nda == NonDailyActivityNonWorkingDay.Local:
                return NonDailyActivity.Local
            elif nda == NonDailyActivityNonWorkingDay.Travel:
                return NonDailyActivity.Travel
            elif nda == NonDailyActivityNonWorkingDay.Sick:
                return NonDailyActivity.Sick
        else:
            return None
    
class Action(IntEnum):
    Home = 1
    Sleep = 2
    WakeUp = 3
    Work = 4
    School = 5
    LocalActivity = 6
    Airport = 7
    Breakfast = 8
    Transport = 9
    Test = 10
    Vaccine = 11
    Hospital = 12

class WeekDay(IntEnum):
    Monday = 1
    Tuesday = 2
    Wednesday = 3
    Thursday = 4
    Friday = 5
    Saturday = 6
    Sunday = 7

class NonDailyActivity(IntEnum):
    NormalWorkOrSchoolDay = 1
    Local = 2
    Travel = 3
    Sick = 4

class NonDailyActivityEmployed(IntEnum):
    NormalWorkingDay = 1
    VacationLocal = 2
    VacationTravel = 3
    SickLeave = 4

class NonDailyActivityStudent(IntEnum):
    NormalSchoolDay = 1
    Sick = 2

class NonDailyActivityNonWorkingDay(IntEnum):
    Local = 1
    Travel = 2
    Sick = 3

class Activity(IntEnum):
    BarsNightClubs = 1
    Food = 2
    EntertainmentIndoor = 3
    EntertainmentOutdoor = 4
    Gym = 5
    Sport = 6
    Shopping = 7
    Religious = 8
    StayHome = 9
    OtherResidenceVisit = 10

class Industry(IntEnum):
    AgricultureForestryFishing = 1
    MiningQuarrying = 2
    Manufacturing = 3
    ElectricityGasSteamACSupply = 4
    WaterSupplySewerageWasteManagementRemediationActivities = 5
    Construction = 6
    WholesaleRetailTrade = 7
    TransportationStorage = 8
    AccommodationFoodServiceActivities = 9
    InformationCommunication = 10
    FinancialInsuranceActivities = 11
    RealEstateActivities = 12
    ProfessionalScientificTechnicalActivities = 13
    AdministrativeSupportServiceActivities = 14
    PublicAdministrationDefenseCompulsorySocialSecurity = 15
    Education = 16
    HumanHealthSocialWorkActivities = 17
    ArtEntertainmentRecreation = 18
    OtherServiceActivities = 19
    ActivitiesOfHouseholdsAsEmployers = 20
    ActivitiesOfExtraterritorialOrganisationsBodies = 21


