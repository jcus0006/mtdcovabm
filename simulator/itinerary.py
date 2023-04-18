import numpy as np
from enum import Enum
from enum import IntEnum
from copy import deepcopy
from copy import copy

class Itinerary:
    def __init__(self, params, timestepmins, cells, industries, workplaces, cells_schools, cells_hospital, cells_entertainment, cells_religious, cells_households):
        self.cells_agents_timesteps = {} # to be filled in during itinerary generation. key is cellid, value is (agentid, starttimestep, endtimestep)

        self.params = params
        self.timestepmins = timestepmins
        self.timesteps_in_hour = round(60 / self.timestepmins)
        self.cells = cells
        self.industries = industries
        self.workplaces = workplaces
        self.cells_schools = cells_schools
        self.cells_hospital = cells_hospital
        self.cells_entertainment = cells_entertainment
        self.cells_religious = cells_religious
        self.cells_households = cells_households

        self.non_daily_activities_employed_distribution = self.params["non_daily_activities_employed_distribution"]
        self.non_daily_activities_schools_distribution = self.params["non_daily_activities_schools_distribution"]
        self.non_daily_activities_nonworkingday_distribution = self.params["non_daily_activities_nonworkingday_distribution"]
        self.non_daily_activities_num_days = self.params["non_daily_activities_num_days"]

        self.industries_working_hours = self.params["industries_working_hours"]
        self.activities_working_hours = self.params["activities_workplaces_working_hours_overrides"]
        self.working_categories_mindays = self.params["working_categories_mindays"]
        self.shift_working_hours = self.params["shift_working_hours"]
        self.sleeping_hours_by_age_groups = self.params["sleeping_hours_by_age_groups"]
        self.religious_ceremonies_hours = self.params["religious_ceremonies_hours"]
        self.sleeping_hours_range = self.params["sleeping_hours_range"]
        self.activities_by_week_days_distribution = self.params["activities_by_week_days_distribution"]
        self.activities_by_agerange_distribution = self.params["activities_by_agerange_distribution"]
        self.activities_duration_hours = self.params["activities_duration_hours"]

        self.age_brackets = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in self.sleeping_hours_by_age_groups] # [[0, 4], [5, 9], ...]
        self.age_brackets_workingages = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in self.non_daily_activities_employed_distribution] # [[15, 19], [20, 24], ...]

        # Calculate probability of each activity for agent
        self.activities_by_week_days_by_age_groups = {}

        for age_bracket_index, activity_probs_by_agerange_dist in enumerate(self.activities_by_agerange_distribution):
            activity_probs_for_agerange = activity_probs_by_agerange_dist[2:]

            activities_by_week_days_for_agerange = []

            for activity_index, activity_prob in enumerate(activity_probs_for_agerange):
                activity_id = activity_index + 1
                activity_by_week_days_for_age_range = [activity_id]

                for day in range(1, 8):
                    prob_product = self.activities_by_week_days_distribution[activity_index][day] * activity_prob
                    activity_by_week_days_for_age_range.append(prob_product)

                # Normalize probabilities so they sum to 1
                # total_probability = sum(activity_by_week_days_for_age_range[1:])

                # normalized_activity_by_week_days_for_age_range = []
                # if total_probability > 0:
                #     normalized_activity_by_week_days_for_age_range.append(activity_id)

                #     for joint_prob in activity_by_week_days_for_age_range[1:]:
                #         joint_prob /= total_probability
                #         normalized_activity_by_week_days_for_age_range.append(joint_prob)
                # else:
                #     normalized_activity_by_week_days_for_age_range = activity_by_week_days_for_age_range

                activities_by_week_days_for_agerange.append(activity_by_week_days_for_age_range)

            # remove activity id from 2d matrix
            sliced_arr = [sublst[1:] for sublst in activities_by_week_days_for_agerange]

            sliced_arr = np.array(sliced_arr)

            # Divide each element by the sum of the column (maintain sum to 1)
            normalized_arr = sliced_arr / sliced_arr.sum(axis=0)

            self.activities_by_week_days_by_age_groups[age_bracket_index] = normalized_arr

    # to be called at the beginning of a new week
    def generate_working_days_for_week(self, agent):
        if agent["empstatus"] == 0: # 0: employed, 1: unemployed, 2: inactive
            # employed
            agent["working_schedule"] = {} # {workingday:(start,end)}

            working_schedule = agent["working_schedule"]
            
            agent_industry = Industry(agent["empind"])

            industry_working_hours_by_ind = self.industries_working_hours[agent_industry - 1]
            industry_working_week_start_day, industry_working_week_end_day, industry_working_days = industry_working_hours_by_ind[1], industry_working_hours_by_ind[2], industry_working_hours_by_ind[3]

            if agent_industry == Industry.ArtEntertainmentRecreation and "ent_activity" in agent and agent["ent_activity"] > -1:
                activity_working_hours_overrides = self.activities_working_hours[agent["ent_activity"] - 1]
                industry_start_work_hour, industry_end_work_hour, industry_working_hours = activity_working_hours_overrides[2], activity_working_hours_overrides[3], activity_working_hours_overrides[4]
            else:
                industry_start_work_hour, industry_end_work_hour, industry_working_hours = industry_working_hours_by_ind[4], industry_working_hours_by_ind[5], industry_working_hours_by_ind[6]
            
            if "isshiftbased" not in agent:

                is_shift_based = industry_working_hours == 24

                agent["isshiftbased"] = is_shift_based
            else:
                is_shift_based = agent["isshiftbased"]

            working_days = []

            if is_shift_based:
                min_working_days = self.working_categories_mindays[1][1]

                max_working_days = min_working_days + 1

                num_working_days = np.random.choice(np.arange(min_working_days, max_working_days + 1), size=1)[0]

                working_days_range = np.arange(industry_working_week_start_day, industry_working_week_end_day + 1)

                working_days = np.random.choice(working_days_range, size=num_working_days, replace=False)
                
                working_days = sorted(working_days)

                for index, day in enumerate(working_days):
                    if index != 0 and working_days[index - 1] == day-1:
                        previous_day_schedule = working_schedule[working_days[index - 1]]
                        previous_day_start_hour = previous_day_schedule[0]

                        for shift_working_hour_option in self.shift_working_hours:
                            if shift_working_hour_option[1] != previous_day_start_hour:
                                working_schedule[day] = (shift_working_hour_option[1], shift_working_hour_option[2])
                    else:
                        working_hours_options = np.arange(len(self.shift_working_hours))
                        
                        sampled_working_hours_option = np.random.choice(working_hours_options, size=1)[0]

                        working_schedule[day] = (self.shift_working_hours[sampled_working_hours_option][1], self.shift_working_hours[sampled_working_hours_option][2])
            else:
                is_full_time = agent["empftpt"] == 0 or agent["empftpt"] == 1

                if is_full_time:
                    min_working_days = self.working_categories_mindays[0][1]
                else:
                    min_working_days = self.working_categories_mindays[2][1]
            
                max_working_days = industry_working_days

                num_working_days = np.random.choice(np.arange(min_working_days, max_working_days + 1), size=1)[0]

                working_days_range = np.arange(industry_working_week_start_day, industry_working_week_end_day + 1)

                working_days = np.random.choice(working_days_range, size=num_working_days, replace=False)

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
                            sampled_option = np.random.choice(options, size=1)[0]

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
                        sampled_option = np.random.choice(options, size=1)[0]

                        start_hour = sampled_option * 4

                        # if start_hour > 0:
                        #     start_hour += 1

                        end_hour = start_hour + 4

                        actual_start_hour = working_hours_range[start_hour]
                        actual_end_hour = working_hours_range[end_hour]

                        if actual_end_hour > 24:
                            actual_end_hour = actual_end_hour - 24

                        working_schedule[day] = (actual_start_hour, actual_end_hour)

    def generate_itinerary_hh(self, simday, weekday, agents, agents_ids_by_ages, resident_uids):
        cohab_agents_ids_by_ages = {}
        for agentid in resident_uids:
            cohab_agents_ids_by_ages[agentid] = agents_ids_by_ages[agentid]

        cohab_agents_ids_by_ages = sorted(cohab_agents_ids_by_ages.items(), key=lambda y: y[1], reverse=True)

        for agentid, age in cohab_agents_ids_by_ages:
            agent = agents[agentid]

            age_bracket_index = agent["age_bracket_index"]
            working_age_bracket_index = agent["working_age_bracket_index"]

            sampled_non_daily_activity = None
            prevday_non_daily_activity = None

            guardian = None
            if age < 15 and "guardian_id" in agent:
                guardian = agents[agent["guardian_id"]]

            if "non_daily_activity_recurring" in agent and agent["non_daily_activity_recurring"] is not None:
                non_daily_activity_recurring = agent["non_daily_activity_recurring"]

                if simday in non_daily_activity_recurring:
                    sampled_non_daily_activity = non_daily_activity_recurring[simday]

                if sampled_non_daily_activity is None:
                    if simday-1 in non_daily_activity_recurring:
                        prevday_non_daily_activity = copy(non_daily_activity_recurring[simday-1])
                        agent["non_daily_activity_recurring"] = None

            # skip if within recurring days of sampled_non_daily_activity
            if sampled_non_daily_activity is None:                
                # get previous night sleep time
                prev_night_sleep_hour, prev_night_sleep_timestep, same_day_sleep_hour, same_day_sleep_timestep = None, None, None, None
                overnight_end_work_ts, overnight_end_activity_ts, activity_overnight_cellid = None, None, None
                
                if simday == 1 or prevday_non_daily_activity == NonDailyActivity.Travel: # sample previous night sleeptimestep for first simday and if previous day was Travel
                    prev_weekday = weekday - 1

                    if prev_weekday < 0:
                        prev_weekday = 7

                    sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                    min_start_sleep_hour, max_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend, param_max = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7], sleeping_hours_by_age_group[8], sleeping_hours_by_age_group[9]
                    
                    alpha, beta = alpha_weekday, beta_weekday
                    if prev_weekday == 6 or prev_weekday == 7: # weekend
                        alpha, beta = alpha_weekend, beta_weekend

                    sampled_sleep_hour_from_range = round(np.random.beta(alpha, beta, 1)[0] * start_hour_range + 1)

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
                else:
                    if len(agent["itinerary_nextday"]) > 0: # overnight itinerary
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
                        for timestep, (action, _) in agent["itinerary"].items():
                            if action == Action.Sleep:
                                prev_night_sleep_timestep = timestep
                                prev_night_sleep_hour = self.get_hour_by_timestep(prev_night_sleep_timestep)

                # initialise Itinerary
                agent["itinerary"] = {} # {timestep: cellindex}
                agent["itinerary_nextday"] = {}

                if overnight_end_work_ts is not None:
                    agent["itinerary"][1] = (Action.Work, agent["work_cellid"])
                    agent["itinerary"][overnight_end_work_ts] = (Action.Home, agent["res_cellid"])
                elif overnight_end_activity_ts is not None:
                    agent["itinerary"][1] = (Action.LocalActivity, activity_overnight_cellid)
                    agent["itinerary"][overnight_end_activity_ts] = (Action.Home, agent["res_cellid"])

                # set wake up hour
                min_sleep_hours, max_sleep_hours = self.sleeping_hours_range[0], self.sleeping_hours_range[1]

                start_work_school_hour = None
                wakeup_hour = None
                wakeup_timestep = None
                working_schedule = None

                if agent["empstatus"] == 0 or agent["sc_student"] == 1:
                    if agent["empstatus"] == 0:
                        working_schedule = agent["working_schedule"]

                        if weekday in working_schedule: # will not be possible 2 days after each other for shift
                            start_work_school_hour = working_schedule[weekday][0]

                            if prev_night_sleep_hour is not None:
                                latest_wake_up_hour = prev_night_sleep_hour + max_sleep_hours

                                if latest_wake_up_hour >= 24:
                                    if latest_wake_up_hour == 24:
                                        latest_wake_up_hour = 0
                                    else:
                                        latest_wake_up_hour -= 24 
                            else:
                                latest_wake_up_hour = same_day_sleep_hour + max_sleep_hours

                            if latest_wake_up_hour <= 24 and latest_wake_up_hour >= start_work_school_hour - 1:
                                wakeup_hour = start_work_school_hour - 1
                                wakeup_timestep = self.get_timestep_by_hour(wakeup_hour) # force wake up before work
                            
                    else: # student
                        if weekday >= 1 and weekday <= 5: # weekday
                            start_work_school_hour = 8

                            if prev_night_sleep_hour is not None:
                                latest_wake_up_hour = prev_night_sleep_hour + max_sleep_hours

                                if latest_wake_up_hour >= 24:
                                    if latest_wake_up_hour == 24:
                                        latest_wake_up_hour = 0
                                    else:
                                        latest_wake_up_hour -= 24 
                            else:
                                latest_wake_up_hour = same_day_sleep_hour + max_sleep_hours

                            if latest_wake_up_hour <= 24 and latest_wake_up_hour >= start_work_school_hour - 1:
                                wakeup_hour = start_work_school_hour - 1
                                wakeup_timestep = self.get_timestep_by_hour(wakeup_hour) # force wake up before school

                if wakeup_timestep is None:
                    sleep_hours_range = np.arange(min_sleep_hours, max_sleep_hours + 1)

                    # Calculate the middle index of the array
                    mid = len(sleep_hours_range) // 2

                    sigma = 1.0
                    probs = np.exp(-(np.arange(len(sleep_hours_range)) - mid)**2 / (2*sigma**2))
                    probs /= probs.sum()

                    # Sample from the array with probabilities favouring the middle range (normal dist)
                    sampled_sleep_hours_duration = np.random.choice(sleep_hours_range, size=1, replace=False, p=probs)[0]

                    if prev_night_sleep_hour is not None:
                        wakeup_hour = prev_night_sleep_hour + sampled_sleep_hours_duration
                    else:
                        wakeup_hour = same_day_sleep_hour + sampled_sleep_hours_duration

                    if wakeup_hour > 24:
                        wakeup_hour = wakeup_hour - 24

                    wakeup_timestep = self.get_timestep_by_hour(wakeup_hour)

                agent["itinerary"][wakeup_timestep] = (Action.WakeUp, agent["res_cellid"])

                is_work_or_school_day = False

                end_work_next_day = False
                if sampled_non_daily_activity is None:
                    if guardian is not None and "non_daily_activity_recurring" in guardian: # travel with guardian override
                        non_daily_activity_recurring = guardian["non_daily_activity_recurring"]

                        if non_daily_activity_recurring is not None:
                            guardian_sampled_non_daily_activity = None
                            if simday in non_daily_activity_recurring and simday-1 not in non_daily_activity_recurring: # if first day of non_daily_activity_recurring
                                guardian_sampled_non_daily_activity = non_daily_activity_recurring[simday]

                            if guardian_sampled_non_daily_activity is not None and guardian_sampled_non_daily_activity == NonDailyActivity.Travel:
                                agent["non_daily_activity_recurring"] = deepcopy(guardian["non_daily_activity_recurring"])
                                sampled_non_daily_activity = guardian_sampled_non_daily_activity

                if sampled_non_daily_activity is None:
                    # set the working / school hours
                    if agent["empstatus"] == 0: # 0: employed, 1: unemployed, 2: inactive
                        # employed. consider workingday/ vacationlocal/ vacationtravel/ sickleave
                        
                        if weekday in working_schedule: # working day
                            is_work_or_school_day = True

                            non_daily_activities_dist_by_ab = self.non_daily_activities_employed_distribution[working_age_bracket_index]
                            non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                            non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                            sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                            sampled_non_daily_activity = NonDailyActivityEmployed(sampled_non_daily_activity_index + 1)

                            if sampled_non_daily_activity == NonDailyActivityEmployed.NormalWorkingDay: # sampled normal working day
                                working_hours = working_schedule[weekday]
                                start_work_hour = working_hours[0]
                                end_work_hour = working_hours[1]

                                # if end_work_hour < start_work_hour: # overnight

                                start_work_timestep_with_leeway, end_work_timestep_with_leeway = self.get_timestep_by_hour(start_work_hour, 2, min_ts=wakeup_timestep+1), self.get_timestep_by_hour(end_work_hour, 3)

                                # self.cells_agents_timesteps[agent["res_cellid"]] = (agentid, 0, start_work_timestep_with_leeway)
                                if start_work_timestep_with_leeway <= 143:
                                    agent["itinerary"][start_work_timestep_with_leeway] = (Action.Work, agent["work_cellid"])
                                else:
                                    agent["itinerary_nextday"][start_work_timestep_with_leeway - 143] = (Action.Work, agent["work_cellid"])  

                                # self.cells_agents_timesteps[agent["work_cellid"]] = (agentid, start_work_timestep_with_leeway, end_work_timestep_with_leeway)
                                if end_work_timestep_with_leeway <= 143 and end_work_timestep_with_leeway > start_work_timestep_with_leeway:
                                    agent["itinerary"][end_work_timestep_with_leeway] = (Action.Home, agent["res_cellid"])
                                else:
                                    end_work_next_day = True
                                    agent["itinerary_nextday"][end_work_timestep_with_leeway] = (Action.Home, agent["res_cellid"])                               
                            # elif sampled_non_daily_activity == NonDailyActivityEmployed.VacationLocal:
                            #     print("vacation local")
                            # elif sampled_non_daily_activity == NonDailyActivityEmployed.VacationTravel:
                            #     print("vacation travel")
                            # elif sampled_non_daily_activity == NonDailyActivityEmployed.SickLeave:
                            #     print("sick leave")

                            sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                        else: 
                            # non working day
                            # unemployed/inactive. only consider local / travel / sick
                            non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[working_age_bracket_index]
                            non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                            non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                            sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                            sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                            sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                    elif agent["sc_student"] == 1:
                        if weekday <= 5: # monday to friday
                            # students. only consider schoolday / sick
                            non_daily_activities_dist_by_ab = self.non_daily_activities_schools_distribution[age_bracket_index]
                            non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                            non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                            sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                            sampled_non_daily_activity = NonDailyActivityStudent(sampled_non_daily_activity_index + 1)

                            if sampled_non_daily_activity == NonDailyActivityStudent.NormalSchoolDay: # sampled normal working day
                                is_work_or_school_day = True

                                start_school_hour = 8
                                end_school_hour = 15 # students end 1 hour before teachers

                                start_school_timestep, end_school_timestep = self.get_timestep_by_hour(start_school_hour), self.get_timestep_by_hour(end_school_hour)

                                # self.cells_agents_timesteps[agent["res_cellid"]] = (agentid, 0, start_school_timestep)
                                agent["itinerary"][start_school_timestep] = (Action.School, agent["school_cellid"])
                                # self.cells_agents_timesteps[agent["school_cellid"]] = (agentid, start_school_timestep, end_school_timestep)
                                agent["itinerary"][end_school_timestep] = (Action.Home, agent["res_cellid"])                              
                            # elif sampled_non_daily_activity == NonDailyActivityStudent.Sick:
                            #     print("sick school day - stay home")    

                            sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                        else:
                            non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[working_age_bracket_index]
                            non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                            non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                            sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                            sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                            sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                    else:
                        # unemployed/inactive. only consider local / travel / sick
                        non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[working_age_bracket_index]
                        non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                        non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                        sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                        sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                        # if sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Local: # sampled normal working day
                        #     print("unemployed/inactive local")
                        # elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Travel:
                        #     print("unemployed/inactive travel")                           
                        # elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Sick:
                        #     print("unemployed/inactive sick")      

                        sampled_non_daily_activity = self.convert_to_generic_non_daily_activity(sampled_non_daily_activity)
                            
                    # generate number of days for non daily activities i.e. 
                    if sampled_non_daily_activity != NonDailyActivity.NormalWorkOrSchoolDay:
                        non_daily_activities_means = self.non_daily_activities_num_days[age_bracket_index]

                        mean_num_days = 0

                        if sampled_non_daily_activity == NonDailyActivity.Local:
                            mean_num_days = non_daily_activities_means[2]
                        if sampled_non_daily_activity == NonDailyActivity.Travel:
                            mean_num_days = non_daily_activities_means[3]
                        elif sampled_non_daily_activity == NonDailyActivity.Sick:
                            mean_num_days = non_daily_activities_means[4]

                        sampled_num_days = np.random.poisson(mean_num_days)

                        agent["non_daily_activity_recurring"] = {}
                        non_daily_activity_recurring = agent["non_daily_activity_recurring"]
                        for day in range(simday, simday+sampled_num_days+1):
                            non_daily_activity_recurring[day] = sampled_non_daily_activity

            # sample sleep & activities for the day, if not sick and not travelling (for now fill in activity_timestep_ranges to sample activities in next step)
            if sampled_non_daily_activity == NonDailyActivity.Local or sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay:
                # schedule sleeping hours
                sleep_hour = None
                sleep_timestep = None

                if agent["empstatus"] == 0:
                    if sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay and agent["isshiftbased"]:
                        set_sleeping_hour = False
                        # sample a timestep from 30 mins to 2 hours randomly
                        timesteps_options = np.arange(round(self.timesteps_in_hour / 2), round((self.timesteps_in_hour * 2) + 1))
                        sampled_timestep = np.random.choice(timesteps_options, size=1)[0]

                        sleep_timestep = end_work_timestep_with_leeway + sampled_timestep
                        sleep_hour = sleep_timestep / self.timesteps_in_hour

                        if not end_work_next_day and sleep_timestep <= 143:
                            agent["itinerary"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])
                        else:
                            if sleep_timestep > 143: # might have skipped midnight or work might have ended overnight
                                sleep_timestep -= 143

                            agent["itinerary_nextday"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])

                if sleep_timestep is None:
                    # set sleeping hours by age brackets

                    sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                    min_start_sleep_hour, max_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend, param_max = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7], sleeping_hours_by_age_group[8], sleeping_hours_by_age_group[9]
                    
                    alpha, beta = alpha_weekday, beta_weekday
                    if weekday == 6 or weekday == 7: # weekend
                        alpha, beta = alpha_weekend, beta_weekend

                    sampled_sleep_hour_from_range = round(np.random.beta(alpha, beta, 1)[0] * start_hour_range + 1)

                    sleep_hour = min_start_sleep_hour + (sampled_sleep_hour_from_range - 1) # this is 1 based; if sampled_sleep_hour_from_range is 1, sleep_hour should be min_start_sleep_hour

                    sleep_timestep = self.get_timestep_by_hour(sleep_hour)

                    if sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay:
                        if agent["empstatus"] == 0:
                            end_work_school_ts = end_work_timestep_with_leeway
                        elif  agent["sc_student"] == 1:
                            end_work_school_ts = end_school_timestep

                        if sleep_timestep <= end_work_school_ts: # if sampled a time which is earlier or same time as end work school, schedule sleep for 30 mins from work/school end
                            sleep_timestep = end_work_school_ts + round(self.timesteps_in_hour / 2) # sleep 30 mins after work/school end
                                
                    if sleep_timestep <= 143: # am
                        agent["itinerary"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])
                    else:
                        sleep_timestep -= 143

                        agent["itinerary_nextday"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])

                # find out the activity_timestep_ranges (to be filled in further down)
                wakeup_ts, work_ts, end_work_ts, sleep_ts, overnight_end_work_ts, overnight_sleep_ts = None, None, None, None, None, None
                activity_timestep_ranges = []

                if sampled_non_daily_activity == NonDailyActivity.NormalWorkOrSchoolDay:
                    # find activity timestep ranges i.e. time between wake up and work/school, and time between end work/school and sleep

                    for timestep, (action, cellid) in agent["itinerary"].items():
                        if action == Action.WakeUp:
                            wakeup_ts = timestep
                        elif action == Action.Work or action == Action.School:
                            work_ts = timestep
                        elif action == Action.Home:
                            end_work_ts = timestep
                        elif action == Action.Sleep:
                            sleep_ts = timestep

                    for timestep, (action, cellid) in agent["itinerary_nextday"].items():
                        if action == Action.Home:
                            overnight_end_work_ts = timestep
                        elif action == Action.Sleep:
                            sleep_ts = 143 + timestep
                    
                    # if work is scheduled for this day and more than 2 hours between wakeup hour and work, sample activities for the range
                    if work_ts is not None:
                        wakeup_until_work_ts = work_ts - wakeup_ts

                        wakeup_until_work_hours = self.get_hour_by_timestep(wakeup_until_work_ts)

                        if wakeup_until_work_hours > 2:
                            activity_timestep_ranges.append(range(wakeup_ts+1, work_ts+1))

                    if overnight_end_work_ts is None: # non-overnight (normal case)
                        # if more than 2 hours between end work hour and sleep
                        endwork_until_sleep_ts = sleep_ts - end_work_ts
                        endwork_until_sleep_hours = self.get_hour_by_timestep(endwork_until_sleep_ts)

                        if endwork_until_sleep_hours > 2:
                            activity_timestep_ranges.append(range(end_work_ts+1, sleep_ts+1))
                    else: # ends work after midnight (will be handled next day), check whether activities for normal day are possible
                        if sleep_ts is not None and wakeup_ts is not None:
                            wakeup_until_sleep_ts = sleep_ts - wakeup_ts

                            activity_timestep_ranges.append(range(wakeup_ts+1, sleep_ts+1))
                else:
                    # find activity timestep ranges for non-workers or non-work-day
                    for timestep, (action, cellid) in agent["itinerary"].items():
                        if action == Action.WakeUp:
                            wakeup_ts = timestep
                        elif action == Action.Sleep:
                            sleep_ts = timestep

                    for timestep, (action, cellid) in agent["itinerary_nextday"].items():
                        if action == Action.Sleep:
                            sleep_ts = 143
                            overnight_sleep_ts = timestep
                            sleep_ts += overnight_sleep_ts

                    wakeup_until_sleep_ts = sleep_ts - wakeup_ts

                    activity_timestep_ranges.append(range(wakeup_ts+1, sleep_ts+1))

                # fill in acitivity_timestep_ranges with actual acitivities
                for timestep_range in activity_timestep_ranges:
                    activities_slot_ts = sum([1 for i in timestep_range])

                    activities_slot_hours = self.get_hour_by_timestep(activities_slot_ts)

                    next_timestep = timestep_range[0]
                    # repeat until no more hours to fill
                    while activities_slot_hours > 0:
                        # sample activity from activities_by_week_days_distribution X activities_by_agerange_distribution (pre-compute in constructor)

                        activities_probs_for_agegroup_and_day = self.activities_by_week_days_by_age_groups[age_bracket_index][:, weekday-1]
                        activities_indices = np.arange(len(activities_probs_for_agegroup_and_day))

                        sampled_activity_index = np.random.choice(activities_indices, 1, False, p=activities_probs_for_agegroup_and_day)[0]
                        sampled_activity_id = sampled_activity_index + 1

                        activity_working_hours_overrides = self.activities_working_hours[sampled_activity_index]

                        # sample numhours from activities_duration_hours, 
                        # where if sampled_num_hours is > then activities_slot_hours, sampled_num_hours = activities_slot_hours
                        # and if activities_slot_hours - sampled_num_hours < 1 hour, sampled_num_hours = sampled_num_hours + (activities_slot_hours - sampled_num_hours)

                        min_hours, max_hours = self.activities_duration_hours[sampled_activity_index][1], self.activities_duration_hours[sampled_activity_index][2]

                        hours_range = np.arange(min_hours, max_hours+1)

                        sampled_num_hours = np.random.choice(hours_range, size=1)[0]

                        if sampled_num_hours > activities_slot_hours:
                            sampled_num_hours = activities_slot_hours # sampled a larger value than the remaining hours available, hence, go for remaining hours
                        elif activities_slot_hours - sampled_num_hours < 1:
                            sampled_num_hours = sampled_num_hours + (activities_slot_hours - sampled_num_hours) # less than an hour would be left after adding activity, hence add it to this activity

                        action_type = Action.LocalActivity
                        if sampled_activity_index in list(self.cells_entertainment.keys()): # if this is an entertainment activity
                            potential_cells = list(self.cells_entertainment[sampled_activity_index].keys())
                        else:  # non entertainment activities
                            industry_id = activity_working_hours_overrides[1]

                            if industry_id != 0: # workplaces as venues, e.g. food / shopping
                                potential_venues_by_industry = list(self.industries[industry_id].keys())

                                sampled_wp_id = np.random.choice(potential_venues_by_industry)

                                potential_cells = list(self.industries[industry_id][sampled_wp_id].keys())
                            elif sampled_activity_id == 8: # religious
                                potential_cells = list(self.cells_religious.keys())
                            elif sampled_activity_id == 9: # stay home
                                potential_cells = [agent["res_cellid"]]
                                action_type = Action.Home
                            elif sampled_activity_id == 10: # other residence visit
                                potential_cells = list(self.cells_households.keys())

                        sampled_cell_id = np.random.choice(potential_cells)

                        if next_timestep <= 143:
                            agent["itinerary"][next_timestep] = (action_type, sampled_cell_id)
                        else:
                            agent["itinerary_nextday"][next_timestep - 143] = (action_type, sampled_cell_id)  

                        next_timestep += self.get_timestep_by_hour(sampled_num_hours)

                        activities_slot_hours -= sampled_num_hours
                    
                    if next_timestep <= 143: # always return back home after last sampled activity
                        if next_timestep < sleep_timestep: # next_timestep could be slightly larger than sleep_timestep due to rounding, in that case ignore going back Home
                            agent["itinerary"][next_timestep] = (Action.Home, agent["res_cellid"])
                    else:
                        if next_timestep - 143 < sleep_timestep: # next_timestep could be slightly larger than sleep_timestep due to rounding, in that case ignore going back Home
                            agent["itinerary_nextday"][next_timestep - 143] = (Action.Home, agent["res_cellid"])

                if guardian is not None: # kids with a guardian
                    wakeup_ts = None
                    home_ts = None
                    sleep_ts = None
                    home_ts_arr = []
                    # get earliest home_ts and sleep_ts indicating the range of timesteps to replace with guardian activities
                    # wakeup_ts replaces home_ts, if the latter is not found
                    for timestep, (action, cellid) in agent["itinerary"].items():
                        if action == Action.Home:
                            if home_ts is None:
                                home_ts = timestep
                            else:
                                if timestep < home_ts:
                                    home_ts = timestep

                            home_ts_arr.append(timestep)

                        if action == Action.WakeUp:
                            wakeup_ts = timestep

                        if action == Action.Sleep:
                            sleep_ts = timestep

                    if home_ts is None:
                        home_ts = wakeup_ts

                    if sleep_ts is None:
                        sleep_ts = 143 # in the context of kids, only sleep is possible after midnight i.e. itinerary_nextday

                    # clear the extra activities to be replaced by guardian itinerary activities
                    if home_ts is not None:
                        home_ts_arr.remove(home_ts)

                        to_remove = []
                        for timestep, (action, cellid) in agent["itinerary"].items():
                            if timestep in home_ts_arr:
                                to_remove.append(timestep)

                        for tr in to_remove:
                            del agent["itinerary"][tr]
                    else:
                        print("big problem")
                    
                    # fill in kid itinerary by guardian activities
                    for timestep, (action, cellid) in guardian["itinerary"].items():
                        if action == Action.LocalActivity:
                            if timestep >= home_ts and timestep < sleep_ts:
                                agent["itinerary"][timestep] = (action, cellid)               

                start_timesteps = sorted(list(agent["itinerary"].keys()))

                prev_cell_id = -1
                for index, curr_ts in enumerate(start_timesteps):
                    curr_itinerary = agent["itinerary"][curr_ts]
                    start_ts = curr_ts
                    curr_action, curr_cell_id = curr_itinerary[0], curr_itinerary[1]

                    if index == 0: # first: guarantees at least 1 iteration
                        start_ts = 0
                        end_ts = 143
                        if len(start_timesteps) > 0:
                            next_itinerary = start_timesteps[index+1] # assume at least 2 start_timesteps in a day - might cause problems
                            end_ts = next_itinerary
                    elif index == len(start_timesteps) - 1: # last: guarantees at least 2 iterations
                        # prev_itinerary = start_timesteps[index-1]
                        # start_ts = prev_itinerary
                        end_ts = 143
                    else: # mid: guarantees at least 3 iterations
                        # prev_itinerary = start_timesteps[index-1]
                        next_itinerary = start_timesteps[index+1]

                        # start_ts = prev_itinerary
                        end_ts = next_itinerary

                    agent_cell_timestep_range = (agentid, start_ts, end_ts)

                    if curr_cell_id not in self.cells_agents_timesteps:
                        self.cells_agents_timesteps[curr_cell_id] = []

                    if prev_cell_id == -1 or prev_cell_id != curr_cell_id:
                        self.cells_agents_timesteps[curr_cell_id].append(agent_cell_timestep_range)

                        if agent_cell_timestep_range[1] > agent_cell_timestep_range[2]:
                            print("problemos")
                    else:
                        temp_agent_cell_ts_range = self.cells_agents_timesteps[curr_cell_id][-1]
                        temp_agent_cell_ts_range = (temp_agent_cell_ts_range[0], temp_agent_cell_ts_range[1], end_ts)
                        self.cells_agents_timesteps[curr_cell_id][-1] = temp_agent_cell_ts_range

                        if temp_agent_cell_ts_range[1] > temp_agent_cell_ts_range[2]:
                            print("problemos")

                    prev_cell_id = curr_cell_id

                # if self.cells_agents_timesteps[curr_cell_id][-1][2] != 143: # if still home, end cell range at home
                #     temp_agent_cell_ts_range = self.cells_agents_timesteps[curr_cell_id][-1]
                #     temp_agent_cell_ts_range = (temp_agent_cell_ts_range[0], temp_agent_cell_ts_range[1], 143)
                #     self.cells_agents_timesteps[curr_cell_id[-1]] = temp_agent_cell_ts_range
            else:
                # sick or travel
                if sampled_non_daily_activity == NonDailyActivity.Sick: # stay home all day, simply sample sleep - to refer to on the next day itinerary
                    sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                    min_start_sleep_hour, max_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend, param_max = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7], sleeping_hours_by_age_group[8], sleeping_hours_by_age_group[9]
                    
                    alpha, beta = alpha_weekday, beta_weekday
                    if weekday == 6 or weekday == 7: # weekend
                        alpha, beta = alpha_weekend, beta_weekend

                    sampled_sleep_hour_from_range = round(np.random.beta(alpha, beta, 1)[0] * start_hour_range + 1)

                    sleep_hour = min_start_sleep_hour + (sampled_sleep_hour_from_range - 1) # this is 1 based; if sampled_sleep_hour_from_range is 1, sleep_hour should be min_start_sleep_hour

                    sleep_timestep = self.get_timestep_by_hour(sleep_hour)

                    if sleep_timestep <= 143: # am
                        agent["itinerary"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])
                    else:
                        sleep_timestep -= 143

                        agent["itinerary_nextday"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])
            
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
            sampled_leeway = np.random.choice(leeway_range, size=1, replace=False, p=probs)[0]

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
    Travel = 7

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


