import numpy as np
from enum import Enum
from enum import IntEnum

class Itinerary:
    def __init__(self, params, timestepmins, cells, industries, workplaces, cells_schools, cells_hospital, cells_entertainment):
        self.params = params
        self.timestepmins = timestepmins
        self.cells = cells
        self.industries = industries
        self.workplaces = workplaces
        self.cells_schools = cells_schools
        self.cells_hospital = cells_hospital
        self.cells_entertainment = cells_entertainment

        self.non_daily_activities_employed_distribution = self.params["non_daily_activities_employed_distribution"]
        self.non_daily_activities_schools_distribution = self.params["non_daily_activities_schools_distribution"]
        self.non_daily_activities_nonworkingday_distribution = self.params["non_daily_activities_nonworkingday_distribution"]
        self.non_daily_activities_num_days = self.params["non_daily_activities_num_days"]
        self.age_brackets = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in self.non_daily_activities_employed_distribution] # [[0, 4], [5, 9], ...]

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
    
    # to be called at the beginning of a new week
    def generate_working_days_for_week(self, agent):
        if agent["empstatus"] == 0: # 0: employed, 1: unemployed, 2: inactive
            # employed
            agent["working_schedule"] = {} # {workingday:(start,end)}

            working_schedule = agent["working_schedule"]
            
            if "isshiftbased" not in agent:
                agent_industry = Industry(agent["empind"])

                industry_working_hours_by_ind = self.industries_working_hours[agent_industry - 1]
                industry_working_week_start_day, industry_working_week_end_day, industry_working_days = industry_working_hours_by_ind[1], industry_working_hours_by_ind[2], industry_working_hours_by_ind[3]

                if agent_industry == Industry.ArtEntertainmentRecreation and "ent_activity" in agent and agent["ent_activity"] > -1:
                    activity_working_hours_overrides = self.activities_working_hours[agent["ent_activity"] - 1]
                    industry_start_work_hour, industry_end_work_hour, industry_working_hours = activity_working_hours_overrides[2], activity_working_hours_overrides[3], activity_working_hours_overrides[4]
                else:
                    industry_start_work_hour, industry_end_work_hour, industry_working_hours = industry_working_hours_by_ind[4], industry_working_hours_by_ind[5], industry_working_hours_by_ind[6]

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

    def generate_itinerary_hh(self, weekday, hh_agents):
        print("generate_itinerary_hh")

        for agentid, agent in hh_agents.items():
            agent["itinerary"] = {} # {timestep: cellindex}

            age_bracket_index = -1 # maybe extract this to start of sim.py and save into "agents"
            for i, ab in enumerate(self.age_brackets):
                if agent["age"] >= ab[0] and agent["age"] <= ab[1]:
                    age_bracket_index = i
                    break
            
            sampled_non_daily_activity = None
            working_schedule = None
            # set the working / school hours
            if agent["empstatus"] == 0: # 0: employed, 1: unemployed, 2: inactive
                # employed. consider workingday/ vacationlocal/ vacationtravel/ sickleave
                working_schedule = agent["working_schedule"]

                if weekday in working_schedule: # working day
                    non_daily_activities_dist_by_ab = self.non_daily_activities_employed_distribution[age_bracket_index]
                    non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                    non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                    sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                    sampled_non_daily_activity = NonDailyActivityEmployed(sampled_non_daily_activity_index + 1)

                    if sampled_non_daily_activity == NonDailyActivityEmployed.NormalWorkingDay: # sampled normal working day
                        print("normal working day")

                        working_hours = working_schedule[weekday]
                        start_work_timestep_with_leeway, end_work_timestep_with_leeway = self.get_timestep_by_hour(working_hours[0], 3), self.get_timestep_by_hour(working_hours[1], 3)

                        agent["itinerary"][start_work_timestep_with_leeway] = (Action.Work, agent["work_cellid"])
                        agent["itinerary"][end_work_timestep_with_leeway] = (Action.Home, agent["res_cellid"])                                   
                    elif sampled_non_daily_activity == NonDailyActivityEmployed.VacationLocal:
                        print("vacation local")
                    elif sampled_non_daily_activity == NonDailyActivityEmployed.VacationTravel:
                        print("vacation travel")
                    elif sampled_non_daily_activity == NonDailyActivityEmployed.SickLeave:
                        print("sick leave")
                else: 
                    # non working day
                    # unemployed/inactive. only consider local / travel / sick
                    non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[age_bracket_index]
                    non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                    non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                    sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                    sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                    if sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Local: # sampled normal working day
                        print("unemployed/inactive local")
                    elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Travel:
                        print("unemployed/inactive travel")                           
                    elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Sick:
                        print("unemployed/inactive sick")
            elif agent["sc_student"] == 1:
                # students. only consider schoolday / sick
                non_daily_activities_dist_by_ab = self.non_daily_activities_schools_distribution[age_bracket_index]
                non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                sampled_non_daily_activity = NonDailyActivityStudent(sampled_non_daily_activity_index + 1)

                if sampled_non_daily_activity == NonDailyActivityStudent.NormalSchoolDay: # sampled normal working day
                    print("normal school day")

                    start_school_hour = 8
                    end_school_hour = 3 # students end 1 hour before teachers

                    start_school_timestep, end_school_timestep = self.get_timestep_by_hour(start_school_hour), self.get_timestep_by_hour(end_school_hour)

                    agent["itinerary"][start_school_timestep] = (Action.School, agent["work_cellid"])
                    agent["itinerary"][end_school_timestep] = (Action.Home, agent["res_cellid"])                              
                elif sampled_non_daily_activity == NonDailyActivityStudent.Sick:
                    print("sick school day - stay home")      
            else:
                # unemployed/inactive. only consider local / travel / sick
                non_daily_activities_dist_by_ab = self.non_daily_activities_nonworkingday_distribution[age_bracket_index]
                non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab[2:]))
                non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                sampled_non_daily_activity_index = np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0]
                sampled_non_daily_activity = NonDailyActivityNonWorkingDay(sampled_non_daily_activity_index + 1)

                if sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Local: # sampled normal working day
                    print("unemployed/inactive local")
                elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Travel:
                    print("unemployed/inactive travel")                           
                elif sampled_non_daily_activity == NonDailyActivityNonWorkingDay.Sick:
                    print("unemployed/inactive sick")      

            # schedule sleeping hours
            sleep_hour = None
            sleep_timestep = None

            if agent["empstatus"] == 0:
                if sampled_non_daily_activity == NonDailyActivityEmployed.NormalWorkingDay and agent["isshiftbased"]:
                    set_sleeping_hour = False
                    # sample a timestep in 2 hours randomly
                    timesteps_in_hour = round(60 / self.timestepmins)
                    timesteps_options = np.arange(timesteps_in_hour * 2)
                    sampled_timestep = np.random.choice(timesteps_options, size=1)[0]

                    sleep_timestep = end_work_timestep_with_leeway + sampled_timestep
                    sleep_hour = sleep_timestep / timesteps_in_hour
                    agent["itinerary"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])

            if sleep_timestep is None:
                # set sleeping hours by age brackets
                print("set sleeping hour by ab")

                sleeping_hours_by_age_group = self.sleeping_hours_by_age_groups[age_bracket_index]
                min_start_sleep_hour, max_start_sleep_hour, start_hour_range, alpha_weekday, beta_weekday, alpha_weekend, beta_weekend, param_max = sleeping_hours_by_age_group[2], sleeping_hours_by_age_group[3], sleeping_hours_by_age_group[4], sleeping_hours_by_age_group[5], sleeping_hours_by_age_group[6], sleeping_hours_by_age_group[7], sleeping_hours_by_age_group[8], sleeping_hours_by_age_group[9]
                
                alpha, beta = alpha_weekday, beta_weekday
                if weekday == 6 or weekday == 7: # weekend
                    alpha, beta = alpha_weekend, beta_weekend

                sampled_sleep_hour_from_range = round(np.random.beta(alpha, beta, 1)[0] * start_hour_range + 1)

                sleep_hour = min_start_sleep_hour + (sampled_sleep_hour_from_range - 1) # this is 1 based; if sampled_sleep_hour_from_range is 1, sleep_hour should be min_start_sleep_hour

                sleep_timestep = self.get_timestep_by_hour(sleep_hour)
                agent["itinerary"][sleep_timestep] = (Action.Sleep, agent["res_cellid"])

            # set end sleeping hour
            min_sleep_hours, max_sleep_hours = self.sleeping_hours_range[0], self.sleeping_hours_range[1]

            next_day_start_work_school_hour = None
            wakeup_hour = None
            wakeup_timestep = None
            if agent["empstatus"] == 0 or agent["sc_student"] == 1:
                if agent["empstatus"] == 0:
                    if weekday + 1 in working_schedule:
                        next_day_start_work_school_hour = working_schedule[weekday+1][1]

                        latest_wake_up_hour = sleep_hour + max_sleep_hours

                        if latest_wake_up_hour >= next_day_start_work_school_hour - 1:
                            wakeup_hour = next_day_start_work_school_hour - 1
                            wakeup_timestep = self.get_timestep_by_hour(wakeup_hour)
                else: # student
                    if weekday >= 1 and weekday <= 5: # weekday
                        next_day_start_work_school_hour = 8

                        latest_wake_up_hour = sleep_hour + max_sleep_hours

                        if latest_wake_up_hour >= next_day_start_work_school_hour - 1:
                            wakeup_hour = next_day_start_work_school_hour - 1
                            wakeup_timestep = self.get_timestep_by_hour(wakeup_hour)

            if wakeup_timestep is None:
                sleep_hours_range = np.arange(min_sleep_hours, max_sleep_hours + 1)

                # Calculate the middle index of the array
                mid = len(sleep_hours_range) // 2

                sigma = 1.0
                probs = np.exp(-(np.arange(len(sleep_hours_range)) - mid)**2 / (2*sigma**2))
                probs /= probs.sum()

                # Sample from the array with probabilities favouring the middle range (normal dist)
                sampled_sleep_hours_duration = np.random.choice(sleep_hours_range, size=1, replace=False, p=probs)[0]

                wakeup_hour = sleep_hour + sampled_sleep_hours_duration

                if wakeup_hour > 24:
                    wakeup_hour = wakeup_hour - 24

                wakeup_timestep = self.get_timestep_by_hour(wakeup_hour)

            agent["itinerary"][wakeup_timestep] = (Action.WakeUp, agent["res_cellid"])

    def get_timestep_by_hour(self, hr, leeway=-1):
        timesteps_in_hour = round(60 / self.timestepmins)
        actual_timestep = (hr * timesteps_in_hour) + 1
        timestep_with_leeway = actual_timestep

        if leeway > -1:
            leeway_range = np.arange(-leeway, leeway + 1)

            # Calculate the middle index of the array
            mid = len(leeway_range) // 2

            sigma = 1.0
            probs = np.exp(-(np.arange(len(leeway_range)) - mid)**2 / (2*sigma**2))
            probs /= probs.sum()

            # Sample from the array with probabilities favouring the middle range (normal dist)
            sampled_leeway = np.random.choice(leeway_range, size=1, replace=False, p=probs)[0]

            timestep_with_leeway = actual_timestep + sampled_leeway

        return timestep_with_leeway
    
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

