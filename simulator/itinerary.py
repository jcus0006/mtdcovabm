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

        self.non_daily_activities_dist = self.params["non_daily_activities_distribution"]
        self.age_brackets = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in self.non_daily_activities_dist] # [[0, 4], [5, 9], ...]

        self.industries_working_hours = self.params["industries_working_hours"]
        self.activities_working_hours = self.params["activities_workplaces_working_hours_overrides"]
        self.working_categories_mindays = self.params["working_categories_mindays"]
        self.shift_working_hours = self.params["shift_working_hours"]
    
    # to be called at the beginning of a new week
    def generate_working_days_for_week(self, agent):
        if agent["empstatus"] == 0: # 0: employed, 1: unemployed, 2: inactive
            # employed
            agent["working_schedule"] = {} # {workingday:(start,end)}

            working_schedule = agent["working_schedule"]
            
            agent_industry = Industry(agent["empind"])

            industry_working_hours_by_ind = self.industries_working_hours[agent_industry - 1]
            industry_working_week_start_day, industry_working_week_end_day, industry_working_days = industry_working_hours_by_ind[1], industry_working_hours_by_ind[2], industry_working_hours_by_ind[3]

            if agent_industry == Industry.ArtEntertainmentRecreation and agent["ent_activity"] > -1:
                activity_working_hours_overrides = self.activities_working_hours[agent["ent_activity"] - 1]
                industry_start_work_hour, industry_end_work_hour, industry_working_hours = activity_working_hours_overrides[2], activity_working_hours_overrides[3], activity_working_hours_overrides[4]
            else:
                industry_start_work_hour, industry_end_work_hour, industry_working_hours = industry_working_hours_by_ind[4], industry_working_hours_by_ind[5], industry_working_hours_by_ind[6]

            is_shift_based = industry_working_hours == 24

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

                working_hours_range = np.arange(industry_start_work_hour, industry_end_work_hour + 1)
                
                for day in working_days:
                    if is_full_time: # assumed 8 hours
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

                        working_schedule[day] = (working_hours_range[start_hour], working_hours_range[end_hour])

    def generate_itinerary_hh(self, weekday, hh_agents):
        print("generate_itinerary_hh")

        for agent in hh_agents:
            agent["itinerary"] = {} # {timestep: cellindex}

            age_bracket_index = -1 # maybe extract this to start of sim.py and save into "agents"
            for i, ab in enumerate(self.age_brackets):
                if agent["age"] >= ab[0] and agent["age"] <= ab[1]:
                    age_bracket_index = i
                    break

            if agent["empstatus"] == 0: # 0: employed, 1: unemployed, 2: inactive
                # employed
                agent_industry = Industry(agent["empind"])

                industry_working_hours_by_ind = self.industries_working_hours[agent_industry - 1]
                industry_working_week_start_day, industry_working_week_end_day = industry_working_hours_by_ind[1], industry_working_hours_by_ind[2]

                if weekday >= industry_working_week_start_day and weekday <= industry_working_week_end_day: # working day
                    non_daily_activities_dist_by_ab = self.non_daily_activities_dist[age_bracket_index]
                    non_daily_activities_dist_by_ab_options = np.arange(len(non_daily_activities_dist_by_ab))
                    non_daily_activities_dist_by_ab_weights = np.array(non_daily_activities_dist_by_ab[2:])
                    sampled_non_daily_activity = NonDailyActivity(np.random.choice(non_daily_activities_dist_by_ab_options, size=1, p=non_daily_activities_dist_by_ab_weights)[0])

                    if sampled_non_daily_activity == NonDailyActivity.NormalWorkingDay: # sampled normal working day
                        print("normal working day")

                        if agent_industry == Industry.ArtEntertainmentRecreation and agent["ent_activity"] > -1:
                            activity_working_hours_overrides = self.activities_working_hours[agent["ent_activity"] - 1]
                            industry_start_work_hour, industry_end_work_hour, industry_working_hours = activity_working_hours_overrides[2], activity_working_hours_overrides[3], activity_working_hours_overrides[4]
                        else:
                            industry_start_work_hour, industry_end_work_hour, industry_working_hours = industry_working_hours_by_ind[4], industry_working_hours_by_ind[5], industry_working_hours_by_ind[6]

                        is_shift_based = industry_working_hours == 24

                        industry_start_work_hour, industry_end_work_hour = self.get_timestep_by_hour(industry_start_work_hour, 3), self.get_timestep_by_hour(industry_end_work_hour, 3)

                        agent["itinerary"][industry_start_work_hour] = (Action.Work, agent["work_cellid"])
                        agent["itinerary"][industry_end_work_hour] = (Action.Home, agent["res_cellid"])

                        if is_shift_based:
                            # sample a timestep in 2 hours randomly
                            timesteps_in_hour = 60 / self.timestepmins
                            timesteps_options = np.arange(timesteps_in_hour * 2)
                            sampled_timestep = np.random.choice(timesteps_options, size=1)[0]

                            sleep_hour = agent["itinerary"][industry_end_work_hour] + sampled_timestep
                            agent["itinerary"][sleep_hour] = (Action.Sleep, agent["res_cellid"])

                        
                    elif sampled_non_daily_activity == NonDailyActivity.VacationLocal:
                        print("vacation local")
                    elif sampled_non_daily_activity == NonDailyActivity.VacationTravel:
                        print("vacation travel")
                    elif sampled_non_daily_activity == NonDailyActivity.SickLeave:
                        print("sick leave")
                else: 
                    print("non working day (industry)")
            else:
                print("unemployed or inactive")

    def get_timestep_by_hour(self, hr, leeway=-1):
        timesteps_in_hour = 60 / self.timestepmins
        actual_timestep = (hr * timesteps_in_hour) + 1
        timestep_with_leeway = actual_timestep

        if leeway > -1:
            leeway_range = np.arange(range(-leeway, leeway + 1))

            # Calculate the middle index of the array
            mid = len(leeway_range) // 2

            sigma = 0.5
            probs = np.exp(-(np.arange(len(leeway_range)) - mid)**2 / (2*sigma**2))
            probs /= probs.sum()

            # Sample from the array with probabilities favouring the middle range (normal dist)
            sampled_leeway = np.random.choice(leeway_range, size=5, replace=False, p=probs)[0]

            timestep_with_leeway = actual_timestep + sampled_leeway

        return timestep_with_leeway
    
class Action(IntEnum):
    Home = 1
    Sleep = 2
    Work = 3
    School = 4
    LocalActivity = 5
    Travel = 6

class WeekDay(IntEnum):
    Monday = 1
    Tuesday = 2
    Wednesday = 3
    Thursday = 4
    Friday = 5
    Saturday = 6
    Sunday = 7

class NonDailyActivity(IntEnum):
    NormalWorkingDay = 1
    VacationLocal = 2
    VacationTravel = 3
    SickLeave = 4

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


