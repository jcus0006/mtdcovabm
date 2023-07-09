import json
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random
import traceback
from cells import Cells
from simulator import util, itinerary_mp, contactnetwork_mp, tourism, seirstateutil, vars
from simulator.epidemiology import SEIRState
from simulator.dynamicparams import DynamicParams
import multiprocessing as mp

def main():
    params = {  "popsubfolder": "1kagents2ktourists2019", # empty takes root (was 500kagents2mtourists2019 / 1kagents2ktourists2019)
                "timestepmins": 10,
                "loadagents": True,
                "loadhouseholds": True,
                "loadinstitutions": True,
                "loadworkplaces": True,
                "loadschools": True,
                "loadtourism": True,
                "religiouscells": True,
                "year": 2021,
                "quickdebug": False,
                "quicktourismrun": False,
                "quickitineraryrun": False,
                "visualise": False,
                "fullpop": 519562,
                "numprocesses": 4
            }

    figure_count = 0

    cellindex = 0
    cells = {}

    cellsfile = open("./data/cells.json")
    cellsparams = json.load(cellsfile)

    itineraryfile = open("./data/itinerary.json")
    itineraryparams = json.load(itineraryfile)

    sleeping_hours_by_age_groups = itineraryparams["sleeping_hours_by_age_groups"]
    non_daily_activities_employed_distribution = itineraryparams["non_daily_activities_employed_distribution"]
    age_brackets = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in sleeping_hours_by_age_groups] # [[0, 4], [5, 9], ...]
    age_brackets_workingages = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in non_daily_activities_employed_distribution] # [[15, 19], [20, 24], ...]

    contactnetworkfile = open("./data/contactnetwork.json")
    contactnetworkparams = json.load(contactnetworkfile)

    sociability_rate_min_max = contactnetworkparams["sociabilityrateminmax"]
    sociability_rate_min, sociability_rate_max = sociability_rate_min_max[0], sociability_rate_min_max[1]
    powerlaw_distribution_parameters = contactnetworkparams["powerlawdistributionparameters"]
    # sociability_rate_options = np.arange(len(sociability_rate_distribution))

    epidemiologyfile = open("./data/epidemiology.json")
    epidemiologyparams = json.load(epidemiologyfile)
    initial_seir_state_distribution = epidemiologyparams["initialseirstatedistribution"]
    tourist_entry_infection_probability = epidemiologyparams["tourist_entry_infection_probability"]

    tourismfile = open("./data/tourism.json")
    tourismparams = json.load(tourismfile)

    population_sub_folder = ""

    if params["quickdebug"]:
        params["popsubfolder"] = "10kagents"

    if len(params["popsubfolder"]) > 0:
        population_sub_folder = params["popsubfolder"] + "/"

    # load agents and all relevant JSON files on each node
    agents = {}
    agents_ids_by_ages = {}
    agents_ids_by_agebrackets = {i:[] for i in range(len(age_brackets))}

    # # contact network model
    # cells_agents_timesteps = {} # {cellid: (agentid, starttimestep, endtimestep)}

    # # transmission model
    agents_seir_state = [] # whole population with following states, 0: undefined, 1: susceptible, 2: exposed, 3: infectious, 4: recovered, 5: deceased
    agents_seir_state_transition_for_day = {} # handled as dict, because it will only apply for a subset of agents per day
    agents_infection_type = {} # handled as dict, because not every agent will be infected
    agents_infection_severity = {} # handled as dict, because not every agent will be infected
    agents_vaccination_doses = [] # number of doses per agent

    # tourism
    tourists_arrivals_departures_for_day = {} # handles both incoming and outgoing, arrivals and departures. handled as a dict, as only represents day
    tourists_arrivals_departures_for_nextday = {}
    n_tourists = 0

    if params["loadtourism"]:
        touristsfile = open("./population/" + population_sub_folder + "tourists.json")
        tourists = json.load(touristsfile)
        tourists = {tour["tourid"]:{"groupid":tour["groupid"], "subgroupid":tour["subgroupid"], "age":tour["age"], "gender": tour["gender"]} for tour in tourists}
        
        n_tourists = len(tourists)

        touristsgroupsfile = open("./population/" + population_sub_folder + "touristsgroups.json")
        touristsgroups = json.load(touristsgroupsfile)
        touristsgroups = {tg["groupid"]:{"subgroupsmemberids":tg["subgroupsmemberids"], "accominfo":tg["accominfo"], "reftourid":tg["reftourid"], "arr": tg["arr"], "dep": tg["dep"], "purpose": tg["purpose"], "accomtype": tg["accomtype"]} for tg in touristsgroups}

        touristsgroupsdaysfile = open("./population/" + population_sub_folder + "touristsgroupsdays.json")
        touristsgroupsdays = json.load(touristsgroupsdaysfile)
        touristsgroupsdays = {day["dayid"]:day["member_uids"] for day in touristsgroupsdays}

    if params["loadagents"]:
        agentsfile = open("./population/" + population_sub_folder + "agents.json")
        agents = json.load(agentsfile)

        n_locals = len(agents)

        # if params["quickdebug"]:
        #     agents = {str(i):agents[str(i)] for i in range(1000)}

        temp_agents = {int(k): v for k, v in agents.items()}

        agents_vaccination_doses = np.array([0 for i in range(n_locals)])

        locals_ratio_to_full_pop = n_locals / params["fullpop"]

        if params["loadtourism"]:
            largest_agent_id = sorted(list(temp_agents.keys()), reverse=True)[0]

            for i in range(len(tourists)):
                temp_agents[largest_agent_id+1] = {}
                largest_agent_id += 1

        agents_seir_state = np.array([SEIRState(0) for i in range(len(temp_agents))])

        # contactnetwork_sum_time_taken = 0
        # contactnetwork_util = contactnetwork.ContactNetwork(n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, cells, cells_agents_timesteps, contactnetworkparams, epidemiologyparams, contactnetwork_sum_time_taken, False, False, params["numprocesses"])
        # epi_util = contactnetwork_util.epi_util

        for index, (agent_uid, agent) in enumerate(temp_agents.items()):
            if index < n_locals: # ignore tourists for now
                agent["curr_cellid"] = -1
                agent["res_cellid"] = -1
                agent["work_cellid"] = -1
                agent["school_cellid"] = -1
                agent["inst_cellid"] = -1
                # agent["symptomatic"] = False
                agent["tourist_id"] = None 
                agent["state_transition_by_day"] = {}
                # intervention_events_by_day
                agent["test_day"] = [] # [day, timestep]
                agent["test_result_day"] = [] # [day, timestep]
                agent["quarantine_days"] = [] # [[[startday, timestep], [endday, timestep]]] -> [startday, timestep, endday]
                agent["vaccination_days"] = [] # [[day, timestep]]
                agent["hospitalisation_days"] = [] # [[startday, timestep], [endday, timestep]] -> [startday, timestep, endday]

                agent, age, agents_ids_by_ages, agents_ids_by_agebrackets = util.set_age_brackets(agent, agents_ids_by_ages, agent_uid, age_brackets, age_brackets_workingages, agents_ids_by_agebrackets)

                agent["epi_age_bracket_index"] = util.get_sus_mort_prog_age_bracket_index(age)

                agent = util.set_public_transport_regular(agent, itineraryparams["public_transport_usage_probability"][0])
            else:
                break

            # agent["soc_rate"] = np.random.choice(sociability_rate_options, size=1, p=sociability_rate_distribution)[0]

        temp_agents = util.generate_sociability_rate_powerlaw_dist(temp_agents, agents_ids_by_agebrackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count)

        agents_seir_state = seirstateutil.initialize_agent_states(n_locals, initial_seir_state_distribution, agents_seir_state)

        agents = temp_agents

        temp_agents = None

        # contactnetwork_util.agents = None
        # epi_util.agents = None
        # contactnetwork_util.agents = agents
        # epi_util.agents = agents

        # maleagents = {k:v for k, v in agents.items() if v["gender"] == 0}
        # femaleagents = {k:v for k, v in agents.items() if v["gender"] == 1}

        # agentsbyages = {}
        # malesbyages = {}
        # femalesbyages = {}

        # for age in range(101):
        #     agentsbyage = {k:v for k, v in agents.items() if v["age"] == age}
        #     malesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 0}
        #     femalesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 1}

        #     agentsbyages[age] = agentsbyage
        #     malesbyages[age] = malesbyage
        #     femalesbyages[age] = femalesbyage

        # agentsemployed = {k:v for k, v in agents.items() if v["empstatus"] == 0}
        # malesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 0}
        # femalesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 1}

        # agentsbyindustries = {}
        # malesbyindustries = {}
        # femalesbyindustries = {}

        # for industry in range(1,22):
        #     agentsbyindustry = {k:v for k, v in agentsemployed.items() if v["empind"] == industry}
        #     malesbyindustry = {k:v for k, v in malesemployed.items() if v["empind"] == industry}
        #     femalesbyindustry = {k:v for k, v in femalesemployed.items() if v["empind"] == industry}

        #     agentsbyindustries[industry] = agentsbyindustry
        #     malesbyindustries[industry] = malesbyindustry
        #     femalesbyindustries[industry] = femalesbyindustry

    cells_util = Cells(agents, cells, cellindex)

    if params["loadhouseholds"]:
        householdsfile = open("./population/" + population_sub_folder + "households.json")
        households_original = json.load(householdsfile)

        workplaces = []
        workplaces_cells_params = []

        if params["loadworkplaces"]:
            workplacesfile = open("./population/" + population_sub_folder + "workplaces.json")
            workplaces = json.load(workplacesfile)

            workplaces_cells_params = cellsparams["workplaces"]

        households, cells_households, householdsworkplaces, cells_householdsworkplaces = cells_util.convert_households(households_original, workplaces, workplaces_cells_params)

        # contactnetwork_util.epi_util.cells_households = cells_households
        
    if params["loadinstitutions"]:
        institutiontypesfile = open("./population/" + population_sub_folder + "institutiontypes.json")
        institutiontypes_original = json.load(institutiontypesfile)

        institutionsfile = open("./population/" + population_sub_folder + "institutions.json")
        institutions = json.load(institutionsfile)

        institutions_cells_params = cellsparams["institutions"]

        institutiontypes, cells_institutions = cells_util.split_institutions_by_cellsize(institutions, institutions_cells_params[0], institutions_cells_params[1])  

        # contactnetwork_util.epi_util.cells_institutions = cells_institutions  

    hh_insts = []
    if params["loadhouseholds"]:
        for hh in households.values():
            hh_inst = {"id": hh["hhid"], "is_hh": True, "resident_uids": hh["resident_uids"]}
            hh_insts.append(hh_inst)

    if params["loadinstitutions"]:
        for inst in institutions:
            hh_inst = {"id": inst["instid"], "is_hh": False, "resident_uids": inst["resident_uids"]}
            hh_insts.append(hh_inst)

    if hh_insts is not None:
        for hh_inst in hh_insts:
            for agentid in hh_inst["resident_uids"]:
                agent = agents[agentid]

                if agent["age"] < 15: # assign parent/guardian at random
                    other_resident_ages_by_uids = {uid:agents[uid]["age"] for uid in hh_inst["resident_uids"] if uid != agentid}

                    other_resident_uids = list(other_resident_ages_by_uids.keys())

                    other_resident_ages = list(other_resident_ages_by_uids.values())

                    other_resident_uids_indices = np.arange(len(other_resident_uids))

                    # give preference to agents with a difference in age of between 15 and 40 years (from kid in iteration)
                    other_resident_uids_guardian_pool = np.array([other_resident_uids[i] for i in range(len(other_resident_uids_indices)) if other_resident_ages[i] - agent["age"] >= 15 and other_resident_ages[i] - agent["age"] <= 40])

                    # if none found, settle for any agent of 15 years or older
                    if len(other_resident_uids_guardian_pool) == 0:
                        other_resident_uids_guardian_pool = np.array([other_resident_uids[i] for i in range(len(other_resident_uids_indices)) if other_resident_ages[i] >= 15])

                    other_resident_uids_guardian_pool_indices = np.arange(len(other_resident_uids_guardian_pool))
                    sampled_index  = np.random.choice(other_resident_uids_guardian_pool_indices, size=1)[0]
                    sampled_uid = other_resident_uids_guardian_pool[sampled_index]
                
                    if sampled_uid is not None:
                        agent["guardian_id"] = sampled_uid
                    else:
                        print("big problem")


    if params["loadworkplaces"]:
        if len(workplaces) == 0:
            workplacesfile = open("./population/" + population_sub_folder + "workplaces.json")
            workplaces = json.load(workplacesfile)

        if len(workplaces_cells_params) == 0:
            workplaces_cells_params = cellsparams["workplaces"]

        hospital_cells_params = cellsparams["hospital"]
        testing_hubs_cells_params = cellsparams["testinghubs"]
        vaccinations_hubs_cells_params = cellsparams["vaccinationhubs"]
        transport_cells_params = cellsparams["transport"]
        airport_cells_params = cellsparams["airport"]
        accom_cells_params = cellsparams["accommodation"]
        entertainment_acitvity_dist = cellsparams["entertainmentactivitydistribution"]

        transport, cells_transport = cells_util.create_transport_cells(transport_cells_params[2], transport_cells_params[0], transport_cells_params[1], transport_cells_params[3], transport_cells_params[4])

        # cells_accommodation_by_accomid = {} # {accomid: [{cellid: {cellinfo}}]}
        accommodations = []
        roomsizes_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomsize: [member_uids]}}} - here member_uids represents room ids
        rooms_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomid: {"roomsize":1, "member_uids":[]}}}} - here member_uids represents tourist ids
        tourists_active_groupids, tourists_active_ids = [], []
        accomgroups = None

        if params["loadtourism"]:
            accommodationsfile = open("./population/" + population_sub_folder + "accommodations.json")
            accommodations = json.load(accommodationsfile)

            for accom in accommodations:
                if accom["accomtypeid"] not in roomsizes_by_accomid_by_accomtype:
                    roomsizes_by_accomid_by_accomtype[accom["accomtypeid"]] = {}

                if accom["accomtypeid"] not in rooms_by_accomid_by_accomtype:
                    rooms_by_accomid_by_accomtype[accom["accomtypeid"]] = {}

                roomsizes_accoms_by_type = roomsizes_by_accomid_by_accomtype[accom["accomtypeid"]]
                rooms_accoms_by_type = rooms_by_accomid_by_accomtype[accom["accomtypeid"]]

                if accom["accomid"] not in roomsizes_accoms_by_type:
                    roomsizes_accoms_by_type[accom["accomid"]] = {}

                if accom["accomid"] not in rooms_accoms_by_type:
                    rooms_accoms_by_type[accom["accomid"]] = {}

                roomsizes_accom_by_id = roomsizes_accoms_by_type[accom["accomid"]]
                roomsizes_accom_by_id[accom["roomsize"]] = accom["member_uids"] # guaranteed to be only 1 room size

                rooms_accom_by_id = rooms_accoms_by_type[accom["accomid"]]

                for roomid in accom["member_uids"]:
                    rooms_accom_by_id[roomid] = {}

        # handle cell splitting (on workplaces & accommodations)
        cells_industries_by_indid_by_wpid, cells_industries, cells_restaurants, cells_accommodation, cells_accommodation_by_accomid, cells_breakfast_by_accomid, rooms_by_accomid_by_accomtype, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_airport = cells_util.split_workplaces_by_cellsize(workplaces, roomsizes_by_accomid_by_accomtype, rooms_by_accomid_by_accomtype, workplaces_cells_params, hospital_cells_params, testing_hubs_cells_params, vaccinations_hubs_cells_params, airport_cells_params, accom_cells_params, transport, entertainment_acitvity_dist)

        # contactnetwork_util.epi_util.cells_accommodation = cells_accommodation
        # airport_cells_params = cellsparams["airport"]

        # cell.create_airport_cell()

        # cellindex += 1

    if params["loadschools"]:
        schoolsfile = open("./population/" + population_sub_folder + "schools.json")
        schools = json.load(schoolsfile)

        schools_cells_params = cellsparams["schools"]

        min_nts_size, max_nts_size, min_classroom_size, max_classroom_size = cells_util.get_min_max_school_sizes(schools)

        print("Min classroom size: " + str(min_classroom_size) + ", Max classroom size: " + str(max_classroom_size))
        print("Min non-teaching staff size: " + str(min_nts_size) + ", Max classroom size: " + str(max_nts_size))

        schooltypes, cells_schools, cells_classrooms = cells_util.split_schools_by_cellsize(schools, schools_cells_params[0], schools_cells_params[1])

    if params["religiouscells"]:
        religious_cells_params = cellsparams["religious"]

        churches, cells_religious = cells_util.create_religious_cells(religious_cells_params[2], religious_cells_params[0], religious_cells_params[1], religious_cells_params[3], religious_cells_params[4])

    # this might cause problems when referring to related agents, by household, workplaces etc
    # if params["quickdebug"]:
    #     agents = {i:agents[i] for i in range(10_000)}

    vars_util = vars.Vars()
    vars_util.populate(agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses)
    tourist_util = tourism.Tourism(tourismparams, cells, n_locals, tourists, agents, agents_seir_state, touristsgroupsdays, touristsgroups, rooms_by_accomid_by_accomtype, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids, tourists_active_ids, age_brackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count, initial_seir_state_distribution)
    dyn_params = DynamicParams(n_locals, n_tourists, epidemiologyparams)
    try:
        itinerary_sum_time_taken = 0
        tourist_itinerary_sum_time_taken = 0
        contactnetwork_sum_time_taken = 0
        
        for day in range(1, 365+1):
            day_start = time.time()

            weekday, weekdaystr = util.day_of_year_to_day_of_week(day, params["year"])

            # itinerary_util.cells_agents_timesteps = {}
            # itinerary_util.epi_util = epi_util
            # contactnetwork_util.epi_util = epi_util
            # contactnetwork_util.cells_agents_timesteps = itinerary_util.cells_agents_timesteps
            # agents_seir_state_transition_for_day = {} # always cleared for a new day, will be filled in itinerary, and used in direct contact simulation (epi)
            # agents_directcontacts_by_simcelltype_by_day = {}

            # itinerary_util = itinerary.Itinerary(itineraryparams, params["timestepmins"], n_locals, n_tourists, locals_ratio_to_full_pop, agents, tourists, cells, industries, workplaces, cells_restaurants, cells_schools, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment, cells_religious, cells_households, cells_accommodation_by_accomid, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, cells_agents_timesteps, tourist_entry_infection_probability, epidemiologyparams, dyn_params, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_directcontacts_by_simcelltype_by_day, agents_vaccination_doses, tourists_active_ids)

            # if params["loadtourism"]:
            #     print("generate_tourist_itinerary for simday " + str(day) + ", weekday " + str(weekday))
            #     start = time.time()
            #     agents, tourists, cells, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids = tourist_util.initialize_foreign_arrivals_departures_for_day(day)
                
            #     itinerary_util.generate_tourist_itinerary(day, weekday, touristsgroups, tourists_active_groupids, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday)
                
            #     time_taken = time.time() - start
            #     tourist_itinerary_sum_time_taken += time_taken
            #     avg_time_taken = tourist_itinerary_sum_time_taken / day
            #     print("generate_tourist_itinerary for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))

            # epi_util.tourists_active_ids = tourist_util.tourists_active_ids

            if not params["quickitineraryrun"]:
                if day == 1: # from day 2 onwards always calculated at eod
                    dyn_params.refresh_dynamic_parameters(day, agents_seir_state, tourists_active_ids) # TO REVIEW

            if not params["quicktourismrun"]:
                start = time.time()  

                itinerary_mp.localitinerary_parallel(day, 
                                                    weekday, 
                                                    weekdaystr, 
                                                    itineraryparams, 
                                                    params["timestepmins"], 
                                                    n_locals, 
                                                    n_tourists, 
                                                    locals_ratio_to_full_pop, 
                                                    agents,
                                                    agents_ids_by_ages,                                                      
                                                    tourists, 
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
                                                    tourist_entry_infection_probability, 
                                                    epidemiologyparams, 
                                                    dyn_params, 
                                                    tourists_active_ids, 
                                                    hh_insts, 
                                                    params["numprocesses"])
                
                time_taken = time.time() - start
                itinerary_sum_time_taken += time_taken
                avg_time_taken = itinerary_sum_time_taken / day
                print("localitinerary_parallel for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken)) 

                if not params["quickitineraryrun"]:
                    print("simulate_contact_network for simday " + str(day) + ", weekday " + str(weekday))
                    start = time.time()       

                    contactnetwork_mp.contactnetwork_parallel(day, weekday, n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_directcontacts_by_simcelltype_by_day, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, tourists_active_ids, cells, cells_households, cells_institutions, cells_accommodation, cells_agents_timesteps, contactnetworkparams, epidemiologyparams, dyn_params, contactnetwork_sum_time_taken, params["numprocesses"])

                    time_taken = time.time() - start
                    contactnetwork_sum_time_taken += time_taken
                    avg_time_taken = contactnetwork_sum_time_taken / day
                    print("simulate_contact_network for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))                 

                # contact tracing
                print("contact_tracing for simday " + str(day) + ", weekday " + str(weekday))
                start = time.time()

                # epi_util.contact_tracing(day) # TO REVIEW

                time_taken = time.time() - start
                print("contact_tracing time taken: " + str(time_taken))

                # vaccinations
                print("schedule_vaccinations for simday " + str(day) + ", weekday " + str(weekday))
                start = time.time()

                # epi_util.schedule_vaccinations(day) # TO REVIEW

                time_taken = time.time() - start
                print("schedule_vaccinations time taken: " + str(time_taken))

                print("refresh_dynamic_parameters for simday " + str(day) + ", weekday " + str(weekday))
                start = time.time()
                dyn_params.refresh_dynamic_parameters(day, agents_seir_state, tourists_active_ids)

                time_taken = time.time() - start
                print("refresh_dynamic_parameters time taken: " + str(time_taken))

            day_time_taken = time.time() - day_start
            print("simulation day: " + str(day) + ", weekday " + str(weekday) + ", curr infectious rate: " + str(round(dyn_params.infectious_rate, 2)) + ", time taken: " + str(day_time_taken))
    except:
        with open('stack_trace.txt', 'w') as f:
            traceback.print_exc(file=f)

    print(len(agents))

if __name__ == '__main__':
    main()
