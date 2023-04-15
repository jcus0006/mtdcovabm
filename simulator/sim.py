import json
import numpy as np
import math
import time
from cells import Cells
from simulator import util
from simulator import itinerary

# to establish set of static parameters such as cellsize, but grouped in dict

params = {  "popsubfolder": "500kagents1mtourists", # empty takes root
            "timestepmins": 10,
            "loadagents": True,
            "loadhouseholds": True,
            "loadinstitutions": True,
            "loadworkplaces": True,
            "loadschools": True,
            "loadtourism": True,
            "religiouscells": True,
            "year": 2021,
            "quickdebug": True
         }

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

contactnetworkfile = open("./data/itinerary.json")
contactnetworkparams = json.load(contactnetworkfile)

sociability_rate_distribution = contactnetworkparams["sociabilityratedistribution"]
sociability_rate_options = np.arange(len(sociability_rate_distribution))
population_sub_folder = ""

if params["quickdebug"]:
    params["popsubfolder"] = "10kagents"

if len(params["popsubfolder"]) > 0:
    population_sub_folder = params["popsubfolder"] + "/"

# load agents and all relevant JSON files on each node
agents = {}
agents_ids_by_ages = {}
if params["loadagents"]:
    agentsfile = open("./population/" + population_sub_folder + "agents.json")
    agents = json.load(agentsfile)

    # if params["quickdebug"]:
    #     agents = {str(i):agents[str(i)] for i in range(1000)}

    temp_agents = {int(k): v for k, v in agents.items()}

    for agent_uid, agent in temp_agents.items():
        agent["curr_cellid"] = -1
        agent["res_cellid"] = -1
        agent["work_cellid"] = -1
        agent["school_cellid"] = -1
        agent["inst_cellid"] = -1

        age = agent["age"]
        agents_ids_by_ages[agent_uid] = agent["age"]

        agent["age_bracket_index"] = -1
        for i, ab in enumerate(age_brackets):
            if age >= ab[0] and age <= ab[1]:
                agent["age_bracket_index"] = i
                break
        
        agent["working_age_bracket_index"] = -1
        for i, ab in enumerate(age_brackets_workingages):
            if age >= ab[0] and age <= ab[1]:
                agent["working_age_bracket_index"] = i
                break

        agent["soc_rate"] = np.random.choice(sociability_rate_options, size=1, p=sociability_rate_distribution)[0]

    agents = temp_agents

    temp_agents = None

    n = len(agents)

    maleagents = {k:v for k, v in agents.items() if v["gender"] == 0}
    femaleagents = {k:v for k, v in agents.items() if v["gender"] == 1}

    agentsbyages = {}
    malesbyages = {}
    femalesbyages = {}

    for age in range(101):
        agentsbyage = {k:v for k, v in agents.items() if v["age"] == age}
        malesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 0}
        femalesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 1}

        agentsbyages[age] = agentsbyage
        malesbyages[age] = malesbyage
        femalesbyages[age] = femalesbyage

    agentsemployed = {k:v for k, v in agents.items() if v["empstatus"] == 0}
    malesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 0}
    femalesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 1}

    agentsbyindustries = {}
    malesbyindustries = {}
    femalesbyindustries = {}

    for industry in range(1,22):
        agentsbyindustry = {k:v for k, v in agentsemployed.items() if v["empind"] == industry}
        malesbyindustry = {k:v for k, v in malesemployed.items() if v["empind"] == industry}
        femalesbyindustry = {k:v for k, v in femalesemployed.items() if v["empind"] == industry}

        agentsbyindustries[industry] = agentsbyindustry
        malesbyindustries[industry] = malesbyindustry
        femalesbyindustries[industry] = femalesbyindustry

cell = Cells(agents, cells, cellindex)

if params["loadhouseholds"]:
    householdsfile = open("./population/" + population_sub_folder + "households.json")
    households_original = json.load(householdsfile)

    workplaces = []
    workplaces_cells_params = []

    if params["loadworkplaces"]:
        workplacesfile = open("./population/" + population_sub_folder + "workplaces.json")
        workplaces = json.load(workplacesfile)

        workplaces_cells_params = cellsparams["workplaces"]

    households, cells_households, householdsworkplaces, cells_householdsworkplaces = cell.convert_households(households_original, workplaces, workplaces_cells_params)

if params["loadinstitutions"]:
    institutiontypesfile = open("./population/" + population_sub_folder + "institutiontypes.json")
    institutiontypes_original = json.load(institutiontypesfile)

    institutionsfile = open("./population/" + population_sub_folder + "institutions.json")
    institutions = json.load(institutionsfile)

    institutions_cells_params = cellsparams["institutions"]

    institutiontypes, cells_institutions = cell.split_institutions_by_cellsize(institutions, institutions_cells_params[0], institutions_cells_params[1])    

hh_insts = []
if params["loadhouseholds"]:
    for hh in households.values():
        hh_inst = {"resident_uids": hh["resident_uids"]}
        hh_insts.append(hh_inst)

if params["loadinstitutions"]:
    for inst in institutions:
        hh_inst = {"resident_uids": inst["resident_uids"]}
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
    transport_cells_params = cellsparams["transport"]
    entertainment_acitvity_dist = cellsparams["entertainmentactivitydistribution"]

    transport, cells_transport = cell.create_transport_cells(transport_cells_params[2], transport_cells_params[0], transport_cells_params[1], transport_cells_params[3], transport_cells_params[4])

    accommodations = []
    roomsizes_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomsize: [member_uids]}}} - here member_uids represents room ids
    rooms_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomid: {"roomsize":1, "member_uids":[]}}}} - here member_uids represents tourist ids
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
    industries, cells_industries, cells_accommodations, rooms_by_accomid_by_accomtype, cells_hospital, cells_entertainment = cell.split_workplaces_by_cellsize(workplaces, roomsizes_by_accomid_by_accomtype, rooms_by_accomid_by_accomtype, workplaces_cells_params, hospital_cells_params, transport, entertainment_acitvity_dist)

if params["loadtourism"]:
    touristsfile = open("./population/" + population_sub_folder + "tourists.json")
    tourists = json.load(touristsfile)
    tourists = {tour["tourid"]:{"groupid":tour["groupid"], "subgroupid":tour["subgroupid"], "age":tour["age"], "gender": tour["gender"]} for tour in tourists}

    touristsgroupsfile = open("./population/" + population_sub_folder + "touristsgroups.json")
    touristsgroups = json.load(touristsgroupsfile)
    touristsgroups = {tg["groupid"]:{"subgroupsmemberids":tg["subgroupsmemberids"], "accominfo":tg["accominfo"], "reftourid":tg["reftourid"], "arr": tg["arr"], "dep": tg["dep"], "purpose": tg["purpose"], "accomtype": tg["accomtype"]} for tg in touristsgroups}

    touristsgroupsdaysfile = open("./population/" + population_sub_folder + "touristsgroupsdays.json")
    touristsgroupsdays = json.load(touristsgroupsdaysfile)
    touristsgroupsdays = {day["dayid"]:day["member_uids"] for day in touristsgroupsdays}

    airport_cells_params = cellsparams["airport"]

    cell.create_airport_cell()

    cellindex += 1

if params["loadschools"]:
    schoolsfile = open("./population/" + population_sub_folder + "schools.json")
    schools = json.load(schoolsfile)

    schools_cells_params = cellsparams["schools"]

    min_nts_size, max_nts_size, min_classroom_size, max_classroom_size = cell.get_min_max_school_sizes(schools)

    print("Min classroom size: " + str(min_classroom_size) + ", Max classroom size: " + str(max_classroom_size))
    print("Min non-teaching staff size: " + str(min_nts_size) + ", Max classroom size: " + str(max_nts_size))

    schooltypes, cells_schools, cells_classrooms = cell.split_schools_by_cellsize(schools, schools_cells_params[0], schools_cells_params[1])

if params["religiouscells"]:
    religious_cells_params = cellsparams["religious"]

    churches, cells_religious = cell.create_religious_cells(religious_cells_params[2], religious_cells_params[0], religious_cells_params[1], religious_cells_params[3], religious_cells_params[4])


# this might cause problems when referring to related agents, by household, workplaces etc
# if params["quickdebug"]:
#     agents = {i:agents[i] for i in range(10_000)}

itinerary_util = itinerary.Itinerary(itineraryparams, params["timestepmins"], cells, industries, workplaces, cells_schools, cells_hospital, cells_entertainment, cells_religious, cells_households)
sum_time_taken = 0
for day in range(1, 365+1):
    weekday, weekdaystr = util.day_of_year_to_day_of_week(day, params["year"])

    # workout itineraries
    # assign tourists (this is the first prototype which assumes tourists checkout/in at 00.00
    # but with itinerary we can generate random timestep at which tourists checkout/in
    if params["loadtourism"]:
        tourist_groupids_by_day = touristsgroupsdays[day]

        for tour_group_id in tourist_groupids_by_day:
            tourists_group = touristsgroups[tour_group_id]

            accomtype = tourists_group["accomtype"]
            accominfo = tourists_group["accominfo"]
            arrivalday = tourists_group["arr"]
            departureday = tourists_group["dep"]
            purpose = tourists_group["purpose"]
            subgroupsmemberids = tourists_group["subgroupsmemberids"]

            if arrivalday == day or departureday == day:
                for accinfoindex, accinfo in enumerate(accominfo):
                    accomid, roomid, roomsize = accinfo[0], accinfo[1], accinfo[2]

                    subgroupmmembers = subgroupsmemberids[accinfoindex]

                    cellindex = rooms_by_accomid_by_accomtype[accomtype][accomid][roomid]["cellindex"]

                    if arrivalday == day:
                        cells[cellindex]["place"]["member_uids"] = subgroupmmembers
                    else:
                        cells[cellindex]["place"]["member_uids"] = []

    # should be cell based, but for testing purposes, traversing all agents here

    if day == 1 or weekdaystr == "Monday":
        for agentid, agent in agents.items():
            print("day " + str(day) + ", agent id: " + str(agentid))
            itinerary_util.generate_working_days_for_week(agent)

    itinerary_util.cells_agents_timesteps = {}

    print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday))
    start = time.time()
    for hh_inst in hh_insts:
        itinerary_util.generate_itinerary_hh(day, weekday, agents, agents_ids_by_ages, hh_inst["resident_uids"])

    time_taken = time.time() - start
    sum_time_taken += time_taken
    avg_time_taken = sum_time_taken / day
    print("generate_itinerary_hh for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time.time() - start) + ", avg time taken: " + str(avg_time_taken))

    # for timestep in range(144): # 0 - 143
    #     print("timestep " + str(timestep))

print(len(agents))
