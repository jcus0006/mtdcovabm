import json
import numpy as np
import math
from cells import Cells

# to establish set of static parameters such as cellsize, but grouped in dict

params = { "cellsize": 10,
           "cellsizespare": 0.1,
           "loadagents": True,
           "loadhouseholds": True,
           "loadinstitutions": True,
           "loadworkplaces": True,
           "loadschools": True
         }

cellindex = 0
cells = {}

# load agents and all relevant JSON files on each node
agents = {}
if params["loadagents"]:
    agentsfile = open("./population/agents.json")
    agents = json.load(agentsfile)

    temp_agents = {int(k): v for k, v in agents.items()}

    for agent_uid, agent in temp_agents.items():
        agent["curr_cellid"] = -1
        agent["res_cellid"] = -1
        agent["work_cellid"] = -1
        agent["school_cellid"] = -1
        agent["inst_cellid"] = -1

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
    householdsfile = open("./population/households.json")
    households_original = json.load(householdsfile)

    households, cells_households = cell.convert_households(households_original)

if params["loadinstitutions"]:
    institutiontypesfile = open("./population/institutiontypes.json")
    institutiontypes_original = json.load(institutiontypesfile)

    institutionsfile = open("./population/institutions.json")
    institutions = json.load(institutionsfile)

    institutiontypes, cells_institutions = cell.split_institutions_by_cellsize(institutions, params["cellsize"], params["cellsizespare"])

if params["loadworkplaces"]:
    workplacesfile = open("./population/workplaces.json")
    workplaces = json.load(workplacesfile)

    accommodationsfile = open("./population/accommodations.json")
    accommodations = json.load(accommodationsfile)

    accommodations_by_id_by_type = {} # to fix here (rooms from same accom id are replacing each other)
    for accom in accommodations:
        if accom["accomtypeid"] not in accommodations_by_id_by_type:
            accommodations_by_id_by_type[accom["accomtypeid"]] = {}

        accoms_by_type = accommodations_by_id_by_type[accom["accomtypeid"]]
        accoms_by_type[accom["accomid"]] = {"member_uids":accom["member_uids"], "accomtypeid": accom["accomtypeid"], "roomsize": accom["roomsize"]}

    # handle cell splitting (on workplaces & accommodations)
    industries, cells_industries, cells_accommodations = cell.split_workplaces_by_cellsize(workplaces, accommodations_by_id_by_type, params["cellsize"], params["cellsizespare"])

if params["loadschools"]:
    schoolsfile = open("./population/schools.json")
    schools = json.load(schoolsfile)

    min_nts_size, max_nts_size, min_classroom_size, max_classroom_size = cell.get_min_max_school_sizes(schools)

    print("Min classroom size: " + str(min_classroom_size) + ", Max classroom size: " + str(max_classroom_size))
    print("Min non-teaching staff size: " + str(min_nts_size) + ", Max classroom size: " + str(max_nts_size))

    schooltypes, cells_schools, cells_classrooms = cell.split_schools_by_cellsize(schools, params["cellsize"], params["cellsizespare"])

print(len(agents))
