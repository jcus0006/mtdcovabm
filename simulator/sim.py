import sys
# sys.path.append('~/AppsPy/mtdcovabm/simulator')
# sys.path.insert(0, '~/AppsPy/mtdcovabm/simulator')
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # add root to sys.path

import shutil
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import traceback
from cells import Cells
import util, itinerary, epidemiology, itinerary_mp, itinerary_dist, itinerary_dask, contactnetwork_mp, tourism, vars, agentsutil, static, shared_mp
from utility.npencoder import NpEncoder
from dynamicparams import DynamicParams
import multiprocessing as mp
from dask.distributed import Client, Worker, LocalCluster, SSHCluster
# from dask.distributed import WorkerPlugin
import dask.dataframe as df
from functools import partial
import gc
from memory_profiler import profile

params = {  "popsubfolder": "1kagents2ktourists2019", # empty takes root (was 500kagents2mtourists2019 / 1kagents2ktourists2019)
            "timestepmins": 10,
            "simulationdays": 1, # 365
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
            "numprocesses": 4, # vm given 10 cores, limiting to X for now (represents processes or workers, depending on mp or dask)
            "numthreads": 1,
            "proc_usepool": 3, # Pool apply_async 0, Process 1, ProcessPoolExecutor = 2, Pool IMap 3, Dask MP Scheduler = 4
            "sync_usethreads": False, # Threads True, Processes False,
            "sync_usequeue": False,
            "use_mp": False, # if this is true, single node multiprocessing is used
            "use_mp_rawarray": False, # this is applicable for any case of mp (if not using mp, set to False)
            "dask_use_mp": False, # if this is true, dask is used, and multiprocessing is used in each node. if use_mp and dask_use_mp are False, dask workers are used for parallelisation each node
            "dask_use_fg": True,
            "keep_processes_open": True,
            "itinerary_normal_weight": 1,
            "itinerary_worker_student_weight": 1.12,
            "logsubfoldername": "logs",
            "logfilename": "output_dask_500k_distmp_4_celloptimisation.txt"
        }

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# class ReadOnlyData(WorkerPlugin):
#     def __init__(self, agentsfilepath):
#         with open(agentsfilepath, "r") as read_file:          
#             self.persons = json.load(read_file)
#             self.persons_len = len(self.persons)

def read_only_data(dask_worker: Worker, agents_ids_by_ages, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, use_shm):
    import os

    dask_worker.data["agents_ids_by_ages"] = agents_ids_by_ages
    dask_worker.data["timestepmins"] = timestepmins
    dask_worker.data["n_locals"] = n_locals
    dask_worker.data["n_tourists"] = n_tourists
    dask_worker.data["locals_ratio_to_full_pop"] = locals_ratio_to_full_pop

    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "itinerary.json"), "itineraryparams")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "epidemiology.json"), "epidemiologyparams")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_households_updated.json"), "cells_households")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_institutions_updated.json"), "cells_institutions")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_transport_updated.json"), "cells_transport")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_industries_by_indid_by_wpid_updated.json"), "cells_industries_by_indid_by_wpid")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_restaurants_updated.json"), "cells_restaurants")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_hospital_updated.json"), "cells_hospital")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_testinghub_updated.json"), "cells_testinghub")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_vaccinationhub_updated.json"), "cells_vaccinationhub")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_entertainment_by_activityid_updated.json"), "cells_entertainment_by_activityid")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_breakfast_by_accomid_updated.json"), "cells_breakfast_by_accomid")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_airport_updated.json"), "cells_airport")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_accommodation_updated.json"), "cells_accommodation")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_religious_updated.json"), "cells_religious")

    agentsupdatedfilepath = os.path.join(dask_worker.local_directory, "agents_updated.json")
    with open(agentsupdatedfilepath, "r") as read_file: 
        temp_agents = json.load(read_file)

        agents_static = static.Static()
        agents_static.populate(temp_agents, n_locals, n_tourists, is_shm=use_shm)
        
        dask_worker.data["agents_static"] = agents_static

def load_dask_worker_data(dask_worker, filepath, propname):
    with open(filepath, "r") as read_file: 
        temp = json.load(read_file, object_hook=jsonKeys2int)
        dask_worker.data[propname] = temp

def jsonKeys2int(x):
    new_dict = {}
    for k, v in x.items():
        try:
            new_key = int(k)
        except:
            new_key = k
        if type(v) == dict:
            v = jsonKeys2int(v)
        new_dict[new_key] = v
    return new_dict
# def initialize_mp(): # agents_static
#     manager = mp.Manager()
#     pool = mp.Pool()
#     # pool = mp.Pool(initializer=shared_mp.init_pool_processes, initargs=(agents_static,))
#     return manager, pool

# fp = open("memory_profiler.log", "w+")
# @profile(stream=fp)
def main():
    original_stdout = sys.stdout

    subfolder_name = params["logsubfoldername"]

    current_directory = os.getcwd()

    subfolder_name = params["logfilename"].replace(".txt", "")

    # Path to the subfolder
    subfolder_path = os.path.join(current_directory, params["logsubfoldername"], subfolder_name)

    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    else:
        shutil.rmtree(subfolder_path)
        os.makedirs(subfolder_path)

    log_file_name = os.path.join(subfolder_path, params["logfilename"])

    f = open(log_file_name, "w")

    sys.stdout = f

    print("interpreter: " + os.path.dirname(sys.executable))
    print("current working directory: " + os.getcwd())

    json_paths_to_upload = [] # to be uploaded to remote nodes
    
    figure_count = 0

    cellindex = 0
    cells = {}

    cellsfile = open(os.path.join(current_directory, "data", "cells.json"))
    cellsparams = json.load(cellsfile)

    itineraryjson = os.path.join(current_directory, "data", "itinerary.json")
    json_paths_to_upload.append(itineraryjson)
    itineraryfile = open(itineraryjson)
    itineraryparams = json.load(itineraryfile)

    sleeping_hours_by_age_groups = itineraryparams["sleeping_hours_by_age_groups"]
    non_daily_activities_employed_distribution = itineraryparams["non_daily_activities_employed_distribution"]
    age_brackets = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in sleeping_hours_by_age_groups] # [[0, 4], [5, 9], ...]
    age_brackets_workingages = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in non_daily_activities_employed_distribution] # [[15, 19], [20, 24], ...]

    contactnetworkfile = open(os.path.join(current_directory, "data", "contactnetwork.json"))
    contactnetworkparams = json.load(contactnetworkfile)

    sociability_rate_min_max = contactnetworkparams["sociabilityrateminmax"]
    sociability_rate_min, sociability_rate_max = sociability_rate_min_max[0], sociability_rate_min_max[1]
    powerlaw_distribution_parameters = contactnetworkparams["powerlawdistributionparameters"]
    # sociability_rate_options = np.arange(len(sociability_rate_distribution))

    epidemiologyjsonpath = os.path.join(current_directory, "data", "epidemiology.json")
    json_paths_to_upload.append(epidemiologyjsonpath)
    epidemiologyfile = open(epidemiologyjsonpath)
    epidemiologyparams = json.load(epidemiologyfile)

    initial_seir_state_distribution = epidemiologyparams["initialseirstatedistribution"]
    tourist_entry_infection_probability = epidemiologyparams["tourist_entry_infection_probability"]

    tourismfile = open(os.path.join(current_directory, "data", "tourism.json"))
    tourismparams = json.load(tourismfile)

    population_sub_folder = ""

    if params["quickdebug"]:
        params["popsubfolder"] = "10kagents"

    if len(params["popsubfolder"]) > 0:
        population_sub_folder = params["popsubfolder"]

    # load agents and all relevant JSON files on each node
    agents_dynamic = {}
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
        touristsfile = open(os.path.join(current_directory, "population", population_sub_folder, "tourists.json")) # 
        tourists = json.load(touristsfile)
        tourists = {tour["tourid"]:{"groupid":tour["groupid"], "subgroupid":tour["subgroupid"], "age":tour["age"], "gender": tour["gender"]} for tour in tourists}
        
        n_tourists = len(tourists)

        touristsgroupsfile = open(os.path.join(current_directory, "population", population_sub_folder, "touristsgroups.json"))
        touristsgroups = json.load(touristsgroupsfile)
        touristsgroups = {tg["groupid"]:{"subgroupsmemberids":tg["subgroupsmemberids"], "accominfo":tg["accominfo"], "reftourid":tg["reftourid"], "arr": tg["arr"], "dep": tg["dep"], "purpose": tg["purpose"], "accomtype": tg["accomtype"]} for tg in touristsgroups}

        touristsgroupsdaysfile = open(os.path.join(current_directory, "population", population_sub_folder, "touristsgroupsdays.json"))
        touristsgroupsdays = json.load(touristsgroupsdaysfile)
        touristsgroupsdays = {day["dayid"]:day["member_uids"] for day in touristsgroupsdays}

    if params["loadagents"]:
        agents = None
        agentsfilepath = os.path.join(current_directory, "population", population_sub_folder, "agents.json")
        with open(agentsfilepath, "r") as read_file:
            agents = json.load(read_file)

        # agentsfile = open(os.path.join(current_directory, "population", population_sub_folder, "agents.json"))
        # agents = json.load(agentsfile)

        n_locals = len(agents)

        agents, agents_seir_state, agents_vaccination_doses, locals_ratio_to_full_pop, figure_count = agentsutil.initialize_agents(agents, agents_ids_by_ages, agents_ids_by_agebrackets, tourists, params, itineraryparams, powerlaw_distribution_parameters, sociability_rate_min, sociability_rate_max, initial_seir_state_distribution, figure_count, n_locals, age_brackets, age_brackets_workingages)

    cells_util = Cells(agents, cells, cellindex)

    if params["loadhouseholds"]:
        householdsfile = open(os.path.join(current_directory, "population", population_sub_folder, "households.json"))
        households_original = json.load(householdsfile)

        workplaces = []
        workplaces_cells_params = []

        if params["loadworkplaces"]:
            workplacesfile = open(os.path.join(current_directory, "population", population_sub_folder, "workplaces.json"))
            workplaces = json.load(workplacesfile)

            workplaces_cells_params = cellsparams["workplaces"]

        households, cells_households, _, _ = cells_util.convert_households(households_original, workplaces, workplaces_cells_params)
        
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_households_updated.json", cells_households))

        # contactnetwork_util.epi_util.cells_households = cells_households
        
    if params["loadinstitutions"]:
        # institutiontypesfile = open(os.path.join(current_directory, "population", population_sub_folder, "institutiontypes.json"))
        # institutiontypes_original = json.load(institutiontypesfile)

        institutionsfile = open(os.path.join(current_directory, "population", population_sub_folder, "institutions.json"))
        institutions = json.load(institutionsfile)

        institutions_cells_params = cellsparams["institutions"]

        _, cells_institutions = cells_util.split_institutions_by_cellsize(institutions, institutions_cells_params[0], institutions_cells_params[1])  
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_institutions_updated.json", cells_institutions))
        # contactnetwork_util.epi_util.cells_institutions = cells_institutions  

    hh_insts = []
    if params["loadhouseholds"]:
        for hh in households.values():
            hh_inst = {"id": hh["hhid"], "is_hh": True, "resident_uids": hh["resident_uids"], "lb_weight": 0}
            hh_insts.append(hh_inst)

    if params["loadinstitutions"]:
        for inst in institutions:
            hh_inst = {"id": inst["instid"], "is_hh": False, "resident_uids": inst["resident_uids"], "lb_weight": 0}
            hh_insts.append(hh_inst)

    # hh_insts_partial = []
    # max_agents_count = 100000
    # curr_agents_count = 0

    if hh_insts is not None:
        for hh_inst in hh_insts:
            lb_weight = 0
            for agentid in hh_inst["resident_uids"]:
                agent = agents[agentid]         

                if agent["empstatus"] == 0 or agent["sc_student"] == 1:
                    lb_weight += params["itinerary_worker_student_weight"]
                else:
                    lb_weight += params["itinerary_normal_weight"]

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

            hh_inst["lb_weight"] = lb_weight

    if params["loadworkplaces"]:
        if len(workplaces) == 0:
            workplacesfile = open(os.path.join(current_directory, "population", population_sub_folder, "workplaces.json"))
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
        
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_transport_updated.json", cells_transport))

        # cells_accommodation_by_accomid = {} # {accomid: [{cellid: {cellinfo}}]}
        accommodations = []
        roomsizes_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomsize: [member_uids]}}} - here member_uids represents room ids
        rooms_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomid: {"roomsize":1, "member_uids":[]}}}} - here member_uids represents tourist ids
        tourists_active_groupids, tourists_active_ids = [], []
        accomgroups = None

        if params["loadtourism"]:
            accommodationsfile = open(os.path.join(current_directory, "population", population_sub_folder, "accommodations.json"))
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
        cells_industries_by_indid_by_wpid, _, cells_restaurants, cells_accommodation, _, cells_breakfast_by_accomid, rooms_by_accomid_by_accomtype, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_airport = cells_util.split_workplaces_by_cellsize(workplaces, roomsizes_by_accomid_by_accomtype, rooms_by_accomid_by_accomtype, workplaces_cells_params, hospital_cells_params, testing_hubs_cells_params, vaccinations_hubs_cells_params, airport_cells_params, accom_cells_params, transport, entertainment_acitvity_dist, itineraryparams)

        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_industries_by_indid_by_wpid_updated.json", cells_industries_by_indid_by_wpid))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_restaurants_updated.json", cells_restaurants))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_hospital_updated.json", cells_hospital))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_testinghub_updated.json", cells_testinghub))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_vaccinationhub_updated.json", cells_vaccinationhub))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_entertainment_by_activityid_updated.json", cells_entertainment_by_activityid))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_breakfast_by_accomid_updated.json", cells_breakfast_by_accomid))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_airport_updated.json", cells_airport))
        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_accommodation_updated.json", cells_accommodation))
        # contactnetwork_util.epi_util.cells_accommodation = cells_accommodation
        # airport_cells_params = cellsparams["airport"]

        # cell.create_airport_cell()

        # cellindex += 1

    if params["loadschools"]:
        schoolsfile = open(os.path.join(current_directory, "population", population_sub_folder, "schools.json"))
        schools = json.load(schoolsfile)

        schools_cells_params = cellsparams["schools"]

        min_nts_size, max_nts_size, min_classroom_size, max_classroom_size = cells_util.get_min_max_school_sizes(schools)

        print("Min classroom size: " + str(min_classroom_size) + ", Max classroom size: " + str(max_classroom_size))
        print("Min non-teaching staff size: " + str(min_nts_size) + ", Max classroom size: " + str(max_nts_size))

        cells_util.split_schools_by_cellsize(schools, schools_cells_params[0], schools_cells_params[1])

    if params["religiouscells"]:
        religious_cells_params = cellsparams["religious"]

        _, cells_religious = cells_util.create_religious_cells(religious_cells_params[2], religious_cells_params[0], religious_cells_params[1], religious_cells_params[3], religious_cells_params[4])

        json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "cells_religious_updated.json", cells_religious))

    # this might cause problems when referring to related agents, by household, workplaces etc
    # if params["quickdebug"]:
    #     agents = {i:agents[i] for i in range(10_000)}

    json_paths_to_upload.append(convert_to_json_file(current_directory, "population", population_sub_folder, "agents_updated.json", agents))

    agents_static = static.Static()
    agents_static.populate(agents, n_locals, n_tourists, is_shm=params["use_mp_rawarray"]) # for now trying without multiprocessing.RawArray

    agents_dynamic = agentsutil.initialize_agents_dict_dynamic(agents)

    del agents

    vars_util = vars.Vars()
    vars_util.populate(agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses)

    tourist_util = tourism.Tourism(tourismparams, cells, n_locals, tourists, agents_static, agents_dynamic, agents_seir_state, touristsgroupsdays, touristsgroups, rooms_by_accomid_by_accomtype, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids, tourists_active_ids, age_brackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count, initial_seir_state_distribution)
    dyn_params = DynamicParams(n_locals, n_tourists, epidemiologyparams)
    
    gc.collect()
    
    client = None
    try:
        if params["use_mp"]:
            manager = mp.Manager()
            pool = mp.Pool(initializer=shared_mp.init_pool_processes, initargs=(agents_static,))
        else:
            start = time.time()
            # cluster = LocalCluster()

            num_workers = 0
            if params["dask_use_mp"]:
                num_workers = 1
            else:
                num_workers = params["numprocesses"]

            cluster = SSHCluster(["localhost", "localhost"], # LAPTOP-FDQJ136P / localhost
                            connect_options={"known_hosts": None},
                            worker_options={"n_workers": num_workers, "local_directory": config["worker_working_directory"], }, # "memory_limit": "3GB" (in worker_options)
                            scheduler_options={"port": 0, "dashboard_address": ":8797", "local_directory": config["scheduler_working_directory"],},) # local_directory in scheduler_options has no effect
                            # worker_class="distributed.Worker", 
                            # remote_python="~/AppsPy/mtdcovabm/bin/python3.11")
            time_taken = time.time() - start
            print("cluster generation: " + str(time_taken))
            # cluster.scale(params["numprocesses"])

            # Set the working directory for the scheduler and workers
            # os.chdir(config["scheduler_working_directory"])  # Change the current working directory for the scheduler
            # cluster.worker_options['local_directory'] = config["worker_working_directory"]  # Worker working directory from config

            start = time.time()
            client = Client(cluster)
            time_taken = time.time() - start
            print("client generation: " + str(time_taken))

            # plugin = ReadOnlyData(agentsfilepath)
            # client.register_worker_plugin(plugin, name="read-only-data")

            # versions = client.get_versions(check=True)
            # print("versions: " + str(versions))

            start = time.time()
            client.upload_file('simulator/epidemiologyclasses.py')
            client.upload_file('simulator/shared_mp.py')
            client.upload_file('simulator/static.py')
            client.upload_file('simulator/seirstateutil.py')
            client.upload_file('simulator/util.py')
            client.upload_file('simulator/epidemiology.py')
            client.upload_file('simulator/dynamicparams.py')
            client.upload_file('simulator/seirstateutil.py')
            client.upload_file('simulator/vars.py')
            client.upload_file('simulator/itinerary.py')
            # client.upload_file('simulator/itinerary_dask.py')
            client.upload_file('simulator/itinerary_mp.py')
            client.upload_file('simulator/itinerary_dist.py')
            # client.upload_file(os.path.join("population", population_sub_folder, "agents.json"))

            for dynamicjsonpath in json_paths_to_upload:
                client.upload_file(dynamicjsonpath)
            
            callback = partial(read_only_data, 
                            agents_ids_by_ages=agents_ids_by_ages, 
                            timestepmins=params["timestepmins"], 
                            n_locals=n_locals, 
                            n_tourists=n_tourists, 
                            locals_ratio_to_full_pop=locals_ratio_to_full_pop,
                            use_shm=params["use_mp_rawarray"])
            
            client.register_worker_callbacks(callback)

            time_taken = time.time() - start
            print("upload modules remotely: " + str(time_taken))

        itinerary_sum_time_taken = 0
        tourist_itinerary_sum_time_taken = 0
        contactnetwork_sum_time_taken = 0
        
        for day in range(1, params["simulationdays"] + 1): # 365 + 1 / 1 + 1
            day_start = time.time()

            weekday, weekdaystr = util.day_of_year_to_day_of_week(day, params["year"])

            vars_util.contact_tracing_agent_ids = set()

            # itinerary_util.cells_agents_timesteps = {}
            # itinerary_util.epi_util = epi_util
            # contactnetwork_util.epi_util = epi_util
            # contactnetwork_util.cells_agents_timesteps = itinerary_util.cells_agents_timesteps
            # agents_seir_state_transition_for_day = {} # always cleared for a new day, will be filled in itinerary, and used in direct contact simulation (epi)
            # agents_directcontacts_by_simcelltype_by_day = {}

            if params["loadtourism"]:
                # single process tourism section
                itinerary_util = itinerary.Itinerary(itineraryparams, 
                                                    params["timestepmins"], 
                                                    n_locals, 
                                                    n_tourists,
                                                    locals_ratio_to_full_pop,
                                                    agents_static,
                                                    agents_dynamic,
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
                                                    dyn_params,
                                                    tourists,
                                                    tourist_entry_infection_probability)
                
                print("generate_tourist_itinerary for simday " + str(day) + ", weekday " + str(weekday))
                start = time.time()
                agents_dynamic, tourists, cells, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids = tourist_util.initialize_foreign_arrivals_departures_for_day(day)
                
                itinerary_util.generate_tourist_itinerary(day, weekday, touristsgroups, tourists_active_groupids, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday)
                
                time_taken = time.time() - start
                tourist_itinerary_sum_time_taken += time_taken
                avg_time_taken = tourist_itinerary_sum_time_taken / day
                print("generate_tourist_itinerary for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))

            # epi_util.tourists_active_ids = tourist_util.tourists_active_ids

            if not params["quickitineraryrun"]:
                if day == 1: # from day 2 onwards always calculated at eod
                    dyn_params.refresh_dynamic_parameters(day, agents_seir_state, tourists_active_ids) # TO REVIEW

            # partialising agents for multiprocessing
            start = time.time()
            it_agents = agentsutil.initialize_agents_dict_it(agents_dynamic)
            time_taken = time.time() - start
            print("initialize_agents_dict_it, time_taken: " + str(time_taken))

            start = time.time()
            cn_agents = agentsutil.initialize_agents_dict_cn(agents_dynamic)
            time_taken = time.time() - start
            print("initialize_agents_dict_cn, time_taken: " + str(time_taken))

            # start = time.time()
            # ct_agents = agentsutil.initialize_agents_dict_ct(agents_dynamic)
            # time_taken = time.time() - start
            # print("initialize_agents_dict_ct, time_taken: " + str(time_taken))

            if not params["quicktourismrun"]:
                start = time.time()  

                if params["use_mp"]:
                    itinerary_mp.localitinerary_parallel(manager,
                                                        pool,
                                                        day,
                                                        weekday,
                                                        weekdaystr,
                                                        itineraryparams,
                                                        params["timestepmins"],
                                                        n_locals,
                                                        n_tourists,
                                                        locals_ratio_to_full_pop,
                                                        it_agents,
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
                                                        dyn_params,
                                                        hh_insts, 
                                                        params["numprocesses"],
                                                        params["numthreads"],
                                                        params["proc_usepool"],
                                                        params["sync_usethreads"],
                                                        params["sync_usequeue"],
                                                        params["keep_processes_open"],
                                                        log_file_name)
                else:
                    if params["dask_use_fg"]:
                        itinerary_dist.localitinerary_finegrained_distributed(client,
                                                                              hh_insts,
                                                                              day,
                                                                              weekday,
                                                                              weekdaystr,
                                                                              agents_dynamic,
                                                                              vars_util,
                                                                              dyn_params,
                                                                              True,
                                                                              log_file_name)
                    else:
                        itinerary_dist.localitinerary_distributed(client,
                                                            day, 
                                                            weekday, 
                                                            weekdaystr, 
                                                            itineraryparams, 
                                                            params["timestepmins"], 
                                                            n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            it_agents,
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
                                                            dyn_params, 
                                                            hh_insts, 
                                                            params["numprocesses"], # to cleanup from here onwards, possibly to pass info about workers rather than these params
                                                            params["numthreads"],
                                                            params["proc_usepool"],                                                   
                                                            params["sync_usethreads"],
                                                            params["sync_usequeue"],
                                                            params["keep_processes_open"],
                                                            params["dask_use_mp"],
                                                            log_file_name)
                
                time_taken = time.time() - start
                itinerary_sum_time_taken += time_taken
                avg_time_taken = itinerary_sum_time_taken / day
                print("localitinerary_parallel for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken)) 

                if not params["quickitineraryrun"]:
                    print("simulate_contact_network for simday " + str(day) + ", weekday " + str(weekday))
                    start = time.time()       

                    contactnetwork_mp.contactnetwork_parallel(manager,
                                                            pool,
                                                            day, 
                                                            weekday, 
                                                            n_locals, 
                                                            n_tourists, 
                                                            locals_ratio_to_full_pop, 
                                                            cn_agents, 
                                                            vars_util,
                                                            cells, 
                                                            cells_households, 
                                                            cells_institutions,
                                                            cells_accommodation,
                                                            contactnetworkparams, 
                                                            epidemiologyparams, 
                                                            dyn_params, 
                                                            contactnetwork_sum_time_taken, 
                                                            params["numprocesses"],
                                                            params["numthreads"],
                                                            params["keep_processes_open"],
                                                            log_file_name)

                    time_taken = time.time() - start
                    contactnetwork_sum_time_taken += time_taken
                    avg_time_taken = contactnetwork_sum_time_taken / day
                    print("simulate_contact_network for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))                 

                # contact tracing
                print("contact_tracing for simday " + str(day) + ", weekday " + str(weekday))
                start = time.time()

                vars_util.dc_by_sct_by_day_agent1_index.sort(key=lambda x:x[0],reverse=False)
                vars_util.dc_by_sct_by_day_agent2_index.sort(key=lambda x:x[1],reverse=False)

                epi_util = epidemiology.Epidemiology(epidemiologyparams, 
                                                    n_locals, 
                                                    n_tourists, 
                                                    locals_ratio_to_full_pop,
                                                    agents_static,
                                                    agents_dynamic, 
                                                    vars_util, 
                                                    cells_households, 
                                                    cells_institutions, 
                                                    cells_accommodation, 
                                                    dyn_params)
                
                process_index, updated_agent_ids, agents_dynamic, vars_util = epi_util.contact_tracing(day)

                # contacttracing_mp.contacttracing_parallel(manager, 
                #                                         pool, 
                #                                         day, 
                #                                         epidemiologyparams, 
                #                                         n_locals, 
                #                                         n_tourists, 
                #                                         locals_ratio_to_full_pop, 
                #                                         ct_agents, 
                #                                         vars_util, 
                #                                         cells_households, 
                #                                         cells_institutions, 
                #                                         cells_accommodation, 
                #                                         dyn_params, 
                #                                         params["numprocesses"], 
                #                                         params["numthreads"], 
                #                                         params["keep_processes_open"], 
                #                                         log_file_name)

                time_taken = time.time() - start
                print("contact_tracing time taken: " + str(time_taken))

                # vaccinations
                print("schedule_vaccinations for simday " + str(day) + ", weekday " + str(weekday))
                start = time.time()

                # epi_util = epidemiology.Epidemiology(epidemiologyparams, n_locals, n_tourists, locals_ratio_to_full_pop, agents_static, agents_dynamic, vars_util, cells_households, cells_institutions, cells_accommodation, dyn_params)
                epi_util.schedule_vaccinations(day)
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
        with open(os.path.join(current_directory, params["logsubfoldername"], subfolder_name, "stack_trace.txt"), 'w') as f:
            traceback.print_exc(file=f)
    finally:
        # print(len(agents))
        sys.stdout = original_stdout
        f.close()

        if client is not None:
            client.shutdown()

def convert_to_json_file(current_directory, folder, subfolder, filename, array):
    temp_filepath = os.path.join(current_directory, folder, subfolder, filename)
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)

    with open(temp_filepath, "w", encoding="utf-8") as fp:
        json.dump(array, fp, ensure_ascii=False, indent=4, cls=NpEncoder)

    return temp_filepath

if __name__ == '__main__':
    main()
