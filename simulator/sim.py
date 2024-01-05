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
import util, itinerary, epidemiology, itinerary_mp, itinerary_dist, contactnetwork_mp, contactnetwork_dist, contacttracing_dist, tourism, vars, agentsutil, static, shared_mp, jsonutil, customdict
from actor_dist_mp import ActorDistMP
from actor_dist import ActorDist
from dynamicparams import DynamicParams
import multiprocessing as mp
from dask.distributed import Client, Worker, SSHCluster, performance_report, as_completed
# from dask.distributed import WorkerPlugin
from functools import partial
import gc
from memory_profiler import profile
# import dask.dataframe as df
import pandas as pd
from pympler import asizeof
from copy import copy, deepcopy
import psutil

params = {  "popsubfolder": "10kagents40ktourists2019_decupd_v4", # empty takes root (was 500kagents2mtourists2019_decupd_v4 / 100kagents400ktourists2019_decupd_v4 / 10kagents40ktourists2019_decupd_v4 / 1kagents2ktourists2019_decupd_v4)
            "timestepmins": 10,
            "simulationdays": 24, # 365/20
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
            "fullpop": 519562, # 519562 / 100000 / 10000 / 1000
            "fulltourpop": 2173531, # 2173531 / 400000 / 40000 / 4000
            "numprocesses": 8, # only used for multiprocessing, refer to dask_nodes and dask_nodes_n_workers for Dask Distributed processing
            "numthreads": -1,
            "proc_usepool": 3, # Pool apply_async 0, Process 1, ProcessPoolExecutor = 2, Pool IMap 3, Dask MP Scheduler = 4
            "sync_usethreads": False, # Threads True, Processes False,
            "sync_usequeue": False,
            "use_mp": False, # if this is true, single node multiprocessing is used, if False, Dask is used (use_shm must be True - currently)
            "use_shm": False, # use_mp_rawarray: this is applicable for any case of mp (if not using mp, it is set to False by default)
            "dask_use_mp": False, # when True, dask is used with multiprocessing in each node. if use_mp and dask_use_mp are False, dask workers are used for parallelisation each node
            "dask_full_stateful": True,
            "dask_actors_innerproc_assignment": False, # when True, assigns work based on the inner-processes within the Dask worker, when set to False, assigns work based on the number of nodes. this only works when dask_usemp = True
            "use_static_dict_tourists": True, # force this!
            "use_static_dict_locals": False,
            "dask_mode": 0, # 0 client.submit, 1 dask.delayed (client.compute) 2 dask.delayed (dask.compute - local) 3 client.map
            "dask_use_fg": False, # will use fine grained method that tackles single item (with single residence, batch sizes, and recurring)
            "dask_numtasks": -1, # only works with dask_use_fg = False and dask_use_chunking = False. the number of tasks to split all computation into
            "dask_use_chunking": False, # only works with dask_use_fg = False, refer to dask_chunk_size below for more info
            "dask_chunk_size": 10240, # defines the chunk size to send to the scheduler at one go (to avoid overload), set as 1 to send everything in 1 go
            "dask_single_item_per_task": False, # if set as True, will send "dask_chunk_size" tasks at one go, if set as False, will send "dask_chunk_size" items in each task and num_tasks / dask_chunk_size tasks
            "dask_map_batching": False, # if set as True, will use client.map with batch_size. set dask_batch_size, requires dask_use_fg = False, does not consider dask_mode
            "dask_map_batched_results": True, # if set as True, will get batched results (works with dask_map_matching only)
            "dask_full_array_mapping": False, # if True, agents_seir_state, mapped to partial dicts (this has been optimized, as was previously using index method with o(n) time complexity)
            "dask_scatter": False, # scattering of data into distributed memory (works with dask_mode 0, not sure with rest)
            "dask_submit": False, # submit data into distributed memory, potential alternative to scatter (set dask_scatter = False)
            "dask_collections": False, # NOT USED: dask.bag and dask.array, where applicable.
            "dask_partition_size": 128, # NOT USED
            "dask_persist": False, # NOT USED: persist data (with dask collections and delayed library)
            "dask_scheduler_node": "localhost",
            "dask_scheduler_host": "localhost", # try to force dask to start the scheduler on this IP
            "dask_nodes": ["localhost"], # 192.168.1.24
            "dask_nodes_n_workers": [4], # 3, 11
            # "dask_scheduler_node": "localhost",
            # "dask_scheduler_host": "192.168.1.17", # try to force dask to start the scheduler on this IP
            # "dask_nodes": ["localhost", "192.168.1.18", "192.168.1.19", "192.168.1.21", "192.168.1.23"], # (to be called with numprocesses = 1) [scheduler, worker1, worker2, ...] 192.168.1.18 
            # "dask_nodes_n_workers": [3, 4, 4, 6, 3], # num of workers on each node - 4, 4, 4, 4, 4, 3
            "dask_nodes_cpu_scores": None, # [13803, 7681, 6137, 3649, 6153, 2503] if specified, static load balancing is applied based on these values 
            "dask_dynamic_load_balancing": False,
            # "dask_nodes_time_taken": [0.13, 0.24, 0.15, 0.13, 0.15, 0.21], # [0.13, 0.24, 0.15, 0.21, 0.13, 0.15] - refined / [0.17, 0.22, 0.15, 0.20, 0.12, 0.14] - varied - used on day 1 and adapted dynamically. If specified, and dask_nodes_cpu_scores is None, will be used as inverted weights for load balancing
            "dask_batch_size": 86, # (last tried with 2 workers and batches of 2, still unbalanced) 113797 residences means that many calls. recursion limit is 999 calls. 113797 / 999 = 113.9, use batches of 120+ to be safe
            "dask_batch_recurring": False, # if recurring send batch size recurringly, else, send batch size the first time, then 1 task per "as_completed" future 
            "dask_cluster_restart_days": -1, # supposedly would restart cluster every X days; but was crashing. need further investigation
            "dask_autoclose_cluster": True,
            "keep_processes_open": True,
            "itinerary_normal_weight": 1,
            "itinerary_worker_student_weight": 1.12,
            "contacttracing_distributed": False,
            "logsubfoldername": "logs",
            "datasubfoldername": "data",
            "remotelogsubfoldername": "AppsPy/mtdcovabm/logs",
            "logmemoryinfo": True,
            "logfilename": "daskstrat3.txt" # dask_5n_20w_500k_3d_opt.txt
        }

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# class ReadOnlyData(WorkerPlugin)
#     def __init__(self, agentsfilepath):
#         with open(agentsfilepath, "r") as read_file:          
#             self.persons = json.load(read_file)
#             self.persons_len = len(self.persons)

def read_only_data(dask_worker: Worker, dask_strategy, agents_ids_by_ages, timestepmins, n_locals, n_tourists, locals_ratio_to_full_pop, use_shm, use_static_dict_locals, use_static_dict_tourists, logsubfoldername, logfilename):
    import os
    import platform

    dask_use_mp = False
    dask_actors = False

    if dask_strategy == 1 or dask_strategy == 2:
        dask_actors = True

        if dask_strategy == 1:
            dask_use_mp = True

    if dask_actors and platform.system() != "Linux": # mac osx uses spawn in recent Python versions; was causing "cannot pickle '_thread.lock' object" error
        mp.set_start_method("fork")

    current_directory = os.getcwd()
    subfolder_name = logfilename.replace(".txt", "")
    log_subfolder_path = os.path.join(current_directory, logsubfoldername, subfolder_name)

    # Create the subfolder if it doesn't exist
    try:
        if not os.path.exists(log_subfolder_path):
            os.makedirs(log_subfolder_path)
    except:
        pass # might fail if already created from another worker or process within same node, but should not matter

    dask_worker.data["agents_ids_by_ages"] = agents_ids_by_ages
    dask_worker.data["timestepmins"] = timestepmins
    dask_worker.data["n_locals"] = n_locals
    dask_worker.data["n_tourists"] = n_tourists
    dask_worker.data["locals_ratio_to_full_pop"] = locals_ratio_to_full_pop

    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "itinerary.json"), "itineraryparams")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "contactnetwork.json"), "contactnetworkparams")
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
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "cells_type_updated.json"), "cells_type")
    load_dask_worker_data(dask_worker, os.path.join(dask_worker.local_directory, "indids_by_cellid_updated.json"), "indids_by_cellid")

    agentsupdatedfilepath = os.path.join(dask_worker.local_directory, "agents_updated.json")
    with open(agentsupdatedfilepath, "r") as read_file: 
        temp_agents = json.load(read_file)

        agents_static = static.Static()
        agents_static.populate(temp_agents, n_locals, n_tourists, use_shm, use_static_dict_locals, use_static_dict_tourists, True, dask_use_mp)
        
        dask_worker.data["agents_static"] = agents_static

def load_dask_worker_data(dask_worker, filepath, propname):
    with open(filepath, "r") as read_file: 
        temp = json.load(read_file, object_hook=jsonutil.jsonKeys2int)
        dask_worker.data[propname] = temp

    # worker.data["new_tourists"] = new_tourists

# def initialize_mp(): # agents_static
#     manager = mp.Manager()
#     pool = mp.Pool()
#     # pool = mp.Pool(initializer=shared_mp.init_pool_processes, initargs=(agents_static,))
#     return manager, pool

# fp = open("memory_profiler_memissues_10k_3d.log", "w+")
# @profile(stream=fp)
def main():
    global params
    
    if not params["use_mp"] and not params["dask_use_mp"] and len(params["dask_nodes"]) == 0 and params["numprocesses"] == 1:
        params["use_mp"] = True # single process is handled in itinerary_mp and contactnetwork_mp, but does not actually use multiprocessing

    if params["use_mp"] and params["numprocesses"] == 1: # don't use shm in single process cases
       params["use_shm"] = False

    if not params["use_mp"] and not params["dask_use_mp"]: # don't use shm if not using multiprocessing
        params["use_shm"] = False

    dask_strategy = None
    if not params["use_mp"]:
        if not params["dask_use_mp"] and not params["dask_full_stateful"]:
            dask_strategy = 0
        elif params["dask_use_mp"]:
            dask_strategy = 1
        elif params["dask_full_stateful"]:
            dask_strategy = 2
            
    data_load_start_time = time.time()

    simdays_range = range(1, params["simulationdays"] + 1)
    perf_timings_df = pd.DataFrame(index=simdays_range, columns=["tourismitinerary_day", 
                                                              "localitinerary_day", 
                                                              "contactnetwork_day", 
                                                              "contacttracing_day", 
                                                              "vaccination_day", 
                                                              "refreshdynamicparams_day",
                                                              "tourismitinerary_avg", 
                                                              "localitinerary_avg", 
                                                              "contactnetwork_avg", 
                                                              "contacttracing_avg", 
                                                              "vaccination_avg", 
                                                              "refreshdynamicparams_avg"])
    
    mem_logs_df = pd.DataFrame(index=simdays_range, columns=["general_day",
                                                            "it_agents_day",
                                                            "working_schedule_day",
                                                            "itinerary_day",
                                                            "itinerary_nextday_day",
                                                            "non_daily_activity_recurring_day",
                                                            "prevday_non_daily_activity_recurring_day",
                                                            "tourist_ids_day",
                                                            "initial_tourist_day",
                                                            "epi_agents_day",
                                                            "state_transition_by_day_day",
                                                            "test_day_day",
                                                            "test_result_day_day",
                                                            "hospitalisation_days_day",
                                                            "quarantine_days_day",
                                                            "vaccination_days_day",
                                                            "vars_util_day",
                                                            "cells_agents_timesteps_day",
                                                            "seir_state_day",
                                                            "seir_state_trans_for_day_day",
                                                            "infection_type_day",
                                                            "infection_severity_day",
                                                            "vacc_doses_day",
                                                            "direct_contacts_day",
                                                            "direct_contacts_index1_day",
                                                            "direct_contacts_index2_day",
                                                            "general_avg",
                                                            "it_agents_avg",
                                                            "working_schedule_avg",
                                                            "itinerary_avg",
                                                            "itinerary_nextday_avg",
                                                            "non_daily_activity_recurring_avg",
                                                            "prevday_non_daily_activity_recurring_avg",
                                                            "tourist_ids_avg",
                                                            "iniitial_tourist_avg",
                                                            "epi_agents_avg",
                                                            "state_transition_by_day_avg",
                                                            "test_day_avg",
                                                            "test_result_day_avg",
                                                            "hospitalisation_days_avg",
                                                            "quarantine_days_avg",
                                                            "vaccination_days_avg",
                                                            "vars_util_avg",
                                                            "cells_agents_timesteps_avg",
                                                            "seir_state_avg",
                                                            "seir_state_trans_for_day_day", 
                                                            "infection_type_avg",
                                                            "infection_severity_avg",
                                                            "vacc_doses_avg",
                                                            "direct_contacts_avg",
                                                            "direct_contacts_index1_avg",
                                                            "direct_contacts_index2_avg"])
    
    interventions_logs_df = pd.DataFrame(index=simdays_range, columns=["quarantine_enabled",
                                                                    "testing_enabled",
                                                                    "contact_tracing_enabled",
                                                                    "workplaces_lockdown",
                                                                    "schools_lockdown",
                                                                    "entertainment_lockdown",
                                                                    "masks_hygiene_distancing_multiplier",
                                                                    "vaccination_propensity",
                                                                    "last_vaccination_propensity"])
    
    statistics_logs_df = pd.DataFrame(index=simdays_range, columns=["total_active_population",
                                                                    "total_locals",
                                                                    "total_active_tourists",
                                                                    "total_arriving_tourists",
                                                                    "total_arriving_nextday_tourists",
                                                                    "total_departing_tourists",
                                                                    "total_exposed",
                                                                    "total_susceptible",
                                                                    "total_infectious",
                                                                    "total_recovered",
                                                                    "total_deceased",
                                                                    "new_exposed",
                                                                    "new_susceptible",
                                                                    "new_infectious",
                                                                    "new_recovered",
                                                                    "new_deaths",
                                                                    "infectious_rate",
                                                                    "recovery_rate",
                                                                    "mortality_rate",
                                                                    "basic_reproduction_number",
                                                                    "effective_reproduction_number",
                                                                    "total_vaccinations",
                                                                    "total_tests",
                                                                    "total_contacttraced",
                                                                    "total_quarantined",
                                                                    "total_hospitalized",
                                                                    "total_to_be_vaccinated",
                                                                    "new_vaccinations",
                                                                    "new_tests",
                                                                    "new_contacttraced",
                                                                    "new_quarantined",
                                                                    "new_hospitalized",
                                                                    "average_contacts_per_person"])
    subfolder_name = params["logsubfoldername"]

    current_directory = os.getcwd()

    subfolder_name = params["logfilename"].replace(".txt", "")

    # Path to the subfolder
    log_subfolder_path = os.path.join(current_directory, params["logsubfoldername"], subfolder_name)

    # Create the subfolder if it doesn't exist
    if not os.path.exists(log_subfolder_path):
        os.makedirs(log_subfolder_path)
    else:
        shutil.rmtree(log_subfolder_path)
        os.makedirs(log_subfolder_path)

    data_subfolder_path = os.path.join(current_directory, params["datasubfoldername"], subfolder_name)

    if not os.path.exists(data_subfolder_path):
        os.makedirs(data_subfolder_path)
    else:
        shutil.rmtree(data_subfolder_path)
        os.makedirs(data_subfolder_path)

    log_file_name = os.path.join(log_subfolder_path, params["logfilename"])
    perf_timings_file_name = os.path.join(log_subfolder_path, params["logfilename"].replace(".txt", "_perf_timings.csv"))
    mem_logs_file_name = os.path.join(log_subfolder_path, params["logfilename"].replace(".txt", "_mem_logs.csv"))
    interventions_logs_file_name = os.path.join(log_subfolder_path, params["logfilename"].replace(".txt", "_interventions_logs.csv"))
    statistics_logs_file_name = os.path.join(log_subfolder_path, params["logfilename"].replace(".txt", "_statistics_logs.csv"))
    
    f = open(log_file_name, "w")
    original_stdout = sys.stdout
    sys.stdout = f

    util.log_memory_usage(f, "Created data frames. Loading data.. ")

    print("interpreter: " + os.path.dirname(sys.executable))
    print("current working directory: " + os.getcwd())
    if f is not None:
        f.flush()

    manager, pool, agents_static, static_agents_dict = None, None, None, None # mp         
    client = None # dask
    # if params["use_mp"] and params["numprocesses"] > 1:
    #     agents_static_start = time.time()
    #     agents = {i:None for i in range(params["fullpop"])}
    #     agents_static = static.Static()
    #     agents_static.populate(agents, params["fullpop"], params["fulltourpop"], params["use_shm"], params["use_static_dict_locals"], params["use_static_dict_tourists"]) # for now trying without multiprocessing.RawArray
    #     agents_static_time_taken = time.time() - agents_static_start

    #     if params["use_shm"]:
    #         print(f"initializing empty shm: {agents_static_time_taken}")
    #     else:
    #         print(f"initializing empty np.array: {agents_static_time_taken}")

    #     manager = mp.Manager()
    #     static_agents_dict = manager.dict()
    #     pool = mp.Pool(processes=params["numprocesses"], initializer=shared_mp.init_pool_processes, initargs=(agents_static,))
    #     agents_static.static_agents_dict = static_agents_dict

    json_paths_to_upload = [] # to be uploaded to remote nodes
    
    figure_count = 0

    cellindex = 0
    cells = {}

    cellsfile = open(os.path.join(current_directory, params["datasubfoldername"], "cells.json"))
    cellsparams = json.load(cellsfile)
    cellsfile.close()
    cellsparams_size = util.asizeof_formatted(cellsparams) # mem: cellsparams_size
    print(f"Cells params size {cellsparams_size}")

    itineraryjson = os.path.join(current_directory, params["datasubfoldername"], "itinerary.json")
    if not params["use_mp"]:
        json_paths_to_upload.append(itineraryjson)
    itineraryfile = open(itineraryjson)
    itineraryparams = json.load(itineraryfile)
    itineraryparams_size = util.asizeof_formatted(itineraryparams) # mem: itineraryparams_size
    print(f"itinerary params size {itineraryparams_size}")

    sleeping_hours_by_age_groups = itineraryparams["sleeping_hours_by_age_groups"]
    non_daily_activities_employed_distribution = itineraryparams["non_daily_activities_employed_distribution"]
    age_brackets = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in sleeping_hours_by_age_groups] # [[0, 4], [5, 9], ...]
    age_brackets_workingages = [[age_group_dist[0], age_group_dist[1]] for age_group_dist in non_daily_activities_employed_distribution] # [[15, 19], [20, 24], ...]

    contactnetworkjsonpath = os.path.join(current_directory, params["datasubfoldername"], "contactnetwork.json")
    if not params["use_mp"]:
        json_paths_to_upload.append(contactnetworkjsonpath)
    contactnetworkfile = open(contactnetworkjsonpath)
    contactnetworkparams = json.load(contactnetworkfile)

    contactnetworkparams_size = util.asizeof_formatted(contactnetworkparams) # mem: contactnetworkparams_size
    print(f"Contact Network params size {contactnetworkparams_size}")

    sociability_rate_min_max = contactnetworkparams["sociabilityrateminmax"]
    sociability_rate_min, sociability_rate_max = sociability_rate_min_max[0], sociability_rate_min_max[1]
    powerlaw_distribution_parameters = contactnetworkparams["powerlawdistributionparameters"]
    # sociability_rate_options = np.arange(len(sociability_rate_distribution))

    epidemiologyjsonpath = os.path.join(current_directory, params["datasubfoldername"], "epidemiology.json")
    if not params["use_mp"]:
        json_paths_to_upload.append(epidemiologyjsonpath)
    epidemiologyfile = open(epidemiologyjsonpath)
    epidemiologyparams = json.load(epidemiologyfile)

    epidemiologyparams_size = util.asizeof_formatted(epidemiologyparams) # mem: epidemiologyparams_size
    print(f"Epidemiology params size {epidemiologyparams_size}")

    initial_seir_state_distribution = epidemiologyparams["initialseirstatedistribution"]

    tourismfile = open(os.path.join(current_directory, params["datasubfoldername"], "tourism.json"))
    tourismparams = json.load(tourismfile)
    tourismfile.close()

    tourismparams_size = util.asizeof_formatted(tourismparams) # mem: tourismparams_size
    print(f"Tourism params size {tourismparams_size}")

    data_load_time_taken = time.time() - data_load_start_time

    print("loading params, time taken: " + str(data_load_time_taken))
    if f is not None:
        f.flush()

    cells_generation_time_taken = 0

    population_sub_folder = ""

    if params["quickdebug"]:
        params["popsubfolder"] = "10kagents"

    if len(params["popsubfolder"]) > 0:
        population_sub_folder = params["popsubfolder"]

    # load agents and all relevant JSON files on each node
    agents_dynamic = customdict.CustomDict()
    agents_ids_by_ages = {}
    agents_ids_by_agebrackets = {i:[] for i in range(len(age_brackets))}

    # # contact network model
    # cells_agents_timesteps = {} # {cellid: (agentid, starttimestep, endtimestep)}

    # # transmission model
    agents_seir_state = customdict.CustomDict() # whole population with following states, 0: undefined, 1: susceptible, 2: exposed, 3: infectious, 4: recovered, 5: deceased
    agents_seir_state_transition_for_day = customdict.CustomDict() # handled as dict, because it will only apply for a subset of agents per day
    agents_infection_type = customdict.CustomDict() # handled as dict, because not every agent will be infected
    agents_infection_severity = customdict.CustomDict() # handled as dict, because not every agent will be infected
    agents_vaccination_doses = customdict.CustomDict() # number of doses per agent

    # tourism
    tourists = {}
    tourists_arrivals_departures_for_day = {} # handles both incoming and outgoing, arrivals and departures. handled as a dict, as only represents day
    tourists_arrivals_departures_for_nextday = {}
    n_tourists = 0
    n_tourists_initial = 0

    if params["loadtourism"]:
        tourists_start = time.time()

        touristsfile = open(os.path.join(current_directory, "population", population_sub_folder, "tourists.json")) # 
        tourists = json.load(touristsfile)
        touristsfile.close()
        tourists = {tour["tourid"]:{"groupid":tour["groupid"], "subgroupid":tour["subgroupid"], "age":tour["age"], "gender": tour["gender"]} for tour in tourists}
        
        tourists_size = util.asizeof_formatted(tourists) # mem: tourists_size
        print(f"Tourists size {tourists_size}")

        n_tourists = len(tourists)

        touristsgroupsfile = open(os.path.join(current_directory, "population", population_sub_folder, "touristsgroups.json"))
        touristsgroups = json.load(touristsgroupsfile)
        touristsgroupsfile.close()
        touristsgroupsids_initial = [tg["groupid"] for tg in touristsgroups if tg["arr"] <= 0 and tg["dep"] > 0]
        touristsgroups = {tg["groupid"]:{"subgroupsmemberids":tg["subgroupsmemberids"], "accominfo":tg["accominfo"], "reftourid": tg["reftourid"], "arr": tg["arr"], "dep": tg["dep"], "purpose": tg["purpose"], "accomtype": tg["accomtype"]} for tg in touristsgroups if tg["dep"] > 0}
        touristsids_initial = [id for grpid in touristsgroupsids_initial for subgrp in touristsgroups[grpid]["subgroupsmemberids"] for id in subgrp]
        
        touristsgroups_size = util.asizeof_formatted(touristsgroups) # mem: touristsgroups_size
        touristsgroupsids_initial_size = util.asizeof_formatted(touristsgroupsids_initial) # mem: touristsgroupsids_initial_size
        touristsids_initial_size = util.asizeof_formatted(touristsids_initial) # mem: touristsids_initial_size

        n_tourists_initial = len(touristsids_initial)

        touristsgroupsdaysfile = open(os.path.join(current_directory, "population", population_sub_folder, "touristsgroupsdays.json"))
        touristsgroupsdays = json.load(touristsgroupsdaysfile)
        touristsgroupsdaysfile.close()
        touristsgroupsdays = {day["dayid"]:day["member_uids"] for day in touristsgroupsdays}

        touristsgroupsbydays_size = util.asizeof_formatted(touristsgroupsdays) # mem: touristsgroupsbydays_size

        print(f"Tourists size {tourists_size}, Tourists groups size {touristsgroups_size}, Tourists groups ids initial size {touristsgroupsids_initial_size}, Tourists ids initial size {touristsids_initial_size}, Tourists groups by days size {touristsgroupsbydays_size}")

        tourists_time_taken = time.time() - tourists_start
        print("loading tourists, time_taken: " + str(tourists_time_taken))
        if f is not None:
            f.flush()

        data_load_time_taken += tourists_time_taken

    if params["loadagents"]:
        agents_start = time.time()
        agents = None
        agentsfilepath = os.path.join(current_directory, "population", population_sub_folder, "agents.json")
        with open(agentsfilepath, "r") as read_file:
            agents = json.load(read_file)

        agents_size = util.asizeof_formatted(agents) # mem: agents_size
        print(f"Agents json size: {agents_size}")

        agents_time_taken = time.time() - agents_start
        print("loading tourists, time_taken: " + str(agents_time_taken))
        if f is not None:
            f.flush()
        data_load_time_taken += agents_time_taken
        # agentsfile = open(os.path.join(current_directory, "population", population_sub_folder, "agents.json"))
        # agents = json.load(agentsfile)

        n_locals = len(agents)

        agents_initialize_start = time.time()
        agents, agents_seir_state, agents_vaccination_doses, locals_ratio_to_full_pop, figure_count = agentsutil.initialize_agents(agents, agents_ids_by_ages, agents_ids_by_agebrackets, tourists, params, itineraryparams, powerlaw_distribution_parameters, sociability_rate_min, sociability_rate_max, initial_seir_state_distribution, figure_count, n_locals, age_brackets, age_brackets_workingages)
        
        agents_size = util.asizeof_formatted(agents) # mem: agents_size
        agents_seir_state_size = util.asizeof_formatted(agents_seir_state) # mem: agents_seir_state_size
        agents_vaccination_doses_size = util.asizeof_formatted(agents_vaccination_doses) # mem: agents_vaccination_doses_size

        print(f"Agents (updated) json size: {agents_size}, Agents seir state size: {agents_seir_state_size}, Agents vacc doses size: {agents_vaccination_doses_size}")

        agents_initialize_time_taken = time.time() - agents_initialize_start
        print("agents dict initialize, time taken: " + str(agents_initialize_time_taken))
        if f is not None:
            f.flush()

    cells_util = Cells(agents, cells, cellindex)
    
    num_households, num_institutions = 0, 0
    if params["loadhouseholds"]:
        hh_start = time.time()
        householdsfile = open(os.path.join(current_directory, "population", population_sub_folder, "households.json"))
        households_original = json.load(householdsfile)
        householdsfile.close()
        households_original_size = util.asizeof_formatted(households_original) # mem: households_original_size
        print(f"households original size: {households_original_size}")
        hh_time_taken = time.time() - hh_start
        data_load_time_taken += hh_time_taken

        workplaces = []
        workplaces_cells_params = []

        if params["loadworkplaces"]:
            wp_start = time.time()
            workplacesfile = open(os.path.join(current_directory, "population", population_sub_folder, "workplaces.json"))
            workplaces = json.load(workplacesfile)
            workplacesfile.close()
            workplaces_size = util.asizeof_formatted(workplaces) # mem: workplaces_size
            print(f"workplaces size: {workplaces_size}")
            wp_time_taken = time.time() - wp_start 
            data_load_time_taken += wp_time_taken

            workplaces_cells_params = cellsparams["workplaces"]

        hh_cells_start = time.time()
        households, _, cells_households, _ = cells_util.convert_households(households_original, workplaces, workplaces_cells_params)
        num_households = len(households)
        del households_original
        households_size = util.asizeof_formatted(households) # mem: households_size
        cells_households_size = util.asizeof_formatted(cells_households) # mem: cells_households_size
        print(f"households size: {households_size}, cells_households_size: {cells_households_size}")
        hh_cells_time_taken = hh_cells_start - time.time()
        cells_generation_time_taken += hh_cells_time_taken

        if not params["use_mp"]:
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_households_updated.json", cells_households))

        # contactnetwork_util.epi_util.cells_households = cells_households
        
    if params["loadinstitutions"]:
        # institutiontypesfile = open(os.path.join(current_directory, "population", population_sub_folder, "institutiontypes.json"))
        # institutiontypes_original = json.load(institutiontypesfile)
        inst_start = time.time()
        institutionsfile = open(os.path.join(current_directory, "population", population_sub_folder, "institutions.json"))
        institutions = json.load(institutionsfile)
        institutionsfile.close()

        num_institutions = len(institutions)
        inst_time_taken = time.time() - inst_start 
        data_load_time_taken += inst_time_taken

        institutions_cells_params = cellsparams["institutions"]
        
        inst_cells_start = time.time()
        _, inst_ids_by_cellid, cells_institutions, cell_ids_by_inst_id = cells_util.split_institutions_by_cellsize(institutions, institutions_cells_params[0], institutions_cells_params[1])  
        institutions_size = util.asizeof_formatted(institutions) # mem: institutions_size
        cells_institutions_size = util.asizeof_formatted(cells_institutions) # mem: cells_institutions_size
        print(f"institutions size: {institutions_size}, cells_institutions_size: {cells_institutions_size}")
        inst_cells_time_taken = time.time() - inst_cells_start
        cells_generation_time_taken += inst_cells_time_taken

        if not params["use_mp"]:
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_institutions_updated.json", cells_institutions))
        # contactnetwork_util.epi_util.cells_institutions = cells_institutions  

    guardian_assignment_start = time.time()

    hh_insts = []
    if params["loadhouseholds"]:
        for hh in households.values():
            hh_inst = {"id": hh["hhid"], "is_hh": True, "resident_uids": hh["resident_uids"], "lb_weight": 0}
            hh_insts.append(hh_inst)

    if params["loadinstitutions"]:
        for inst in institutions:
            hh_inst = {"id": inst["instid"], "is_hh": False, "resident_uids": inst["resident_uids"], "lb_weight": 0}
            hh_insts.append(hh_inst)

    del households
    del institutions

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

    agents_size = util.asizeof_formatted(agents) # mem: agents_size
    print(f"agents size (updated): {agents_size}")

    guardian_assignment_time_taken = time.time() - guardian_assignment_start
    print("guardian & load balancing weight assignment time_taken: " + str(guardian_assignment_time_taken))

    if params["loadworkplaces"]:
        if len(workplaces) == 0:
            wp_start = time.time()
            workplacesfile = open(os.path.join(current_directory, "population", population_sub_folder, "workplaces.json"))
            workplaces = json.load(workplacesfile)
            workplacesfile.close()
            wp_time_taken = time.time() - wp_start
            data_load_time_taken += wp_time_taken

        wp_cells_start = time.time()
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
        transport_size = util.asizeof_formatted(transport) # mem: transport_size
        cells_transport_size = util.asizeof_formatted(cells_transport) # mem: cells_transport_size
        print(f"transport size: {transport_size}, cells_transport size: {cells_transport_size}")

        if not params["use_mp"]:
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_transport_updated.json", cells_transport))

        # cells_accommodation_by_accomid = {} # {accomid: [{cellid: {cellinfo}}]}
        accommodations = []
        roomsizes_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomsize: [member_uids]}}} - here member_uids represents room ids
        rooms_by_accomid_by_accomtype = {} # {typeid: {accomid: {roomid: {"roomsize":1, "member_uids":[]}}}} - here member_uids represents tourist ids
        tourists_active_groupids, tourists_active_ids = [], []
        accomgroups = None

        if params["loadtourism"]:
            acc_start = time.time()
            accommodationsfile = open(os.path.join(current_directory, "population", population_sub_folder, "accommodations.json"))
            accommodations = json.load(accommodationsfile)
            accommodationsfile.close()
            accommodations_size = util.asizeof_formatted(accommodations) # mem: accommodations_size
            print(f"accommodations size: {accommodations_size}")
            acc_time_taken = time.time() - acc_start
            data_load_time_taken += acc_time_taken

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
        cells_industries_by_indid_by_wpid, _, cells_restaurants, cells_accommodation, _, cells_breakfast_by_accomid, rooms_by_accomid_by_accomtype, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_airport, indids_by_cellid = cells_util.split_workplaces_by_cellsize(workplaces, roomsizes_by_accomid_by_accomtype, rooms_by_accomid_by_accomtype, workplaces_cells_params, hospital_cells_params, testing_hubs_cells_params, vaccinations_hubs_cells_params, airport_cells_params, accom_cells_params, transport, entertainment_acitvity_dist, itineraryparams)
        
        cells_industries_by_indid_by_wpid_size = util.asizeof_formatted(cells_industries_by_indid_by_wpid) # mem: cells_industries_by_indid_by_wpid_size
        cells_restaurants_size = util.asizeof_formatted(cells_restaurants) # mem: cells_restaurants_size
        cells_accommodation_size = util.asizeof_formatted(cells_accommodation) # mem: cells_accommodation_size
        cells_breakfast_by_accomid_size = util.asizeof_formatted(cells_breakfast_by_accomid) # mem: cells_breakfast_by_accomid_size
        roomsizes_by_accomid_by_accomtype_size = util.asizeof_formatted(roomsizes_by_accomid_by_accomtype) # mem: roomsizes_by_accomid_by_accomtype_size
        cells_hospital_size = util.asizeof_formatted(cells_hospital) # mem: cells_hospital_size
        cells_testinghub_size = util.asizeof_formatted(cells_testinghub) # mem: cells_testinghub_size
        cells_vaccinationhub_size = util.asizeof_formatted(cells_vaccinationhub) # mem: cells_vaccinationhub_size
        cells_entertainment_by_activityid_size = util.asizeof_formatted(cells_entertainment_by_activityid) # mem: cells_entertainment_by_activityid_size
        cells_airport_size = util.asizeof_formatted(cells_airport) # mem: cells_airport_size
        indids_by_cellid_size = util.asizeof_formatted(indids_by_cellid) # mem: indids_by_cellid_size
        rooms_by_accomid_by_accomtype_size = util.asizeof_formatted(rooms_by_accomid_by_accomtype) # mem: rooms_by_accomid_by_accomtype_size
        print(f"roomsizes_by_accomid_by_accomtype size: {roomsizes_by_accomid_by_accomtype_size}, rooms_by_accomid_by_accomtype size: {rooms_by_accomid_by_accomtype_size}")
        print(f"cells_industries_by_indid_by_wpid size: {cells_industries_by_indid_by_wpid_size}, cells_restaurants size: {cells_restaurants_size}, cells_accommodation_size: {cells_accommodation_size}")
        print(f"cells_breakfast_by_accomid size: {cells_breakfast_by_accomid_size}, cells_hospital size: {cells_hospital_size}, cells_testinghub size: {cells_testinghub_size}, cells_vaccinationhub size: {cells_vaccinationhub_size}")
        print(f"cells_entertainment_by_activityid size: {cells_entertainment_by_activityid_size}, cells_airport size: {cells_airport_size}, indids_by_cellid size: {indids_by_cellid_size}")
        
        del accommodations
        del workplaces
        del roomsizes_by_accomid_by_accomtype_size

        wp_cells_time_taken = time.time() - wp_cells_start
        cells_generation_time_taken += wp_cells_time_taken

        if not params["use_mp"]:
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_industries_by_indid_by_wpid_updated.json", cells_industries_by_indid_by_wpid))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_restaurants_updated.json", cells_restaurants))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_hospital_updated.json", cells_hospital))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_testinghub_updated.json", cells_testinghub))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_vaccinationhub_updated.json", cells_vaccinationhub))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_entertainment_by_activityid_updated.json", cells_entertainment_by_activityid))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_breakfast_by_accomid_updated.json", cells_breakfast_by_accomid))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_airport_updated.json", cells_airport))
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_accommodation_updated.json", cells_accommodation))
        
        # contactnetwork_util.epi_util.cells_accommodation = cells_accommodation
        # airport_cells_params = cellsparams["airport"]

        # cell.create_airport_cell()

        # cellindex += 1

    if params["loadschools"]:
        schools_start = time.time()
        schoolsfile = open(os.path.join(current_directory, "population", population_sub_folder, "schools.json"))
        schools = json.load(schoolsfile)
        schoolsfile.close()
        schools_size = util.asizeof_formatted(schools) # mem: schools_size
        print(f"schools size: {schools_size}")
        schools_time_taken = time.time() - schools_start
        data_load_time_taken += schools_time_taken

        schools_cells_params = cellsparams["schools"]

        # min_nts_size, max_nts_size, min_classroom_size, max_classroom_size = cells_util.get_min_max_school_sizes(schools)

        # print("Min classroom size: " + str(min_classroom_size) + ", Max classroom size: " + str(max_classroom_size))
        # print("Min non-teaching staff size: " + str(min_nts_size) + ", Max classroom size: " + str(max_nts_size))
        # f.flush()
        
        schools_cells_start = time.time()
        cells_util.split_schools_by_cellsize(schools, schools_cells_params[0], schools_cells_params[1])
        del schools
        schools_cells_time_taken = time.time() - schools_cells_start
        cells_generation_time_taken += schools_cells_time_taken

    if params["religiouscells"]:
        religious_cells_start = time.time()
        religious_cells_params = cellsparams["religious"]

        _, cells_religious = cells_util.create_religious_cells(religious_cells_params[2], religious_cells_params[0], religious_cells_params[1], religious_cells_params[3], religious_cells_params[4])
        cells_religious_size = util.asizeof_formatted(cells_religious) # mem: cells_religious_size
        print(f"cells_religious size: {cells_religious_size}")

        if not params["use_mp"]:
            json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_religious_updated.json", cells_religious))
        
        religious_cells_time_taken = time.time() - religious_cells_start
        cells_generation_time_taken += religious_cells_time_taken

    # this might cause problems when referring to related agents, by household, workplaces etc
    # if params["quickdebug"]:
    #     agents = {i:agents[i] for i in range(10_000)}

    del cellsparams

    cells_type_start = time.time()
    cells_type = {cellid: props["type"] for cellid, props in cells_util.cells.items()}
    cells_type_size = util.asizeof_formatted(cells_type) # mem: cells_type_size
    print(f"cells_type size: {cells_type_size}")
    cells_type_time_taken = time.time() - cells_type_start
    cells_generation_time_taken += cells_type_time_taken

    if not params["use_mp"]:
        json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "cells_type_updated.json", cells_type))
        json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "indids_by_cellid_updated.json", indids_by_cellid))
        json_paths_to_upload.append(jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "agents_updated.json", agents))

    # if params["use_mp"] and params["numprocesses"] > 1:
    #     agents_static_start = time.time()
    #     agents_static.populate_shm(agents) # for now trying without multiprocessing.RawArray
    #     agents_static_time_taken = time.time() - agents_static_start
    # else: # might not be needed for Dask cases, if the tourism itinerary is also handled remotely. in that case, remote handling might be enough
    #     agents_static_start = time.time()
    #     agents_static = static.Static()
    #     agents_static.populate(agents, n_locals, n_tourists, params["use_shm"], params["use_static_dict_locals"], params["use_static_dict_tourists"]) # for now trying without multiprocessing.RawArray
    #     agents_static_time_taken = time.time() - agents_static_start

    #  might not be needed for Dask cases, if the tourism itinerary is also handled remotely. in that case, remote handling might be enough
    agents_static_start = time.time()
    agents_static = static.Static()
    agents_static.populate(agents, n_locals, n_tourists, params["use_shm"], params["use_static_dict_locals"], params["use_static_dict_tourists"]) # for now trying without multiprocessing.RawArray
    agents_static_time_taken = time.time() - agents_static_start

    agents_static_size = util.asizeof_formatted(agents_static) # mem: agents_static_size
    print(f"agents_static size: {agents_static_size}")

    agents_dynamic_start = time.time()
    agents_dynamic = agentsutil.initialize_agents_dict_dynamic(agents, agents_dynamic)
    agents_dynamic_time_taken = time.time() - agents_dynamic_start

    agents_dynamic_size = util.asizeof_formatted(agents_dynamic) # mem: agents_dynamic_size
    print(f"agents_dynamic size: {agents_dynamic_size}")

    # jsonutil.convert_to_json_file(current_directory, "population", population_sub_folder, "agents_dynamic_updated.json", agents_dynamic)

    del agents

    vars_util_start = time.time()
    vars_util = vars.Vars()
    vars_util.populate(agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses)
    vars_util_time_taken = time.time() - vars_util_start

    vars_util_size = util.asizeof_formatted(vars_util) # mem: vars_util_size
    print(f"vars_util size: {vars_util_size}")

    dyn_params = DynamicParams(n_locals, n_tourists, n_tourists_initial, epidemiologyparams)

    dyn_params_size = util.asizeof_formatted(dyn_params) # mem: vars_util_size
    print(f"dyn_params size: {dyn_params_size}")

    init_time_taken = time.time() - data_load_start_time

    print("data_load {0}, cells_generation {1}, init_total {2}, agents_static populate {3}, agents_dynamic initialize {4}, vars_util populate {5}".format(str(data_load_time_taken), str(cells_generation_time_taken), str(init_time_taken), str(agents_static_time_taken), str(agents_dynamic_time_taken), str(vars_util_time_taken)))
    if f is not None:
        f.flush()

    util.log_memory_usage(f, "Loaded data. Before gc.collect() ")
    gc_start = time.time()
    gc.collect()
    gc_time_taken = time.time() - gc_start
    print("gc time_taken: " + str(gc_time_taken))
    util.log_memory_usage(f, "Loaded data. After gc.collect() ")
    
    dask_combined_scores_nworkers = None
    dask_it_workers_time_taken, dask_cn_workers_time_taken = {}, {}
    dask_mp_it_processes_time_taken, dask_mp_cn_processes_time_taken = {}, {}
    workers = []
    actors = []
    try:
        if params["use_mp"] and params["numprocesses"] > 1:
            manager = mp.Manager()
            static_agents_dict = manager.dict()
            static_params = agents_static, agents_ids_by_ages, params["timestepmins"], n_locals, n_tourists, locals_ratio_to_full_pop, itineraryparams, epidemiologyparams, cells_industries_by_indid_by_wpid, cells_restaurants, cells_hospital, cells_testinghub, cells_vaccinationhub, cells_entertainment_by_activityid, cells_religious, cells_households, cells_breakfast_by_accomid, cells_airport, cells_transport, cells_institutions, cells_accommodation, contactnetworkparams, cells_type, indids_by_cellid
            pool = mp.Pool(processes=params["numprocesses"], initializer=shared_mp.init_pool_processes, initargs=(static_params,))
            agents_static.static_agents_dict = static_agents_dict
        elif not params["use_mp"]:
            start = time.time()
            # cluster = LocalCluster()

            num_workers = 1 # this is set to 1, because now always expecting "dask_nodes_n_workers" to be specified for any Dask scenario
            # if params["dask_use_mp"]:
            #     num_workers = 1
            # else:
            #     num_workers = params["numprocesses"]

            num_threads = None
            if params["numthreads"] > 0:
                num_threads = params["numthreads"]

            dask_nodes = []

            if not params["dask_use_mp"]:
                dask_nodes.append(params["dask_scheduler_node"])

                for index, node in enumerate(params["dask_nodes"]):
                    num_workers_this_node = params["dask_nodes_n_workers"][index]

                    for i in range(num_workers_this_node):
                        dask_nodes.append(node)
            else:
                dask_nodes.append(params["dask_scheduler_node"])

                for dask_node in params["dask_nodes"]:
                    dask_nodes.append(dask_node)

            if params["dask_nodes_cpu_scores"] is not None:
                dask_nodes_cpu_scores = np.array(params["dask_nodes_cpu_scores"])
                dask_nodes_n_workers = np.array(params["dask_nodes_n_workers"])
                dask_combined_scores_nworkers = dask_nodes_cpu_scores * dask_nodes_n_workers
            
            worker_index = 0
            for node_index, node in enumerate(params["dask_nodes"]):
                if not params["dask_use_mp"]:
                    # time_taken_this_node = params["dask_nodes_time_taken"][index]
                    num_workers_this_node = params["dask_nodes_n_workers"][node_index]
                    # time_taken_each_node = time_taken_this_node / num_workers_this_node

                    for _ in range(num_workers_this_node):
                        dask_it_workers_time_taken[(node_index, worker_index)] = 1 # default to 1 second each for first day, load balancing starts from second day
                        dask_cn_workers_time_taken[(node_index, worker_index)] = 1
                        worker_index += 1
                else:
                    dask_it_workers_time_taken[node_index] = 1
                    dask_cn_workers_time_taken[node_index] = 1

            worker_class = ""
            if not params["dask_use_mp"] and not params["dask_full_stateful"]:
                worker_class = "distributed.Nanny"
            else:
                worker_class = "distributed.Worker"

            print("dask_nodes {0}, num_workers {1}, nthreads {2}, worker_class {3}".format(str(dask_nodes), str(num_workers), str(num_threads), str(worker_class)))

            cluster = SSHCluster(dask_nodes, 
                            connect_options={"known_hosts": None},
                            worker_options={"n_workers": num_workers, "nthreads": num_threads, "local_directory": config["worker_working_directory"] }, # "memory_limit": "3GB" (in worker_options)
                            scheduler_options={"host": params["dask_scheduler_host"], "port": 0, "dashboard_address": ":8797", "local_directory": config["scheduler_working_directory"] }, # local_directory in scheduler_options has no effect
                            worker_class=worker_class, 
                            remote_python=config["worker_remote_python"])
            
            time_taken = time.time() - start
            print("cluster generation: " + str(time_taken))
            if f is not None:
                f.flush()

            if params["dask_use_mp"] or params["dask_full_stateful"]:
                print("sleeping for 5 seconds to make sure the cluster and inner processes have started successfully")
                time.sleep(5)

            # cluster.scale(params["numprocesses"])

            # Set the working directory for the scheduler and workers
            # os.chdir(config["scheduler_working_directory"])  # Change the current working directory for the scheduler
            # cluster.worker_options['local_directory'] = config["worker_working_directory"]  # Worker working directory from config

            start = time.time()
            client = Client(cluster)
            time_taken = time.time() - start
            print("client generation: " + str(time_taken))
            if f is not None:
                f.flush()

            workers = client.scheduler_info()["workers"]
            print("num_workers: " + str(len(workers)))
            workers_keys = list(workers.keys())
            print("workers_keys: " + str(workers_keys))
            # client_ip = workers_keys[0].replace("tcp://", "").split(':', 1)[0]
            # print("client_ip: " + str(client_ip))

            # plugin = ReadOnlyData(agentsfilepath)
            # client.register_worker_plugin(plugin, name="read-only-data")

            # versions = client.get_versions(check=True)
            # print("versions: " + str(versions))
            
            dask_init_file_name = os.path.join(log_subfolder_path, "dask_init.html")

            with performance_report(filename=dask_init_file_name):
                start = time.time()
                client.upload_file('simulator/cellsclasses.py')
                client.upload_file('simulator/customdict.py')
                client.upload_file('simulator/npencoder.py')
                client.upload_file('simulator/jsonutil.py')
                client.upload_file('simulator/epidemiologyclasses.py')
                client.upload_file('simulator/shared_mp.py')
                client.upload_file('simulator/static.py')
                client.upload_file('simulator/seirstateutil.py')
                client.upload_file('simulator/util.py')
                client.upload_file('simulator/daskutil.py')
                client.upload_file('simulator/tourism_dist.py')
                client.upload_file('simulator/epidemiology.py')
                client.upload_file('simulator/seirstateutil.py')
                client.upload_file('simulator/vars.py')
                client.upload_file('simulator/dynamicstatistics.py')
                client.upload_file('simulator/dynamicparams.py')
                client.upload_file('simulator/itinerary.py')
                client.upload_file('simulator/contactnetwork.py')
                # client.upload_file('simulator/itinerary_dask.py')

                if params["dask_use_mp"]:
                    client.upload_file('simulator/actor_dist_mp.py')
                elif params["dask_full_stateful"]:
                    client.upload_file('simulator/actor_dist.py')

                client.upload_file('simulator/itinerary_mp.py')
                client.upload_file('simulator/itinerary_dist.py')      
                client.upload_file('simulator/contactnetwork_dist.py')
                client.upload_file('simulator/contacttracing_dist.py')
                client.upload_file('simulator/tourism_dist.py')

                for dynamicjsonpath in json_paths_to_upload:
                    client.upload_file(dynamicjsonpath)
                
                callback = partial(read_only_data, 
                                dask_strategy=dask_strategy,
                                agents_ids_by_ages=agents_ids_by_ages, 
                                timestepmins=params["timestepmins"], 
                                n_locals=n_locals, 
                                n_tourists=n_tourists, 
                                locals_ratio_to_full_pop=locals_ratio_to_full_pop,
                                use_shm=params["use_shm"],
                                use_static_dict_locals=params["use_static_dict_locals"],
                                use_static_dict_tourists=params["use_static_dict_tourists"],
                                logsubfoldername=params["remotelogsubfoldername"],
                                logfilename=params["logfilename"])
                
                client.register_worker_callbacks(callback)

                time_taken = time.time() - start
                print("upload modules remotely: " + str(time_taken))

                if f is not None:
                    f.flush()   

        itinerary_sum_time_taken = 0
        tourist_itinerary_sum_time_taken = 0
        contactnetwork_sum_time_taken = 0
        contactracing_sum_time_taken = 0
        vaccination_sum_time_taken = 0
        refresh_dyn_params_sum_time_taken = 0
        simdays_sum_time_taken = 0

        general_sum_mem = 0
        it_agents_sum_mem = 0
        working_schedule_sum_mem = 0
        itinerary_sum_mem = 0
        itinerary_nextday_sum_mem = 0
        non_daily_activity_recurring_sum_mem = 0
        prevday_non_daily_activity_recurring_sum_mem = 0
        tourist_id_sum_mem = 0
        initial_tourist_sum_mem = 0

        epi_agents_sum_mem = 0
        state_transition_by_day_sum_mem = 0
        test_day_sum_mem = 0
        test_result_day_sum_mem = 0
        hospitalisation_days_sum_mem = 0
        quarantine_days_sum_mem = 0
        vaccination_days_sum_mem = 0

        vars_util_sum_mem = 0
        cat_agents_sum_mem = 0
        seir_state_sum_mem = 0
        seir_state_trans_for_day_sum_mem = 0
        inf_type_sum_mem = 0
        inf_sev_sum_mem = 0
        vacc_doses_sum_mem = 0
        dir_con_sum_mem = 0
        dir_con_idx1_sum_mem = 0
        dir_con_idx2_sum_mem = 0

        memory_sums = general_sum_mem, it_agents_sum_mem, working_schedule_sum_mem, itinerary_sum_mem, itinerary_nextday_sum_mem, non_daily_activity_recurring_sum_mem, prevday_non_daily_activity_recurring_sum_mem, tourist_id_sum_mem, initial_tourist_sum_mem, epi_agents_sum_mem, state_transition_by_day_sum_mem, test_day_sum_mem, test_result_day_sum_mem, hospitalisation_days_sum_mem, quarantine_days_sum_mem, vaccination_days_sum_mem, vars_util_sum_mem, cat_agents_sum_mem, seir_state_sum_mem, seir_state_trans_for_day_sum_mem, inf_type_sum_mem, inf_sev_sum_mem, vacc_doses_sum_mem, dir_con_sum_mem, dir_con_idx1_sum_mem, dir_con_idx2_sum_mem

        epi_keys = ["state_transition_by_day", "test_day", "test_result_day", "hospitalisation_days", "quarantine_days", "vaccination_days"]
        it_keys = ["working_schedule", "non_daily_activity_recurring", "prevday_non_daily_activity_recurring", "itinerary", "itinerary_next_day"]
        
        # new - partialising agents for multiprocessing
        agents_epi = customdict.CustomDict({
            key: {
                inner_key: value[inner_key] for inner_key in epi_keys if inner_key in value
            } for key, value in agents_dynamic.items()
        })
                
        it_agents = customdict.CustomDict({
            key: {
                inner_key: value[inner_key] for inner_key in it_keys if inner_key in value
            } for key, value in agents_dynamic.items()
        })

        agents_epi_size = util.asizeof_formatted(agents_epi) # mem: agents_epi_size
        it_agents_size = util.asizeof_formatted(it_agents) # mem: it_agents_size

        print(f"agents_epi size {agents_epi_size}, it_agents size: {it_agents_size}")

        del agents_dynamic

        tourist_util = None

        if params["loadtourism"]:
            util.log_memory_usage(f, "Loaded data. Before sample_initial_tourists ")
            tourist_util = tourism.Tourism(tourismparams, cells, n_locals, tourists, agents_static, it_agents, agents_epi, agents_seir_state, touristsgroupsdays, touristsgroups, rooms_by_accomid_by_accomtype, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids, tourists_active_ids, age_brackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count, initial_seir_state_distribution) 
            tourist_util.sample_initial_tourists(touristsgroupsids_initial, f)
            util.log_memory_usage(f, "Loaded data. After sample_initial_tourists ")
        
        if params["dask_use_mp"]:
            num_actors = len(workers_keys)
            for worker_index in range(num_actors):
                worker_url = workers_keys[worker_index]
                num_processes = params["dask_nodes_n_workers"][worker_index]
                dmp_params = (num_processes, worker_index, params["remotelogsubfoldername"], params["logfilename"])
                actor_future = client.submit(ActorDistMP, dmp_params, workers=worker_url, actor=True)
                
                actor = actor_future.result()
                actors.append(actor)

        if params["dask_full_stateful"]:
            num_actors = sum(params["dask_nodes_n_workers"])

            # if not params["dask_actors_innerproc_assignment"]:
            #     dask_num_tasks = num_actors
            # else:
            #     dask_num_tasks = num_inner_processes

            # split residences
            hh_inst_split_indices = util.split_residences_by_weight(hh_insts, num_actors)
            
            inst_keys = list(cells_institutions.keys())
            first_inst_key = inst_keys[0]
            last_inst_key = inst_keys[-1]

            # split cells (except households and institutions - for now)
            cells_ids = np.array(list(cells.keys())[last_inst_key+1:]) # from last inst + 1 until end
            np.random.shuffle(cells_ids)

            num_cells_per_actor = util.split_balanced_partitions(len(cells_ids), num_actors)
            
            cells_split_ids = []
            cells_index = 0
            for num_cells_this_actor in num_cells_per_actor:
                this_actor_cells_ids = []

                for _ in range(num_cells_this_actor):
                    this_actor_cells_ids.append(cells_ids[cells_index])
                    cells_index += 1

                cells_split_ids.append(this_actor_cells_ids)

            agent_ids_by_worker_lookup, cell_ids_by_worker_lookup = customdict.CustomDict(), customdict.CustomDict()
            worker_by_res_ids_lookup, worker_by_agent_ids_lookup, worker_by_cell_ids_lookup = customdict.CustomDict(), customdict.CustomDict(), customdict.CustomDict()
            worker_data = customdict.CustomDict()
            
            worker_index = 0
            for _, num_workers in enumerate(params["dask_nodes_n_workers"]):
                for _ in range(num_workers):
                    hh_inst_split_indices_this_worker = hh_inst_split_indices[worker_index]

                    hh_insts_this_worker = []
                    it_agents_this_worker, agents_epi_this_worker = customdict.CustomDict(), customdict.CustomDict()
                    vars_util_this_worker = vars.Vars()

                    agent_ids_this_worker = []
                    hh_inst_cell_ids = []

                    for index in hh_inst_split_indices_this_worker:
                        hh_inst = hh_insts[index]

                        if hh_inst["is_hh"]:
                            hh_inst_cell_ids.append(index) # hh_id == cell_id, set as is
                        else:
                            hh_inst_cell_ids.extend(cell_ids_by_inst_id[hh_inst["id"]]) # set all cell ids per inst (can be more than 1)

                        agent_ids_this_worker.extend(hh_inst["resident_uids"]) # set all agents within this residence in the same worker
                        worker_by_res_ids_lookup[index] = worker_index
                        hh_insts_this_worker.append(hh_inst)

                    for agent_id in agent_ids_this_worker:
                        it_agents_this_worker[agent_id] = it_agents[agent_id]
                        agents_epi_this_worker[agent_id] = agents_epi[agent_id]

                        if agent_id in vars_util.agents_seir_state:
                            vars_util_this_worker.agents_seir_state[agent_id] = vars_util.agents_seir_state[agent_id]
                        if agent_id in vars_util.agents_seir_state_transition_for_day:
                            vars_util_this_worker.agents_seir_state_transition_for_day[agent_id] = vars_util.agents_seir_state_transition_for_day[agent_id]
                        if agent_id in vars_util.agents_infection_type:
                            vars_util_this_worker.agents_infection_type[agent_id] = vars_util.agents_infection_type[agent_id]
                        if agent_id in vars_util.agents_infection_severity:
                            vars_util_this_worker.agents_infection_severity[agent_id] = vars_util.agents_infection_severity[agent_id]
                        if agent_id in vars_util.agents_vaccination_doses:
                            vars_util_this_worker.agents_vaccination_doses[agent_id] = vars_util.agents_vaccination_doses[agent_id]
                        
                        worker_by_agent_ids_lookup[agent_id] = worker_index

                    cell_ids_this_worker = hh_inst_cell_ids # set residence cells
                    cell_ids_this_worker.extend(cells_split_ids[worker_index]) # extend with other cells

                    for cell_id in cell_ids_this_worker:
                        worker_by_cell_ids_lookup[cell_id] = worker_index

                    agent_ids_by_worker_lookup[worker_index] = set(agent_ids_this_worker)
                    cell_ids_by_worker_lookup[worker_index] = set(cell_ids_this_worker)
                    worker_data[worker_index] = [hh_insts_this_worker, it_agents_this_worker, agents_epi_this_worker, vars_util_this_worker]

                    worker_index += 1

            worker_index = 0
            for _, num_workers in enumerate(params["dask_nodes_n_workers"]):
                for _ in range(num_workers):
                    worker_url = workers_keys[worker_index]

                    d_params = (workers_keys, 
                                worker_index, 
                                worker_data[worker_index][0], # hh_insts
                                worker_data[worker_index][1], # it_agents
                                worker_data[worker_index][2], # agents_epi
                                worker_data[worker_index][3], # vars_util
                                dyn_params,
                                agent_ids_by_worker_lookup, 
                                cell_ids_by_worker_lookup, 
                                worker_by_res_ids_lookup, 
                                worker_by_agent_ids_lookup, 
                                worker_by_cell_ids_lookup, 
                                params["remotelogsubfoldername"], 
                                params["logfilename"])
                    
                    actor_future = client.submit(ActorDist, d_params, workers=worker_url, actor=True)
                    
                    actor = actor_future.result()
                    actors.append(actor)          

                    worker_index += 1   

            for ai, actor in enumerate(actors):
                temp_actors = [a if i != ai else None for i, a in enumerate(actors)]
                actor.set_remote_actors(temp_actors)

        for day in simdays_range: # 365 + 1 / 1 + 1
            print("simulating day {0}".format(str(day)))
            if f is not None:
                f.flush()
            
            day_start = time.time()
            
            if day > 1 and not params["use_mp"] and params["dask_cluster_restart_days"] != -1 and day % params["dask_cluster_restart_days"] == 0: # force clean-up every X days
                restart_start = time.time()
                client.restart()
                restart_time_taken = time.time() - restart_start
                print("restart cluster on day {0} time_taken {1}".format(str(day), str(restart_time_taken)))
                if f is not None:
                    f.flush()

            weekday, weekdaystr = util.day_of_year_to_day_of_week(day, params["year"])

            if params["dask_full_stateful"]:
                for actor in actors:
                    actor.reset_day(day, weekday, weekdaystr, dyn_params)

            vars_util.contact_tracing_agent_ids = set()
            vars_util.reset_daily_structures()

            if params["loadtourism"]:
                # single process tourism section
                itinerary_util = itinerary.Itinerary(itineraryparams, 
                                                    params["timestepmins"], 
                                                    n_locals, 
                                                    n_tourists,
                                                    locals_ratio_to_full_pop,
                                                    agents_static,
                                                    it_agents, # it_agents
                                                    agents_epi, # agents_epi ?
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
                                                    tourists)
                
                print("generating_tourist_itinerary for simday " + str(day) + ", weekday " + str(weekday))
                if f is not None:
                    f.flush()

                util.log_memory_usage(f, "Loaded data. Before tourist itinerary ")

                start = time.time()
                
                it_agents, agents_epi, tourists, cells, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, tourists_active_groupids = tourist_util.initialize_foreign_arrivals_departures_for_day(day, f)
                print("initialize_foreign_arrivals_departures_for_day (done) for simday " + str(day) + ", weekday " + str(weekday))
                if f is not None:
                    f.flush()

                itinerary_util.generate_tourist_itinerary(day, weekday, touristsgroups, tourists_active_groupids, tourists_arrivals_departures_for_day, tourists_arrivals_departures_for_nextday, log_file_name, f)
                print("generate_tourist_itinerary (done) for simday " + str(day) + ", weekday " + str(weekday))
                if f is not None:
                    f.flush()

                tourist_util.sync_and_clean_tourist_data(day, client, actors, params["remotelogsubfoldername"], params["logfilename"], params["dask_full_stateful"], f)
                print("sync_and_clean_tourist_data (done) for simday " + str(day) + ", weekday " + str(weekday))
                if f is not None:
                    f.flush()

                util.log_memory_usage(f, "Loaded data. After tourist itinerary ")

                time_taken = time.time() - start
                tourist_itinerary_sum_time_taken += time_taken
                avg_time_taken = tourist_itinerary_sum_time_taken / day
                perf_timings_df.loc[day, "tourismitinerary_day"] = round(time_taken, 2)
                perf_timings_df.loc[day, "tourismitinerary_avg"] = round(avg_time_taken, 2)

                print("generate_tourist_itinerary for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))
                if f is not None:
                    f.flush()

            # epi_util.tourists_active_ids = tourist_util.tourists_active_ids

            if day == 1: # from day 2 onwards always calculated at eod
                util.log_memory_usage(f, "Loaded data. Before refreshing dynamic parameters ")

                num_arrivals, num_departures = 0, 0

                if params["loadtourism"]:
                    num_departures = len(tourist_util.departing_tourists_agents_ids[day])
                    num_arrivals = len(tourist_util.arriving_tourists_agents_ids)
                    num_arrivals_nextday = len(tourist_util.arriving_tourists_next_day_agents_ids)
            
                dyn_params.refresh_dynamic_parameters(day, num_arrivals, num_arrivals_nextday, num_departures, tourists_active_ids, vars_util)
                util.log_memory_usage(f, "Loaded data. After refreshing dynamic parameters ")

            if not params["quicktourismrun"]:
                start = time.time()  
                util.log_memory_usage(f, "Loaded data. Before local itinerary ")
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
                                                        dyn_params,
                                                        hh_insts, 
                                                        params["numprocesses"],
                                                        params["numthreads"],
                                                        params["proc_usepool"],
                                                        params["use_shm"],
                                                        params["sync_usethreads"],
                                                        params["sync_usequeue"],
                                                        params["keep_processes_open"],
                                                        log_file_name,
                                                        False,
                                                        agents_static,
                                                        static_agents_dict)
                else:
                    if params["dask_use_fg"]:
                        itinerary_dist.localitinerary_distributed_finegrained(client,
                                                                            hh_insts,
                                                                            day,
                                                                            weekday,
                                                                            weekdaystr,
                                                                            it_agents,
                                                                            agents_epi,
                                                                            vars_util,
                                                                            dyn_params,
                                                                            params["keep_processes_open"],
                                                                            params["dask_mode"],
                                                                            params["dask_scatter"],
                                                                            params["dask_batch_size"],
                                                                            params["dask_batch_recurring"],
                                                                            log_file_name)
                    else:
                        if params["dask_use_chunking"]:
                            itinerary_dist.localitinerary_distributed_finegrained_chunks(client,
                                                                                        hh_insts,
                                                                                        day,
                                                                                        weekday,
                                                                                        weekdaystr,
                                                                                        it_agents,
                                                                                        agents_epi,
                                                                                        vars_util,
                                                                                        dyn_params,
                                                                                        params["keep_processes_open"],
                                                                                        params["dask_mode"],
                                                                                        params["dask_chunk_size"],
                                                                                        params["dask_single_item_per_task"],
                                                                                        params["dask_full_array_mapping"],
                                                                                        log_file_name)    
                        elif params["dask_map_batching"]:
                            itinerary_dist.localitinerary_distributed_map_batched(client,
                                                                                hh_insts,
                                                                                day,
                                                                                weekday,
                                                                                weekdaystr,
                                                                                it_agents,
                                                                                agents_epi,
                                                                                vars_util,
                                                                                dyn_params,
                                                                                params["keep_processes_open"],
                                                                                params["dask_batch_size"],
                                                                                params["dask_full_array_mapping"],
                                                                                params["dask_scatter"],
                                                                                params["dask_submit"],
                                                                                params["dask_map_batched_results"],
                                                                                log_file_name)
                        elif not params["dask_full_stateful"]:
                            itinerary_dist.localitinerary_distributed(client,
                                                                    day, 
                                                                    weekday, 
                                                                    weekdaystr, 
                                                                    it_agents,
                                                                    agents_epi,
                                                                    vars_util,
                                                                    dyn_params, 
                                                                    hh_insts, 
                                                                    params["keep_processes_open"],
                                                                    params["dask_use_mp"],
                                                                    params["dask_numtasks"],
                                                                    params["dask_mode"],
                                                                    params["dask_full_array_mapping"],
                                                                    params["dask_nodes_n_workers"],
                                                                    dask_combined_scores_nworkers,
                                                                    dask_it_workers_time_taken,
                                                                    dask_mp_it_processes_time_taken,
                                                                    params["dask_dynamic_load_balancing"],
                                                                    params["dask_actors_innerproc_assignment"],
                                                                    f,
                                                                    actors,
                                                                    log_file_name)
                        else: # full stateful
                            futures = []
                            for actor in actors:
                                futures.append(actor.itinerary())

                            for future in as_completed(futures):
                                a_worker_index, contact_tracing_agent_ids_partial, a_tt, a_results_tt, a_avg_tt = future.result()

                                if len(contact_tracing_agent_ids_partial) > 0:
                                    vars_util.contact_tracing_agent_ids.update(contact_tracing_agent_ids_partial)
                                
                                print(f"actor worker index {a_worker_index}, contact tracing agent ids: {len(vars_util.contact_tracing_agent_ids)}, time taken: {a_tt}, send results time taken: {a_results_tt}, avg time taken: {a_avg_tt}")
                
                # may use dask_workers_time_taken and dask_mp_processes_time_taken for historical performance data

                util.log_memory_usage(f, "After itinerary. Before gc.collect() ")
                gc_start = time.time()
                gc.collect()
                gc_time_taken = time.time() - gc_start
                print("gc time_taken: " + str(gc_time_taken))
                util.log_memory_usage(f, "After itinerary. After gc.collect() ")

                time_taken = time.time() - start
                itinerary_sum_time_taken += time_taken
                avg_time_taken = itinerary_sum_time_taken / day
                perf_timings_df.loc[day, "localitinerary_day"] = round(time_taken, 2)
                perf_timings_df.loc[day, "localitinerary_avg"] = round(avg_time_taken, 2)
                print("localitinerary_parallel for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))
                print("main workers: time results [start, time_taken]: {0}".format(str(dask_it_workers_time_taken)))

                if dask_mp_it_processes_time_taken is not None and len(dask_mp_it_processes_time_taken) > 0:
                    print("inner mp processes: time results [start, time_taken]: {0}".format(str(dask_mp_it_processes_time_taken)))
                
                if f is not None:
                    f.flush()

                if not params["quickitineraryrun"]:
                    print("simulate_contact_network for simday " + str(day) + ", weekday " + str(weekday))
                    if f is not None:
                        f.flush()

                    util.log_memory_usage(f, "Loaded data. Before contact network ")

                    start = time.time()       

                    if params["use_mp"]:
                        contactnetwork_mp.contactnetwork_parallel(manager,
                                                                pool,
                                                                day, 
                                                                weekday, 
                                                                n_locals, 
                                                                n_tourists, 
                                                                locals_ratio_to_full_pop, 
                                                                agents_epi, 
                                                                vars_util,
                                                                cells_type, 
                                                                indids_by_cellid,
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
                                                                log_file_name,
                                                                agents_static,
                                                                static_agents_dict)
                    else:
                        contactnetwork_dist.contactnetwork_distributed(client,
                                                                day,
                                                                weekday,
                                                                agents_epi,
                                                                vars_util,
                                                                dyn_params,
                                                                params["dask_mode"],
                                                                params["dask_numtasks"],
                                                                params["dask_full_array_mapping"],
                                                                params["keep_processes_open"],
                                                                params["dask_use_mp"],
                                                                params["dask_nodes_n_workers"],
                                                                dask_cn_workers_time_taken,
                                                                dask_mp_cn_processes_time_taken,
                                                                params["dask_actors_innerproc_assignment"],
                                                                f,
                                                                actors,
                                                                log_file_name)

                    util.log_memory_usage(f, "After contact network. Before gc.collect() ")
                    gc_start = time.time()
                    gc.collect()
                    gc_time_taken = time.time() - gc_start
                    print("gc time_taken: " + str(gc_time_taken))
                    util.log_memory_usage(f, "After contact network. After gc.collect() ")

                    time_taken = time.time() - start
                    contactnetwork_sum_time_taken += time_taken
                    avg_time_taken = contactnetwork_sum_time_taken / day
                    perf_timings_df.loc[day, "contactnetwork_day"] = round(time_taken, 2)
                    perf_timings_df.loc[day, "contactnetwork_avg"] = round(avg_time_taken, 2)
                    print("simulate_contact_network for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))                 
                    print("main workers: time results [start, time_taken]: {0}".format(str(dask_cn_workers_time_taken)))

                    if dask_mp_cn_processes_time_taken is not None and len(dask_mp_cn_processes_time_taken) > 0:
                        print("inner mp processes: time results [start, time_taken]: {0}".format(str(dask_mp_cn_processes_time_taken)))

                    if f is not None:
                        f.flush()

                # contact tracing
                print("contact_tracing for simday " + str(day) + ", weekday " + str(weekday))
                if f is not None:
                    f.flush()

                util.log_memory_usage(f, "Loaded data. Before contact tracing ")

                start = time.time()

                # vars_util.dc_by_sct_by_day_agent1_index.sort(key=lambda x:x[0],reverse=False)
                # vars_util.dc_by_sct_by_day_agent2_index.sort(key=lambda x:x[0],reverse=False)

                epi_util = epidemiology.Epidemiology(epidemiologyparams, 
                                                    n_locals, 
                                                    n_tourists, 
                                                    locals_ratio_to_full_pop,
                                                    agents_static,
                                                    agents_epi, 
                                                    vars_util, 
                                                    cells_households, 
                                                    cells_institutions, 
                                                    cells_accommodation, 
                                                    dyn_params)

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

                if not params["contacttracing_distributed"]:
                    _, _updated_agent_ids, agents_epi, vars_util = epi_util.contact_tracing(day, f=f) # process_index, updated_agent_ids
                else:
                    vars_util.reset_daily_structures()

                    contacttracing_dist.contacttracing_distributed(client, 
                                                                day, 
                                                                epi_util,
                                                                agents_epi, 
                                                                vars_util, 
                                                                dyn_params, 
                                                                params["dask_mode"],
                                                                params["dask_numtasks"],
                                                                params["dask_full_array_mapping"],
                                                                params["keep_processes_open"],
                                                                f,
                                                                log_file_name)

                util.log_memory_usage(f, "After contact tracing. Before gc.collect() ")
                gc_start = time.time()
                gc.collect()
                gc_time_taken = time.time() - gc_start
                print("gc time_taken: " + str(gc_time_taken))
                util.log_memory_usage(f, "After contact tracing. After gc.collect() ")

                time_taken = time.time() - start
                contactracing_sum_time_taken += time_taken
                avg_time_taken = contactracing_sum_time_taken / day
                perf_timings_df.loc[day, "contacttracing_day"] = round(time_taken, 2)
                perf_timings_df.loc[day, "contacttracing_avg"] = round(avg_time_taken, 2)
                print("contact_tracing time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))

                if f is not None:
                    f.flush()   

                # vaccinations
                print("schedule_vaccinations for simday " + str(day) + ", weekday " + str(weekday))
                if f is not None:
                    f.flush()

                util.log_memory_usage(f, "Loaded data. Before vaccinations ")
                start = time.time()
                epi_util.schedule_vaccinations(day)
                util.log_memory_usage(f, "Loaded data. After vaccinations ")
                time_taken = time.time() - start
                vaccination_sum_time_taken += time_taken
                avg_time_taken = vaccination_sum_time_taken / day
                perf_timings_df.loc[day, "vaccination_day"] = round(time_taken, 2)
                perf_timings_df.loc[day, "vaccination_avg"] = round(avg_time_taken, 2)
                print("schedule_vaccinations time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))
                if f is not None:
                    f.flush()

                print("refresh_dynamic_parameters for simday " + str(day) + ", weekday " + str(weekday))
                if f is not None:
                    f.flush()

                start = time.time()
                util.log_memory_usage(f, "Loaded data. Before refreshing dynamic parameters and updating statistics ")

                num_arrivals, num_departures = 0, 0

                if params["loadtourism"]:
                    num_departures = len(tourist_util.departing_tourists_agents_ids[day])
                    num_arrivals = len(tourist_util.arriving_tourists_agents_ids)
                    num_arrivals_nextday = len(tourist_util.arriving_tourists_next_day_agents_ids)
                    
                dyn_params.refresh_dynamic_parameters(day, num_arrivals, num_arrivals_nextday, num_departures, tourists_active_ids, vars_util)
                interventions_logs_df, statistics_logs_df = dyn_params.update_logs_df(day, interventions_logs_df, statistics_logs_df)
                util.log_memory_usage(f, "Loaded data. After refreshing dynamic parameters and updating statistics ")
                time_taken = time.time() - start
                refresh_dyn_params_sum_time_taken += time_taken
                avg_time_taken = refresh_dyn_params_sum_time_taken / day
                perf_timings_df.loc[day, "refreshdynamicparams_day"] = round(time_taken, 2)
                perf_timings_df.loc[day, "refreshdynamicparams_avg"] = round(avg_time_taken, 2)
                print("refresh_dynamic_parameters time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken))
                if f is not None:
                    f.flush()

            mem_start = time.time()
            util.log_memory_usage(f, "Loaded data. Before calculating memory info ")
            memory_sums, mem_logs_df = calculate_memory_info(day, params["logmemoryinfo"], it_agents, agents_epi, vars_util, memory_sums, f, mem_logs_df)
            util.log_memory_usage(f, "Loaded data. After calculating memory info ")
            mem_time_taken = time.time() - mem_start
            print("log memory info time_taken: " + str(mem_time_taken))

            util.log_memory_usage(f, "End of sim day. Before gc.collect() ")
            gc_start = time.time()
            gc.collect()
            gc_time_taken = time.time() - gc_start
            print("gc time_taken: " + str(gc_time_taken))
            util.log_memory_usage(f, "End of sim day. After gc.collect() ")
                              
            day_time_taken = time.time() - day_start
            simdays_sum_time_taken += day_time_taken
            simdays_avg_time_taken = simdays_sum_time_taken / day

            print("simulation day: " + str(day) + ", weekday " + str(weekday) + ", curr infectious rate: " + str(round(dyn_params.statistics.infectious_rate, 2)) + ", time taken: " + str(day_time_taken) + ", avg time taken: " + str(simdays_avg_time_taken))
            if f is not None:
                f.flush()

            perf_timings_df.to_csv(perf_timings_file_name, index_label="day")
            mem_logs_df.to_csv(mem_logs_file_name, index_label="day")
            interventions_logs_df.to_csv(interventions_logs_file_name, index_label="day")
            statistics_logs_df.to_csv(statistics_logs_file_name, index_label="day")
    except:
        with open(os.path.join(current_directory, params["logsubfoldername"], subfolder_name, "stack_trace.txt"), 'w') as ef:
            traceback.print_exc(file=ef)
    finally:
        sys.stdout = original_stdout
        f.close()

        if params["dask_autoclose_cluster"] and client is not None:
            if params["dask_use_mp"] and len(actors) > 0:
                for worker_index in range(len(workers_keys)):
                    actor = actors[worker_index]
                    actor.close_pool()

            client.shutdown()

def calculate_memory_info(day, log_memory_info, it_agents, agents_epi, vars_util, sums=None, f=None, df=None):
    if log_memory_info:
        start = time.time()
        memory_info = psutil.virtual_memory()
        
        general_mem = round(memory_info.used / (1024 * 1024), 2)

        it_agents_mem = util.asizeof_formatted(it_agents)
        working_schedule_mem = util.asizeof_formatted([props["working_schedule"] for props in it_agents.values() if "working_schedule" in props])
        it_mem = util.asizeof_formatted([props["itinerary"] for props in it_agents.values() if "itinerary" in props])
        it_nd_mem = util.asizeof_formatted([props["itinerary_nextday"] for props in it_agents.values() if "itinerary_nextday" in props])
        ndar_mem = util.asizeof_formatted([props["non_daily_activity_recurring"] for props in it_agents.values() if "non_daily_activity_recurring" in props])
        pd_ndar_mem = util.asizeof_formatted([props["prevday_non_daily_activity_recurring"] for props in it_agents.values() if "prevday_non_daily_activity_recurring" in props])
        tour_ids_mem = util.asizeof_formatted([props["tourist_id"] for props in it_agents.values() if "tourist_id" in props])
        initial_tourist_mem = util.asizeof_formatted([props["initial_tourist"] for props in it_agents.values() if "initial_tourist" in props])
        
        epi_agents_mem = util.asizeof_formatted(agents_epi)
        state_transition_by_day_mem = util.asizeof_formatted([props["state_transition_by_day"] for props in agents_epi.values() if "state_transition_by_day" in props])
        test_day_mem = util.asizeof_formatted([props["test_day"] for props in agents_epi.values() if "test_day" in props])
        test_result_day_mem = util.asizeof_formatted([props["test_result_day"] for props in agents_epi.values() if "test_result_day" in props])
        hosp_days_mem = util.asizeof_formatted([props["hospitalisation_days"] for props in agents_epi.values() if "hospitalisation_days" in props])
        quar_days_mem = util.asizeof_formatted([props["quarantine_days"] for props in agents_epi.values() if "quarantine_days" in props])
        vacc_days_mem = util.asizeof_formatted([props["vaccination_days"] for props in agents_epi.values() if "vaccination_days" in props])
        
        vars_util_mem = util.asizeof_formatted(vars_util)
        cat_mem = util.asizeof_formatted(vars_util.cells_agents_timesteps)
        seir_state_mem = util.asizeof_formatted(vars_util.agents_seir_state)
        seir_state_trans_for_day_mem = util.asizeof_formatted(vars_util.agents_seir_state_transition_for_day)
        inf_type_mem = util.asizeof_formatted(vars_util.agents_infection_type)
        inf_sev_mem = util.asizeof_formatted(vars_util.agents_infection_severity)
        vacc_doses_mem = util.asizeof_formatted(vars_util.agents_vaccination_doses)
        dir_con_mem = util.asizeof_formatted(vars_util.directcontacts_by_simcelltype_by_day)
        dir_con_idx1_mem = 0
        dir_con_idx2_mem = 0
        # dir_con_idx1_mem = util.asizeof_formatted(vars_util.dc_by_sct_by_day_agent1_index)
        # dir_con_idx2_mem = util.asizeof_formatted(vars_util.dc_by_sct_by_day_agent2_index)

        # it_agents_mem = round(sum([sys.getsizeof(d) for a in it_agents.values() for d in a.values()]) / (1024 * 1024), 2)
        # epi_agents_mem = round(sum([sys.getsizeof(i) for c in agents_epi.values() for i in c.values()]) / (1024 * 1024), 2)
        # cat_mem = round(sum([sys.getsizeof(i) for c in vars_util.cells_agents_timesteps.values() for i in c]) / (1024 * 1024), 2)
        # seir_state_mem = round(sum([sys.getsizeof(k) + sys.getsizeof(v) for k, v in vars_util.agents_seir_state.items()]) / (1024 * 1024), 2)
        # inf_type_mem = round(sum([sys.getsizeof(s) for s in vars_util.agents_infection_type.values()]) / (1024 * 1024), 2)
        # inf_sev_mem = round(sum([sys.getsizeof(s) for s in vars_util.agents_infection_severity.values()]) / (1024 * 1024), 2)
        # dir_con_mem = round(sum([sys.getsizeof(i) for i in vars_util.directcontacts_by_simcelltype_by_day]) / (1024 * 1024), 2)
        # dir_con_idx1_mem = round(sum([sys.getsizeof(i) for i in vars_util.dc_by_sct_by_day_agent1_index]) / (1024 * 1024), 2)
        # dir_con_idx2_mem = round(sum([sys.getsizeof(i) for i in vars_util.dc_by_sct_by_day_agent2_index]) / (1024 * 1024), 2)
        time_taken = time.time() - start

        print("memory footprint. it_agents: {0}, epi_agents: {1}, cat: {2}, seir_state: {3}, inf_type: {4}, inf_sev: {5}, dir_con: {6}, dir_con_idx1: {7}, dir_con_idx2: {8}, time_taken: {9}".format(str(it_agents_mem), str(epi_agents_mem), str(cat_mem), str(seir_state_mem), str(inf_type_mem), str(inf_sev_mem), str(dir_con_mem), str(dir_con_idx1_mem), str(dir_con_idx2_mem), str(time_taken)))
        if f is not None:
            f.flush()
        
        if sums is not None:
            general_sum, it_agents_sum, working_schedule_sum, itinerary_sum, itinerary_nextday_sum, ndar_sum, pd_ndar_sum, tourists_ids_sum, initial_tourist_sum, epi_agents_sum, state_transition_by_day_sum, test_day_sum, test_result_day_sum, hosp_days_sum, quar_days_sum, vacc_days_sum, vars_util_sum, cat_sum, seir_state_sum, seir_state_trans_for_day_sum, inf_type_sum, inf_sev_sum, vacc_doses_sum, dir_con_sum, dir_con_idx1_sum, dir_con_idx2_sum = sums
            general_sum += general_mem
            it_agents_sum += it_agents_mem
            working_schedule_sum += working_schedule_mem
            itinerary_sum += it_mem
            itinerary_nextday_sum += it_nd_mem
            ndar_sum += ndar_mem
            pd_ndar_sum += pd_ndar_mem
            tourists_ids_sum += tour_ids_mem
            initial_tourist_sum += initial_tourist_mem
            epi_agents_sum += epi_agents_mem
            state_transition_by_day_sum += state_transition_by_day_mem
            test_day_sum += test_day_mem
            test_result_day_sum += test_result_day_mem
            hosp_days_sum += hosp_days_mem
            quar_days_sum += quar_days_mem
            vacc_days_sum += vacc_days_mem
            vars_util_sum += vars_util_mem
            cat_sum += cat_mem
            seir_state_sum += seir_state_mem
            seir_state_trans_for_day_sum += seir_state_trans_for_day_mem
            inf_type_sum += inf_type_mem
            inf_sev_sum += inf_sev_mem
            dir_con_sum += dir_con_mem
            dir_con_idx1_sum += dir_con_idx1_mem
            dir_con_idx2_sum += dir_con_idx2_mem
            vacc_doses_sum += vacc_doses_mem

            sums = general_sum, it_agents_sum, working_schedule_sum, itinerary_sum, itinerary_nextday_sum, ndar_sum, pd_ndar_sum, tourists_ids_sum, initial_tourist_sum, epi_agents_sum, state_transition_by_day_sum, test_day_sum, test_result_day_sum, hosp_days_sum, quar_days_sum, vacc_days_sum, vars_util_sum, cat_sum, seir_state_sum, seir_state_trans_for_day_sum, inf_type_sum, inf_sev_sum, vacc_doses_sum, dir_con_sum, dir_con_idx1_sum, dir_con_idx2_sum

        if df is not None:
            start = time.time()
            general_avg = round(general_sum / day, 2)
            it_agents_avg = round(it_agents_sum / day, 2)
            working_schedule_avg = round(working_schedule_sum / day, 2)
            it_avg = round(itinerary_sum / day, 2)
            it_nd_avg = round(itinerary_nextday_sum / day, 2)
            ndar_avg = round(ndar_sum / day, 2)
            pd_ndar_avg = round(pd_ndar_sum / day, 2)
            tour_ids_avg = round(tourists_ids_sum / day, 2)
            initial_tourist_avg = round(initial_tourist_sum / day, 2)

            epi_agents_avg = round(epi_agents_sum / day, 2)
            state_transition_by_day_avg = round(state_transition_by_day_sum / day, 2)
            test_day_avg = round(test_day_sum / day, 2)
            test_result_day_avg = round(test_result_day_sum / day, 2)
            hosp_days_avg = round(hosp_days_sum / day, 2)
            quar_days_avg = round(quar_days_sum / day, 2)
            vacc_days_avg = round(vacc_days_sum / day, 2)

            vars_util_avg = round(vars_util_sum / day, 2)
            cat_avg = round(cat_sum / day, 2)
            seir_state_avg = round(seir_state_sum / day, 2)
            seir_state_trans_for_day_avg = round(seir_state_trans_for_day_sum / day, 2)
            inf_type_avg = round(inf_type_sum / day, 2)
            inf_sev_avg = round(inf_sev_sum / day, 2)
            vacc_doses_avg = round(vacc_doses_sum / day, 2)
            dir_con_avg = round(dir_con_sum / day, 2)
            dir_con_idx1_avg = round(dir_con_idx1_sum / day, 2)
            dir_con_idx2_avg = round(dir_con_idx2_sum / day, 2)

            df.loc[day, "general_day"] = general_mem
            df.loc[day, "general_avg"] = general_avg
            df.loc[day, "it_agents_day"] = it_agents_mem
            df.loc[day, "it_agents_avg"] = it_agents_avg
            df.loc[day, "working_schedule_day"] = working_schedule_mem
            df.loc[day, "working_schedule_avg"] = working_schedule_avg
            df.loc[day, "itinerary_day"] = it_mem
            df.loc[day, "itinerary_avg"] = it_avg
            df.loc[day, "itinerary_nextday_day"] = it_nd_mem
            df.loc[day, "itinerary_nextday_avg"] = it_nd_avg
            df.loc[day, "non_daily_activity_recurring_day"] = ndar_mem
            df.loc[day, "non_daily_activity_recurring_avg"] = ndar_avg
            df.loc[day, "prevday_non_daily_activity_recurring_day"] = pd_ndar_mem
            df.loc[day, "prevday_non_daily_activity_recurring_avg"] = pd_ndar_avg
            df.loc[day, "tourist_ids_day"] = tour_ids_mem
            df.loc[day, "tourist_ids_avg"] = tour_ids_avg
            df.loc[day, "initial_tourist_day"] = initial_tourist_mem
            df.loc[day, "initial_tourist_avg"] = initial_tourist_avg
            df.loc[day, "epi_agents_day"] = epi_agents_mem
            df.loc[day, "epi_agents_avg"] = epi_agents_avg
            df.loc[day, "state_transition_by_day_day"] = state_transition_by_day_mem
            df.loc[day, "state_transition_by_day_avg"] = state_transition_by_day_avg
            df.loc[day, "test_day_day"] = test_day_mem
            df.loc[day, "test_day_avg"] = test_day_avg
            df.loc[day, "test_result_day_day"] = test_result_day_mem
            df.loc[day, "test_result_day_avg"] = test_result_day_avg
            df.loc[day, "hospitalisation_days_day"] = hosp_days_mem
            df.loc[day, "hospitalisation_days_avg"] = hosp_days_avg
            df.loc[day, "quarantine_days_day"] = quar_days_mem
            df.loc[day, "quarantine_days_avg"] = quar_days_avg
            df.loc[day, "vaccination_days_day"] = vacc_days_mem
            df.loc[day, "vaccination_days_avg"] = vacc_days_avg
            df.loc[day, "vars_util_day"] = vars_util_mem
            df.loc[day, "vars_util_avg"] = vars_util_avg
            df.loc[day, "cells_agents_timesteps_day"] = cat_mem
            df.loc[day, "cells_agents_timesteps_avg"] = cat_avg
            df.loc[day, "seir_state_day"] = seir_state_mem
            df.loc[day, "seir_state_avg"] = seir_state_avg
            df.loc[day, "seir_state_trans_for_day_day"] = seir_state_trans_for_day_mem
            df.loc[day, "seir_state_trans_for_day_avg"] = seir_state_trans_for_day_avg
            df.loc[day, "infection_type_day"] = inf_type_mem
            df.loc[day, "infection_type_avg"] = inf_type_avg
            df.loc[day, "infection_severity_day"] = inf_sev_mem    
            df.loc[day, "infection_severity_avg"] = inf_sev_avg
            df.loc[day, "vacc_doses_day"] = vacc_doses_mem
            df.loc[day, "vacc_doses_avg"] = vacc_doses_avg
            df.loc[day, "direct_contacts_day"] = dir_con_mem
            df.loc[day, "direct_contacts_avg"] = dir_con_avg
            df.loc[day, "direct_contacts_index1_day"] = dir_con_idx1_mem
            df.loc[day, "direct_contacts_index1_avg"] = dir_con_idx1_avg
            df.loc[day, "direct_contacts_index2_day"] = dir_con_idx2_mem
            df.loc[day, "direct_contacts_index2_avg"] = dir_con_idx2_avg

            time_taken = time.time() - start

            print("memory footprint (averages). it_agents: {0}, epi_agents: {1}, cat: {2}, seir_state: {3}, inf_type: {4}, inf_sev: {5}, dir_con: {6}, dir_con_idx1: {7}, dir_con_idx2: {8}, time_taken: {9}".format(str(it_agents_avg), str(epi_agents_avg), str(cat_avg), str(seir_state_avg), str(inf_type_avg), str(inf_sev_avg), str(dir_con_avg), str(dir_con_idx1_avg), str(dir_con_idx2_avg), str(time_taken)))
            if f is not None:
                f.flush()

    return sums, df

if __name__ == '__main__':
    main()
