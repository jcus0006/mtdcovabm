import os
import sys
import time
import traceback
from dask.distributed import get_worker, get_client, as_completed
import customdict, vars, tourism, tourism_dist, util, itinerary, contactnetwork
from cellsclasses import CellType
from enum import Enum
import gc
import psutil

# the client will start each stage on each actor
# it will also maintain a flag to monitor progress (for each stage: itinerary, contact network)
# certain methods like contact tracing still happen locally (and direct contacts are still synced with the main node because of this)
class ActorDist:
    def __init__(self, params):
        self.worker = get_worker()
        self.client = get_client()

        self.remote_actors = [] # [actor_index, actor]

        workers_keys, workerindex, hh_insts, it_agents, agents_epi, vars_util, tourists, touristsgroups, touristsgroupsids_initial, dyn_params, tourismparams, rooms_by_accomid_by_accomtype, age_brackets, agent_ids_by_worker_lookup, cell_ids_by_worker_lookup, logsubfoldername, logfilename = params

        current_directory = os.getcwd()
        subfolder_name = logfilename.replace(".txt", "")
        log_subfolder_path = os.path.join(current_directory, logsubfoldername, subfolder_name)

        self.logfilename = logfilename
        self.logsubfoldername = logsubfoldername
        self.folder_name = log_subfolder_path

        self.workers_keys = workers_keys
        self.worker_index = workerindex
        self.worker_key = self.workers_keys[self.worker_index]
        self.agent_ids_by_worker_lookup = agent_ids_by_worker_lookup
        self.cell_ids_by_worker_lookup = cell_ids_by_worker_lookup
        self.agent_ids = set(self.agent_ids_by_worker_lookup[self.worker_index])
        self.cell_ids = set(self.cell_ids_by_worker_lookup[self.worker_index])

        self.day = None # 1 - 365
        self.weekday = None
        self.weekday_str = None
        self.simstage = None # 0 itinerary
        # self.workers_sync_flag = None
        # self.init_worker_flags()
        self.dyn_params = dyn_params
        self.tourismparams = tourismparams
        self.rooms_by_accomid_by_accomtype = rooms_by_accomid_by_accomtype
        self.age_brackets = age_brackets
        self.hh_insts = hh_insts
        self.it_agents = it_agents # this does not have to be shared with the remote nodes or returned to the client!
        self.agents_epi = agents_epi
        self.vars_util = vars_util # direct contacts do not have to be shared with the remote nodes, it can be simply returned to the client

        self.tourist_util = None # set in itineraries, cleared after sending result

        # self.tourists = tourists
        # self.touristsgroups = touristsgroups
        # method that loads whole tourists and touristsgroups remotely
        # tourists = self.worker.data["tourists"]
        # self.tourists = {tourid: tour for tourid, tour in tourists.items() if tourid in tourists_ids}
        # touristsgroups = self.worker.data["touristsgroups"]
        # self.touristsgroups = {tourgroupid: tourgroup for tourgroupid, tourgroup in touristsgroups.items() if tourgroupid in touristsgroups_ids}

        # self.worker.data["tourists"] = None
        # self.worker.data["touristsgroups"] = None

        self.tourists = tourists
        self.touristsgroups = touristsgroups
        self.touristsgroupsids_initial = touristsgroupsids_initial
        self.tourists_active_ids = []
        self.tourists_active_groupids = []
        self.tourists_arrivals_departures_for_day = {} # handles both incoming and outgoing, arrivals and departures. handled as a dict, as only represents day
        self.tourists_arrivals_departures_for_nextday = {}

        self.updated_agent_ids = [] # represents updated agent ids in CN (set in CN, cleared after sending result)

        self.it_main_time_taken_sum = 0
        self.it_main_time_taken_avg = 0
        self.cn_time_taken_sum = 0
        self.cn_time_taken_avg = 0

    def set_cpu_affinity(self):
        num_cores = psutil.cpu_count()
        core_index = num_cores - 1 - self.worker_index
        core_indices = []
        if core_index < 0:
            core_indices = [i for i in range(num_cores)]
        else:
            core_indices.append(core_index) # this would become inefficient, might consider throwing an error instead, i.e. would have more workers then available cores

        pid = os.getpid()
        p = psutil.Process(pid)
        p.cpu_affinity(core_indices)
        
    # by setting the remote actors, we enable each actor to communicate with all the other actors
    def set_remote_actors(self, actors):
        self.remote_actors = actors

    def close_pool(self):
        self.pool.close()
        self.pool.join()
        self.manager.shutdown()

    # tourists are handled in the main process (adding new tourists and tourism itinerary). 
    # the distributed stage starts immediately by syncing the cells_agents_timesteps created in the tourism itinerary, 
    # as well as syncing new (initial and arriving) tourists and deleting departing tourists
    # def tourists_sync(self, params):
    #     process_index, success, _, _, _ = tourism_dist.update_tourist_data_remote(params, self.folder_name)

    #     return success
    
    def itineraries(self, touristsgroupsdays_this_day, touristsgroupsdays_next_day):
        f = None
        original_stdout = sys.stdout
        stack_trace_log_file_name = ""
        time_takens = None
        cells_accommodation_to_send_back = None

        try:
            main_start = time.time()

            # self.set_cpu_affinity()

            self.simstage = SimStage.TouristItinerary

            stack_trace_log_file_name = os.path.join(self.folder_name, "it_actor_stack_trace_" + str(self.day) + "_" + str(self.worker_index) + ".txt")

            touristsgroupsdays = customdict.CustomDict()
            touristsgroupsdays[self.day] = touristsgroupsdays_this_day

            if self.day + 1 <= 365:
                touristsgroupsdays[self.day + 1] = touristsgroupsdays_next_day

            log_file_name = os.path.join(self.folder_name, "it_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
            f = open(log_file_name, "w")
            sys.stdout = f

            # load worker data
            if self.worker is None:
                raise TypeError("Worker is none")
            
            if self.worker.data is None or len(self.worker.data) == 0:
                raise TypeError("Worker.data is None or empty")
            else:
                if self.worker.data["itineraryparams"] is None or len(self.worker.data["itineraryparams"]) == 0:
                    raise TypeError("Worker.data['itineraryparams'] is None or empty")

            agents_ids_by_ages = self.worker.data["agents_ids_by_ages"]
            timestepmins = self.worker.data["timestepmins"]
            n_locals = self.worker.data["n_locals"]
            n_tourists = self.worker.data["n_tourists"]
            locals_ratio_to_full_pop = self.worker.data["locals_ratio_to_full_pop"]

            itineraryparams = self.worker.data["itineraryparams"]
            epidemiologyparams = self.worker.data["epidemiologyparams"]
            cells_industries_by_indid_by_wpid = self.worker.data["cells_industries_by_indid_by_wpid"] 
            cells_restaurants = self.worker.data["cells_restaurants"] 
            cells_hospital = self.worker.data["cells_hospital"] 
            cells_testinghub = self.worker.data["cells_testinghub"] 
            cells_vaccinationhub = self.worker.data["cells_vaccinationhub"] 
            cells_entertainment_by_activityid = self.worker.data["cells_entertainment_by_activityid"] 
            cells_religious = self.worker.data["cells_religious"] 
            cells_households = self.worker.data["cells_households"] 
            cells_breakfast_by_accomid = self.worker.data["cells_breakfast_by_accomid"] 
            cells_airport = self.worker.data["cells_airport"] 
            cells_transport = self.worker.data["cells_transport"] 
            cells_institutions = self.worker.data["cells_institutions"] 
            cells_accommodation = self.worker.data["cells_accommodation"] 
            agents_static = self.worker.data["agents_static"]

            tour_start = time.time()
            contactnetworkparams = self.worker.data["contactnetworkparams"]
            sociability_rate_min_max = contactnetworkparams["sociabilityrateminmax"]
            sociability_rate_min, sociability_rate_max = sociability_rate_min_max[0], sociability_rate_min_max[1]
            powerlaw_distribution_parameters = contactnetworkparams["powerlawdistributionparameters"]

            initial_seir_state_distribution = epidemiologyparams["initialseirstatedistribution"]

            if self.day == 1:
                self.tourist_util = tourism.Tourism(self.tourismparams, 
                                                cells_accommodation,
                                                n_locals, 
                                                self.tourists, 
                                                agents_static, 
                                                self.it_agents, 
                                                self.agents_epi, 
                                                self.vars_util, 
                                                touristsgroupsdays, 
                                                self.touristsgroups, 
                                                self.rooms_by_accomid_by_accomtype, 
                                                self.tourists_arrivals_departures_for_day, 
                                                self.tourists_arrivals_departures_for_nextday, 
                                                self.tourists_active_groupids, 
                                                self.tourists_active_ids, 
                                                self.age_brackets, 
                                                powerlaw_distribution_parameters, 
                                                False, # visualise 
                                                sociability_rate_min, 
                                                sociability_rate_max, 
                                                0, 
                                                initial_seir_state_distribution, 
                                                True) # dask_full_stateful
                
                self.tourist_util.sample_initial_tourists(self.touristsgroupsids_initial, f)
                self.touristsgroupsids_initial = None
                
                gc.collect()
            else:
                self.tourist_util.touristsgroupsdays = touristsgroupsdays
                    
            self.it_agents, self.agents_epi, self.tourists, cells_accommodation, self.tourists_arrivals_departures_for_day, self.tourists_arrivals_departures_for_nextday, self.tourists_active_groupids = self.tourist_util.initialize_foreign_arrivals_departures_for_day(self.day, self.dyn_params.airport_lockdown, f)
            print("initialize_foreign_arrivals_departures_for_day (done) for simday " + str(self.day) + ", weekday " + str(self.weekday))

            if f is not None:
                f.flush()

            cells_accommodation_to_send_back = customdict.CustomDict()
            for cellid in self.tourist_util.updated_cell_ids:
                cells_accommodation_to_send_back[cellid] = cells_accommodation[cellid]

            self.worker.data["cells_accommodation"] = cells_accommodation

            itinerary_util = itinerary.Itinerary(itineraryparams, 
                                                timestepmins, 
                                                n_locals, 
                                                n_tourists,
                                                locals_ratio_to_full_pop,
                                                agents_static,
                                                self.it_agents, 
                                                self.agents_epi, 
                                                agents_ids_by_ages,
                                                self.vars_util,
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
                                                self.dyn_params,
                                                self.tourists)

            itinerary_util.generate_tourist_itinerary(self.day, self.weekday, self.touristsgroups, self.tourists_active_groupids, self.tourists_arrivals_departures_for_day, self.tourists_arrivals_departures_for_nextday, log_file_name, f)
            print("generate_tourist_itinerary (done) for simday " + str(self.day) + ", weekday " + str(self.weekday))
            if f is not None:
                f.flush()    

            self.tourist_util.sync_and_clean_tourist_data(self.day, self.client, self.remote_actors, self.logsubfoldername, self.logfilename, True, self.dyn_params.airport_lockdown, f)
            
            print("sync_and_clean_tourist_data (done) for simday " + str(self.day) + ", weekday " + str(self.weekday))
            if f is not None:
                f.flush()

            # num_arrivals, num_arrivals_nextday, num_departures = 0, 0, 0
            num_active = len(self.tourist_util.tourists_active_ids)
            num_arrivals = len(self.tourist_util.arriving_tourists_agents_ids)
            num_arrivals_nextday = len(self.tourist_util.arriving_tourists_next_day_agents_ids)
            num_departures = len(self.tourist_util.departing_tourists_agents_ids[self.day])

            arr_dep_counts = (num_active, num_arrivals, num_arrivals_nextday, num_departures)

            tour_time_taken = time.time() - tour_start
            print("tourism for simday " + str(self.day) + ", weekday " + str(self.weekday) + ", time taken: " + str(tour_time_taken))
            if f is not None:
                f.flush()

            # must return partial part of cells_accommodation, relevant to current tourist set, back to main process, so it can be used with main process

            self.simstage = SimStage.Itinerary

            num_agents_working_schedule = 0
            working_schedule_times_by_resid = {}

            ws_time_taken = 0
            if self.day == 1 or self.weekday_str == "Monday":
                # print("generate_working_days_for_week_residence for simday " + str(day) + ", weekday " + str(weekday))
                ws_main_start = time.time()
                for hh_inst in self.hh_insts:
                    ws_start = time.time()
                    # print("day " + str(day) + ", res id: " + str(hh_inst["id"]) + ", is_hh: " + str(hh_inst["is_hh"]))
                    itinerary_util.generate_working_days_for_week_residence(hh_inst["resident_uids"], hh_inst["is_hh"])
                    ws_time_taken = time.time() - ws_start
                    working_schedule_times_by_resid[hh_inst["id"]] = ws_time_taken
                    num_agents_working_schedule += len(hh_inst["resident_uids"])

                ws_time_taken = time.time() - ws_main_start
                print("generate_working_days_for_week_residence for simday " + str(self.day) + ", weekday " + str(self.weekday) + ", time taken: " + str(ws_time_taken) + ", proc index: " + str(self.worker_index))
                if f is not None:
                    f.flush()

            print("generate_itinerary_hh for simday " + str(self.day) + ", weekday " + str(self.weekday))
            if f is not None:
                f.flush()

            it_start = time.time()                    

            num_agents_itinerary = 0
            itinerary_times_by_resid = {}
            for hh_inst in self.hh_insts:
                res_start = time.time()
                itinerary_util.generate_local_itinerary(self.day, self.weekday, hh_inst["resident_uids"])
                res_time_taken = time.time() - res_start
                itinerary_times_by_resid[hh_inst["id"]] = res_time_taken
                num_agents_itinerary += len(hh_inst["resident_uids"])

            it_time_taken = time.time() - it_start

            main_time_taken = time.time() - main_start

            self.it_main_time_taken_sum += main_time_taken
            self.it_main_time_taken_avg = self.it_main_time_taken_sum / self.day
            print("generate_itinerary_hh for simday " + str(self.day) + ", weekday " + str(self.weekday) + ", time taken: " + str(it_time_taken) + ", proc index: " + str(self.worker_index))

            # return contact_tracing_agent_ids (that are only added to in this context) and time-logging information to client

            util.log_memory_usage(f, "End of itineraries ")

            time_takens = (main_time_taken, tour_time_taken, ws_time_taken, it_time_taken, self.it_main_time_taken_avg)

            print("process " + str(self.worker_index) + ", ended at " + str(time.time()) + ", full time taken: " + str(main_time_taken))
            if f is not None:
                f.flush()

            return self.worker_index, cells_accommodation_to_send_back, arr_dep_counts, self.vars_util.contact_tracing_agent_ids, time_takens
        except Exception as e:
            with open(stack_trace_log_file_name, 'w') as fi:
                traceback.print_exc(file=fi)

            raise
        finally:
            gc.collect()       

            if f is not None:
                # Close the file
                f.close()

            if original_stdout is not None:
                sys.stdout = original_stdout

    def contact_network(self):
        f = None
        original_stdout = sys.stdout
        stack_trace_log_file_name = ""

        try:
            main_start = time.time()

            # self.set_cpu_affinity()

            self.simstage = SimStage.ContactNetwork

            stack_trace_log_file_name = os.path.join(self.folder_name, "cn_actor_stack_trace_" + str(self.day) + "_" + str(self.worker_index) + ".txt")

            log_file_name = os.path.join(self.folder_name, "cn_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
            f = open(log_file_name, "w")
            sys.stdout = f

            print(f"starting contact network on {len(self.vars_util.cells_agents_timesteps)} cells")
            if f is not None:
                f.flush()

            n_locals = self.worker.data["n_locals"]
            n_tourists = self.worker.data["n_tourists"]
            locals_ratio_to_full_pop = self.worker.data["locals_ratio_to_full_pop"]
            contactnetworkparams = self.worker.data["contactnetworkparams"]
            epidemiologyparams = self.worker.data["epidemiologyparams"]
            cells_type = self.worker.data["cells_type"]
            indids_by_cellid = self.worker.data["indids_by_cellid"]
            cells_households = self.worker.data["cells_households"] 
            cells_institutions = self.worker.data["cells_institutions"] 
            cells_accommodation = self.worker.data["cells_accommodation"] 
            agents_static = self.worker.data["agents_static"]

            # print("process " + str(process_index) + " started at " + str(start))

            contact_network_util = contactnetwork.ContactNetwork(n_locals, 
                                                                n_tourists, 
                                                                locals_ratio_to_full_pop, 
                                                                agents_static,
                                                                self.agents_epi,
                                                                self.vars_util,
                                                                cells_type,
                                                                indids_by_cellid,
                                                                cells_households, 
                                                                cells_institutions, 
                                                                cells_accommodation, 
                                                                contactnetworkparams,
                                                                epidemiologyparams, 
                                                                self.dyn_params, 
                                                                process_index=self.worker_index)

            _, self.updated_agent_ids, _, self.vars_util = contact_network_util.simulate_contact_network(self.day, self.weekday)
            
            # certain data does not have to go back because it would not have been updated in this context
            self.vars_util.cells_agents_timesteps = customdict.CustomDict()
            self.vars_util.agents_seir_state_transition_for_day = customdict.CustomDict()
            self.vars_util.agents_vaccination_doses = customdict.CustomDict()
            
            main_time_taken = time.time() - main_start
            
            # return direct contacts and time-logging information to client

            util.log_memory_usage(f, "End of contact networks ")

            return self.worker_index, self.vars_util.directcontacts_by_simcelltype_by_day, main_time_taken
        except Exception as e:
            with open(stack_trace_log_file_name, 'w') as fi:
                traceback.print_exc(file=fi)

            raise
        finally:
            gc.collect()       

            if f is not None:
                # Close the file
                f.close()

            if original_stdout is not None:
                sys.stdout = original_stdout


    def send_results(self):
        # if itinerary, get list of cell ids from cells_agents_timesteps, then get list of cell ids that are not in self.cells_ids (difference with sets)
        # these are the cases for which the contact network will be computed on another remote node
        # iterate on the workers, skipping this one, build a dict of the following structure: {worker_index: (agents_epi, vars_util)} 
        # in vars_util include cells_agents_timesteps that are relevant to each worker and all agent information for agents that show up in those cells_agents_timesteps
        # note any properties that are not relevant for the itinerary and don't include them
        
        # if not is_itinerary_result, would be contact network. get list of contacts that include agents that are not in self.agents_ids (consider BST Search)
        # the same applies, in terms of the structure: {worker_index: (agents_epi, vars_util)}
        # do not include direct contacts, these will be returned directly to the client and synced accordingly (contact tracing happens on the client anyway)
        f = None
        original_stdout = sys.stdout
        stack_trace_log_file_name = ""

        try:        
            send_results_by_worker_id = customdict.CustomDict()

            if self.simstage == SimStage.Itinerary:
                stack_trace_log_file_name = os.path.join(self.folder_name, "sendres_it_actor_stack_trace_" + str(self.day) + "_" + str(self.worker_index) + ".txt")

                log_file_name = os.path.join(self.folder_name, "sendres_it_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
                f = open(log_file_name, "w")
                sys.stdout = f

                print("sending results for itinerary")
                if f is not None:
                    f.flush()

                cells_ids = set(self.vars_util.cells_agents_timesteps.keys()) # convert to set to enable set functions
                cells_ids = cells_ids.difference(self.cell_ids) # get the newly created cell ids that are not local to this worker
                
                for wi, w_cell_ids in self.cell_ids_by_worker_lookup.items():
                    if wi != self.worker_index:
                        cells_ids_to_send = cells_ids.intersection(w_cell_ids) # get the matching cell ids to send to this worker specifically

                        agents_ids_to_send = set()
                        agents_epi_to_send = customdict.CustomDict()
                        vars_util_to_send = vars.Vars()

                        for cell_id in cells_ids_to_send:
                            cats_by_cell_id = self.vars_util.cells_agents_timesteps[cell_id]
                            vars_util_to_send.cells_agents_timesteps[cell_id] = cats_by_cell_id

                            for agent_id, _, _ in cats_by_cell_id:
                                agents_ids_to_send.add(agent_id)

                        # print(f"agents_ids_to_send: {agents_ids_to_send}")
                        # if f is not None:
                        #     f.flush()

                        agents_epi_to_send, vars_util_to_send = util.split_agents_epi_by_agentsids(agents_ids_to_send, self.agents_epi, self.vars_util, agents_epi_to_send, vars_util_to_send)

                        send_results_by_worker_id[wi] = [agents_epi_to_send, vars_util_to_send, self.tourist_util.agents_static_to_sync, self.tourist_util.prev_day_departing_tourists_agents_ids, self.tourist_util.prev_day_departing_tourists_group_ids]
                        # send_results_by_worker_id[wi] = [agents_epi_to_send, vars_util_to_send]

                self.clean_cells_agents_timesteps(cells_ids)

                print("cleaning cells_agents_timesteps")
                if f is not None:
                    f.flush()
            else:
                stack_trace_log_file_name = os.path.join(self.folder_name, "sendres_cn_actor_stack_trace_" + str(self.day) + "_" + str(self.worker_index) + ".txt")

                log_file_name = os.path.join(self.folder_name, "sendres_cn_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
                f = open(log_file_name, "w")
                sys.stdout = f
            
                print("sending results for contact network. updated_agents_ids len: " + str(len(self.updated_agent_ids)))
                if f is not None:
                    f.flush()

                self.updated_agent_ids = set(self.updated_agent_ids) # convert to set to enable set functions

                self.updated_agent_ids = self.updated_agent_ids.difference(self.agent_ids) # get the updated agent ids that are not local to this worker

                for wi, w_agent_ids in self.agent_ids_by_worker_lookup.items():
                    if wi != self.worker_index:
                        agents_ids_to_send = self.updated_agent_ids.intersection(w_agent_ids) # get the matching agent ids to send to this worker specifically
                        # print(f"agents_ids_to_send: {agents_ids_to_send}")
                        # if f is not None:
                        #     f.flush()

                        agents_epi_to_send = customdict.CustomDict()
                        vars_util_to_send = vars.Vars()

                        agents_epi_to_send, vars_util_to_send = util.split_agents_epi_by_agentsids(agents_ids_to_send, self.agents_epi, self.vars_util, agents_epi_to_send, vars_util_to_send)

                        send_results_by_worker_id[wi] = [agents_epi_to_send, vars_util_to_send]
            
            results = []
            for wi in range(len(self.workers_keys)):
                if wi != self.worker_index:
                    print(f"sending results to actor {wi}")
                    if f is not None:
                        f.flush()
                    data = send_results_by_worker_id[wi]
                    params = (self.worker_index, self.simstage, data)
                    result = self.remote_actors[wi].receive_results(params)
                    results.append(result)
            
            send_results_by_worker_id = None

            result_index = 0
            success = True
            for result in results:
                success &= result.result()
                print("Message Result {0}: {1}".format(str(result_index), str(success)))
                if f is not None:
                    f.flush()

                result_index += 1

            self.clean_tourists_agents_static_to_sync()

            print("cleaning tourist_util.agents_static_to_sync")
            if f is not None:
                f.flush()
                
            util.log_memory_usage(f, "End of send_results ")

            if self.simstage == SimStage.Itinerary:
                return success
            elif self.simstage == SimStage.ContactNetwork:
                self.updated_agent_ids = []
                return success, self.agents_epi
        except Exception as e:
            with open(stack_trace_log_file_name, 'w') as fi:
                traceback.print_exc(file=fi)

            raise
        finally:
            gc.collect()       

            if f is not None:
                # Close the file
                f.close()

            if original_stdout is not None:
                sys.stdout = original_stdout
    
    def receive_results(self, params):
        f = None
        original_stdout = sys.stdout
        stack_trace_log_file_name = ""
    
        try:       
            sender_worker_index, simstage, data = params

            n_locals = self.worker.data["n_locals"]
            # sync results after itinerary or contact network

            ss = ""
            if simstage == SimStage.Itinerary:
                ss = "it"
            elif simstage == SimStage.ContactNetwork:
                ss = "cn"
            elif simstage == SimStage.EndOfDaySync:
                ss = "sync"

            stack_trace_log_file_name = os.path.join(self.folder_name, "recvres_" + ss + "_actor_stack_trace_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
            
            log_file_name = os.path.join(self.folder_name, "recvres_" + ss + "_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
            f = open(log_file_name, "w")
            sys.stdout = f

            if simstage == SimStage.Itinerary:
                print("receiving results for itinerary")
                if f is not None:
                    f.flush()
                agents_epi_partial, vars_util_partial, agents_static_to_sync, departed_tourist_agent_ids, departed_tourist_group_ids = data
                start = time.time()
                util.sync_state_info_cells_agents_timesteps(self.vars_util, vars_util_partial)
                time_taken = time.time() - start
                print("sync_state_info cells_agents_timesteps time_taken: " + str(time_taken))
                if f is not None:
                    f.flush()
                start = time.time()
                # print(f"agents {agents_epi_partial.keys()}")
                # if f is not None:
                #     f.flush()
                self.agents_epi, self.vars_util = util.sync_state_info_by_agentsids_agents_epi(agents_epi_partial.keys(), self.agents_epi, self.vars_util, agents_epi_partial, vars_util_partial)
                time_taken = time.time() - start
                print("sync state info agents_epi time_taken: " + str(time_taken))
                if f is not None:
                    f.flush()
                start = time.time()
                params = [self.day, agents_static_to_sync, departed_tourist_agent_ids, departed_tourist_group_ids, sender_worker_index]
                tourism_dist.update_tourist_data_remote(params, self.folder_name, self.worker, self_actor_index=self.worker_index, tourists=self.tourists, touristsgroups=self.touristsgroups, n_locals=n_locals)
                time_taken = time.time() - start
                print("sync state info update_tourist_data_remote time_taken: " + str(time_taken))
                if f is not None:
                    f.flush()
            elif simstage == SimStage.ContactNetwork:
                print("receiving results for contact network")
                if f is not None:
                    f.flush()
                agents_epi_partial, vars_util_partial = data
                start = time.time()
                # print(f"agents {agents_epi_partial.keys()}")
                # if f is not None:
                #     f.flush()
                self.agents_epi, self.vars_util = util.sync_state_info_by_agentsids_agents_epi(agents_epi_partial.keys(), self.agents_epi, self.vars_util, agents_epi_partial, vars_util_partial)
                time_taken = time.time() - start
                print("sync state info agents_epi time_taken: " + str(time_taken))
                if f is not None:
                    f.flush()
            elif simstage == SimStage.EndOfDaySync:
                print("receiving results for end of day sync")
                if f is not None:
                    f.flush()
                start = time.time()
                agents_epi_partial, vars_util_partial = data
                # print(f"agents {agents_epi_partial.keys()}")
                # if f is not None:
                #     f.flush()
                self.agents_epi, self.vars_util = util.sync_state_info_by_agentsids_agents_epi_end_of_day_sync(agents_epi_partial.keys(), self.agents_epi, self.vars_util, agents_epi_partial, vars_util_partial)
                time_taken = time.time() - start
                print("sync state info (end of day) agents_epi time_taken: " + str(time_taken))
                if f is not None:
                    f.flush()
            
            util.log_memory_usage(f, "End of receive_results ")

            return True
        except Exception as e:
            with open(stack_trace_log_file_name, 'w') as fi:
                traceback.print_exc(file=fi)

            raise
        finally:
            if f is not None:
                # Close the file
                f.close()

            if original_stdout is not None:
                sys.stdout = original_stdout

    # def init_worker_flags(self):
    #     self.workers_sync_flag = customdict.CustomDict({
    #         i:False for i in range(len(self.workers_keys)) if i != self.worker_index
    #     })
    
    def reset_day(self, new_day, new_weekday, new_weekday_str, new_dyn_params):
        self.day = new_day
        self.weekday = new_weekday
        self.weekday_str = new_weekday_str
        self.dyn_params = new_dyn_params

        return True

    # clean cells_agents_timesteps, otherwise this worker will compute cells that do not reside on this worker by default (called after sending cat info to default nodes)
    def clean_cells_agents_timesteps(self, keys_to_del):
        for key in keys_to_del:
            del self.vars_util.cells_agents_timesteps[key]

    def clean_tourists_agents_static_to_sync(self):
        self.tourist_util.agents_static_to_sync = customdict.CustomDict()

    # removes any data that does not reside on this worker by default (called at the end of the simulation day)
    def clean_up_and_calculate_seir_states_daily(self):
        # n_locals = self.worker.data["n_locals"]

        f = None
        original_stdout = sys.stdout

        stack_trace_log_file_name = ""
        stack_trace_log_file_name = os.path.join(self.folder_name, "clean_actor_stack_trace_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
        
        # log_file_name = os.path.join(self.folder_name, "clean_actor_" + str(self.day) + "_" + str(self.worker_index) + ".txt")
        # f = open(log_file_name, "w")
        # sys.stdout = f

        self.vars_util.reset_daily_structures()

        for id in list(self.agents_epi.keys()): # can try BST search and compare times
            if id not in self.agent_ids: # if id < n_locals and id not in self.agent_ids: // id not in self.agent_ids
                del self.agents_epi[id]

                try:
                    del self.it_agents[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_seir_state[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_infection_type[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_infection_severity[id]
                except:
                    pass

                try:
                    del self.vars_util.agents_vaccination_doses[id]
                except:
                    pass

        seir_states = self.dyn_params.statistics.calculate_seir_states_counts(self.vars_util)

        # gc.collect()
        # if f is not None:
        #     # Close the file
        #     f.close()

        # if original_stdout is not None:
        #     sys.stdout = original_stdout

        return seir_states

class SimStage(Enum):
    TouristSync = 0
    TouristItinerary = 1
    Itinerary = 2
    ContactNetwork = 3
    EndOfDaySync = 4