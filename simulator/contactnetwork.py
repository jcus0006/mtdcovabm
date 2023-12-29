import numpy as np
import math
import time
from copy import copy
from copy import deepcopy
import util
from cellsclasses import CellType, SimCellType
from epidemiology import Epidemiology
from agents_epi import AgentsEpi
import matplotlib.pyplot as plt

class ContactNetwork:
    def __init__(self, 
                n_locals, 
                n_tourists, 
                locals_ratio_to_full_pop, 
                agents_static,
                agents_epi_util, 
                vars_util, 
                cells_type, 
                indids_by_cellid,
                cells_households, 
                cells_institutions, 
                cells_accommodation, 
                contactnetworkparams, 
                epidemiologyparams, 
                dynparams,
                contact_network_sum_time_taken=0, 
                visualise=False, 
                maintain_directcontacts_count=False, 
                process_index=-1):
        self.agents_static = agents_static
        self.agents_epi_util = agents_epi_util

        self.cells_type = cells_type
        self.indids_by_cellid = indids_by_cellid

        # self.mp_cells_keys = []

        self.vars_util = vars_util

        self.contactnetworkparams = contactnetworkparams
        self.ageactivitycontactmatrix = np.array(self.contactnetworkparams["ageactivitycontactmatrix"])

        self.visualise = visualise
        self.maintain_directcontacts_count = maintain_directcontacts_count
        self.figurecount = 0

        self.contactnetwork_sum_time_taken = contact_network_sum_time_taken

        self.process_index = process_index
        # self.sync_queue = sync_queue

        self.population_per_timestep = [0 for i in range(144)]

        self.agents_seir_indices = None
        # self.agents_seir_indices = {agentid:idx for idx, agentid in enumerate(self.agents_dynamic.keys())}

        # print("agents_seir_indices values: " + str(self.agents_seir_indices.keys()))
        
        # it is possible that this may need to be extracted out of the contact network and handled at the next step
        # because it could be impossible to parallelise otherwise
        self.epi_util = Epidemiology(epidemiologyparams, n_locals, n_tourists, locals_ratio_to_full_pop, agents_static, agents_epi_util, vars_util, cells_households, cells_institutions, cells_accommodation, dynparams, process_index, self.agents_seir_indices)

    # full day, all cells context
    def simulate_contact_network(self, day, weekday):
        if self.process_index != -1:        
            agents_directcontacts_by_simcelltype_by_day  = []
        else:
            agents_directcontacts_by_simcelltype_by_day = self.vars_util.directcontacts_by_simcelltype_by_day

        updated_agents_ids = []
        # if self.process_index >= 0:
        #     sp_cells_keys = self.mp_cells_keys[self.process_index]
        # else:
        #     sp_cells_keys = list(self.vars_util.cells_agents_timesteps.keys()) # single process without multi-processing

        print("generate contact network for " + str(len(self.vars_util.cells_agents_timesteps)) + " cells on process: " + str(self.process_index))
        start = time.time()
        for cellindex, cellid in enumerate(self.vars_util.cells_agents_timesteps.keys()):
            cell_updated_agents_ids, cell_agents_directcontacts, cell_type = self.simulate_contact_network_by_cellid(cellid, day)

            updated_agents_ids.extend(cell_updated_agents_ids)

            if len(cell_agents_directcontacts) > 0:
                # cell_type = cell["type"]

                sim_cell_type = util.convert_celltype_to_simcelltype(cellid, celltype=cell_type)

                # if sim_cell_type not in agents_directcontacts_by_simcelltype_thisday:
                #     agents_directcontacts_by_simcelltype_thisday[sim_cell_type] = set()

                # agents_directcontacts_thissimcelltype_thisday = agents_directcontacts_by_simcelltype_thisday[sim_cell_type]

                # agents_directcontacts_thissimcelltype_thisday += list(cell_agents_directcontacts.keys())

                current_index = len(agents_directcontacts_by_simcelltype_by_day)

                for key in cell_agents_directcontacts.keys():
                    contact_pair_timesteps = cell_agents_directcontacts[key]

                    min_start_ts, max_end_ts = contact_pair_timesteps[0][0], contact_pair_timesteps[len(contact_pair_timesteps) - 1][1]

                    agent1_id, agent2_id = key[0], key[1]
                        
                    agents_directcontacts_by_simcelltype_by_day.append([sim_cell_type, agent1_id, agent2_id, min_start_ts, max_end_ts])
                    # agents_directcontacts_thissimcelltype_thisday.add((key, (min_start_ts, max_end_ts)))

                    current_index += 1

                    if self.process_index == -1:
                        self.vars_util.dc_by_sct_by_day_agent1_index.append([agent1_id, current_index])
                        self.vars_util.dc_by_sct_by_day_agent2_index.append([agent2_id, current_index])

        time_taken = time.time() - start
        self.contactnetwork_sum_time_taken += time_taken
        avg_time_taken = self.contactnetwork_sum_time_taken / day
        print("simulate_contact_network for simday " + str(day) + ", weekday " + str(weekday) + ", time taken: " + str(time_taken) + ", avg time taken: " + str(avg_time_taken) + ", process index: " + str(self.process_index))

        agents_epi_util_partial = AgentsEpi()
        agents_epi_util_partial = self.agents_epi_util.partialize(day, updated_agents_ids, agents_epi_util_partial)
        # agents_partial = {agentid:self.agents_epi_util[agentid] for agentid in updated_agents_ids}

        if self.process_index != -1: # if -1 would already be assigned above
            self.vars_util.directcontacts_by_simcelltype_by_day = agents_directcontacts_by_simcelltype_by_day

        return self.process_index, updated_agents_ids, agents_epi_util_partial, self.vars_util
    
    # full day, single cell context
    def simulate_contact_network_by_cellid(self, cellid, day):
        agents_directcontacts, cell_type = self.generate_contact_network(cellid)

        updated_agents_ids = self.epi_util.simulate_direct_contacts(agents_directcontacts, cellid, cell_type, day)

        return updated_agents_ids, agents_directcontacts, cell_type

    def generate_contact_network(self, cellid):
        # print("generating contact network for cell " + str(cellid))
        agents_ids = [] # each index represents an agent ID
        agents_degrees = [] # each index represents an agent degree, maps to agents_ids indices
        agents_total_timesteps = {} # {agentid : total_num_of_timesteps}
        agents_potentialcontacts_count = {} # {agentid: total_num_of_potential_contacts}
        agents_potentialcontacts = {} # {(agentid1, agentid2) : [ (start_ts1, end_ts1), (start_ts2, end_ts2) ]}
        agents_directcontacts = {} # {(agentid1, agentid2) : [ (start_ts1, end_ts1), (start_ts2, end_ts2) ]}
        agents_directcontacts_count = {} # {agentid: contact_count}

        # cell = None
        # if cellid in self.vars_util.cells_agents_timesteps:
        cell_type = self.cells_type[cellid]

        cell_agents_timesteps = self.vars_util.cells_agents_timesteps[cellid]

        indid = None

        if cell_type == CellType.Workplace:
            if cellid in self.indids_by_cellid: # "indid" in cell["place"]
                indid = self.indids_by_cellid[cellid]

        ageactivitycontact_cm_activityid = self.convert_celltype_to_ageactivitycontactmatrixtype(cellid, cell_type, indid)
        
        agents_timesteps_sum = 0

        for agentid, starttimestep, endtimestep in cell_agents_timesteps:
            if agentid not in agents_ids:
                agents_ids.append(agentid)
                agents_degrees.append(0)

            if agentid not in agents_total_timesteps:
                agents_total_timesteps[agentid] = 0

            for timestep in range(starttimestep, endtimestep+1):
                self.population_per_timestep[timestep] += 1

            total_timesteps = len(range(starttimestep, endtimestep+1))
            agents_timesteps_sum += total_timesteps

            agents_total_timesteps[agentid] += total_timesteps
        
            # pre computation of potential contacts
            for ag_id, st_ts, end_ts in cell_agents_timesteps:
                if ag_id != agentid and not self.pair_already_computed_in_agentspotentialcontacts(agents_potentialcontacts, agentid, ag_id):
                    overlapping_range = self.get_overlapping_range(starttimestep, endtimestep, st_ts, end_ts)

                    if agentid not in agents_potentialcontacts_count:
                        agents_potentialcontacts_count[agentid] = 0

                    if ag_id not in agents_potentialcontacts_count:
                        agents_potentialcontacts_count[ag_id] = 0
                    
                    if overlapping_range is not None:
                        if not self.pair_exists_in_agentspotentialcontacts(agents_potentialcontacts, agentid, ag_id):
                            agents_potentialcontacts_count[agentid] += 1
                            agents_potentialcontacts_count[ag_id] += 1

                        pair_key = (agentid, ag_id)
                        if pair_key not in agents_potentialcontacts:
                            agents_potentialcontacts[pair_key] = []

                        agents_potentialcontacts[pair_key].append(overlapping_range)

        if len(agents_potentialcontacts) > 0:
            if self.visualise and len(agents_ids) > 500:
                self.figurecount += 1
                plt.figure(self.figurecount)
                plt.hist(agents_potentialcontacts_count.values(), bins=10)
                plt.xlabel("Potential Contacts")
                plt.ylabel("Count")
                plt.show(block=False)
        
            # self.population_per_timestep_no_zeros = np.array([pop_per_ts for pop_per_ts in self.population_per_timestep if pop_per_ts > 0])

            # n = round(np.median(self.population_per_timestep_no_zeros)) # max/median population during the day (if using median remove 0s as they would be outliers and take middle value)      
            n = len(agents_ids)

            avg_contacts_activity_agegroups_col = np.array(self.ageactivitycontactmatrix[:, 2 + ageactivitycontact_cm_activityid]) # start from 4th column onwards, ageactivitycontact_cm_activityid min = 1

            k = round(np.mean(avg_contacts_activity_agegroups_col)) # average for activity across all age groups

            max_degree = round((k * n)) # should this be k * n / 2 ?

            avg_agents_timestep_counts = agents_timesteps_sum / n

            agents_potentialcontacts_count_np = np.array(list(agents_potentialcontacts_count.values()))
            avg_potential_contacts_count = np.mean(agents_potentialcontacts_count_np)
            # std_potential_contacts_count = np.std(agents_potentialcontacts_count_np)
            
            # create degrees per agent
            for agentindex, agentid in enumerate(agents_ids):
                agent_potentialcontacts_count = agents_potentialcontacts_count[agentid]

                if agent_potentialcontacts_count > 0:
                    # agent = self.agents_dynamic[agentid]

                    agent_timestep_count = agents_total_timesteps[agentid]

                    age_bracket_index = self.agents_static.get(agentid, "age_bracket_index")

                    avg_contacts_by_age_activity = self.ageactivitycontactmatrix[age_bracket_index, 2 + ageactivitycontact_cm_activityid]

                    timestep_multiplier = 1.0
                    
                    if agent_timestep_count != avg_agents_timestep_counts and avg_agents_timestep_counts > 1:
                        timestep_multiplier = math.log(agent_timestep_count, avg_agents_timestep_counts)

                    potential_contacts_count_multiplier = 1.0
                    
                    if agent_potentialcontacts_count != avg_potential_contacts_count and avg_potential_contacts_count > 1:
                        potential_contacts_count_multiplier = math.log(agent_potentialcontacts_count, avg_potential_contacts_count)

                    soc_rate = self.agents_static.get(agentid, "soc_rate")

                    if soc_rate is None:
                        print("agentindex {0}, agentid {1}, avg_contacts_by_age_activity {2}, timestep_multiplier {3}, potential_contacts_count_multiplier {4}, soc_rate {5}, masks_hygiene_distancing_multiplier {6}".format(str(agentindex), str(agentid), str(avg_contacts_by_age_activity), str(timestep_multiplier), str(potential_contacts_count_multiplier), str(soc_rate), str(self.epi_util.dyn_params.masks_hygiene_distancing_multiplier)))
                    
                    agents_degrees[agentindex] = avg_contacts_by_age_activity * timestep_multiplier * potential_contacts_count_multiplier * soc_rate * (1 - self.epi_util.dyn_params.masks_hygiene_distancing_multiplier)

            agents_degrees_sum = round(np.sum(agents_degrees))
            
            # normalize against "max_degree" i.e. k * n, and ensure agent specific degrees do not exceed number of potential contacts
            for i in range(len(agents_degrees)):
                degree = agents_degrees[i]

                agentid = agents_ids[i]

                agent_potentialcontacts_count = agents_potentialcontacts_count[agentid]

                if agents_degrees_sum > max_degree:
                    degree = min(agent_potentialcontacts_count, round((degree / agents_degrees_sum) * max_degree))
                else:
                    degree = min(agent_potentialcontacts_count, round(degree))

                agents_degrees[i] = degree

            if self.visualise and len(agents_ids) > 500:
                self.figurecount += 1
                plt.figure(self.figurecount)
                plt.hist(agents_degrees, bins=10)
                plt.xlabel("Expected Direct Contacts")
                plt.ylabel("Count")
                plt.show(block=False)

            # commented as not being used
            # agents_degrees_backup = copy(agents_degrees) # these are final degrees, to save. agents_degrees will be deducted from below.
            # agents_potentialcontacts_backup = deepcopy(agents_potentialcontacts)

            agents_degrees, agents_ids = zip(*sorted(zip(agents_degrees, agents_ids), reverse=True))

            agents_degrees, agents_ids = list(agents_degrees), list(agents_ids)

            for i in range(len(agents_degrees)):
                agent_id = agents_ids[i]
                degree = agents_degrees[i]

                if degree > 0:
                    potential_contacts = util.get_all_contacts_ids_by_id(agent_id, agents_potentialcontacts.keys(), agents_ids, agents_degrees, shuffle=True)

                    if potential_contacts is not None:
                        if degree > len(potential_contacts):
                            # print("expected to find " + str(degree) + " direct contacts for agent: " + str(agent_id) + ", found: " + str(len(potential_contacts)))
                            degree = len(potential_contacts)

                        direct_contacts_ids = np.random.choice(potential_contacts, size=degree, replace=False)

                        agents_directcontacts, agents_potentialcontacts, agents_directcontacts_count = self.create_contacts(agents_ids, agents_degrees, i, agent_id, degree, direct_contacts_ids, agents_potentialcontacts, agents_directcontacts, agents_directcontacts_count, self.maintain_directcontacts_count)
                    # else:
                        # print("expected to find " + str(degree) + " direct contacts for agent: " + str(agent_id) + ", found: 0")
            
            if self.visualise and self.maintain_directcontacts_count and len(agents_ids) > 500:
                self.figurecount += 1
                plt.figure(self.figurecount)
                plt.hist(agents_directcontacts_count.values(), bins=10)
                plt.xlabel("Actual Direct Contacts")
                plt.ylabel("Count")
                plt.show(block=False)

            # print(str(len(agents_directcontacts)) + " contacts created from a pool of " + str(n) + " agents and " + str(len(agents_potentialcontacts_backup)) + " potential contacts")

        return agents_directcontacts, cell_type

    def convert_celltype_to_ageactivitycontactmatrixtype(self, cellid, celltype=None, indid=None):
        if celltype is None:
            celltype = self.cells_type[cellid]
            # celltype = cell["type"]

        if indid == 9 and celltype != CellType.Accommodation:
            return 5

        match celltype:
            case CellType.Household:
                return 1
            case CellType.Workplace:
                return 3
            case CellType.Accommodation:
                return 1
            case CellType.Hospital:
                return 5
            case CellType.Entertainment:
                return 5
            case CellType.School:
                return 2
            case CellType.Institution:
                return 1
            case CellType.Transport:
                return 4
            case CellType.Religion:
                return 5
            case CellType.Airport:
                return 5
            case _:
                return 0 # total
            
    # this determines whether all the overlapping timesteps for this pair have been computed, hence, allowing the same checks to be done only once
    def pair_already_computed_in_agentspotentialcontacts(self, agents_potentialcontacts, id1, id2):
        return (id2, id1) in agents_potentialcontacts

    # this determines whether at least one overlapping timestep for this pair has already been computed (at least once), 
    # hence, ensuring that multiple overlapping timesteps are not considered as multiple edges between nodes
    def pair_exists_in_agentspotentialcontacts(self, agents_potentialcontacts, id1, id2):
        return (id1, id2) in agents_potentialcontacts or (id2, id1) in agents_potentialcontacts
    
    def get_key_if_pair_exists_in_agentspotentialcontacts(self, agents_potentialcontacts, id1, id2):
        if (id1, id2) in agents_potentialcontacts:
            return (id1, id2)
            
        if (id2, id1) in agents_potentialcontacts:
            return (id2, id1)
        
        return None
        
    def delete_all_pairs_by_id(self, agents_potentialcontacts, id):
        for pairid in list(agents_potentialcontacts.keys()):
            if id in pairid:
                try:
                    del agents_potentialcontacts[pairid]
                    # print("deleted id {0} from pairid {1}".format(str(id), str(pairid)))
                except:
                    if pairid not in agents_potentialcontacts:
                        print("pairid {0} does not exist in agents_potentialcontacts!")

                    raise

        return agents_potentialcontacts
    
    def create_contacts(self, agents_ids, agents_degrees, agent_index, agent_id, main_agent_degree, direct_contacts_ids, agents_potentialcontacts, agents_directcontacts, agents_directcontacts_count, maintain_agents_directcontacts_count=False):       
        main_agent_degree -= len(direct_contacts_ids)

        agents_degrees[agent_index] = main_agent_degree

        if maintain_agents_directcontacts_count:
            if agent_id not in agents_directcontacts_count:
                agents_directcontacts_count[agent_id] = len(direct_contacts_ids)
            else:
                agents_directcontacts_count[agent_id] += len(direct_contacts_ids)

        for contact_agent_id in direct_contacts_ids:
            pair_id = self.get_key_if_pair_exists_in_agentspotentialcontacts(agents_potentialcontacts, agent_id, contact_agent_id)

            timesteps = agents_potentialcontacts[pair_id]

            agents_directcontacts[pair_id] = copy(timesteps)

            contact_agent_index = agents_ids.index(contact_agent_id)

            contact_agent_degree = agents_degrees[contact_agent_index]

            contact_agent_degree -= 1

            agents_degrees[contact_agent_index] = contact_agent_degree

            if maintain_agents_directcontacts_count:
                if contact_agent_id not in agents_directcontacts_count:
                    agents_directcontacts_count[contact_agent_id] = 1
                else:
                    agents_directcontacts_count[contact_agent_id] += 1

            if contact_agent_degree == 0:
                self.delete_all_pairs_by_id(agents_potentialcontacts, contact_agent_id)

        if main_agent_degree == 0:
            agents_potentialcontacts = self.delete_all_pairs_by_id(agents_potentialcontacts, agent_id)

        return agents_directcontacts, agents_potentialcontacts, agents_directcontacts_count
    
    def get_overlapping_range(self, a1, a2, b1, b2):
        lower = max(a1, b1)
        upper = min(a2, b2)
        if lower <= upper:
            return lower, upper
        else:
            return None

