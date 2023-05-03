import numpy as np
import math
from copy import copy
from copy import deepcopy
from simulator.epidemiology import Epidemiology
import matplotlib.pyplot as plt

class ContactNetwork:
    def __init__(self, agents, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, cells, cells_agents_timesteps, contactnetworkparams, epidemiologyparams, visualise=False, maintain_directcontacts_count=False):
        self.agents = agents

        self.cells = cells
        self.cells_agents_timesteps = cells_agents_timesteps # {cellid: [(agentid, starttimestep, endtimestep)]}
        self.contactnetworkparams = contactnetworkparams
        self.ageactivitycontactmatrix = np.array(self.contactnetworkparams["ageactivitycontactmatrix"])

        self.visualise = visualise
        self.maintain_directcontacts_count = maintain_directcontacts_count
        self.figurecount = 0

        self.epi_util = Epidemiology(epidemiologyparams, agents, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity)

    def simulate_contact_network(self, cellid, day):
        agents_directcontacts, cell = self.generate_contact_network(cellid)

        self.epi_util.simulate_direct_contacts(agents_directcontacts, cell, day)

    def generate_contact_network(self, cellid):
        print("generating contact network for cell " + str(cellid))
        agents_ids = [] # each index represents an agent ID
        agents_degrees = [] # each index represents an agent degree, maps to agents_ids indices
        agents_total_timesteps = {} # {agentid : total_num_of_timesteps}
        agents_potentialcontacts_count = {} # {agentid: total_num_of_potential_contats}
        agents_potentialcontacts = {} # {(agentid1, agentid2) : [ (start_ts1, end_ts1), (start_ts2, end_ts2) ]}
        agents_directcontacts = {}
        agents_directcontacts_count = {}
        population_per_timestep = [0 for i in range(144)]

        cell = None
        if cellid in self.cells_agents_timesteps:
            cell = self.cells[cellid]

            cell_agents_timesteps = self.cells_agents_timesteps[cellid]

            indid = None

            if cell["type"] == "workplace":
                if "indid" in cell["place"]:
                    indid = cell["place"]["indid"]

            ageactivitycontact_cm_activityid = self.convert_celltype_to_ageactivitycontactmatrixtype(cellid, cell["type"], indid)
            
            agents_timesteps_sum = 0

            for agentid, starttimestep, endtimestep in cell_agents_timesteps:
                if agentid not in agents_ids:
                    agents_ids.append(agentid)
                    agents_degrees.append(0)

                if agentid not in agents_total_timesteps:
                    agents_total_timesteps[agentid] = 0

                for timestep in range(starttimestep, endtimestep+1):
                    population_per_timestep[timestep] += 1

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
            
                # population_per_timestep_no_zeros = np.array([pop_per_ts for pop_per_ts in population_per_timestep if pop_per_ts > 0])

                # n = round(np.median(population_per_timestep_no_zeros)) # max/median population during the day (if using median remove 0s as they would be outliers and take middle value)      
                n = len(agents_ids)

                avg_contacts_activity_agegroups_col = np.array(self.ageactivitycontactmatrix[:, 2 + ageactivitycontact_cm_activityid]) # start from 4th column onwards, ageactivitycontact_cm_activityid min = 1

                k = round(np.mean(avg_contacts_activity_agegroups_col)) # average for activity across all age groups

                max_degree = round((k * n)) # should this be k * n / 2 ?

                avg_agents_timestep_counts = agents_timesteps_sum / n

                agents_potentialcontacts_count_np = np.array(list(agents_potentialcontacts_count.values()))
                avg_potential_contacts_count = np.mean(agents_potentialcontacts_count_np)
                std_potential_contacts_count = np.std(agents_potentialcontacts_count_np)
                
                # create degrees per agent
                for agentindex, agentid in enumerate(agents_ids):
                    agent_potentialcontacts_count = agents_potentialcontacts_count[agentid]

                    if agent_potentialcontacts_count > 0:
                        agent = self.agents[agentid]

                        agent_timestep_count = agents_total_timesteps[agentid]

                        avg_contacts_by_age_activity = self.ageactivitycontactmatrix[agent["age_bracket_index"], 2 + ageactivitycontact_cm_activityid]

                        timestep_multiplier = math.log(agent_timestep_count, avg_agents_timestep_counts)

                        potential_contacts_count_multiplier = 1
                        
                        if agent_potentialcontacts_count != avg_potential_contacts_count and avg_potential_contacts_count > 1:
                            potential_contacts_count_multiplier = math.log(agent_potentialcontacts_count, avg_potential_contacts_count)

                        agents_degrees[agentindex] = avg_contacts_by_age_activity * timestep_multiplier * potential_contacts_count_multiplier * agent["soc_rate"]

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

                agents_degrees_backup = copy(agents_degrees) # these are final degrees, to save. agents_degrees will be deducted from below.
                agents_potentialcontacts_backup = deepcopy(agents_potentialcontacts)

                agents_degrees, agents_ids = zip(*sorted(zip(agents_degrees, agents_ids), reverse=True))

                agents_degrees, agents_ids = list(agents_degrees), list(agents_ids)

                for i in range(len(agents_degrees)):
                    agent_id = agents_ids[i]
                    degree = agents_degrees[i]

                    if degree > 0:
                        potential_contacts = self.get_all_potentialcontacts_ids(agents_ids, agents_potentialcontacts, agents_degrees, agent_id, shuffle=True)

                        if potential_contacts is not None:
                            if degree > len(potential_contacts):
                                print("expected to find " + str(degree) + " direct contacts for agent: " + str(agent_id) + ", found: " + str(len(potential_contacts)))
                                degree = len(potential_contacts)

                            direct_contacts_ids = np.random.choice(potential_contacts, size=degree, replace=False)

                            agents_directcontacts, agents_potentialcontacts, agents_directcontacts_count = self.create_contacts(agents_ids, agents_degrees, i, agent_id, degree, direct_contacts_ids, agents_potentialcontacts, agents_directcontacts, agents_directcontacts_count, self.maintain_directcontacts_count)
                        else:
                            print("expected to find " + str(degree) + " direct contacts for agent: " + str(agent_id) + ", found: 0")
                
                if self.visualise and self.maintain_directcontacts_count and len(agents_ids) > 500:
                    self.figurecount += 1
                    plt.figure(self.figurecount)
                    plt.hist(agents_directcontacts_count.values(), bins=10)
                    plt.xlabel("Actual Direct Contacts")
                    plt.ylabel("Count")
                    plt.show(block=False)

                print(str(len(agents_directcontacts)) + " contacts created from a pool of " + str(n) + " agents and " + str(len(agents_potentialcontacts_backup)) + " potential contacts")

                # print("direct contacts: " + str(agents_directcontacts))

        return agents_directcontacts, cell

    def convert_celltype_to_ageactivitycontactmatrixtype(self, cellid, celltype=None, indid=None):
        if celltype is None:
            cell = self.cells[cellid]
            celltype = cell["type"]

        if indid == 9 and celltype != "accom":
            return 5

        match celltype:
            case "household":
                return 1
            case "workplace":
                return 3
            case "accom":
                return 1
            case "hospital":
                return 5
            case "entertainment":
                return 5
            case "school":
                return 2
            case "institution":
                return 1
            case "transport":
                return 4
            case "religion":
                return 5
            case "airport":
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
    
    def get_all_potentialcontacts_ids(self, agents_ids, agents_potentialcontacts, agents_degrees, id, shuffle=True):
        potential_contacts_ids = []
        for pairid in agents_potentialcontacts.keys():
            if id in pairid:
                potential_contact_id = pairid[1] if pairid[0] == id else pairid[0]
                potential_contact_index = agents_ids.index(potential_contact_id)
                potential_contact_degree = agents_degrees[potential_contact_index]

                if potential_contact_degree > 0:
                    potential_contacts_ids.append(potential_contact_id)

        if len(potential_contacts_ids) == 0:
            return None
        
        potential_contacts_ids = np.array(potential_contacts_ids)

        if shuffle:
            np.random.shuffle(potential_contacts_ids)

        return potential_contacts_ids
        
    def delete_all_pairs_by_id(self, agents_potentialcontacts, id):
        for pairid in list(agents_potentialcontacts.keys()):
            if id in pairid:
                del agents_potentialcontacts[pairid]

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

