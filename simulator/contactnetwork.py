import numpy as np
import math
import random
import json
import powerlaw

class ContactNetwork:
    def __init__(self, agents, cells, contactnetworkparams):
        self.agents = agents
        self.cells = cells
        self.contactnetworkparams = contactnetworkparams
        self.powerlawdist_exponent = self.contactnetworkparams["powerlawdistributionparameters"][0]
        self.powerlawdist_xmin = self.contactnetworkparams["powerlawdistributionparameters"][1]
        self.sociabilityratemultipliers = self.contactnetworkparams["sociabilityratemultipliers"] 
        self.ageactivitycontactmatrix = np.array(self.contactnetworkparams["ageactivitycontactmatrix"])  

    def simulate_contact_network(self, cellid, agent_cell_timestep_ranges): # to create class and initialise stuff in init
        agents_ids = []
        agents_degrees = []
        population_per_timestep = [0 for i in range(144)]

        for agent_cell_time_range in agent_cell_timestep_ranges:
            agentid, starttimestep, endtimestep = agent_cell_time_range[0], agent_cell_time_range[1], agent_cell_time_range[2]

            if agentid not in agents_ids:
                agents_ids.append(agentid)
                agents_degrees.append(0)

            for timestep in range(starttimestep, endtimestep+1):
                population_per_timestep[timestep] += 1

        population_per_timestep_no_zeros = np.array([pop_per_ts for pop_per_ts in population_per_timestep if pop_per_ts > 0])

        n = np.median(population_per_timestep_no_zeros) # max/median population during the day (if using median remove 0s as they would be outliers and take middle value)

        num_agents_whole_day = len(agents_ids)

        ageactivitycontact_cm_activityid = self.convert_celltype_to_ageactivitycontactmatrixtype(cellid)
        
        avg_contacts_activity_agegroups_col = np.array(self.ageactivitycontactmatrix[:, 2 + ageactivitycontact_cm_activityid]) # start from 4th column onwards, ageactivitycontact_cm_activityid min = 1

        k = np.mean(avg_contacts_activity_agegroups_col) # average for activity across all age groups

        max_degree = (k * n) / 2

        # Sample "num_agents_whole_day" numbers from a power law distribution representing degrees for every agent
        dist = powerlaw.Power_Law(xmin=self.powerlawdist_xmin, parameters=[self.powerlawdist_exponent])

        agents_contact_propensity = dist.generate_random(num_agents_whole_day)

        agents_contact_propensity_sum = sum(agents_contact_propensity)

        agents_contact_propensity_normalized = [degree/agents_contact_propensity_sum for degree in agents_contact_propensity]

        agents_contact_propensity_cdf = np.cumsum(agents_contact_propensity_normalized)

        for i in range(max_degree):
            sampled_agent_index = np.argmax(agents_contact_propensity_cdf > np.random.rand())
            agents_degrees[sampled_agent_index] += 1

        for index, degree in enumerate(agents_degrees):
            agentid = agents_ids[index]
            agent = self.agents[agentid]
            degree = round(math.log(degree, k) * agent["soc_rate"])
        

    def convert_celltype_to_ageactivitycontactmatrixtype(self, cellid):
        cell = self.cells[cellid]
        celltype = cell["type"]

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



