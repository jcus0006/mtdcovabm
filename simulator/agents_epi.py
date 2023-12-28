class AgentsEpi:
    def __init__(self, agents_epi):
        self.properties = {"state_transition": 0, "test_day": 1, "test_result_day": 2, "hospitalisation_days" : 3, "quarantine_days" : 4, "vaccination_days": 5}
        self.agents_epi = agents_epi

    def get(self, day, id, key):
        if day in self.agents_epi and id in self.agents_epi[day]:
            return self.agents_epi[day][self.properties[key]]
        return None
    
    def set(self, day, id, key, value):
        if day not in self.agents_epi:
            self.agents_epi[day] = {}

        if id not in self.agents_epi[day]:
            self.agents_epi[day][id] = [None, None, None, None, None]

        self.agents_epi[day][key][self.properties[key]] = value

    def partialize(self, day, agent_ids):
        temp_agents_epi = {}
        temp_agents_epi[day] = {}

        if day in self.agents_epi:
            agents_epi_for_day = self.agents_epi[day]
            for id in agent_ids:
                if id in agents_epi_for_day:
                    temp_agents_epi[day][id] = agents_epi_for_day[day][id]

        return AgentsEpi(temp_agents_epi)
