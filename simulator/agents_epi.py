class AgentsEpi:
    def __init__(self, agents_epi=None):
        self.properties = {"state_transition": 0, "test_day": 1, "test_result_day": 2, "hospitalisation_days" : 3, "quarantine_days" : 4, "vaccination_days": 5}
        
        if agents_epi is not None:
            self.agents_epi = agents_epi
        else:
            self.agents_epi = {}

    def get(self, day, id, key):
        if day in self.agents_epi and id in self.agents_epi[day]:
            return self.agents_epi[day][id][self.properties[key]]
        return None
    
    def set(self, day, id, key, value):
        if day not in self.agents_epi:
            self.agents_epi[day] = {}

        if id not in self.agents_epi[day]:
            self.agents_epi[day][id] = [None, None, None, None, None, None]

        self.agents_epi[day][id][self.properties[key]] = value

    # would crash if the value within the passed key is None and is not an array
    # would convert any None property to an array, even if it is not meant to be an array
    # def append(self, day, id, key, value):
    #     if day not in self.agents_epi:
    #         self.agents_epi[day] = {}

    #     if id not in self.agents_epi[day]:
    #         self.agents_epi[day][id] = [None, None, None, None, None]

    #     if self.agents_epi[day][key][self.properties[key]] is None:
    #         self.agents_epi[day][key][self.properties[key]] = []

    #     self.agents_epi[day][key][self.properties[key]].append(value)

    def partialize(self, day, agent_ids):
        temp_agents_epi = {}
        temp_agents_epi[day] = {}

        if day in self.agents_epi:
            agents_epi_for_day = self.agents_epi[day]
            for id in agent_ids:
                if id in agents_epi_for_day:
                    temp_agents_epi[day][id] = agents_epi_for_day[day][id]

        return AgentsEpi(temp_agents_epi)
    
    def convert_agents_epi(agents_epi):
        temp_agents_epi = {}
        agents_epi_util = AgentsEpi(temp_agents_epi)

        for k, v in agents_epi.items():
            if v is not None:
                if v["state_transition_by_day"] is not None:
                    for params in v["state_transition_by_day"]:
                        agents_epi_util.set(params[0], k, "state_transition", [params[1], params[2]])
                
                if v["test_day"] is not None:
                    agents_epi_util.set(v["test_day"][0], k, "test_day", v["test_day"][1])

                if v["test_result_day"] is not None:
                    agents_epi_util.set(v["test_result_day"][0], k, "test_result_day", v["test_result_day"][1])

                if v["hospitalisation_days"] is not None:
                    agents_epi_util.set(v["hospitalisation_days"][0], k, "hospitalisation_days", [True, v["hospitalisation_days"][1]]) # start_day
                    agents_epi_util.set(v["hospitalisation_days"][2], k, "hospitalisation_days", [False, v["hospitalisation_days"][1]]) # end day

                if v["quarantine_days"] is not None:
                    agents_epi_util.set(v["quarantine_days"][0], k, "quarantine_days", [True, v["quarantine_days"][1]]) # start_day
                    agents_epi_util.set(v["quarantine_days"][2], k, "quarantine_days", [False, v["quarantine_days"][1]]) # end day

                if "vaccination_days" in v and v["vaccination_days"] is not None:
                    for vacc_day in v["vaccination_days"]:
                        agents_epi_util.set(vacc_day[0], k, "vaccination_days", vacc_day[1])

        return agents_epi_util
