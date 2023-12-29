import gc
import time
from enum import Enum
from copy import copy
import customdict

class AgentsEpi:
    def __init__(self, agents_epi=None):
        self.properties = {"state_transition": 0, "test_day": 1, "test_result_day": 2, "hospitalisation_days" : 3, "quarantine_days" : 4, "vaccination_days": 5}
        
        if agents_epi is not None:
            self.agents_epi = agents_epi
        else:
            self.agents_epi = customdict.CustomDict()

    def get(self, day, id, key):
        if day in self.agents_epi and id in self.agents_epi[day]:
            return self.agents_epi[day][id][self.properties[key]]
        return None
    
    def set(self, day, id, key, value):
        if day not in self.agents_epi:
            self.agents_epi[day] = customdict.CustomDict()

        if id not in self.agents_epi[day]:
            self.agents_epi[day][id] = [None, None, None, None, None, None]

        self.agents_epi[day][id][self.properties[key]] = value

    def get_props(self, day, id):
        if day in self.agents_epi and id in self.agents_epi[day]:
            return self.agents_epi[day][id]
        return None

    def set_props(self, day, id, props):
        if day not in self.agents_epi:
            self.agents_epi[day] = customdict.CustomDict()

        self.agents_epi[day][id] = props

    # takes a day -> agent id combination to avoid repeated lookup
    # must call get_props first with day and agent_id
    def set_props_existing(self, props, key, value):
        props[self.properties[key]] = value

    # applicable to ongoing events, i.e. hospitalisation and quarantine
    def set_range(self, start_day, end_day, timestep, id, key):
        for day in range(start_day, end_day + 1):
            if day == start_day:
                self.set(day, id, key, [HospQuarDays.Start, timestep])
            elif day == end_day:
                self.set(day, id, key, [HospQuarDays.End, timestep])
            else:
                self.set(day, id, key, [HospQuarDays.Ongoing, None])

        return [start_day, timestep, end_day]

    def set_test_range(self, test_day, timestep, range_before_after, id):
        for day in range(test_day - range_before_after, test_day + range_before_after + 1):
            if day < test_day:
                self.set(day, id, "test_day", [TestDay.Before, None])
            elif day == test_day:
                self.set(day, id, "test_day", [TestDay.Actual, timestep])
            else:
                self.set(day, id, "test_day", [TestDay.After, None])

    def delete(self, day, id, key):
        if day in self.agents_epi and id in self.agents_epi[day]:
            self.agents_epi[day][id][self.properties[key]] = None

    def delete_day(self, day, force_gc=False):
        if day in self.agents_epi:
            del self.agents_epi[day]

            if force_gc:
                gc.collect()

    # danger - this might be slow
    def delete_agent(self, id):
        start = time.time()
        days = list(self.agents_epi.keys())

        for day in days:
            agent_props = self.agents_epi[day]

            if id in agent_props:
                del agent_props[id]

        time_taken = time.time() - start
        print(f"delete_agent from agents_epi {time_taken}")

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

    def partialize(self, day, agent_ids, temp_agents_epi_util, force_copy=False):
        # temp_agents_epi = {}
        temp_agents_epi = temp_agents_epi_util.agents_epi
        
        if day not in temp_agents_epi:
            temp_agents_epi[day] = customdict.CustomDict()

        if day in self.agents_epi:
            agents_epi_for_day = self.agents_epi[day]

            for id in agent_ids:
                if id in agents_epi_for_day:
                    if not force_copy:
                        temp_agents_epi[day][id] = agents_epi_for_day[id]
                    else:
                        temp_agents_epi[day][id] = copy(agents_epi_for_day[id])

        return temp_agents_epi_util

    def sync(self, simday, agents_epi_util_partial, force_gc=False):
        for day, agents_props in agents_epi_util_partial.agents_epi.items():
            for id, props in agents_props.items():
                self.set_props(day, id, props)

        if force_gc:
            gc.collect()
    
    def convert_agents_epi(agents_epi):
        temp_agents_epi = customdict.CustomDict()
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

class HospQuarDays(Enum):
    Start = 0
    Ongoing = 1
    End = 2

class TestDay(Enum):
    Before = 0
    Actual = 1
    After = 2