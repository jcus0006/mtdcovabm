import customdict

class Vars:
    def __init__(self, 
                cells_agents_timesteps={}, 
                directcontacts_by_simcelltype_by_day=[], 
                dc_by_sct_by_day_agent1_index=[],
                dc_by_sct_by_day_agent2_index=[],
                contact_tracing_agent_ids=set(),
                agents_seir_state=[],
                agents_seir_state_transition_for_day=customdict.CustomDict(),
                agents_infection_type=customdict.CustomDict(),
                agents_infection_severity=customdict.CustomDict(),
                agents_vaccination_doses=[]) -> None:
        self.cells_agents_timesteps = cells_agents_timesteps # cellid: [[agentid, starttimestep, endtimestep]] - IT
        self.directcontacts_by_simcelltype_by_day = directcontacts_by_simcelltype_by_day # [day, simcelltype, agent1_id, agent2_id, start_ts, end_ts] - CN for Contact tracing (was set)
        self.dc_by_sct_by_day_agent1_index = dc_by_sct_by_day_agent1_index # [agent1id, index_in_directcontacts_by_simcelltype_by_day]
        self.dc_by_sct_by_day_agent2_index = dc_by_sct_by_day_agent2_index # [agent2id, index_in_directcontacts_by_simcelltype_by_day]
        self.contact_tracing_agent_ids = contact_tracing_agent_ids # (agentid, start_timestep) - CN for Contact tracing

        # transmission model
        self.agents_seir_state = agents_seir_state # whole population with following states, 0: undefined, 1: susceptible, 2: exposed, 3: infectious, 4: recovered, 5: deceased
        self.agents_seir_state_transition_for_day = agents_seir_state_transition_for_day # handled as dict, represents during day transitions. value: [new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep]
        self.agents_infection_type = agents_infection_type # handled as dict, because not every agent will be infected
        self.agents_infection_severity = agents_infection_severity # handled as dict, because not every agent will be infected
        self.agents_vaccination_doses = agents_vaccination_doses # number of doses per agent

    def __reduce__(self):
        return (self.__class__, (self.cells_agents_timesteps, self.directcontacts_by_simcelltype_by_day, self.dc_by_sct_by_day_agent1_index, self.dc_by_sct_by_day_agent2_index, self.contact_tracing_agent_ids, self.agents_seir_state, self.agents_seir_state_transition_for_day, self.agents_infection_type, self.agents_infection_severity, self.agents_vaccination_doses))
    
    def populate(self, ag_seir_state, ag_seir_state_transition_for_day, ag_infection_type, ag_infection_severity, ag_vaccination_doses):
        self.agents_seir_state = ag_seir_state
        self.agents_seir_state_transition_for_day = ag_seir_state_transition_for_day
        self.agents_infection_type = ag_infection_type
        self.agents_infection_severity = ag_infection_severity
        self.agents_vaccination_doses = ag_vaccination_doses

    def reset_cells_agents_timesteps(self):
        self.cells_agents_timesteps = {}

    # def update_cells_agents_timesteps(self, cellid, agent_cell_timestep_ranges):
    #     if cellid not in self.cells_agents_timesteps:
    #         self.cells_agents_timesteps[cellid] = []

    #     self.cells_agents_timesteps[cellid] += agent_cell_timestep_ranges

    def update(self, name, value, index=None):
        if name == "directcontacts_by_simcelltype_by_day":
            self.directcontacts_by_simcelltype_by_day.append(value)
        elif name == "contact_tracing_agent_ids":
            self.contact_tracing_agent_ids.add(value)
        elif name == "agents_seir_state":
            try:
                self.agents_seir_state[index] = value
            except:
                return
        elif name == "agents_seir_state_transition_for_day":
            try:
                self.agents_seir_state_transition_for_day[index] = value
            except:
                return
        elif name == "agents_infection_type":
            try:
                self.agents_infection_type[index] = value
            except:
                return
        elif name == "agents_infection_severity":
            try:
                self.agents_infection_severity[index] = value
            except:
                return
        elif name == "agents_vaccination_doses":
            try:
                self.agents_vaccination_doses[index] = value
            except:
                return