import customdict
# import pandas as pd
# import dask.dataframe as df
# import dask.array as da

class Vars:
    def __init__(self, 
                cells_agents_timesteps=None, 
                directcontacts_by_simcelltype_by_day=None, 
                dc_by_sct_by_day_agent1_index=None,
                dc_by_sct_by_day_agent2_index=None,
                directcontacts_by_simcelltype_by_day_start_marker=None,
                contact_tracing_agent_ids=None,
                agents_seir_state=None,
                agents_seir_state_transition_for_day=None,
                agents_infection_type=None,
                agents_infection_severity=None,
                agents_vaccination_doses=None) -> None:
        if cells_agents_timesteps is None:
            cells_agents_timesteps = customdict.CustomDict()

        if directcontacts_by_simcelltype_by_day is None:
            directcontacts_by_simcelltype_by_day = []

        if dc_by_sct_by_day_agent1_index is None:
            dc_by_sct_by_day_agent1_index = []

        if dc_by_sct_by_day_agent2_index is None:
            dc_by_sct_by_day_agent2_index = []

        if directcontacts_by_simcelltype_by_day_start_marker is None:
            directcontacts_by_simcelltype_by_day_start_marker = customdict.CustomDict()

        if contact_tracing_agent_ids is None:
            contact_tracing_agent_ids = set()

        if agents_seir_state is None:
            agents_seir_state = customdict.CustomDict()

        if agents_seir_state_transition_for_day is None:
            agents_seir_state_transition_for_day = customdict.CustomDict()

        if agents_infection_type is None:
            agents_infection_type = customdict.CustomDict()

        if agents_infection_severity is None:
            agents_infection_severity = customdict.CustomDict()

        if agents_vaccination_doses is None:
            agents_vaccination_doses = customdict.CustomDict()
        
        self.cells_agents_timesteps = cells_agents_timesteps # {cellid: [[agentid, starttimestep, endtimestep]]} - IT
        self.directcontacts_by_simcelltype_by_day = directcontacts_by_simcelltype_by_day # [[simcelltype, agent1id, agent2id, start_ts, end_ts]] - CN for Contact tracing (was set)
        self.dc_by_sct_by_day_agent1_index = dc_by_sct_by_day_agent1_index # [[agent1id, index_in_directcontacts_by_simcelltype_by_day]] - agent1 index
        self.dc_by_sct_by_day_agent2_index = dc_by_sct_by_day_agent2_index # [[agent2id, index_in_directcontacts_by_simcelltype_by_day]] - agent2 index
        self.directcontacts_by_simcelltype_by_day_start_marker = directcontacts_by_simcelltype_by_day_start_marker # {day: startindex} index matches main array only, agent indexes need re-sorting - day-mark index
        self.contact_tracing_agent_ids = contact_tracing_agent_ids # {(agentid, start_timestep)} - CT for Contact tracing

        # transmission model
        self.agents_seir_state = agents_seir_state # whole population with following states, 0: undefined, 1: susceptible, 2: exposed, 3: infectious, 4: recovered, 5: deceased
        self.agents_seir_state_transition_for_day = agents_seir_state_transition_for_day # handled as dict, represents during day transitions, set in itinerary and used in contact network. value: [new_seir_state, old_seir_state, new_infection_type, new_infection_severity, seir_state_transition, new_state_timestep]
        self.agents_infection_type = agents_infection_type # handled as dict, because not every agent will be infected
        self.agents_infection_severity = agents_infection_severity # handled as dict, because not every agent will be infected
        self.agents_vaccination_doses = agents_vaccination_doses # handled as dict, days on which each agent has been administered a dose. index represents dose1, dose2, etc

    def __reduce__(self):
        return (self.__class__, (self.cells_agents_timesteps, self.directcontacts_by_simcelltype_by_day, self.dc_by_sct_by_day_agent1_index, self.dc_by_sct_by_day_agent2_index, self.directcontacts_by_simcelltype_by_day_start_marker, self.contact_tracing_agent_ids, self.agents_seir_state, self.agents_seir_state_transition_for_day, self.agents_infection_type, self.agents_infection_severity, self.agents_vaccination_doses))
    
    def populate(self, ag_seir_state, ag_seir_state_transition_for_day, ag_infection_type, ag_infection_severity, ag_vaccination_doses):
        self.agents_seir_state = ag_seir_state
        self.agents_seir_state_transition_for_day = ag_seir_state_transition_for_day
        self.agents_infection_type = ag_infection_type
        self.agents_infection_severity = ag_infection_severity
        self.agents_vaccination_doses = ag_vaccination_doses

    def reset_daily_structures(self):
        self.cells_agents_timesteps = customdict.CustomDict()
        self.agents_seir_state_transition_for_day = customdict.CustomDict()

    # def convert_to_dask_collections(self, partition_size):
    #     # self.cells_agents_timesteps = db.from_sequence(self.cells_agents_timesteps.items(), partition_size=128)
    #     # self.directcontacts_by_simcelltype_by_day = da.from_array(self.directcontacts_by_simcelltype_by_day, chunks=128)
    #     # self.contact_tracing_agent_ids = db.from_sequence(self.contact_tracing_agent_ids.items(), partition_size=128)
    #     self.agents_seir_state = da.from_array(self.agents_seir_state, chunks=partition_size)
    #     self.agents_infection_type = df.from_pandas(pd.DataFrame(self.agents_infection_type), npartitions=partition_size)
    #     self.agents_infection_severity = df.from_pandas(pd.DataFrame(self.agents_infection_severity), npartitions=partition_size)

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