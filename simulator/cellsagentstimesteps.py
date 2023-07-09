class CAT:
    def __init__(self) -> None:
        self.cells_agents_timesteps = {} # [[agentid, starttimestep, endtimestep]]

    def reset_cells_agents_timesteps(self):
        self.cells_agents_timesteps = {}

    def update_cells_agents_timesteps(self, cellid, agent_cell_timestep_ranges):
        if cellid not in self.cells_agents_timesteps:
            self.cells_agents_timesteps[cellid] = []

        self.cells_agents_timesteps[cellid] += agent_cell_timestep_ranges