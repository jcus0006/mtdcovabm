class Vars:
    def __init__(self) -> None:
        self.directcontacts_by_simcelltype_by_day = set() # [day, simcelltype, agent1_id, agent2_id, start_ts, end_ts]
        self.contact_tracing_agent_ids = set() # flat set of ids

    def update(self, name, value):
        if name == "directcontacts_by_simcelltype_by_day":
            self.directcontacts_by_simcelltype_by_day.update(value)
        elif name == "contact_tracing_agent_ids":
            self.contact_tracing_agent_ids.update(value)