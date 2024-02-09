import numpy as np
import random
from epidemiologyclasses import SEIRState, SEIRStateTransition, InfectionType, Severity

def initialize_agent_states(n, initial_seir_state_distribution, agents_seir_state):
    for seirindex, seirstate_percent in enumerate(initial_seir_state_distribution):
        seirid = seirindex + 1

        total_to_assign_this_state = round(n * seirstate_percent)

        seirstate = SEIRState(seirid)

        undefined_ids = [id for id, x in agents_seir_state.items() if x == SEIRState.Undefined]

        if len(undefined_ids) > 0:
            undefined_ids = np.array(undefined_ids)

        this_state_ids = np.random.choice(undefined_ids, size=total_to_assign_this_state, replace=False)

        for id in this_state_ids:
            agents_seir_state[id] = seirstate

    undefined_ids = [id for id, x in agents_seir_state.items() if x == SEIRState.Undefined]

    for id in undefined_ids:
        random_state = random.randint(1, 4)

        random_seir_state = SEIRState(random_state)

        agents_seir_state[id] = random_seir_state

    return agents_seir_state

# to be called at the beginning of itinerary generation per day, per agent
# if day is in agent["state_transition_by_day"]
#   - update agent state (as required) - to be logged
#   - clear agent["state_transition_by_day"] for day
def update_agent_state(agents_seir_state, agents_infection_type, agents_infection_severity, agentid, agent, agentindex, day):
    today_index = -1
    if agent["state_transition_by_day"] is not None:
        for index, day_entry in enumerate(agent["state_transition_by_day"]): # should always be a short list
            if day_entry[0] == day:
                today_index = index
                break
    
    if today_index > -1:
        _, seir_state_transition, timestep = agent["state_transition_by_day"][today_index]
        current_seir_state = agents_seir_state_get(agents_seir_state, agentid) #agentindex
        try:
            current_infection_type = agents_infection_type[agentid]
        except:
            print(f"infection type does not exist for agent id: {agentid}, will crash. seir state is {current_seir_state}")
            raise

        current_infection_severity = agents_infection_severity[agentid]

        new_seir_state, new_infection_type, new_infection_severity = convert_state_transition_to_new_state(current_seir_state, current_infection_type, current_infection_severity, seir_state_transition)

        if new_seir_state != current_seir_state: # to be logged
            agents_seir_state = agents_seir_state_update(agents_seir_state, new_seir_state, agentid)

        if new_infection_type != current_infection_type: # to be logged
            agents_infection_type[agentid] = new_infection_type

        if new_infection_severity != current_infection_severity: # to be logged
            agents_infection_severity[agentid] = new_infection_severity
            
        del agent["state_transition_by_day"][today_index] # clean up

        return new_seir_state, SEIRState(current_seir_state), new_infection_type, new_infection_severity, seir_state_transition, timestep

    return None

# this depends on the context where it is called from. may be called with agentid directly or with agentindex (in global space, agentid == agentindex, but not in partial cases)
def agents_seir_state_get(agents_seir_state, agent_id_or_idx, agents_seir_indices=None):
    return agents_seir_state[agent_id_or_idx]

    # if agents_seir_indices is None:
    #     return agents_seir_state[agent_id_or_idx]
    # else:
    #     return agents_seir_state[agents_seir_indices[agent_id_or_idx]]
    
    # if agentids is None:
    #     return agents_seir_state[agentid]
    # else:
    #     return agents_seir_state[agentids.index(agentid)] # extremely slow

# this depends on the context where it is called from. may be called with agentid directly or with agentindex (in global space, agentid == agentindex, but not in partial cases)
def agents_seir_state_update(agents_seir_state, new_seir_state, agent_id_or_idx, agents_seir_indices=None):
    agents_seir_state[agent_id_or_idx] = new_seir_state

    # if agents_seir_indices is None:
    #     agents_seir_state[agent_id_or_idx] = new_seir_state
    # else:
    #     agents_seir_state[agents_seir_indices[agent_id_or_idx]]

    # if agentids is None:
    #     agents_seir_state[agentid] = new_seir_state
    # else:
    #     agents_seir_state[agentids.index(agentid)] = new_seir_state # extremely slow

    return agents_seir_state

# returns new: (state, infection_type, severity)
def convert_state_transition_to_new_state(current_seir_state, current_infection_type, current_infection_severity, seir_state_transition):
    match seir_state_transition:
        case SEIRStateTransition.ExposedToInfectious:
            new_infection_type = current_infection_type
            if current_infection_type == InfectionType.PreAsymptomatic:
                new_infection_type = InfectionType.Asymptomatic
            # if PreSymptomatic, infection type will already be assigned, but is only to be considered infectious, if SEIR State is Infectious
            return SEIRState.Infectious, new_infection_type, Severity.Mild
        case SEIRStateTransition.InfectiousToSymptomatic:
            new_infection_type = current_infection_type
            if current_infection_type == InfectionType.PreSymptomatic:
                new_infection_type = InfectionType.Symptomatic

            return InfectionType.Symptomatic, new_infection_type, current_infection_severity
        case SEIRStateTransition.SymptomaticToSevere:
            return current_seir_state, current_infection_type, Severity.Severe
        case SEIRStateTransition.SevereToCritical:
            return current_seir_state, current_infection_type, Severity.Critical
        case SEIRStateTransition.CriticalToDeath:
            return SEIRState.Deceased, current_infection_type, current_infection_severity
        case SEIRStateTransition.AsymptomaticToRecovery:
            return SEIRState.Recovered, InfectionType.Undefined, Severity.Undefined
        case SEIRStateTransition.MildToRecovery:
            return SEIRState.Recovered, InfectionType.Undefined, Severity.Undefined
        case SEIRStateTransition.SevereToRecovery:
            return SEIRState.Recovered, InfectionType.Undefined, Severity.Undefined
        case SEIRStateTransition.CriticalToRecovery:
            return SEIRState.Recovered, InfectionType.Undefined, Severity.Undefined
        case SEIRStateTransition.RecoveredToSusceptible:
            return SEIRState.Susceptible, current_infection_type, current_infection_severity
        case _:
            return SEIRState.Undefined, InfectionType.Undefined, Severity.Undefined