import numpy as np
import random
from epidemiology import SEIRState, SEIRStateTransition, InfectionType, Severity

def initialize_agent_states(n, initial_seir_state_distribution, agents_seir_state):
    partial_agents_seir_state = agents_seir_state[:n]

    for seirindex, seirstate_percent in enumerate(initial_seir_state_distribution):
        seirid = seirindex + 1

        total_to_assign_this_state = round(n * seirstate_percent)

        seirstate = SEIRState(seirid)

        undefined_indices = [i for i, x in enumerate(partial_agents_seir_state) if x == SEIRState.Undefined]

        if len(undefined_indices) > 0:
            undefined_indices = np.array(undefined_indices)

        this_state_indices = np.random.choice(undefined_indices, size=total_to_assign_this_state, replace=False)

        if len(this_state_indices) > 0:
            partial_agents_seir_state[this_state_indices] = np.array([seirstate for i in range(total_to_assign_this_state)])
        else:
            partial_agents_seir_state = []

    undefined_indices = [i for i, x in enumerate(partial_agents_seir_state) if x == SEIRState.Undefined]

    for index in undefined_indices:
        random_state = random.randint(1, 4)

        random_seir_state = SEIRState(random_state)

        partial_agents_seir_state[index] = random_seir_state

    return agents_seir_state

# to be called at the beginning of itinerary generation per day, per agent
# if day is in agent["state_transition_by_day"]
#   - update agent state (as required) - to be logged
#   - clear agent["state_transition_by_day"] for day
def update_agent_state(agents_seir_state, agents_infection_type, agents_infection_severity, agentid, agent, day):
    if day in agent["state_transition_by_day"]:
        seir_state_transition, timestep = agent["state_transition_by_day"][day]
        current_seir_state = agents_seir_state[agentid]
        current_infection_type = agents_infection_type[agentid]
        current_infection_severity = agents_infection_severity[agentid]

        new_seir_state, new_infection_type, new_infection_severity = convert_state_transition_to_new_state(current_seir_state, current_infection_type, current_infection_severity, seir_state_transition)

        if new_seir_state != current_seir_state: # to be logged
            agents_seir_state[agentid] = new_seir_state

        if new_infection_type != current_infection_type: # to be logged
            agents_infection_type[agentid] = new_infection_type

        if new_infection_severity != current_infection_severity: # to be logged
            agents_infection_severity[agentid] = new_infection_severity
            
        del agent["state_transition_by_day"][day]

        return new_seir_state, SEIRState(current_seir_state), new_infection_type, new_infection_severity, seir_state_transition, timestep

    return None

# returns new: (state, infection_type, severity)
def convert_state_transition_to_new_state(self, current_seir_state, current_infection_type, current_infection_severity, seir_state_transition):
    match seir_state_transition:
        case SEIRStateTransition.ExposedToInfectious:
            new_infection_type = current_infection_type
            if current_infection_type == InfectionType.PreAsymptomatic:
                new_infection_type = InfectionType.Asymptomatic
            # if PreSymptomatic, infection type will already be assigned, but is only to be considered infectious, if SEIR State is Infectious
            return SEIRState.Infectious, new_infection_type, current_infection_severity
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
        case SEIRStateTransition.RecoveredToExposed:
            return SEIRState.Exposed, current_infection_type, current_infection_severity
        case _:
            return SEIRState.Undefined, InfectionType.Undefined, Severity.Undefined