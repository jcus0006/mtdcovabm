import numpy as np
import math
import random
from simulator import util
from enum import IntEnum

class Epidemiology:
    def __init__(self, epidemiologyparams, agents, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity):
        self.household_infection_probability = epidemiologyparams["household_infection_probability"]
        self.workplace_school_infection_probability = epidemiologyparams["workplace_school_infection_probability"]
        self.community_infection_probability = epidemiologyparams["community_infection_probability"]
        self.susceptibility_progression_mortality_probs_by_age = epidemiologyparams["susceptibility_progression_mortality_probs_by_age"]
        durations_days_distribution_parameters = epidemiologyparams["durations_days_distribution_parameters"]

        exp_to_inf_dist_params, inf_to_symp_dist_params, symp_to_severe_dist_params, sev_to_cri_dist_params, cri_to_death_dist_params = durations_days_distribution_parameters[0], durations_days_distribution_parameters[1], durations_days_distribution_parameters[2], durations_days_distribution_parameters[3], durations_days_distribution_parameters[4]
        asymp_to_rec_dist_params, mild_to_rec_dist_params, sev_to_rec_dist_params, cri_to_rec_dist_params, rec_to_exp_min_max = durations_days_distribution_parameters[5], durations_days_distribution_parameters[6], durations_days_distribution_parameters[7], durations_days_distribution_parameters[8], durations_days_distribution_parameters[9]
        self.exp_to_inf_mean, self.exp_to_inf_std = exp_to_inf_dist_params[0], exp_to_inf_dist_params[1]
        self.inf_to_symp_mean, self.inf_to_symp_std = inf_to_symp_dist_params[0], inf_to_symp_dist_params[1]
        self.symp_to_sev_mean, self.symp_to_sev_std = symp_to_severe_dist_params[0], symp_to_severe_dist_params[1]
        self.sev_to_cri_mean, self.sev_to_cri_std = sev_to_cri_dist_params[0], sev_to_cri_dist_params[1]
        self.cri_to_death_mean, self.cri_to_death_std = cri_to_death_dist_params[0], cri_to_death_dist_params[1]
        self.asymp_to_rec_mean, self.asymp_to_rec_std = asymp_to_rec_dist_params[0], asymp_to_rec_dist_params[1]
        self.mild_to_rec_mean, self.mild_to_rec_std = mild_to_rec_dist_params[0], mild_to_rec_dist_params[1]
        self.sev_to_rec_mean, self.sev_to_rec_std = sev_to_rec_dist_params[0], sev_to_rec_dist_params[1]
        self.cri_to_rec_mean, self.cri_to_rec_std = cri_to_rec_dist_params[0], cri_to_rec_dist_params[1]
        self.rec_to_exp_mean, self.rec_to_exp_std = rec_to_exp_min_max[0], rec_to_exp_min_max[1]

        self.agents = agents
        self.agents_seir_state = agents_seir_state
        self.agents_seir_state_transition_for_day = agents_seir_state_transition_for_day
        self.agents_infection_type = agents_infection_type
        self.agents_infection_severity = agents_infection_severity

    def initialize_agent_states(self, n, initial_seir_state_distribution, agents_seir_state):
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
    def update_agent_state(self, agentid, agent, day):
        if day in agent["state_transition_by_day"]:
            seir_state_transition, timestep = agent["state_transition_by_day"][day]
            current_seir_state = self.agents_seir_state[agentid]
            current_infection_type = self.agents_infection_type[agentid]
            current_infection_severity = self.agents_infection_severity[agentid]

            new_seir_state, new_infection_type, new_infection_severity = self.convert_state_transition_to_new_state(current_seir_state, current_infection_type, current_infection_severity, seir_state_transition)

            if new_seir_state != current_seir_state: # to be logged
                self.agents_seir_state[agentid] = new_seir_state

            if new_infection_type != current_infection_type: # to be logged
                self.agents_infection_type[agentid] = new_infection_type

            if new_infection_severity != current_infection_severity: # to be logged
                self.agents_infection_severity[agentid] = new_infection_severity
                
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

    def simulate_direct_contacts(self, agents_directcontacts, cell, day):
        for pairid, timesteps in agents_directcontacts.items():
            primary_agent_id, secondary_agent_id = pairid[0], pairid[1]

            primary_agent_state, secondary_agent_state = self.agents_seir_state[primary_agent_id], self.agents_seir_state[secondary_agent_id]

            primary_agent_new_seir_state, primary_agent_old_seir_state, primary_agent_state_transition_timestep = None, None, None
            if primary_agent_id in self.agents_seir_state_transition_for_day:
                primary_agent_state_transition_for_day = self.agents_seir_state_transition_for_day[primary_agent_id]
                primary_agent_new_seir_state, primary_agent_old_seir_state, primary_agent_state_transition_timestep = primary_agent_state_transition_for_day[0], primary_agent_state_transition_for_day[1], primary_agent_state_transition_for_day[5]

            secondary_agent_new_seir_state, secondary_agent_old_seir_state, secondary_agent_state_transition_timestep = None, None, None
            if secondary_agent_id in self.agents_seir_state_transition_for_day:
                secondary_agent_state_transition_for_day = self.agents_seir_state_transition_for_day[secondary_agent_id]
                secondary_agent_new_seir_state, secondary_agent_old_seir_state, secondary_agent_state_transition_timestep = secondary_agent_state_transition_for_day[0], secondary_agent_state_transition_for_day[1], secondary_agent_state_transition_for_day[5]

            if (((primary_agent_state == SEIRState.Infectious and secondary_agent_state == SEIRState.Susceptible) or
                (primary_agent_state == SEIRState.Susceptible and secondary_agent_state == SEIRState.Infectious)) or 
                ((primary_agent_new_seir_state == SEIRState.Infectious or primary_agent_old_seir_state == SEIRState.Infectious) and
                (secondary_agent_new_seir_state == SEIRState.Susceptible or secondary_agent_old_seir_state == SEIRState.Susceptible)) or
                ((primary_agent_new_seir_state == SEIRState.Susceptible or primary_agent_old_seir_state == SEIRState.Susceptible) and
                (secondary_agent_new_seir_state == SEIRState.Infectious or secondary_agent_old_seir_state == SEIRState.Infectious))): # 1 infected and 1 susceptible (either or)  
                contact_duration = 0

                # (Timestep, True/False) True = Transition To Type, False = Transition From Type 
                # e.g. (20, True) not applicable prior to 20 as had not been transitioned to exposed/infected yet (consider from timestep 20 onwards)
                # e.g. (20, False) applicable prior to 20 as was still exposed/infected prior (applicable until 20)
                primary_timestep_cutoff, primary_transition_to_type = None, None 
                
                if primary_agent_new_seir_state is not None:
                    if primary_agent_old_seir_state == SEIRState.Exposed and primary_agent_new_seir_state == SEIRState.Infectious:
                        primary_timestep_cutoff, primary_transition_to_type = (primary_agent_state_transition_timestep, True)
                    elif primary_agent_old_seir_state == SEIRState.Infectious and primary_agent_new_seir_state == SEIRState.Recovered:
                        primary_timestep_cutoff, primary_transition_to_type = (primary_agent_state_transition_timestep, False)
                    elif primary_agent_old_seir_state == SEIRState.Recovered and primary_agent_new_seir_state == SEIRState.Exposed:
                        primary_timestep_cutoff, primary_transition_to_type = (primary_agent_state_transition_timestep, True)

                    primary_agent_state = primary_agent_new_seir_state

                secondary_timestep_cutoff, secondary_timestep_higher = None, None # (Timestep, True/False) True: Higher, False: Lower, e.g. (20, True) consider from timestep 20 onwards
                
                if secondary_agent_new_seir_state is not None:
                    if secondary_agent_old_seir_state == SEIRState.Exposed and secondary_agent_new_seir_state == SEIRState.Infectious:
                        secondary_timestep_cutoff, secondary_timestep_higher = (secondary_agent_state_transition_timestep, True)
                    elif secondary_agent_old_seir_state == SEIRState.Infectious and secondary_agent_new_seir_state == SEIRState.Recovered:
                        secondary_timestep_cutoff, secondary_timestep_higher = (secondary_agent_state_transition_timestep, False)
                    elif secondary_agent_old_seir_state == SEIRState.Recovered and secondary_agent_new_seir_state == SEIRState.Exposed:
                        secondary_timestep_cutoff, secondary_timestep_higher = (secondary_agent_state_transition_timestep, True)

                    secondary_agent_state = secondary_agent_new_seir_state

                overlapping_timesteps = []
                for start_timestep, end_timestep in timesteps:
                    add_timesteps = True

                    if primary_timestep_cutoff is not None:
                        if primary_transition_to_type:
                            if primary_timestep_cutoff > end_timestep: # transition happened "too-late", hence, does not apply
                                add_timesteps = False
                            elif primary_timestep_cutoff > start_timestep and primary_timestep_cutoff < end_timestep: # if co-inciding, transition time is the start_timestep (transition time until end_timestep)
                                start_timestep = primary_timestep_cutoff
                        else:
                            if primary_timestep_cutoff < start_timestep: # transition happened before, hence, does not apply
                                add_timesteps = False
                            elif primary_timestep_cutoff > start_timestep and primary_timestep_cutoff < end_timestep: # if co-inciding, transition time is the end_timestep (start_timestep until transition time)
                                end_timestep = primary_timestep_cutoff

                    if secondary_timestep_cutoff is not None:
                        if secondary_timestep_higher:
                            if secondary_timestep_cutoff > end_timestep:
                                add_timesteps = False
                            elif secondary_timestep_cutoff > start_timestep and secondary_timestep_cutoff < end_timestep:
                                start_timestep = secondary_timestep_cutoff
                        else:
                            if secondary_timestep_cutoff < start_timestep:
                                add_timesteps = False
                            elif secondary_timestep_cutoff > start_timestep and secondary_timestep_cutoff < end_timestep:
                                end_timestep = secondary_timestep_cutoff

                    if add_timesteps:
                        contact_duration += (end_timestep - start_timestep) + 1 # ensures that if overlapping range is the same (just 10 mins), it is considered 1 timestep
                        overlapping_timesteps.append(np.arange(start_timestep, end_timestep + 1))

                if contact_duration > 0:
                    overlapping_timesteps = [ts for ot in overlapping_timesteps for ts in ot] # flatten

                    if primary_agent_state == SEIRState.Susceptible:
                        exposed_agent_id = primary_agent_id
                        infectious_agent_id = secondary_agent_id
                    else:
                        exposed_agent_id = secondary_agent_id
                        infectious_agent_id = primary_agent_id

                    exposed_agent = self.agents[exposed_agent_id]

                    agent_state_transition_by_day = exposed_agent["state_transition_by_day"]
                    agent_epi_age_bracket_index = exposed_agent["epi_age_bracket_index"]

                    susceptibility_multiplier = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SusceptibilityMultiplier][agent_epi_age_bracket_index]
                        
                    infection_probability = self.convert_celltype_to_base_prob(None, cell["type"])

                    infection_multiplier = max(1, math.log(contact_duration)) # to see how this affects covasim base probs

                    infection_probability *= infection_multiplier * susceptibility_multiplier

                    exposed_rand = random.random()

                    recovered = False # if below condition is hit, False means Dead, True means recovered. 
                    incremental_days = day
                    if exposed_rand < infection_probability: # exposed (infected but not yet infectious)                  
                        self.agents_seir_state[exposed_agent_id] = SEIRState.Exposed
                        self.agents_infection_severity[exposed_agent_id] = Severity.Undefined

                        sampled_exposed_timestep = np.random.choice(overlapping_timesteps, size=1)[0]
                        
                        exp_to_inf_days = util.sample_log_normal(self.exp_to_inf_mean, self.exp_to_inf_std, 1, True)

                        incremental_days += exp_to_inf_days

                        agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.ExposedToInfectious, sampled_exposed_timestep)

                        symptomatic_rand = random.random()

                        symptomatic_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SymptomaticProbability][agent_epi_age_bracket_index]

                        if symptomatic_rand < symptomatic_probability:
                            self.agents_infection_type[exposed_agent_id] = InfectionType.PreSymptomatic # Pre-Symptomatic is infectious, but only applies with Infectious state (and not Exposed)

                            inf_to_symp_days = util.sample_log_normal(self.inf_to_symp_mean, self.inf_to_symp_std, 1, True)

                            incremental_days += inf_to_symp_days
                            agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.InfectiousToSymptomatic, sampled_exposed_timestep)

                            severe_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SevereProbability][agent_epi_age_bracket_index]

                            severe_rand = random.random()

                            if severe_rand < severe_probability:
                                symp_to_sev_days = util.sample_log_normal(self.symp_to_sev_mean, self.symp_to_sev_std, 1, True)

                                incremental_days += symp_to_sev_days

                                agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.SymptomaticToSevere, sampled_exposed_timestep)

                                critical_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.CriticalProbability][agent_epi_age_bracket_index]
                                
                                critical_rand = random.random()

                                if critical_rand < critical_probability:
                                    sev_to_cri_days = util.sample_log_normal(self.sev_to_cri_mean, self.sev_to_cri_std, 1, True)

                                    incremental_days += sev_to_cri_days

                                    agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.SevereToCritical, sampled_exposed_timestep)

                                    deceased_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.DeceasedProbability][exposed_agent["epi_age_bracket_index"]]
                        
                                    deceased_rand = random.random()

                                    if deceased_rand < deceased_probability:
                                        cri_to_death_days = util.sample_log_normal(self.cri_to_death_mean, self.cri_to_death_std, 1, True)

                                        incremental_days += cri_to_death_days

                                        agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.CriticalToDeath, sampled_exposed_timestep)
                                    else:
                                        # critical
                                        cri_to_rec_days = util.sample_log_normal(self.cri_to_rec_mean, self.cri_to_rec_std, 1, True)

                                        incremental_days += cri_to_rec_days

                                        agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.CriticalToRecovery, sampled_exposed_timestep)

                                        recovered = True
                                else:
                                    # severe
                                    sev_to_rec_days = util.sample_log_normal(self.sev_to_rec_mean, self.sev_to_rec_std, 1, True)

                                    incremental_days += sev_to_rec_days

                                    agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.SevereToRecovery, sampled_exposed_timestep)

                                    recovered = True
                            else:
                                # mild
                                mild_to_rec_days = util.sample_log_normal(self.mild_to_rec_mean, self.mild_to_rec_std, 1, True)

                                incremental_days += mild_to_rec_days

                                agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.MildToRecovery, sampled_exposed_timestep)

                                recovered = True
                        else:
                            # asymptomatic
                            self.agents_infection_type[exposed_agent_id] = InfectionType.PreAsymptomatic # Pre-Asymptomatic is infectious, but only applies with Infectious state (and not Exposed)

                            asymp_to_rec_days = util.sample_log_normal(self.asymp_to_rec_mean, self.asymp_to_rec_std, 1, True)

                            incremental_days += asymp_to_rec_days

                            agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.AsymptomaticToRecovery, sampled_exposed_timestep)

                            recovered = True

                        if recovered: # else means deceased
                            rec_to_exp_days = util.sample_log_normal(self.rec_to_exp_mean, self.rec_to_exp_std, 1, True)

                            incremental_days += rec_to_exp_days

                            agent_state_transition_by_day[incremental_days] = (SEIRStateTransition.RecoveredToExposed, sampled_exposed_timestep)

    def convert_celltype_to_base_prob(self, cellid, celltype=None):
        if celltype is None:
            cell = self.cells[cellid]
            celltype = cell["type"]

        match celltype:
            case "household":
                return self.household_infection_probability
            case "workplace":
                return self.workplace_school_infection_probability
            case "accom":
                return self.household_infection_probability
            case "hospital":
                return self.community_infection_probability
            case "entertainment":
                return self.community_infection_probability
            case "school":
                return self.workplace_school_infection_probability
            case "institution":
                return self.household_infection_probability
            case "transport":
                return self.community_infection_probability
            case "religion":
                return self.community_infection_probability
            case "airport":
                return self.community_infection_probability
            case _:
                return self.community_infection_probability
            
    def get_sus_mort_prog_age_bracket_index(self, age):
        if age < 0 or age > 100:
            return None
        elif age < 10:
            return 0
        elif age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 40:
            return 3
        elif age < 50:
            return 4
        elif age < 60:
            return 5
        elif age < 70:
            return 6
        elif age < 80:
            return 7
        elif age < 90:
            return 8
        else:
            return 9

class SEIRState(IntEnum):
    Undefined = 0
    Susceptible = 1 # not infected, susceptible to become exposed
    Exposed = 2 # infected, not yet infectious
    Infectious = 3 # infected and infectious
    Recovered = 4 # recovered, immune for an immunity period
    Deceased = 5 # deceased

class InfectionType(IntEnum):
    Undefined = 0
    PreAsymptomatic = 1 # infected, no symptoms yet, will not have symptoms (not infectious yet but will be)
    PreSymptomatic = 2 # infected, no symptoms yet, will have symptoms (not infectious yet but will be)
    Asymptomatic = 3 # infected, no symptoms, infectious
    Symptomatic = 4 # infected, with symptoms, infectious

class Severity(IntEnum):
    Undefined = 0
    Mild = 1 # mild symptoms
    Severe = 2 # severe symptoms
    Critical = 3 # critical symptoms

class EpidemiologyProbabilities(IntEnum):
    SusceptibilityMultiplier = 0
    SymptomaticProbability = 1
    SevereProbability = 2
    CriticalProbability = 3
    DeceasedProbability = 4

# state_transition_by_day - handled per agent: { day : (state_transition, timestep)}
# in itinerary, if current day exists in state_transition_by_day, switch the state in the agents_seir_state (with the new SEIRState enum), and update the agents_seir_state_transition (with the timestep)
# when an infected agent meets a susceptible agent, the virus transmission model below is activated for the pair
# handle InfectionType and Severity in agents_infection_type, and agents_infection_severity dicts (which only contains the keys of currently infected agents)
# to do - include the gene + the LTI (as multipliers within the infection probability)
# to handle - mask-wearing + quarantine + testing + vaccination + contact tracing
class SEIRStateTransition(IntEnum):
    ExposedToInfectious = 0, # in the case of a direct contact, (base_prob * susc_multiplier) chance of being exposed: if exposed (infected, not yet infectious), sample ExpToInfDays
    InfectiousToSymptomatic = 1, # if exposed/infected, compute symptomatic_probability: if symptomatic, assign "Presymptomatic", sample InfToSymp, assign "Mild" after InfToSymp, else, assign "Asymptomatic"
    SymptomaticToSevere = 2, # if symptomatic, compute severe_probability: if severe, sample SympToSev, assign "Severe" after InfToSymp + SympToSev
    SevereToCritical = 3, # if severe, compute critical_probability: if critical, sample SevToCri, assign "Critical" after InfToSymp + SympToSev + SevToCri
    CriticalToDeath = 4, # if critical, compute death_probability: if dead, sample CriToDea, send to "dead cell" after InfToSymp + SympToSev + SevToCri + CriToDea
    AsymptomaticToRecovery = 5, # if asymptomatic (not symptomatic), sample AsympToRec, assign "Recovered" after AsympToRec
    MildToRecovery = 6, # if mild, sample MildToRec, assign "Recovered" after MildToRec
    SevereToRecovery = 7, # if severe, sample SevToRec, assign "Recovered" after SevToRec
    CriticalToRecovery = 8 # if critical, sample CriToRec, assign "Recovered" after CriToRec
    RecoveredToExposed = 9 # if recovered, sample RecToExp (uniform from range e.g. 30-90 days), assign "Exposed" after RecToExp