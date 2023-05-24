import numpy as np
import math
import random
from copy import copy
from simulator import util
from enum import IntEnum

class Epidemiology:
    def __init__(self, epidemiologyparams, agents, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_directcontacts_by_simcelltype_by_day, cells_households=None, cells_institutions=None, cells_accommodation=None):
        self.household_infection_probability = epidemiologyparams["household_infection_probability"]
        self.workplace_infection_probability = epidemiologyparams["workplace_infection_probability"]
        self.school_infection_probability = epidemiologyparams["school_infection_probability"]
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

        # intervention parameters
        self.testing_after_symptoms_probability = epidemiologyparams["testing_after_symptoms_probability"]
        self.testing_days_distribution_parameters = epidemiologyparams["testing_days_distribution_parameters"]
        quarantine_duration_params = epidemiologyparams["quarantine_duration_parameters"]
        self.quarantine_positive_duration, self.quarantine_positive_contact_duration, self.quarantine_secondary_contact_duration = quarantine_duration_params[0], quarantine_duration_params[1], quarantine_duration_params[2]
        self.testing_results_days_distribution_parameters = epidemiologyparams["testing_results_days_distribution_parameters"]
        testing_false_positive_negative_rates = epidemiologyparams["testing_false_positive_negative_rates"]
        self.testing_false_positive_rate, self.testing_false_negative_rate = testing_false_positive_negative_rates[0], testing_false_positive_negative_rates[1]
        self.ignore_quarantine_rules_probability = epidemiologyparams["ignore_quarantine_rules_probability"]
        contact_tracing_params = epidemiologyparams["contact_tracing_parameters"]
        self.contact_tracing_days_back, self.contact_tracing_positive_quarantine_prob, self.contact_tracing_secondary_quarantine_prob, self.contact_tracing_positive_test_prob, self.contact_tracing_secondary_test_prob = contact_tracing_params[0], contact_tracing_params[1], contact_tracing_params[2], contact_tracing_params[3], contact_tracing_params[4]
        self.contact_tracing_positive_delay_days, self.contact_tracing_secondary_delay_days = contact_tracing_params[5], contact_tracing_params[6]
        contact_tracing_success_probability = epidemiologyparams["contact_tracing_success_probability"]
        self.contact_tracing_residence_probability, self.contact_tracing_work_probability, self.contact_tracing_school_probability, self.contact_tracing_community_probability = contact_tracing_success_probability[0], contact_tracing_success_probability[1], contact_tracing_success_probability[2], contact_tracing_success_probability[3]
        self.masks_hygiene_distancing_parameters = epidemiologyparams["masks_hygiene_distancing_parameters"]
        self.daily_total_vaccinations = epidemiologyparams["daily_total_vaccinations"]
        self.vaccination_parameters = epidemiologyparams["vaccination_parameters"]
        self.immunity_after_vaccination_multiplier = epidemiologyparams["immunity_after_vaccination_multiplier"]       

        self.agents = agents
        self.agents_seir_state = agents_seir_state
        self.agents_seir_state_transition_for_day = agents_seir_state_transition_for_day
        self.agents_infection_type = agents_infection_type
        self.agents_infection_severity = agents_infection_severity
        self.agents_directcontacts_by_simcelltype_by_day = agents_directcontacts_by_simcelltype_by_day

        self.cells_households = cells_households
        self.cells_institutions = cells_institutions
        self.cells_accommodation = cells_accommodation

        self.timestep_options = np.arange(42, 121) # 7.00 - 20.00
        self.contact_tracing_agent_ids = []

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

    def simulate_direct_contacts(self, agents_directcontacts, cellid, cell, day):
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
                    
                    infection_probability = self.convert_celltype_to_base_infection_prob(cellid, exposed_agent["res_cellid"], cell["type"])

                    infection_multiplier = max(1, math.log(contact_duration)) # to check how this affects covasim base probs

                    infection_probability *= infection_multiplier * susceptibility_multiplier

                    exposed_rand = random.random()

                    symptomatic_day = -1
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
                            symptomatic_day = incremental_days

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

                        if symptomatic_day > -1:
                            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                            exposed_agent, test_scheduled, test_result_day = self.schedule_test(exposed_agent, exposed_agent_id, symptomatic_day, sampled_timestep, QuarantineType.Positive)
                            
                            start_quarantine_day = None
                            if test_scheduled:
                                if test_result_day < symptomatic_day:
                                    start_quarantine_day = test_result_day
                                else:
                                    start_quarantine_day = symptomatic_day
                            else:
                                start_quarantine_day = symptomatic_day

                            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                            exposed_agent, _ = self.schedule_quarantine(exposed_agent, start_quarantine_day, sampled_timestep, QuarantineType.Positive)

    def schedule_test(self, agent, agent_id, incremental_days, start_timestep, quarantine_type):
        test_already_scheduled = False
        if len(agent["test_day"]) > 0:
            if abs(agent["test_day"][0] - incremental_days) < 5:
                test_already_scheduled

        test_scheduled = False
        if not test_already_scheduled:
            test_after_symptoms_rand = -1

            if quarantine_type == QuarantineType.Positive:
                test_after_symptoms_rand = random.random()
            
            test_result_day = -1
            if test_after_symptoms_rand == -1 or test_after_symptoms_rand < self.testing_after_symptoms_probability:
                days_until_test = 0

                if quarantine_type == QuarantineType.PositiveContact:
                    days_until_test = self.contact_tracing_positive_delay_days
                elif quarantine_type == QuarantineType.SecondaryContact:
                    days_until_test = self.contact_tracing_secondary_delay_days

                days_until_test += util.sample_log_normal(self.testing_days_distribution_parameters[0], self.testing_days_distribution_parameters[1], 1, True)

                testing_day = incremental_days + days_until_test # incremental days here starts from symptomatic day

                agent["test_day"] = [testing_day, start_timestep]

                days_until_test_result = util.sample_log_normal(self.testing_results_days_distribution_parameters[0], self.testing_results_days_distribution_parameters[1], 1, True)

                test_result_day = testing_day + days_until_test_result

                agent["test_result_day"] = [test_result_day, start_timestep] # and we know agent is infected at this point (assume positive result)

                if quarantine_type == QuarantineType.Positive:
                    # to perform contact tracing (as received positive test result)
                    # contact tracing is handled globally at the end of every day, and contact tracing delays are represented in quarantine/testing scheduling
                    self.contact_tracing_agent_ids.append(agent_id) 
                
                test_scheduled = True

        return agent, test_scheduled, test_result_day
    
    def schedule_quarantine(self, agent, start_day, start_timestep, quarantine_type): # True if positive
        quarantine_days = agent["quarantine_days"]
        
        quarantine_scheduled = False

        to_schedule_quarantine = True

        if len(quarantine_days) > 0: # to clear quarantine_days when ready from quaratine (in itinerary)
            st_day_ts = agent["quarantine_days"][0][0]
            st_day = st_day_ts[0]

            if start_day > st_day:
                to_schedule_quarantine = False # don't schedule later

        if to_schedule_quarantine:
            if quarantine_type == QuarantineType.Positive: # positive
                end_day = start_day + self.quarantine_positive_duration
            elif quarantine_type == QuarantineType.PositiveContact:
                start_day += self.contact_tracing_positive_delay_days
                end_day = start_day + self.quarantine_positive_contact_duration
            elif quarantine_type == QuarantineType.SecondaryContact:
                start_day += self.contact_tracing_secondary_delay_days
                end_day = start_day + self.quarantine_secondary_contact_duration

            quarantine_days.append([[start_day, start_timestep], [end_day, start_timestep]])
            quarantine_scheduled = True

        return agent, quarantine_scheduled

    def update_quarantine_end(self, agent, new_end_day, new_end_timestep):
        quarantine_days = agent["quarantine_days"]

        quarantine_days[0, 1] = [new_end_day, new_end_timestep]

        return agent
    
    # the outer loop iterates "positive contacts" on simcelltype (residence, workplace, school, community contacts)
    # the positive agent would have had contacts in different places pertaining to the 4 "simcelltypes"
    # a percentage of these contacts, based on "simcelltype" will be successfully traced
    # a percentage of the secondary contacts, based on the simcelltype "residence" probability will also be traced
    # quarantine and tests will be scheduled accordingly according to the relevant percentages in the "contact_tracing_parameters"
    def contact_tracing(self, day):
        quarantine_scheduled_ids = []
        test_scheduled_ids = []

        for daybackindex in range(self.contact_tracing_days_back):
            dayback = day - daybackindex

            if dayback in self.agents_directcontacts_by_simcelltype_by_day:
                directcontacts_by_simcelltype = self.agents_directcontacts_by_simcelltype_by_day[dayback]

                for simcelltype, directcontacts in directcontacts_by_simcelltype.items(): # purely an iteration per simcelltype (x 4) with all directcontacts for each
                    for agent_id in self.contact_tracing_agent_ids:
                        contact_ids = util.get_all_contacts_ids_by_id(agent_id, directcontacts, shuffle=False)

                        if contact_ids is not None:
                            contact_tracing_success_prob = self.convert_simcelltype_to_contact_tracing_success_prob(simcelltype)

                            num_of_successfully_traced = round(len(contact_ids) * contact_tracing_success_prob)
                            
                            sampled_traced_contact_ids = np.random.choice(np.array(contact_ids), size=num_of_successfully_traced, replace=False)

                            sampled_traced_contact_indices = np.arange(len(sampled_traced_contact_ids))

                            num_of_quarantined = round(num_of_successfully_traced * self.contact_tracing_positive_quarantine_prob)

                            if num_of_quarantined == len(sampled_traced_contact_indices):
                                sampled_quarantine_indices = sampled_traced_contact_indices
                            else:
                                sampled_quarantine_indices = np.random.choice(sampled_traced_contact_indices, size=num_of_quarantined, replace=False)

                            num_of_tests = round(num_of_successfully_traced * self.contact_tracing_positive_test_prob)

                            if num_of_tests == len(sampled_traced_contact_indices):
                                sampled_test_indices = sampled_traced_contact_indices
                            else:
                                sampled_test_indices = np.random.choice(sampled_traced_contact_indices, size=num_of_tests, replace=False)

                            for index, contact_id in enumerate(sampled_traced_contact_ids):
                                positive_contact_agent = None

                                if index in sampled_quarantine_indices and contact_id not in quarantine_scheduled_ids:
                                    positive_contact_agent = self.agents[contact_id]
                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                    positive_contact_agent, quarantine_scheduled = self.schedule_quarantine(positive_contact_agent, day, sampled_timestep, QuarantineType.PositiveContact)

                                    if quarantine_scheduled:
                                        quarantine_scheduled_ids.append(contact_id)

                                if index in sampled_test_indices and contact_id not in test_scheduled_ids:
                                    if positive_contact_agent is None:
                                        positive_contact_agent = self.agents[contact_id]

                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                    positive_contact_agent, test_scheduled, _ = self.schedule_test(positive_contact_agent, contact_id, day, sampled_timestep, QuarantineType.PositiveContact)

                                    if test_scheduled:
                                        test_scheduled_ids.append(contact_id)

                                if simcelltype != "residence" and positive_contact_agent is not None: # compute secondary contacts (residential), if not residential
                                    res_cell_id = positive_contact_agent["res_cellid"]

                                    residence = None

                                    residents_key = ""
                                    staff_key = ""

                                    if res_cell_id in self.cells_households:
                                        residents_key = "resident_uids"
                                        staff_key = "staff_uids"
                                        residence = self.cells_households[res_cell_id]
                                    elif res_cell_id in self.cells_institutions:
                                        residents_key = "resident_uids"
                                        staff_key = "staff_uids"
                                        residence = self.cells_institutions[res_cell_id]["place"]
                                    elif res_cell_id in self.cells_accommodation:
                                        residents_key = "member_uids"
                                        residence = self.cells_accommodation[res_cell_id]["place"]

                                    if residence is not None:
                                        resident_ids = residence[residents_key]
                                        contact_id_index = np.argwhere(resident_ids == contact_id)
                                        resident_ids = np.delete(resident_ids, contact_id_index)

                                        employees_ids = []

                                        if staff_key != "":
                                            employees_ids = residence[staff_key]

                                        secondary_contact_ids = []
                                        if employees_ids is None or len(employees_ids) == 0:
                                            secondary_contact_ids = resident_ids
                                        else:
                                            try:
                                                secondary_contact_ids = np.concatenate((resident_ids, employees_ids))
                                            except Exception as e:
                                                print("problemos: " + e)

                                        if secondary_contact_ids is not None and len(secondary_contact_ids) > 0:
                                            num_of_sec_successfully_traced = round(len(secondary_contact_ids) * self.contact_tracing_residence_probability)
                                
                                            sampled_sec_traced_contact_ids = np.random.choice(np.array(secondary_contact_ids), size=num_of_sec_successfully_traced, replace=False)

                                            secondary_contact_indices = np.arange(len(sampled_sec_traced_contact_ids))

                                            num_of_sec_quarantined = round(num_of_sec_successfully_traced * self.contact_tracing_secondary_quarantine_prob)

                                            if num_of_sec_quarantined == len(secondary_contact_indices):
                                                sampled_sec_quarantine_indices = secondary_contact_indices
                                            else:
                                                sampled_sec_quarantine_indices = np.random.choice(secondary_contact_indices, size=num_of_sec_quarantined, replace=False)

                                            num_of_sec_tests = round(num_of_sec_successfully_traced * self.contact_tracing_secondary_test_prob)

                                            if num_of_sec_tests == len(secondary_contact_indices):
                                                sampled_sec_test_indices = secondary_contact_indices
                                            else:
                                                sampled_sec_test_indices = np.random.choice(secondary_contact_indices, size=num_of_sec_tests, replace=False)

                                            for sec_index, sec_contact_id in enumerate(secondary_contact_ids):
                                                secondary_contact_agent = None

                                                if sec_index in sampled_sec_quarantine_indices and sec_contact_id not in quarantine_scheduled_ids:
                                                    secondary_contact_agent = self.agents[sec_contact_id]
                                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                                    secondary_contact_agent, quarantine_scheduled = self.schedule_quarantine(secondary_contact_agent, day, sampled_timestep, QuarantineType.SecondaryContact)

                                                    if quarantine_scheduled:
                                                        quarantine_scheduled_ids.append(sec_contact_id)

                                                if sec_index in sampled_sec_test_indices and sec_contact_id not in test_scheduled_ids:
                                                    if secondary_contact_agent is None:
                                                        secondary_contact_agent = self.agents[sec_contact_id]

                                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                                    secondary_contact_agent, test_scheduled, _ = self.schedule_test(secondary_contact_agent, sec_contact_id, day, sampled_timestep, QuarantineType.SecondaryContact)

                                                    if test_scheduled:
                                                        test_scheduled_ids.append(contact_id)
        
        # clear for next day
        self.contact_tracing_agent_ids = []

    def schedule_vaccination(self, agent):
        print("to do")

    def convert_simcelltype_to_contact_tracing_success_prob(self, simcelltype):
        contact_tracing_success_prob = 0
        match simcelltype:
            case "residence":
                contact_tracing_success_prob = self.contact_tracing_residence_probability
            case "workplace":
                contact_tracing_success_prob = self.contact_tracing_work_probability
                # if rescellid == cellid:
                #     contact_tracing_success_prob = self.contact_tracing_work_probability
                # else:
                #     contact_tracing_success_prob = self.contact_tracing_community_probability
            case "school":
                contact_tracing_success_prob = self.contact_tracing_school_probability
            case "community":
                contact_tracing_success_prob = self.contact_tracing_community_probability
            case _:
                contact_tracing_success_prob = self.contact_tracing_community_probability

        return contact_tracing_success_prob

    def convert_celltype_to_base_infection_prob(self, cellid, rescellid, celltype=None):
        if celltype is None:
            cell = self.cells[cellid]
            celltype = cell["type"]

        simcelltype = util.convert_celltype_to_simcelltype(cellid, celltype=celltype)

        match simcelltype:
            case "residence":
                return self.household_infection_probability
            case "workplace":
                if cellid == rescellid:
                    return self.workplace_infection_probability # exposed agent is worker at a workplace
                else:
                    return self.community_infection_probability # exposed agent is a a visitor at a workplace (e.g. patrons at a restaurant)
            case "school":
                return self.school_infection_probability
            case "community":
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

class QuarantineType(IntEnum):
    Positive = 0,
    PositiveContact = 1,
    SecondaryContact = 2

# class InterventionAgentEvents(IntEnum):
#     Test = 0,
#     TestResult = 1,
#     Quarantine = 2,
#     ContactTracing = 3,
#     Vaccine = 4

class InterventionSimulationEvents(IntEnum):
    MasksHygeneDistancing = 0,
    ContactTracing = 1,
    PartialLockDown = 2,
    LockDown = 3