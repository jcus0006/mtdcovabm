import numpy as np
import math
import random
import time
from copy import copy
import util
import seirstateutil
from epidemiologyclasses import SEIRState, SEIRStateTransition, InfectionType, Severity, EpidemiologyProbabilities, QuarantineType
from cellsclasses import CellType, SimCellType
import gc
from memory_profiler import profile
from bintrees import FastAVLTree

class Epidemiology:
    def __init__(self, 
                epidemiologyparams, 
                n_locals, 
                n_tourists, 
                locals_ratio_to_full_pop, 
                agents_static,
                agents_epi,
                vars_util,
                cells_households, 
                cells_institutions, 
                cells_accommodation, 
                dyn_params,
                process_index=-1,
                agents_seir_indices=None): # only used when called from contact network
        self.n_locals = n_locals
        self.n_tourists = n_tourists
        self.locals_ratio_to_full_pop = locals_ratio_to_full_pop
        self.household_infection_probability = epidemiologyparams["household_infection_probability"]
        self.workplace_infection_probability = epidemiologyparams["workplace_infection_probability"]
        self.school_infection_probability = epidemiologyparams["school_infection_probability"]
        self.community_infection_probability = epidemiologyparams["community_infection_probability"]
        self.susceptibility_progression_mortality_probs_by_age = epidemiologyparams["susceptibility_progression_mortality_probs_by_age"]
        durations_days_distribution_parameters = epidemiologyparams["durations_days_distribution_parameters"]

        exp_to_inf_dist_params, inf_to_symp_dist_params, symp_to_severe_dist_params, sev_to_cri_dist_params, cri_to_death_dist_params = durations_days_distribution_parameters[0], durations_days_distribution_parameters[1], durations_days_distribution_parameters[2], durations_days_distribution_parameters[3], durations_days_distribution_parameters[4]
        asymp_to_rec_dist_params, mild_to_rec_dist_params, sev_to_rec_dist_params, cri_to_rec_dist_params, rec_to_sus_min_max = durations_days_distribution_parameters[5], durations_days_distribution_parameters[6], durations_days_distribution_parameters[7], durations_days_distribution_parameters[8], durations_days_distribution_parameters[9]
        self.exp_to_inf_mean, self.exp_to_inf_std = exp_to_inf_dist_params[0], exp_to_inf_dist_params[1]
        self.inf_to_symp_mean, self.inf_to_symp_std = inf_to_symp_dist_params[0], inf_to_symp_dist_params[1]
        self.symp_to_sev_mean, self.symp_to_sev_std = symp_to_severe_dist_params[0], symp_to_severe_dist_params[1]
        self.sev_to_cri_mean, self.sev_to_cri_std = sev_to_cri_dist_params[0], sev_to_cri_dist_params[1]
        self.cri_to_death_mean, self.cri_to_death_std = cri_to_death_dist_params[0], cri_to_death_dist_params[1]
        self.asymp_to_rec_mean, self.asymp_to_rec_std = asymp_to_rec_dist_params[0], asymp_to_rec_dist_params[1]
        self.mild_to_rec_mean, self.mild_to_rec_std = mild_to_rec_dist_params[0], mild_to_rec_dist_params[1]
        self.sev_to_rec_mean, self.sev_to_rec_std = sev_to_rec_dist_params[0], sev_to_rec_dist_params[1]
        self.cri_to_rec_mean, self.cri_to_rec_std = cri_to_rec_dist_params[0], cri_to_rec_dist_params[1]
        self.rec_to_sus_mean, self.rec_to_sus_std = rec_to_sus_min_max[0], rec_to_sus_min_max[1]

        # intervention parameters
        self.testing_after_symptoms_probability = epidemiologyparams["testing_after_symptoms_probability"]
        self.testing_days_distribution_parameters = epidemiologyparams["testing_days_distribution_parameters"]
        quarantine_duration_params = epidemiologyparams["quarantine_duration_parameters"]
        self.quarantine_positive_duration, self.quarantine_positive_contact_duration, self.quarantine_secondary_contact_duration = quarantine_duration_params[0], quarantine_duration_params[1], quarantine_duration_params[2]
        self.quarantine_after_symptoms_probability = epidemiologyparams["quarantine_after_symptoms_probability"]
        self.testing_results_days_distribution_parameters = epidemiologyparams["testing_results_days_distribution_parameters"]
        testing_false_positive_negative_rates = epidemiologyparams["testing_false_positive_negative_rates"]
        self.testing_false_positive_rate, self.testing_false_negative_rate = testing_false_positive_negative_rates[0], testing_false_positive_negative_rates[1]
        self.ignore_quarantine_rules_probability = epidemiologyparams["ignore_quarantine_rules_probability"]
        contact_tracing_params = epidemiologyparams["contact_tracing_parameters"]
        self.contact_tracing_days_back, self.contact_tracing_positive_delay_days, self.contact_tracing_secondary_delay_days = contact_tracing_params[0], contact_tracing_params[1], contact_tracing_params[2]
        contact_tracing_success_probability = epidemiologyparams["contact_tracing_success_probability"]
        self.contact_tracing_residence_probability, self.contact_tracing_work_probability, self.contact_tracing_school_probability, self.contact_tracing_community_probability = contact_tracing_success_probability[0], contact_tracing_success_probability[1], contact_tracing_success_probability[2], contact_tracing_success_probability[3]
        self.vaccination_effectiveness_parameters = epidemiologyparams["vaccination_effectiveness_parameters"]
        self.vaccination_immunity_multiplier, self.vaccination_asymptomatic_multiplier, self.vaccination_exp_decay_interval = self.vaccination_effectiveness_parameters[0], self.vaccination_effectiveness_parameters[1], self.vaccination_effectiveness_parameters[2]
        self.vaccination_daily_parameters = epidemiologyparams["vaccination_daily_parameters"]
        self.lockdown_infectiousrate_thresholds = epidemiologyparams["lockdown_infectiousrate_thresholds"]
        self.lockdown_day_thresholds = epidemiologyparams["lockdown_day_thresholds"]

        self.dyn_params = dyn_params 
        # self.sync_queue = sync_queue    

        self.agents_static = agents_static
        self.agents_epi = agents_epi
        self.agents_seir_indices = agents_seir_indices
        self.vars_util = vars_util

        self.cells_households = cells_households
        self.cells_institutions = cells_institutions
        self.cells_accommodation = cells_accommodation

        self.timestep_options = np.arange(42, 121) # 7.00 - 20.00

        self.process_index = process_index

    def simulate_direct_contacts(self, agents_directcontacts, cellid, cell_type, day):
        updated_agents_ids = []

        for pairid, timesteps in agents_directcontacts.items():
            primary_agent_id, secondary_agent_id = pairid[0], pairid[1]

            primary_agent_state, secondary_agent_state = seirstateutil.agents_seir_state_get(self.vars_util.agents_seir_state, primary_agent_id, self.agents_seir_indices), seirstateutil.agents_seir_state_get(self.vars_util.agents_seir_state, secondary_agent_id, self.agents_seir_indices)
            # Imp: Not considering in-day transition anymore. an agent is considered as infectious or suscetible for whole day
            # primary_agent_new_seir_state, primary_agent_old_seir_state, primary_agent_state_transition_timestep = None, None, None
            # if primary_agent_id in self.vars_util.agents_seir_state_transition_for_day:
            #     primary_agent_state_transition_for_day = self.vars_util.agents_seir_state_transition_for_day[primary_agent_id]
            #     primary_agent_new_seir_state, primary_agent_old_seir_state, primary_agent_state_transition_timestep = primary_agent_state_transition_for_day[0], primary_agent_state_transition_for_day[1], primary_agent_state_transition_for_day[5]

            # secondary_agent_new_seir_state, secondary_agent_old_seir_state, secondary_agent_state_transition_timestep = None, None, None
            # if secondary_agent_id in self.vars_util.agents_seir_state_transition_for_day:
            #     secondary_agent_state_transition_for_day = self.vars_util.agents_seir_state_transition_for_day[secondary_agent_id]
            #     secondary_agent_new_seir_state, secondary_agent_old_seir_state, secondary_agent_state_transition_timestep = secondary_agent_state_transition_for_day[0], secondary_agent_state_transition_for_day[1], secondary_agent_state_transition_for_day[5]

            if ((primary_agent_state == SEIRState.Infectious and secondary_agent_state == SEIRState.Susceptible) or
                (primary_agent_state == SEIRState.Susceptible and secondary_agent_state == SEIRState.Infectious)): # or 
                # ((primary_agent_new_seir_state == SEIRState.Infectious or primary_agent_old_seir_state == SEIRState.Infectious) and
                # (secondary_agent_new_seir_state == SEIRState.Susceptible or secondary_agent_old_seir_state == SEIRState.Susceptible)) or
                # ((primary_agent_new_seir_state == SEIRState.Susceptible or primary_agent_old_seir_state == SEIRState.Susceptible) and
                # (secondary_agent_new_seir_state == SEIRState.Infectious or secondary_agent_old_seir_state == SEIRState.Infectious))): # 1 infected and 1 susceptible (either or)  
                contact_duration = 0

                # (Timestep, True/False) True = Transition To Type, False = Transition From Type 
                # e.g. (20, True) not applicable prior to 20 as had not been transitioned to exposed/infected yet (consider from timestep 20 onwards)
                # e.g. (20, False) applicable prior to 20 as was still exposed/infected prior (applicable until 20)
                # primary_timestep_cutoff, primary_transition_to_type = None, None 
                
                # if primary_agent_new_seir_state is not None:
                #     if primary_agent_old_seir_state == SEIRState.Exposed and primary_agent_new_seir_state == SEIRState.Infectious:
                #         primary_timestep_cutoff, primary_transition_to_type = (primary_agent_state_transition_timestep, True)
                #     elif primary_agent_old_seir_state == SEIRState.Infectious and primary_agent_new_seir_state == SEIRState.Recovered:
                #         primary_timestep_cutoff, primary_transition_to_type = (primary_agent_state_transition_timestep, False)
                #     elif primary_agent_old_seir_state == SEIRState.Recovered and primary_agent_new_seir_state == SEIRState.Exposed:
                #         primary_timestep_cutoff, primary_transition_to_type = (primary_agent_state_transition_timestep, True)

                #     primary_agent_state = primary_agent_new_seir_state

                # secondary_timestep_cutoff, secondary_timestep_higher = None, None # (Timestep, True/False) True: Higher, False: Lower, e.g. (20, True) consider from timestep 20 onwards
                
                # if secondary_agent_new_seir_state is not None:
                #     if secondary_agent_old_seir_state == SEIRState.Exposed and secondary_agent_new_seir_state == SEIRState.Infectious:
                #         secondary_timestep_cutoff, secondary_timestep_higher = (secondary_agent_state_transition_timestep, True)
                #     elif secondary_agent_old_seir_state == SEIRState.Infectious and secondary_agent_new_seir_state == SEIRState.Recovered:
                #         secondary_timestep_cutoff, secondary_timestep_higher = (secondary_agent_state_transition_timestep, False)
                #     elif secondary_agent_old_seir_state == SEIRState.Recovered and secondary_agent_new_seir_state == SEIRState.Exposed:
                #         secondary_timestep_cutoff, secondary_timestep_higher = (secondary_agent_state_transition_timestep, True)

                #     secondary_agent_state = secondary_agent_new_seir_state

                overlapping_timesteps = []
                for start_timestep, end_timestep in timesteps:
                    # add_timesteps = True

                    # if primary_timestep_cutoff is not None:
                    #     if primary_transition_to_type:
                    #         if primary_timestep_cutoff > end_timestep: # transition happened "too-late", hence, does not apply
                    #             add_timesteps = False
                    #         elif primary_timestep_cutoff > start_timestep and primary_timestep_cutoff < end_timestep: # if co-inciding, transition time is the start_timestep (transition time until end_timestep)
                    #             start_timestep = primary_timestep_cutoff
                    #     else:
                    #         if primary_timestep_cutoff < start_timestep: # transition happened before, hence, does not apply
                    #             add_timesteps = False
                    #         elif primary_timestep_cutoff > start_timestep and primary_timestep_cutoff < end_timestep: # if co-inciding, transition time is the end_timestep (start_timestep until transition time)
                    #             end_timestep = primary_timestep_cutoff

                    # if secondary_timestep_cutoff is not None:
                    #     if secondary_timestep_higher:
                    #         if secondary_timestep_cutoff > end_timestep:
                    #             add_timesteps = False
                    #         elif secondary_timestep_cutoff > start_timestep and secondary_timestep_cutoff < end_timestep:
                    #             start_timestep = secondary_timestep_cutoff
                    #     else:
                    #         if secondary_timestep_cutoff < start_timestep:
                    #             add_timesteps = False
                    #         elif secondary_timestep_cutoff > start_timestep and secondary_timestep_cutoff < end_timestep:
                    #             end_timestep = secondary_timestep_cutoff

                    # if add_timesteps:
                    contact_duration += (end_timestep - start_timestep) + 1 # ensures that if overlapping range is the same (just 10 mins), it is considered 1 timestep
                    overlapping_timesteps.append(np.arange(start_timestep, end_timestep + 1))

                if contact_duration > 0:
                    overlapping_timesteps = [ts for ot in overlapping_timesteps for ts in ot] # flatten

                    if primary_agent_state == SEIRState.Susceptible:
                        exposed_agent_id = primary_agent_id
                        # infectious_agent_id = secondary_agent_id
                    else:
                        exposed_agent_id = secondary_agent_id
                        # infectious_agent_id = primary_agent_id

                    exposed_agent_epi = self.agents_epi[exposed_agent_id]

                    agent_epi_age_bracket_index = self.agents_static.get(exposed_agent_id, "epi_age_bracket_index")

                    susceptibility_multiplier = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SusceptibilityMultiplier][agent_epi_age_bracket_index]
                    
                    infection_probability = self.convert_celltype_to_base_infection_prob(cellid, self.agents_static.get(exposed_agent_id, "res_cellid"), cell_type)

                    infection_multiplier = max(1, math.log(contact_duration)) # to check how this affects covasim base probs

                    # immunity_multiplier, asymptomatic_multiplier = 1, 1
                    immunity_multiplier, asymptomatic_multiplier, _ = util.calculate_vaccination_multipliers(self.vars_util.agents_vaccination_doses, exposed_agent_id, day, self.vaccination_immunity_multiplier, self.vaccination_asymptomatic_multiplier, self.vaccination_exp_decay_interval)
                    
                    infection_probability *= infection_multiplier * susceptibility_multiplier * (1 - self.dyn_params.masks_hygiene_distancing_multiplier) * immunity_multiplier

                    exposed_rand = random.random()

                    incremental_days = day
                    if exposed_rand < infection_probability: # exposed (infected but not yet infectious)
                        updated_agents_ids.append(exposed_agent_id)

                        # if "state_transition_by_day" not in exposed_agent_epi:
                        #     print(f"exposed_agent_epi {exposed_agent_id} is empty. will crash.")

                        agent_state_transition_by_day = exposed_agent_epi["state_transition_by_day"]
                        agent_quarantine_days = exposed_agent_epi["quarantine_days"]

                        if agent_state_transition_by_day is None:
                            agent_state_transition_by_day = []

                        agent_state_transition_by_day, agent_seir_state, agent_infection_type, agent_infection_severity, _ = self.simulate_seir_state_transition(day, exposed_agent_epi, exposed_agent_id, incremental_days, overlapping_timesteps, agent_state_transition_by_day, agent_epi_age_bracket_index, agent_quarantine_days, asymptomatic_multiplier)

                        exposed_agent_epi["state_transition_by_day"] = agent_state_transition_by_day
                        exposed_agent_epi["quarantine_days"] = agent_quarantine_days
                        # self.agents_mp.set(exposed_agent_id, "state_transition_by_day", agent_state_transition_by_day)
                        # self.sync_queue.put(["a", exposed_agent_id, "state_transition_by_day", agent_state_transition_by_day]) # updated by ref in agent           

                        # self.sync_queue.put(["v", exposed_agent_id, "agents_seir_state", agent_seir_state])
                        self.vars_util.agents_seir_state = seirstateutil.agents_seir_state_update(self.vars_util.agents_seir_state, agent_seir_state, exposed_agent_id, self.agents_seir_indices)
                        # self.sync_queue.put(["v", exposed_agent_id, "agents_infection_type", agent_infection_type])
                        self.vars_util.agents_infection_type[exposed_agent_id] = agent_infection_type
                        # self.sync_queue.put(["v", exposed_agent_id, "agents_infection_severity", agent_infection_severity])
                        self.vars_util.agents_infection_severity[exposed_agent_id] = agent_infection_severity

                        # print(f"updating agent {exposed_agent_id}, state {self.vars_util.agents_seir_state[exposed_agent_id]}, inf_type {self.vars_util.agents_infection_type[exposed_agent_id]}, inf_sev {self.vars_util.agents_infection_severity[exposed_agent_id]}")
                        
        # print(f"vaccination multiplier total time taken {vacc_mult_time_taken}")

        return updated_agents_ids

    def simulate_seir_state_transition(self, current_day, exposed_agent, exposed_agent_id, incremental_days, overlapping_timesteps, agent_state_transition_by_day, agent_epi_age_bracket_index, agent_quarantine_days, asymptomatic_multiplier = 1.0):
        symptomatic_day = -1
        recovered = False # if below condition is hit, False means Dead, True means recovered. 
        start_hosp_day, end_hosp_day = None, None
        
        seir_state = SEIRState.Exposed
        infection_type = InfectionType.Undefined
        infection_severity = Severity.Undefined

        # exposed would be prior to day 1 (and hence would not be accurate), reset to the PREVIOUS states of first set SEIRStateTransition
        reset_states = incremental_days < 1 
        curr_seir_state, curr_inf_type, curr_inf_sev = None, None, None

        sampled_exposed_timestep = np.random.choice(overlapping_timesteps, size=1)[0]
        
        exp_to_inf_days = util.sample_log_normal(self.exp_to_inf_mean, self.exp_to_inf_std, 1, True)

        incremental_days += exp_to_inf_days

        # if agent_state_transition_by_day is None:
        #     agent_state_transition_by_day = []

        if incremental_days >= current_day:
            agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.ExposedToInfectious, sampled_exposed_timestep])
            
            if reset_states:
                if curr_seir_state is None:
                    curr_seir_state = SEIRState.Exposed # was Infectious

                if curr_inf_type is None:
                    curr_inf_type = InfectionType.PreAsymptomatic

                if curr_inf_sev is None:
                    curr_inf_sev = InfectionType.Undefined
        
        symptomatic_rand = random.random()

        symptomatic_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SymptomaticProbability][agent_epi_age_bracket_index] * asymptomatic_multiplier # if effectiveness against symptoms is 0.9, this will be 1 - 0.9 = 0.1

        if symptomatic_rand < symptomatic_probability:
            infection_type = InfectionType.PreSymptomatic # Pre-Symptomatic is infectious, but only applies with Infectious state (and not Exposed)

            inf_to_symp_days = util.sample_log_normal(self.inf_to_symp_mean, self.inf_to_symp_std, 1, True)

            incremental_days += inf_to_symp_days
            symptomatic_day = incremental_days

            if incremental_days >= current_day:
                agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.InfectiousToSymptomatic, sampled_exposed_timestep]) 

                if reset_states:
                    if curr_seir_state is None:
                        curr_seir_state = SEIRState.Infectious

                    if curr_inf_type is None:
                        curr_inf_type = InfectionType.PreSymptomatic

                    if curr_inf_sev is None:
                        curr_inf_sev = Severity.Undefined
            
            severe_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SevereProbability][agent_epi_age_bracket_index]

            severe_rand = random.random()

            if severe_rand < severe_probability:
                symp_to_sev_days = util.sample_log_normal(self.symp_to_sev_mean, self.symp_to_sev_std, 1, True)

                incremental_days += symp_to_sev_days

                start_hosp_day = incremental_days

                if incremental_days >= current_day:
                    agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.SymptomaticToSevere, sampled_exposed_timestep]) 

                    if reset_states:
                        if curr_seir_state is None:
                            curr_seir_state = SEIRState.Infectious

                        if curr_inf_type is None:
                            curr_inf_type = InfectionType.Symptomatic

                        if curr_inf_sev is None:
                            curr_inf_sev = Severity.Mild
                
                critical_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.CriticalProbability][agent_epi_age_bracket_index]
                
                critical_rand = random.random()

                if critical_rand < critical_probability:
                    sev_to_cri_days = util.sample_log_normal(self.sev_to_cri_mean, self.sev_to_cri_std, 1, True)

                    incremental_days += sev_to_cri_days

                    if incremental_days >= current_day:
                        agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.SevereToCritical, sampled_exposed_timestep]) 

                        if reset_states:
                            if curr_seir_state is None:
                                curr_seir_state = SEIRState.Infectious

                            if curr_inf_type is None:
                                curr_inf_type = InfectionType.Symptomatic

                            if curr_inf_sev is None:
                                curr_inf_sev = Severity.Severe             

                    deceased_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.DeceasedProbability][agent_epi_age_bracket_index]
        
                    deceased_rand = random.random()

                    if deceased_rand < deceased_probability:
                        cri_to_death_days = util.sample_log_normal(self.cri_to_death_mean, self.cri_to_death_std, 1, True)

                        incremental_days += cri_to_death_days

                        end_hosp_day = incremental_days

                        if incremental_days >= current_day:
                            agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.CriticalToDeath, sampled_exposed_timestep]) 

                            if reset_states:
                                if curr_seir_state is None:
                                    curr_seir_state = SEIRState.Infectious

                                if curr_inf_type is None:
                                    curr_inf_type = InfectionType.Symptomatic

                                if curr_inf_sev is None:
                                    curr_inf_sev = Severity.Critical
                    else:
                        # critical
                        cri_to_rec_days = util.sample_log_normal(self.cri_to_rec_mean, self.cri_to_rec_std, 1, True)

                        incremental_days += cri_to_rec_days

                        end_hosp_day = incremental_days

                        if incremental_days >= current_day:
                            agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.CriticalToRecovery, sampled_exposed_timestep]) 

                            if reset_states:
                                if curr_seir_state is None:
                                    curr_seir_state = SEIRState.Infectious

                                if curr_inf_type is None:
                                    curr_inf_type = InfectionType.Symptomatic

                                if curr_inf_sev is None:
                                    curr_inf_sev = Severity.Critical

                        recovered = True
                else:
                    # severe
                    sev_to_rec_days = util.sample_log_normal(self.sev_to_rec_mean, self.sev_to_rec_std, 1, True)

                    incremental_days += sev_to_rec_days

                    end_hosp_day = incremental_days

                    if incremental_days >= current_day:
                        agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.SevereToRecovery, sampled_exposed_timestep])

                        if reset_states:
                            if curr_seir_state is None:
                                curr_seir_state = SEIRState.Infectious

                            if curr_inf_type is None:
                                curr_inf_type = InfectionType.Symptomatic

                            if curr_inf_sev is None:
                                curr_inf_sev = Severity.Severe

                    recovered = True
            else:
                # mild
                mild_to_rec_days = util.sample_log_normal(self.mild_to_rec_mean, self.mild_to_rec_std, 1, True)

                incremental_days += mild_to_rec_days
                
                if incremental_days >= current_day:
                    agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.MildToRecovery, sampled_exposed_timestep])

                    if reset_states:
                        if curr_seir_state is None:
                            curr_seir_state = SEIRState.Infectious

                        if curr_inf_type is None:
                            curr_inf_type = InfectionType.Symptomatic

                        if curr_inf_sev is None:
                            curr_inf_sev = Severity.Mild

                recovered = True
        else:
            # asymptomatic
            infection_type = InfectionType.PreAsymptomatic # Pre-Asymptomatic is infectious, but only applies with Infectious state (and not Exposed)                        

            asymp_to_rec_days = util.sample_log_normal(self.asymp_to_rec_mean, self.asymp_to_rec_std, 1, True)

            incremental_days += asymp_to_rec_days
            
            if incremental_days >= current_day:
                agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.AsymptomaticToRecovery, sampled_exposed_timestep])

                if reset_states:
                    if curr_seir_state is None:
                        curr_seir_state = SEIRState.Infectious

                    if curr_inf_type is None:
                        curr_inf_type = InfectionType.Asymptomatic

                    if curr_inf_sev is None:
                        curr_inf_sev = Severity.Undefined

            recovered = True

        if recovered: # else means deceased
            rec_to_sus_days = util.sample_log_normal(self.rec_to_sus_mean, self.rec_to_sus_std, 1, True)

            incremental_days += rec_to_sus_days

            if incremental_days >= current_day:
                agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.RecoveredToSusceptible, sampled_exposed_timestep])

                if reset_states:
                    if curr_seir_state is None:
                        curr_seir_state = SEIRState.Recovered

                    if curr_inf_type is None:
                        curr_inf_type = InfectionType.Undefined

                    if curr_inf_sev is None:
                        curr_inf_sev = Severity.Undefined

        if reset_states: # overwrite for cases when agents were exposed prior to sim start
            if curr_seir_state is not None:
                seir_state = curr_seir_state
            if curr_inf_type is not None:
                infection_type = curr_inf_type
            if curr_inf_sev is not None:
                infection_severity = curr_inf_sev

        if start_hosp_day is not None:
            hospitalisation_days = [start_hosp_day, sampled_exposed_timestep, end_hosp_day]
            self.schedule_hospitalisation(exposed_agent, exposed_agent_id, hospitalisation_days)
        
        if symptomatic_day > -1:
            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
            self.schedule_test(exposed_agent, exposed_agent_id, symptomatic_day, sampled_timestep, QuarantineType.Symptomatic)

            # this was not correct. quarantine based on test_result_day happens when sampling intervention activities in the itinerary
            # in this case, since the agent is symptomatic, we need to sample starting quarantine immediately (based on the relevant probability)
            # start_quarantine_day = None
            # if test_scheduled:
            #     if test_result_day[0] < symptomatic_day:
            #         start_quarantine_day = test_result_day[0]
            #     else:
            #         start_quarantine_day = symptomatic_day
            # else:
            #     start_quarantine_day = symptomatic_day

            start_quarantine_day = symptomatic_day
                
            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                
            self.schedule_quarantine(exposed_agent_id, start_quarantine_day, sampled_timestep, QuarantineType.Symptomatic, quarantine_days=agent_quarantine_days, agent=exposed_agent)

        return agent_state_transition_by_day, seir_state, infection_type, infection_severity, recovered
    
    def schedule_test(self, agent, agent_id, incremental_days, start_timestep, quarantine_type):
        test_scheduled = False
        test_day, test_result_day = None, None

        if self.dyn_params.testing_enabled:
            test_already_scheduled = False

            # if "test_day" not in agent:
            #     keys_in_agent = ""
            #     for key in agent.keys():
            #         keys_in_agent += "," + str(key)

            #     raise ValueError("test_day not in agent {0}, existing_keys: {1}".format(str(agent_id), keys_in_agent))
            
            test_day = agent["test_day"]  
            # test_day = self.agents_mp.get(agent_id, "test_day")

            if test_day is not None and len(test_day) > 0:
                # if abs(test_day[0] - incremental_days) <= 5:
                test_already_scheduled = True # the presence of a test_day is enough (it will only be cleared from the itinerary when it can be cleared)

            if not test_already_scheduled:
                test_after_symptoms_rand = -1

                if quarantine_type == QuarantineType.Symptomatic:
                    test_after_symptoms_rand = random.random()
                
                if test_after_symptoms_rand == -1 or test_after_symptoms_rand < self.testing_after_symptoms_probability:
                    days_until_test = 0

                    if quarantine_type == QuarantineType.PositiveContact:
                        days_until_test = self.contact_tracing_positive_delay_days
                    elif quarantine_type == QuarantineType.SecondaryContact:
                        days_until_test = self.contact_tracing_secondary_delay_days

                    days_until_test += util.sample_log_normal(self.testing_days_distribution_parameters[0], self.testing_days_distribution_parameters[1], 1, True)

                    testing_day = incremental_days + days_until_test # incremental days here starts from symptomatic day

                    # agent["test_day"] = [testing_day, start_timestep]
                    test_day = [testing_day, start_timestep]
                    # self.agents_mp.set(agent_id, "test_day", test_day)
                    # self.sync_queue.put(["a", agent_id, "test_day", test_day]) # update by ref

                    days_until_test_result = util.sample_log_normal(self.testing_results_days_distribution_parameters[0], self.testing_results_days_distribution_parameters[1], 1, True)

                    # agent["test_result_day"] = [test_result_day, start_timestep] # and we know agent is infected at this point (assume positive result)
                    test_result_day = [testing_day + days_until_test_result, start_timestep]
                    # self.agents_mp.set(agent_id, "test_result_day", test_result_day)
                    # self.sync_queue.put(["a", agent_id, "test_result_day", test_result_day])

                    agent["test_day"] = test_day
                    agent["test_result_day"] = test_result_day

                    # this was wrong here. at this point, test_day and test_result_day are being scheduled and not received. 
                    # adding to contact_tracing_agent_ids at this point would be pre-mature
                    # if quarantine_type == QuarantineType.Positive:
                    #     # to perform contact tracing (as received positive test result)
                    #     # contact tracing is handled globally at the end of every day, and contact tracing delays are represented in quarantine/testing scheduling
                    #     self.vars_util.contact_tracing_agent_ids.add((agent_id, start_timestep)) 
                    
                    test_scheduled = True

        return test_scheduled, test_day, test_result_day
    
    # if end_day is not None, assume static start_day and end_day
    def schedule_quarantine(self, agent_id, start_day, start_timestep, quarantine_type, end_day=None, quarantine_days=None, agent=None): # True if positive
        quarantine_scheduled = False

        if self.dyn_params.quarantine_enabled:
            to_schedule_quarantine = False

            if quarantine_type == QuarantineType.Symptomatic:
                quarantine_rand = random.random()

                if quarantine_rand < self.quarantine_after_symptoms_probability:
                    to_schedule_quarantine = True
            else:
                ignore_quarantine_rules_rand = random.random()

                if ignore_quarantine_rules_rand < self.ignore_quarantine_rules_probability:
                    to_schedule_quarantine = False
                else:
                    to_schedule_quarantine = True

            if to_schedule_quarantine:
                if quarantine_days is None:
                    if agent is None:
                        quarantine_days = self.agents_epi[agent_id]["quarantine_days"]
                    else:
                        quarantine_days = agent["quarantine_days"]

                    # quarantine_days = self.agents_mp.get(agent_id, "quarantine_days")

                if end_day is None:
                    if quarantine_type == QuarantineType.Symptomatic or quarantine_type == QuarantineType.Positive: # symptomatic or positive, same duration
                        end_day = start_day + self.quarantine_positive_duration
                    elif quarantine_type == QuarantineType.PositiveContact:
                        start_day += self.contact_tracing_positive_delay_days
                        end_day = start_day + self.quarantine_positive_contact_duration
                    elif quarantine_type == QuarantineType.SecondaryContact:
                        start_day += self.contact_tracing_secondary_delay_days
                        end_day = start_day + self.quarantine_secondary_contact_duration

                if quarantine_days is not None and len(quarantine_days) > 0: # to clear quarantine_days when ready from quaratine (in itinerary)
                    st_day = quarantine_days[0]
                    # st_day = st_day_ts[0]

                    if start_day >= st_day:
                        to_schedule_quarantine = False # don't schedule same or later date or later date (i.e. dont reschedule quarantine if already quarantined, but may start before)

                if to_schedule_quarantine:
                    quarantine_days = [start_day, start_timestep, end_day]

                    # self.agents_mp.set(agent_id, "quarantine_days", quarantine_days)
                    # self.sync_queue.put(["a", agent_id, "quarantine_days", quarantine_days])
                    if agent is not None:
                        agent["quarantine_days"] = quarantine_days
                    else:
                        self.agents_epi[agent_id]["quarantine_days"] = quarantine_days

                    quarantine_scheduled = True

        return quarantine_scheduled, quarantine_days
    
    def update_quarantine(self, agent, new_start_day, new_start_ts, new_end_day, new_end_ts):
        quarantine_days = [new_start_day, new_start_ts, new_end_day]

        agent["quarantine_days"] = quarantine_days
        # quarantine_days.append([[new_start_day, new_start_ts], [new_end_day, new_end_ts]])

        # self.agents_mp.set(agent_id, "quarantine_days", quarantine_days)
        # self.sync_queue.put(["a", agent_id, "quarantine_days", quarantine_days])

    # def update_quarantine_end(self, agent_id, new_end_day, new_end_timestep, quarantine_days=None):
    #     if quarantine_days is None:
    #         quarantine_days = self.agents[agent_id]["quarantine_days"]
    #         # quarantine_days = self.agents_mp.get(agent_id, "quarantine_days")

    #     # quarantine_days[0, 1] = [new_end_day, new_end_timestep]
    #     quarantine_days[1] = new_end_timestep
    #     quarantine_days[2] = new_end_day

    #     self.agents[agent_id]["quarantine_days"] = quarantine_days

    def schedule_hospitalisation(self, agent, agent_id, hospitalisation_days):
        agent_hospitalisation_days = agent["hospitalisation_days"]
        # agent_hospitalisation_days = self.agents_mp.get(agent_id, "hospitalisation_days")

        if agent_hospitalisation_days is not None and len(agent_hospitalisation_days) > 0:
            new_end_ts, new_end_day = hospitalisation_days[1], hospitalisation_days[2]

            start_day = agent_hospitalisation_days[0]

            agent_hospitalisation_days = [start_day, new_end_ts, new_end_day]
        else:
            agent_hospitalisation_days = hospitalisation_days

        agent["hospitalisation_days"] = agent_hospitalisation_days

        # agent_hospitalisation_days.extend(hospitalisation_days)
        # self.agents_mp.set(agent_id, "hospitalisation_days", agent_hospitalisation_days)
        # self.sync_queue.put(["a", agent_id, "hospitalisation_days", agent_hospitalisation_days])


    def contact_tracing(self, day, distributed=False, f=None):
        primary_contact_ids = set()
        secondary_contact_ids = set()
        res_cell_ids = set()

        num_quarantine_scheduled = 0
        num_tests_scheduled = 0

        if self.dyn_params.contact_tracing_enabled and len(self.vars_util.contact_tracing_agent_ids) > 0:
            self.dyn_params.statistics.new_contacttraced = len(self.vars_util.contact_tracing_agent_ids)
            
            for daybackindex in range(self.contact_tracing_days_back + 1): # assume minimum is 1 + 1, i.e. 2 iterations (i.e. full today and full yesterday)
                contact_tracing_day = day - daybackindex # if it's 48 hours, it would be 2 + 1, i.e. 2, i.e. full today, yesterday and day before yesterday

                day_start_index = 0
                day_end_index = len(self.vars_util.directcontacts_by_simcelltype_by_day) # default index is the last unless there is a "next day"

                print("daybackindex: {0}, day_end_index: {1}, directcontacts_by_simcelltype_by_day_start_marker len: {2}".format(str(daybackindex), str(day_end_index), str(len(self.vars_util.directcontacts_by_simcelltype_by_day_start_marker))))
                if f is not None:
                    f.flush()

                if contact_tracing_day in self.vars_util.directcontacts_by_simcelltype_by_day_start_marker:
                    day_start_index = self.vars_util.directcontacts_by_simcelltype_by_day_start_marker[contact_tracing_day]

                    if contact_tracing_day + 1 in self.vars_util.directcontacts_by_simcelltype_by_day_start_marker: # next day case
                        day_end_index = self.vars_util.directcontacts_by_simcelltype_by_day_start_marker[contact_tracing_day + 1] - 1

                    print("contact tracing day: {0}, contact_tracing_agent_ids: {1}, index len for day: {2}".format(str(contact_tracing_day), str(len(self.vars_util.contact_tracing_agent_ids)), str(day_end_index - day_start_index)))
                    if f is not None:
                        f.flush()

                    contact_tracing_tree = FastAVLTree()

                    for agent_id in self.vars_util.contact_tracing_agent_ids: # traced_timestep
                        contact_tracing_tree[agent_id] = None

                    directcontacts_by_simcelltype_by_day_this_day = np.array(self.vars_util.directcontacts_by_simcelltype_by_day[day_start_index:day_end_index])
                    
                    self.dyn_params.statistics.num_direct_contacts = len(directcontacts_by_simcelltype_by_day_this_day)

                    for dircon in directcontacts_by_simcelltype_by_day_this_day:
                        simcelltype, ag1_id, ag2_id = dircon[0], dircon[1], dircon[2]

                        ag1_flag = contact_tracing_tree.__contains__(ag1_id)
                        ag2_flag = contact_tracing_tree.__contains__(ag2_id)

                        if ag1_flag != ag2_flag: # logical xor, if 1 is true and the other is false (either or)
                            contact_tracing_success_prob_rand = random.random()
                            contact_tracing_success_prob = self.convert_simcelltype_to_contact_tracing_success_prob(simcelltype)

                            if contact_tracing_success_prob_rand < contact_tracing_success_prob:
                                if not ag1_flag:    
                                    primary_contact_ids.add(ag1_id)
                                else:
                                    primary_contact_ids.add(ag2_id)

            for contact_id in primary_contact_ids:
                # primary contact tracing
                try:
                    positive_contact_agent = self.agents_epi[contact_id]
                except:
                    positive_contact_agent = None

                if positive_contact_agent is not None and len(positive_contact_agent) > 0:
                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                    quarantine_scheduled, _ = self.schedule_quarantine(contact_id, day, sampled_timestep, QuarantineType.PositiveContact, agent=positive_contact_agent)

                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                    test_scheduled, _, _ = self.schedule_test(positive_contact_agent, contact_id, day, sampled_timestep, QuarantineType.PositiveContact)

                    if quarantine_scheduled:
                        num_quarantine_scheduled += 1

                    if test_scheduled:
                        num_tests_scheduled += 1

                    # secondary contact tracing
                    res_cell_id = -1

                    try:
                        res_cell_id = self.agents_static.get(contact_id, "res_cellid")
                    except:
                        if contact_id < self.n_locals: # if tourist would have departed, skip
                            raise ValueError("Agent {0} not found during contact tracing".format(str(contact_id)))

                    if res_cell_id != -1 and res_cell_id not in res_cell_ids: # skip if already computed
                        res_cell_ids.add(res_cell_id)

                        residence = None

                        residents_key = ""
                        staff_key = ""

                        if res_cell_id in self.cells_households:
                            residents_key = "resident_uids"
                            staff_key = "staff_uids"
                            residence = self.cells_households[res_cell_id] # res_cell_id
                        elif res_cell_id in self.cells_institutions:
                            residents_key = "resident_uids"
                            staff_key = "staff_uids"
                            residence = self.cells_institutions[res_cell_id] # res_cell_id 
                        elif res_cell_id in self.cells_accommodation:
                            residents_key = "member_uids"
                            residence = self.cells_accommodation[res_cell_id] # res_cell_id

                        if residence is not None:
                            resident_ids = residence["place"][residents_key] # self.cells_mp.get(residence, residents_key) 

                            contact_id_index = np.argwhere(resident_ids == contact_id)
                            resident_ids = np.delete(resident_ids, contact_id_index)

                            employees_ids = []

                            if staff_key != "":
                                employees_ids = residence["place"][staff_key] # self.cells_mp.get(residence, staff_key)

                            sec_contact_ids = []
                            if employees_ids is None or len(employees_ids) == 0:
                                sec_contact_ids = resident_ids
                            else:
                                try:
                                    sec_contact_ids = np.concatenate((resident_ids, employees_ids))
                                except Exception as e:
                                    print("problemos: " + e)

                            if sec_contact_ids is not None and len(sec_contact_ids) > 0:
                                num_of_sec_successfully_traced = round(len(sec_contact_ids) * self.contact_tracing_residence_probability)
                    
                                sampled_sec_traced_contact_ids = np.random.choice(sec_contact_ids, size=num_of_sec_successfully_traced, replace=False)

                                for sec_contact_id in sampled_sec_traced_contact_ids:
                                    if sec_contact_id not in secondary_contact_ids and sec_contact_id not in primary_contact_ids: # skip if already handled
                                        try:
                                            secondary_contact_agent = self.agents_epi[contact_id]
                                        except:
                                            secondary_contact_agent = None

                                        if secondary_contact_agent is not None and len(secondary_contact_agent) > 0:
                                            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                            quarantine_scheduled, _ = self.schedule_quarantine(sec_contact_id, day, sampled_timestep, QuarantineType.SecondaryContact, agent=secondary_contact_agent)

                                            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                            test_scheduled, _, _ = self.schedule_test(secondary_contact_agent, contact_id, day, sampled_timestep, QuarantineType.SecondaryContact)

                                            if quarantine_scheduled:
                                                num_quarantine_scheduled += 1

                                            if test_scheduled:
                                                num_tests_scheduled += 1

                                            secondary_contact_ids.add(sec_contact_id)
        else:
            if self.dyn_params.contact_tracing_enabled:
                print("contract tracing skipped because there are no agents to contact trace")
            else:
                print("contact tracing skipped because contact tracing is disabled")

            if f is not None:
                f.flush()

        if not distributed:
            self.contact_tracing_clean_up(day, f)

        updated_agent_ids = list(primary_contact_ids) + list(secondary_contact_ids)

        print(f"total contact traced {len(updated_agent_ids)} primary contact ids {len(primary_contact_ids)}, secondary contact ids {len(secondary_contact_ids)}, num_quarantine_scheduled {num_quarantine_scheduled}, num_tests_scheduled {num_tests_scheduled}")
               
        return self.process_index, updated_agent_ids, self.agents_epi, self.vars_util
      
    # fp = open("memory_profiler_10k_ct_cleanup.log", "w+")
    # @profile(stream=fp)
    def contact_tracing_clean_up(self, day, f=None):
        # every day, clean up an "old" day. e.g. after the second day, delete the first day; because on the third day, second and third day will be used
        day_end_idx = 0

        day_to_clean = day - self.contact_tracing_days_back

        if day_to_clean + 1 in self.vars_util.directcontacts_by_simcelltype_by_day_start_marker:
            day_end_idx = self.vars_util.directcontacts_by_simcelltype_by_day_start_marker[day_to_clean + 1] - 1 # 1 less from the next index

        if day_end_idx > 0:
            start = time.time()
            # # sort by index of the main list again, to re-establish 1 to 1 mapping between order of index and order of main list 
            # self.vars_util.dc_by_sct_by_day_agent1_index.sort(key=lambda x:x[1],reverse=False)
            # self.vars_util.dc_by_sct_by_day_agent2_index.sort(key=lambda x:x[1],reverse=False) 

            # # retain indices from day_end_idx + 1 until the end
            self.vars_util.directcontacts_by_simcelltype_by_day = self.vars_util.directcontacts_by_simcelltype_by_day[day_end_idx+1:] # np.delete(self.vars_util.directcontacts_by_simcelltype_by_day, np.s_[:day_end_idx])
            # self.vars_util.dc_by_sct_by_day_agent1_index = self.vars_util.dc_by_sct_by_day_agent1_index[day_end_idx+1:] # np.delete(self.vars_util.dc_by_sct_by_day_agent1_index, np.s_[:day_end_idx])
            # self.vars_util.dc_by_sct_by_day_agent2_index = self.vars_util.dc_by_sct_by_day_agent2_index[day_end_idx+1:] # np.delete(self.vars_util.dc_by_sct_by_day_agent2_index, np.s_[:day_end_idx])
            
            # delete the day_to_clean entry from day-start marker index
            if day_to_clean in self.vars_util.directcontacts_by_simcelltype_by_day_start_marker: # this should only not be the case when day_to_clean is 0
                del self.vars_util.directcontacts_by_simcelltype_by_day_start_marker[day_to_clean]
            
            # re-compute the index
            # start_index_recompute = time.time()
            # day_end_idx_plus_one = day_end_idx + 1
            # self.vars_util.dc_by_sct_by_day_agent1_index = [[aid, idx - day_end_idx_plus_one] for aid, idx in self.vars_util.dc_by_sct_by_day_agent1_index]
            # self.vars_util.dc_by_sct_by_day_agent2_index = [[aid, idx - day_end_idx_plus_one] for aid, idx in self.vars_util.dc_by_sct_by_day_agent2_index]
            # end_index_recompute = time.time() - start_index_recompute
            # print("day " + str(day) + ": contact-tracing index re-compute: " + str(end_index_recompute))

            if f is not None:
                f.flush()

            temp_dc_by_sct_by_day_start_marker = {}

            # extract the keys and num of keys
            dc_sct_keys = list(self.vars_util.directcontacts_by_simcelltype_by_day_start_marker.keys())
            dc_sct_len = len(dc_sct_keys)

            # re-compute the day-start marker index
            for i in range(dc_sct_len, 0, -1):
                curr_key = dc_sct_keys[i-1]

                prev_key = None
                if i-2 >= 0 and i-2 <= len(dc_sct_keys):
                    prev_key = dc_sct_keys[i-2]

                if prev_key is None:
                    temp_dc_by_sct_by_day_start_marker[curr_key] = 0
                else:
                    temp_dc_by_sct_by_day_start_marker[curr_key] = self.vars_util.directcontacts_by_simcelltype_by_day_start_marker[curr_key] - self.vars_util.directcontacts_by_simcelltype_by_day_start_marker[prev_key]

            self.vars_util.directcontacts_by_simcelltype_by_day_start_marker = temp_dc_by_sct_by_day_start_marker

            gc.collect()

            stats = gc.get_stats()

            for stat in stats:
                print(stat)

            time_taken = time.time() - start

            print("day " + str(day) + ": contact-tracing clean-up and re-indexing: " + str(time_taken))

            if f is not None:
                f.flush()

    # currently handling not vaccinated / not-vaccinated, but can also handle first/second dose
    # agents_vaccination_doses has already been equipped to hold simulation days on which vaccination doses have been administered
    # virus transmission has already been implemented to take this into consideration for immunity + asymptomatic likelihood
    def schedule_vaccinations(self, day):
        updated_agent_ids = []

        if self.dyn_params.vaccination_propensity > 0:
            num_not_vaccinated = self.n_locals - len(self.vars_util.agents_vaccination_doses) # np.where(self.vars_util.agents_vaccination_doses == 0)[0]

            num_already_vaccinated = self.n_locals - num_not_vaccinated

            if self.dyn_params.vaccination_propensity != self.dyn_params.last_vaccination_propensity:
                if self.dyn_params.num_agents_to_be_vaccinated == 0:
                    self.dyn_params.num_agents_to_be_vaccinated = round(num_not_vaccinated * self.dyn_params.vaccination_propensity)

                    if self.dyn_params.num_agents_to_be_vaccinated == 0:
                        self.dyn_params.num_agents_to_be_vaccinated = 1
                else:
                    change_in_propensity = self.dyn_params.vaccination_propensity - self.dyn_params.last_vaccination_propensity

                    diff_num_agents_to_be_vaccinated = round(num_not_vaccinated * change_in_propensity)

                    if diff_num_agents_to_be_vaccinated == 0:
                        diff_num_agents_to_be_vaccinated = 1

                    self.dyn_params.num_agents_to_be_vaccinated += diff_num_agents_to_be_vaccinated

                self.dyn_params.last_vaccination_propensity = self.dyn_params.vaccination_propensity

            self.dyn_params.num_vaccinations_today = round(util.sample_log_normal(self.vaccination_daily_parameters[0], self.vaccination_daily_parameters[1], 1, True) * self.locals_ratio_to_full_pop)

            num_remaining_agents_to_be_vaccinated = self.dyn_params.num_agents_to_be_vaccinated - num_already_vaccinated

            if self.dyn_params.num_vaccinations_today > num_remaining_agents_to_be_vaccinated:
                self.dyn_params.num_vaccinations_today = num_remaining_agents_to_be_vaccinated

            # num_agents_to_be_vaccinated *= self.locals_ratio_to_full_pop # would be 1 if full pop

            if self.dyn_params.num_vaccinations_today > 0:
                all_agent_ids = [agentid for agentid in range(self.n_locals)]
                vaccinated_agent_ids = list(self.vars_util.agents_vaccination_doses.keys())
                not_vaccinated_agent_ids = np.array(list(set(all_agent_ids) ^ set(vaccinated_agent_ids))) # symmetric difference
                
                if num_not_vaccinated < self.dyn_params.num_vaccinations_today:
                    sampled_to_vaccinate_ids = not_vaccinated_agent_ids
                else:
                    sampled_to_vaccinate_ids = np.random.choice(not_vaccinated_agent_ids, size=self.dyn_params.num_vaccinations_today, replace=False)

                for agentid in sampled_to_vaccinate_ids:
                    agent = self.agents_epi[agentid]
                    agent_vaccination_days = agent["vaccination_days"]

                    if agent_vaccination_days is None:
                        agent_vaccination_days = []

                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]

                    agent_vaccination_days.append([day + 1, sampled_timestep])
                    
                    agent["vaccination_days"] = agent_vaccination_days

                    updated_agent_ids.append(agentid)

                    # increment on actual day of vaccination
                    # agent_vaccination_doses = self.vars_util.agents_vaccination_doses[agentid]
                    # agent_vaccination_doses += 1
                    # self.vars_util.agents_vaccination_doses[agentid] = agent_vaccination_doses
                    
        return updated_agent_ids
    
    def convert_simcelltype_to_contact_tracing_success_prob(self, simcelltype):
        contact_tracing_success_prob = 0
        match simcelltype:
            case SimCellType.Residence:
                contact_tracing_success_prob = self.contact_tracing_residence_probability
            case SimCellType.Workplace:
                contact_tracing_success_prob = self.contact_tracing_work_probability
                # if rescellid == cellid:
                #     contact_tracing_success_prob = self.contact_tracing_work_probability
                # else:
                #     contact_tracing_success_prob = self.contact_tracing_community_probability
            case SimCellType.School:
                contact_tracing_success_prob = self.contact_tracing_school_probability
            case SimCellType.Community:
                contact_tracing_success_prob = self.contact_tracing_community_probability
            case _:
                contact_tracing_success_prob = self.contact_tracing_community_probability

        return contact_tracing_success_prob

    def convert_celltype_to_base_infection_prob(self, cellid, rescellid, celltype=None):
        simcelltype = util.convert_celltype_to_simcelltype(cellid, celltype=celltype)

        match simcelltype:
            case SimCellType.Residence:
                return self.household_infection_probability
            case SimCellType.Workplace:
                if cellid == rescellid:
                    return self.workplace_infection_probability # exposed agent is worker at a workplace
                else:
                    return self.community_infection_probability # exposed agent is a a visitor at a workplace (e.g. patrons at a restaurant)
            case SimCellType.School:
                return self.school_infection_probability
            case SimCellType.Community:
                return self.community_infection_probability
            case _:
                return self.community_infection_probability