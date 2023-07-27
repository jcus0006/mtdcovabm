import numpy as np
import math
import random
from copy import copy
from simulator import util
from enum import IntEnum

class Epidemiology:
    def __init__(self, 
                epidemiologyparams, 
                n_locals, 
                n_tourists, 
                locals_ratio_to_full_pop, 
                agents, 
                vars_util,
                cells_households, 
                cells_institutions, 
                cells_accommodation, 
                dyn_params):
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
        self.daily_vaccinations_parameters = epidemiologyparams["daily_vaccinations_parameters"]
        self.immunity_after_vaccination_multiplier = epidemiologyparams["immunity_after_vaccination_multiplier"]  

        self.dyn_params = dyn_params 
        # self.sync_queue = sync_queue    

        self.agents = agents
        self.vars_util = vars_util
        self.agents_seir_state = vars_util.agents_seir_state
        self.agents_seir_state_transition_for_day = vars_util.agents_seir_state_transition_for_day
        self.agents_infection_type = vars_util.agents_infection_type
        self.agents_infection_severity = vars_util.agents_infection_severity
        self.agents_vaccination_doses = vars_util.agents_vaccination_doses
        self.directcontacts_by_simcelltype_by_day = vars_util.directcontacts_by_simcelltype_by_day # will be initialised as empty set
        # self.tourists_active_ids = tourists_active_ids
        self.contact_tracing_agent_ids = vars_util.contact_tracing_agent_ids # will be initialised as empty set

        self.cells_households = cells_households
        self.cells_institutions = cells_institutions
        self.cells_accommodation = cells_accommodation

        self.timestep_options = np.arange(42, 121) # 7.00 - 20.00

    def simulate_direct_contacts(self, agents_directcontacts, cellid, cell, day):
        updated_agents_ids = []
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
                        # infectious_agent_id = secondary_agent_id
                    else:
                        exposed_agent_id = secondary_agent_id
                        # infectious_agent_id = primary_agent_id

                    exposed_agent = self.agents[exposed_agent_id]

                    agent_epi_age_bracket_index = exposed_agent["epi_age_bracket_index"]

                    susceptibility_multiplier = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SusceptibilityMultiplier][agent_epi_age_bracket_index]
                    
                    infection_probability = self.convert_celltype_to_base_infection_prob(cellid, exposed_agent["res_cellid"], cell["type"])

                    infection_multiplier = max(1, math.log(contact_duration)) # to check how this affects covasim base probs

                    infection_probability *= infection_multiplier * susceptibility_multiplier * (1 - self.dyn_params.masks_hygiene_distancing_multiplier)

                    exposed_rand = random.random()

                    incremental_days = day
                    if exposed_rand < infection_probability: # exposed (infected but not yet infectious)
                        updated_agents_ids.append(exposed_agent_id)

                        agent_state_transition_by_day = exposed_agent["state_transition_by_day"]
                        agent_quarantine_days = exposed_agent["quarantine_days"]

                        agent_state_transition_by_day, agent_seir_state, agent_infection_type, agent_infection_severity, recovered = self.simulate_seir_state_transition(exposed_agent, exposed_agent_id, incremental_days, overlapping_timesteps, agent_state_transition_by_day, agent_epi_age_bracket_index, agent_quarantine_days)

                        # self.agents_mp.set(exposed_agent_id, "state_transition_by_day", agent_state_transition_by_day)
                        # self.sync_queue.put(["a", exposed_agent_id, "state_transition_by_day", agent_state_transition_by_day]) # updated by ref in agent           

                        if agent_infection_type != InfectionType.Undefined:
                            # self.sync_queue.put(["v", exposed_agent_id, "agents_seir_state", agent_seir_state])
                            self.agents_seir_state[exposed_agent_id] = agent_seir_state
                            # self.sync_queue.put(["v", exposed_agent_id, "agents_infection_type", agent_infection_type])
                            self.agents_infection_type[exposed_agent_id] = agent_infection_type
                            # self.sync_queue.put(["v", exposed_agent_id, "agents_infection_severity", agent_infection_severity])
                            self.agents_infection_severity[exposed_agent_id] = agent_infection_severity

        return updated_agents_ids
    
    def simulate_seir_state_transition(self, exposed_agent, exposed_agent_id, incremental_days, overlapping_timesteps, agent_state_transition_by_day, agent_epi_age_bracket_index, agent_quarantine_days):
        symptomatic_day = -1
        recovered = False # if below condition is hit, False means Dead, True means recovered. 
        start_hosp_day, end_hosp_day = None, None
        
        seir_state = SEIRState.Exposed
        infection_severity = Severity.Undefined
        infection_type = InfectionType.Undefined

        sampled_exposed_timestep = np.random.choice(overlapping_timesteps, size=1)[0]
        
        exp_to_inf_days = util.sample_log_normal(self.exp_to_inf_mean, self.exp_to_inf_std, 1, True)

        incremental_days += exp_to_inf_days

        if agent_state_transition_by_day is None:
            agent_state_transition_by_day = []

        agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.ExposedToInfectious, sampled_exposed_timestep])

        symptomatic_rand = random.random()

        symptomatic_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SymptomaticProbability][agent_epi_age_bracket_index]

        if symptomatic_rand < symptomatic_probability:
            infection_type = InfectionType.PreSymptomatic # Pre-Symptomatic is infectious, but only applies with Infectious state (and not Exposed)

            inf_to_symp_days = util.sample_log_normal(self.inf_to_symp_mean, self.inf_to_symp_std, 1, True)

            incremental_days += inf_to_symp_days
            symptomatic_day = incremental_days

            agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.InfectiousToSymptomatic, sampled_exposed_timestep]) 

            severe_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.SevereProbability][agent_epi_age_bracket_index]

            severe_rand = random.random()

            if severe_rand < severe_probability:
                symp_to_sev_days = util.sample_log_normal(self.symp_to_sev_mean, self.symp_to_sev_std, 1, True)

                incremental_days += symp_to_sev_days

                start_hosp_day = incremental_days

                # hospitalisation_days.append([incremental_days, sampled_exposed_timestep])

                agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.SymptomaticToSevere, sampled_exposed_timestep]) 

                critical_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.CriticalProbability][agent_epi_age_bracket_index]
                
                critical_rand = random.random()

                if critical_rand < critical_probability:
                    sev_to_cri_days = util.sample_log_normal(self.sev_to_cri_mean, self.sev_to_cri_std, 1, True)

                    incremental_days += sev_to_cri_days

                    agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.SevereToCritical, sampled_exposed_timestep]) 

                    deceased_probability = self.susceptibility_progression_mortality_probs_by_age[EpidemiologyProbabilities.DeceasedProbability][agent_epi_age_bracket_index]
        
                    deceased_rand = random.random()

                    if deceased_rand < deceased_probability:
                        cri_to_death_days = util.sample_log_normal(self.cri_to_death_mean, self.cri_to_death_std, 1, True)

                        incremental_days += cri_to_death_days

                        end_hosp_day = incremental_days

                        # hospitalisation_days.append([incremental_days, sampled_exposed_timestep])

                        agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.CriticalToDeath, sampled_exposed_timestep]) 
                    else:
                        # critical
                        cri_to_rec_days = util.sample_log_normal(self.cri_to_rec_mean, self.cri_to_rec_std, 1, True)

                        incremental_days += cri_to_rec_days

                        end_hosp_day = incremental_days
                        # hospitalisation_days.append([incremental_days, sampled_exposed_timestep])

                        agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.CriticalToRecovery, sampled_exposed_timestep]) 

                        recovered = True
                else:
                    # severe
                    sev_to_rec_days = util.sample_log_normal(self.sev_to_rec_mean, self.sev_to_rec_std, 1, True)

                    incremental_days += sev_to_rec_days

                    end_hosp_day = incremental_days
                    # hospitalisation_days.append([incremental_days, sampled_exposed_timestep])

                    agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.SevereToRecovery, sampled_exposed_timestep])

                    recovered = True
            else:
                # mild
                mild_to_rec_days = util.sample_log_normal(self.mild_to_rec_mean, self.mild_to_rec_std, 1, True)

                incremental_days += mild_to_rec_days
                
                agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.MildToRecovery, sampled_exposed_timestep])

                recovered = True
        else:
            # asymptomatic
            infection_type = InfectionType.PreAsymptomatic # Pre-Asymptomatic is infectious, but only applies with Infectious state (and not Exposed)                        
            
            asymp_to_rec_days = util.sample_log_normal(self.asymp_to_rec_mean, self.asymp_to_rec_std, 1, True)

            incremental_days += asymp_to_rec_days

            agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.AsymptomaticToRecovery, sampled_exposed_timestep])

            recovered = True

        if recovered: # else means deceased
            rec_to_exp_days = util.sample_log_normal(self.rec_to_exp_mean, self.rec_to_exp_std, 1, True)

            incremental_days += rec_to_exp_days

            agent_state_transition_by_day.append([incremental_days, SEIRStateTransition.RecoveredToExposed, sampled_exposed_timestep])

        if start_hosp_day is not None:
            hospitalisation_days = [start_hosp_day, sampled_exposed_timestep, end_hosp_day]
            self.schedule_hospitalisation(exposed_agent, exposed_agent_id, hospitalisation_days)
        
        if symptomatic_day > -1:
            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
            test_scheduled, _, test_result_day = self.schedule_test(exposed_agent, exposed_agent_id, symptomatic_day, sampled_timestep, QuarantineType.Positive)
            
            start_quarantine_day = None
            if test_scheduled:
                if test_result_day < symptomatic_day:
                    start_quarantine_day = test_result_day
                else:
                    start_quarantine_day = symptomatic_day
            else:
                start_quarantine_day = symptomatic_day

            sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
            _, _ = self.schedule_quarantine(exposed_agent, exposed_agent_id, start_quarantine_day, sampled_timestep, QuarantineType.Positive, quarantine_days=agent_quarantine_days)

        return agent_state_transition_by_day, seir_state, infection_type, infection_severity, recovered
    
    def schedule_test(self, agent, agent_id, incremental_days, start_timestep, quarantine_type):
        test_scheduled = False
        test_day, test_result_day = None, None

        if self.dyn_params.testing_enabled:
            test_already_scheduled = False

            test_day = agent["test_day"]  
            # test_day = self.agents_mp.get(agent_id, "test_day")

            if test_day is not None and len(test_day) > 0:
                if abs(test_day[0] - incremental_days) < 5:
                    test_already_scheduled

            if not test_already_scheduled:
                test_after_symptoms_rand = -1

                if quarantine_type == QuarantineType.Positive:
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
                    test_result_day = testing_day + days_until_test_result
                    # self.agents_mp.set(agent_id, "test_result_day", test_result_day)
                    # self.sync_queue.put(["a", agent_id, "test_result_day", test_result_day])
                    agent["test_result_day"] = test_result_day

                    if quarantine_type == QuarantineType.Positive:
                        # to perform contact tracing (as received positive test result)
                        # contact tracing is handled globally at the end of every day, and contact tracing delays are represented in quarantine/testing scheduling
                        self.contact_tracing_agent_ids.add((agent_id, start_timestep)) 
                    
                    test_scheduled = True

        return test_scheduled, test_day, test_result_day
    
    # if end_day is not None, assume static start_day and end_day
    def schedule_quarantine(self, agent_id, start_day, start_timestep, quarantine_type, end_day=None, quarantine_days=None): # True if positive
        quarantine_scheduled = False

        if self.dyn_params.quarantine_enabled:
            to_schedule_quarantine = True

            if to_schedule_quarantine:
                if quarantine_days is None:
                    quarantine_days = self.agents[agent_id]["quarantine_days"]
                    # quarantine_days = self.agents_mp.get(agent_id, "quarantine_days")

                if end_day is None:
                    if quarantine_type == QuarantineType.Positive: # positive
                        end_day = start_day + self.quarantine_positive_duration
                    elif quarantine_type == QuarantineType.PositiveContact:
                        start_day += self.contact_tracing_positive_delay_days
                        end_day = start_day + self.quarantine_positive_contact_duration
                    elif quarantine_type == QuarantineType.SecondaryContact:
                        start_day += self.contact_tracing_secondary_delay_days
                        end_day = start_day + self.quarantine_secondary_contact_duration

                if quarantine_days is not None and len(quarantine_days) > 0: # to clear quarantine_days when ready from quaratine (in itinerary)
                    st_day_ts = quarantine_days[0][0]
                    st_day = st_day_ts[0]

                    if start_day >= st_day:
                        to_schedule_quarantine = False # don't schedule same start date or later start date (i.e. dont reschedule quarantine if already quarantined)

                if to_schedule_quarantine:
                    quarantine_days = [start_day, start_timestep, end_day]

                    # self.agents_mp.set(agent_id, "quarantine_days", quarantine_days)
                    # self.sync_queue.put(["a", agent_id, "quarantine_days", quarantine_days])

                    quarantine_scheduled = True

        return quarantine_scheduled, quarantine_days
    
    def update_quarantine(self, agent_id, new_start_day, new_start_ts, new_end_day, new_end_ts):
        quarantine_days = [new_start_day, new_start_ts, new_end_day]
        # quarantine_days.append([[new_start_day, new_start_ts], [new_end_day, new_end_ts]])

        # self.agents_mp.set(agent_id, "quarantine_days", quarantine_days)
        # self.sync_queue.put(["a", agent_id, "quarantine_days", quarantine_days])

    def update_quarantine_end(self, agent_id, new_end_day, new_end_timestep, quarantine_days=None):
        if quarantine_days is None:
            quarantine_days = self.agents[agent_id]["quarantine_days"]
            # quarantine_days = self.agents_mp.get(agent_id, "quarantine_days")

        # quarantine_days[0, 1] = [new_end_day, new_end_timestep]
        quarantine_days[1] = new_end_timestep
        quarantine_days[2] = new_end_day

        # self.agents_mp.set(agent_id, "quarantine_days", quarantine_days)
        # self.sync_queue.put(["a", agent_id, "quarantine_days", quarantine_days])

    def schedule_hospitalisation(self, agent, agent_id, hospitalisation_days):
        agent_hospitalisation_days = agent["hospitalisation_days"]
        # agent_hospitalisation_days = self.agents_mp.get(agent_id, "hospitalisation_days")

        if agent_hospitalisation_days is not None and len(agent_hospitalisation_days) > 0:
            new_end_ts, new_end_day = hospitalisation_days[1], hospitalisation_days[2]

            start_day = agent_hospitalisation_days[0]

            agent_hospitalisation_days = [start_day, new_end_ts, new_end_day]
        else:
            agent_hospitalisation_days = hospitalisation_days

        # agent_hospitalisation_days.extend(hospitalisation_days)
        # self.agents_mp.set(agent_id, "hospitalisation_days", agent_hospitalisation_days)
        # self.sync_queue.put(["a", agent_id, "hospitalisation_days", agent_hospitalisation_days])
    
    # the outer loop iterates for a number of days back pertaining to the number of days that the public health would attempt to trace back (e.g. 1 day)
    # the next loop iterates direct contacts in each simcelltype (residence, workplace, school, community contacts) i.e. subset of contacts per sim type
    # the next loop iterates the positive contacts to be traced
    # the positive agent would have had contacts in different places pertaining to the 4 "simcelltypes" (which feature a different chance of tracing)
    # a percentage of these contacts, based on "simcelltype" successful tracing probability, will be successfully traced
    # a percentage of the secondary contacts (people that share the same residence), based on the simcelltype "residence" probability will also be traced
    # quarantine and tests will be scheduled accordingly according to the relevant percentages in the "contact_tracing_parameters"
    def contact_tracing(self, day):
        if self.dyn_params.contact_tracing_enabled:
            quarantine_scheduled_ids = []
            test_scheduled_ids = []

            for daybackindex in range(self.contact_tracing_days_back + 1): # assume minimum is 1 + 1, i.e. 2 iterations, i.e. 24 hours
                dayback = day - daybackindex

                trace_back_min_ts, trace_back_max_ts = None, None

                if daybackindex == 0: # first day looks like: traced timestep until 0
                    trace_back_max_ts = None
                    trace_back_min_ts = 0
                elif daybackindex < self.contact_tracing_days_back: # day in between looks like: timestep 143 until 0
                    trace_back_max_ts = 143
                    trace_back_min_ts = 0
                elif daybackindex == self.contact_tracing_days_back: # + 1 - 1 = 0, last day traced back onto looks like: timestep 143 until traced timestep
                    trace_back_max_ts = 143
                    trace_back_min_ts = None

                simcelltypes = ["residence", "workplace", "school", "community"]

                for simcelltype in simcelltypes: # purely an iteration per simcelltype (x 4)
                    directcontacts = {(params[2], params[3]):(params[4], params[5]) for params in self.directcontacts_by_simcelltype_by_day if params[0] == dayback and params[1] == simcelltype}

                    for agent_id, traced_timestep in self.contact_tracing_agent_ids:
                        contact_ids = util.get_all_contacts_ids_by_id_and_timesteprange(agent_id, directcontacts, traced_timestep, trace_back_min_ts, trace_back_max_ts, shuffle=False)

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
                                quarantine_reference_quar_days = None
                                positive_contact_agent_id = None

                                if index in sampled_quarantine_indices and contact_id not in quarantine_scheduled_ids:
                                    # positive_contact_agent = self.agents[contact_id]
                                    positive_contact_agent_id = contact_id
                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                    quarantine_scheduled, quar_days = self.schedule_quarantine(positive_contact_agent_id, day, sampled_timestep, QuarantineType.PositiveContact)

                                    if quarantine_scheduled:
                                        quarantine_scheduled_ids.append(contact_id)
                                        quarantine_reference_quar_days = quar_days

                                if index in sampled_test_indices and contact_id not in test_scheduled_ids:
                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                    test_scheduled, _, _ = self.schedule_test(contact_id, contact_id, day, sampled_timestep, QuarantineType.PositiveContact)

                                    if test_scheduled:
                                        test_scheduled_ids.append(contact_id)

                                if simcelltype != "residence" and positive_contact_agent_id is not None: # compute secondary contacts (residential), if not residential
                                    # res_cell_id = self.agents_mp.get(positive_contact_agent_id, "res_cellid")
                                    res_cell_id = self.agents[positive_contact_agent_id]["res_cellid"]

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
                                        residence = self.cells_institutions[res_cell_id]["place"] # res_cell_id 
                                    elif res_cell_id in self.cells_accommodation:
                                        residents_key = "member_uids"
                                        residence = self.cells_accommodation[res_cell_id]["place"] # res_cell_id

                                    if residence is not None:
                                        resident_ids = residence[residents_key] # self.cells_mp.get(residence, residents_key) 
                                        contact_id_index = np.argwhere(resident_ids == contact_id)
                                        resident_ids = np.delete(resident_ids, contact_id_index)

                                        employees_ids = []

                                        if staff_key != "":
                                            employees_ids = residence[staff_key] # self.cells_mp.get(residence, staff_key)

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

                                            quar_start_day, quar_start_timestep, quar_end_day = None, None, None
                                            if quarantine_reference_quar_days is not None:
                                                quar_start_day, quar_start_timestep, quar_end_day = quarantine_reference_quar_days[0][0][0], quarantine_reference_quar_days[0][0][1], quarantine_reference_quar_days[0][1][0]
                                            else:
                                                quar_start_day = day
                                                quar_start_timestep = np.random.choice(self.timestep_options, size=1)[0]

                                            for sec_index, sec_contact_id in enumerate(secondary_contact_ids):
                                                secondary_contact_agent_id = None

                                                if sec_index in sampled_sec_quarantine_indices and sec_contact_id not in quarantine_scheduled_ids:
                                                    secondary_contact_agent_id = sec_contact_id

                                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                                    quarantine_scheduled, quar_days = self.schedule_quarantine(secondary_contact_agent_id, quar_start_day, quar_start_timestep, QuarantineType.SecondaryContact, quar_end_day)

                                                    if quarantine_scheduled:
                                                        quarantine_scheduled_ids.append(sec_contact_id)

                                                        if quarantine_reference_quar_days is None:
                                                            quarantine_reference_quar_days = quar_days

                                                if sec_index in sampled_sec_test_indices and sec_contact_id not in test_scheduled_ids:
                                                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]
                                                    test_scheduled, _ = self.schedule_test(secondary_contact_agent_id, day, sampled_timestep, QuarantineType.SecondaryContact)

                                                    if test_scheduled:
                                                        test_scheduled_ids.append(contact_id)
        
        # clear for next day (next day will be re initialized so not required anymore)
        # self.contact_tracing_agent_ids = [] # still being cleared for next day regardless to whether contact tracing is enabled or not

    # currently handling not vaccinated / vaccinated, but can also handle first/second dose in a similar manner
    def schedule_vaccinations(self, day):
        if self.dyn_params.vaccination_propensity > 0:
            not_vaccinated_indices = np.where(self.agents_vaccination_doses == 0)[0]

            num_already_vaccinated = self.n_locals - len(not_vaccinated_indices)

            if self.dyn_params.vaccination_propensity != self.dyn_params.last_vaccination_propensity:
                if self.dyn_params.num_agents_to_be_vaccinated == 0:
                    self.dyn_params.num_agents_to_be_vaccinated = round(len(not_vaccinated_indices) * self.dyn_params.vaccination_propensity)

                    if self.dyn_params.num_agents_to_be_vaccinated == 0:
                        self.dyn_params.num_agents_to_be_vaccinated = 1
                else:
                    change_in_propensity = self.dyn_params.vaccination_propensity - self.dyn_params.last_vaccination_propensity

                    diff_num_agents_to_be_vaccinated = round(len(not_vaccinated_indices) * change_in_propensity)

                    if diff_num_agents_to_be_vaccinated == 0:
                        diff_num_agents_to_be_vaccinated = 1

                    self.dyn_params.num_agents_to_be_vaccinated += diff_num_agents_to_be_vaccinated

                self.dyn_params.last_vaccination_propensity = self.dyn_params.vaccination_propensity

            num_vaccinations_today = round(util.sample_log_normal(self.daily_vaccinations_parameters[0], self.daily_vaccinations_parameters[1], 1, True) * self.locals_ratio_to_full_pop)

            num_remaining_agents_to_be_vaccinated = self.dyn_params.num_agents_to_be_vaccinated - num_already_vaccinated

            if num_vaccinations_today > num_remaining_agents_to_be_vaccinated:
                num_vaccinations_today = num_remaining_agents_to_be_vaccinated

            # num_agents_to_be_vaccinated *= self.locals_ratio_to_full_pop # would be 1 if full pop

            if num_vaccinations_today > 0:
                if len(not_vaccinated_indices) < num_vaccinations_today:
                    sampled_to_vaccinate_indices = not_vaccinated_indices
                else:
                    sampled_to_vaccinate_indices = np.random.choice(not_vaccinated_indices, size=num_vaccinations_today, replace=False)

                for agentid in sampled_to_vaccinate_indices:
                    agent = self.agents[agentid]
                    agent_vaccination_days = agent["vaccination_days"]
                    agent_vaccination_doses = self.agents_vaccination_doses[agentid]

                    # agent_vaccination_days = self.agents_mp.get(agentid, "vaccination_days")
                    # agent_vaccination_doses = self.agents_mp.get(agentid, "vaccination_doses")
                    sampled_timestep = np.random.choice(self.timestep_options, size=1)[0]

                    agent_vaccination_days.append([day + 1, sampled_timestep])

                    # self.agents_mp.set(agentid, "vaccination_days", agent_vaccination_days)
                    # self.sync_queue.put(["a", agentid, "vaccination_days", agent_vaccination_days])

                    agent_vaccination_doses += 1
                    # self.sync_queue.put(["v", agentid, "agents_vaccination_doses", agent_vaccination_doses])
                    # self.agents_vaccination_doses[agentid] = agent_vaccination_doses

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