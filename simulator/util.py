import datetime
import numpy as np
# import scipy.stats as stats
import powerlaw
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
import seirstateutil, customdict
from cellsclasses import CellType, SimCellType
from enum import IntEnum
import psutil
import time
from pympler import asizeof

def day_of_year_to_day_of_week(day_of_year, year):
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    weekdaystr = date.strftime("%A")
    weekday = -1
    match weekdaystr:
        case "Monday":
            weekday = 1
        case "Tuesday":
            weekday = 2
        case "Wednesday":
            weekday = 3
        case "Thursday":
            weekday = 4
        case "Friday":
            weekday = 5
        case "Saturday":
            weekday = 6
        case "Sunday":
            weekday = 7

    return weekday, weekdaystr

def sample_gamma(gamma_shape, min, max, k = 1, returnInt = False):
    if min == max:
        return min
    
    gamma_scale = (max - min) / (gamma_shape * k)

    sample = np.random.gamma(gamma_shape, gamma_scale)

    if returnInt:
        return round(sample)
    
    return sample

def sample_gamma_reject_out_of_range(gamma_shape, min, max, k = 1, returnInt = False, useNp = False):
    # if useNp:
    #     sample = min - 1

    #     while sample < min or sample > max:
    #         sample = sample_gamma(gamma_shape, min, max, k, returnInt)
    # else:
    #     scale = (max - min) / (gamma_shape * k)

    #     trunc_gamma = stats.truncnorm((min - scale) / np.sqrt(gamma_shape),
    #                           (max - scale) / np.sqrt(gamma_shape),
    #                           loc=scale, scale=np.sqrt(gamma_shape))
        
    #     sample = trunc_gamma.rvs()

    sample = min - 1

    while sample < min or sample > max:
        sample = sample_gamma(gamma_shape, min, max, k, returnInt)

    return sample

# Generate a variable number of group_sizes in the range min_size to max_size.
# Sample from a gamma distribution. Default gamma params favour smaller values (use gamma_shape > 1 to favour larger values)
def random_group_partition(group_size, min_size, max_size, gamma_shape = 0.5, k=1):
    group_sizes = []
    
    while sum(group_sizes) < group_size:
        sampled_group_size = sample_gamma_reject_out_of_range(gamma_shape, min_size, max_size, k, True, True)

        if sum(group_sizes) + sampled_group_size > group_size:
            sampled_group_size = group_size - sum(group_sizes)

        group_sizes.append(sampled_group_size)
    
    return group_sizes

def sample_log_normal(mean, std, size, isInt=False):
    dist_mean  = np.log(mean**2 / np.sqrt(std**2 + mean**2)) # Computes the mean of the underlying normal distribution
    dist_sigma = np.sqrt(np.log(std**2/mean**2 + 1)) # Computes sigma for the underlying normal distribution
    samples = np.random.lognormal(mean=dist_mean, sigma=dist_sigma, size=size)

    if isInt:
        int_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            int_samples.append(round(sample))

        if size == 1:
            return int_samples[0]
        else:
            return int_samples

    if size == 1:
        return samples[0]
    
    return samples

def calculate_vaccination_multipliers(agents_vaccination_doses, agentid, current_day, vaccine_immunity, vaccine_asymptomatic, exponential_decay_interval, accum_time=None):
    immunity_multiplier = 1.0
    asymptomatic_multiplier = 1.0
    
    if accum_time is not None:
        start = time.time()
    
    if agentid in agents_vaccination_doses:
        doses_days = agents_vaccination_doses[agentid]
        last_dose_day = doses_days[-1]
        days_since_last_dose = current_day - last_dose_day

        if days_since_last_dose == 0: # full effectiveness (otherwise will always result in 1.0 - (0.9 ^ 0) = 0.0 -> 1.0 - (1.0) = 0.0)
            immunity_multiplier = 1.0 - vaccine_immunity
            asymptomatic_multiplier = 1.0 - vaccine_asymptomatic
        else:
            immunity_exp_decay = days_since_last_dose / exponential_decay_interval
            if immunity_exp_decay < 1:
                immunity_exp_decay = 1

            asymptomatic_exp_decay = days_since_last_dose / exponential_decay_interval
            if asymptomatic_exp_decay < 1:
                asymptomatic_exp_decay = 1

            immunity_multiplier = 1.0 - (vaccine_immunity ** immunity_exp_decay) # e.g. 0.9 ^ (60 / 30) = 0.9 ^ 2
            asymptomatic_multiplier = 1.0 - (vaccine_asymptomatic ** asymptomatic_exp_decay) # e.g. 0.9 ^ (60 / 30) = 0.9 ^ 2


    if accum_time is not None:    
        time_taken = time.time() - start

        accum_time += time_taken

    return immunity_multiplier, asymptomatic_multiplier, accum_time

def set_age_brackets(agent, agents_ids_by_ages, agent_uid, age_brackets, age_brackets_workingages, agents_ids_by_agebrackets, set_working_age_bracket=True):
    age = agent["age"]
    agents_ids_by_ages[agent_uid] = agent["age"]

    agent["age_bracket_index"] = -1
    for i, ab in enumerate(age_brackets):
        if age >= ab[0] and age <= ab[1]:
            agent["age_bracket_index"] = i
            
            agents_ids_by_agebrackets[i].append(agent_uid)

            break
    
    if set_working_age_bracket:
        agent["working_age_bracket_index"] = -1
        for i, ab in enumerate(age_brackets_workingages):
            if age >= ab[0] and age <= ab[1]:
                agent["working_age_bracket_index"] = i
                break

    return agent, age, agents_ids_by_ages, agents_ids_by_agebrackets

def set_age_brackets_tourists(age, agents_ids_by_ages, agent_uid, age_brackets, agents_ids_by_agebrackets):
    agents_ids_by_ages[agent_uid] = age

    age_bracket_index = -1
    for i, ab in enumerate(age_brackets):
        if age >= ab[0] and age <= ab[1]:
            age_bracket_index = i
            
            agents_ids_by_agebrackets[i].append(agent_uid)

            break

    return age_bracket_index, agents_ids_by_ages, agents_ids_by_agebrackets

def generate_sociability_rate_powerlaw_dist(temp_agents, agents_ids_by_agebrackets, powerlaw_distribution_parameters, visualise, sociability_rate_min, sociability_rate_max, figure_count):
    for agebracket_index, agents_ids_in_bracket in agents_ids_by_agebrackets.items():
        powerlaw_dist_params = powerlaw_distribution_parameters[agebracket_index]

        exponent, xmin = powerlaw_dist_params[2], powerlaw_dist_params[3]
        # exponent, xmin = 2.64, 4.0
        dist = powerlaw.Power_Law(xmin=xmin, parameters=[exponent])

        if len(agents_ids_in_bracket) > 0:
            if len(agents_ids_in_bracket) > 1:
                agents_contact_propensity = dist.generate_random(len(agents_ids_in_bracket))

                min_arr = np.min(agents_contact_propensity)
                max_arr = np.max(agents_contact_propensity)

                normalized_arr = (agents_contact_propensity - min_arr) / (max_arr - min_arr) * (sociability_rate_max - sociability_rate_min) + sociability_rate_min

                if visualise:
                    figure_count += 1
                    plt.figure(figure_count)
                    bins = np.logspace(np.log10(min(agents_contact_propensity)), np.log10(max(agents_contact_propensity)), 50)
                    plt.hist(agents_contact_propensity, bins=bins, density=True)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.show(block=False)

                for index, agent_id in enumerate(agents_ids_in_bracket):
                    agent = temp_agents[agent_id]

                    agent["soc_rate"] = normalized_arr[index]
            else: # this will never be hit, but in a single case, would favour the lower end of the range with a gamma dist
                single_agent_id = agents_ids_in_bracket[0]

                agent = temp_agents[single_agent_id]             

                agent["soc_rate"] = sample_gamma_reject_out_of_range(0.5, sociability_rate_min, sociability_rate_max, useNp=True)

    return temp_agents

def set_public_transport_regular(agent, usage_probability):
    public_transport_rand = random.random()

    if public_transport_rand < usage_probability:
        agent["pub_transp_reg"] = True
    else:
        agent["pub_transp_reg"] = False

    return agent

def convert_celltype_to_simcelltype(cellid, cells=None, celltype=None):
    if cells is not None and celltype is None:
        cell = cells[cellid]
        celltype = cell["type"]

    match celltype:
        case CellType.Household:
            return SimCellType.Residence
        case CellType.Workplace:
            return SimCellType.Workplace
        case CellType.Accommodation:
            return SimCellType.Residence
        case CellType.Hospital:
            return SimCellType.Community
        case CellType.Entertainment:
            return SimCellType.Community
        case CellType.School:
            return SimCellType.School
        case CellType.Classroom:
            return SimCellType.School
        case CellType.Institution:
            return SimCellType.Residence
        case CellType.Transport:
            return SimCellType.Community
        case CellType.Religion:
            return SimCellType.Community
        case CellType.Airport:
            return SimCellType.Community
        case _:
            return SimCellType.Community

# takes the contact structure in agents_contacts
# can be used for both potential contacts and direct contacts (as they bear the same structure)
# if agents_degree is passed (not none), it will only return the contact id if the relevant degree is greater than 0
# for direct cotacts, agents_degrees may be left out
def get_all_contacts_ids_by_id(id, agents_contacts_keys, agents_ids = None, agents_degrees = None, shuffle=False):
    contacts_ids = []
    for pairid in agents_contacts_keys:
        if id in pairid:
            contact_id = pairid[1] if pairid[0] == id else pairid[0]

            if agents_ids is not None and agents_degrees is not None:
                contact_index = agents_ids.index(contact_id)
                contact_degree = agents_degrees[contact_index]

                if contact_degree > 0:
                    contacts_ids.append(contact_id)
            else:
                contacts_ids.append(contact_id)
            
    if len(contacts_ids) == 0:
        return None
    
    contacts_ids = np.array(contacts_ids)

    if shuffle:
        np.random.shuffle(contacts_ids)

    return contacts_ids

def filter_contacttracing_agents_by_startts_groupby_simcelltype(contact_tracing_info_arr, traced_ts, min_ts, max_ts, shuffle=False):
    if max_ts is None:
        max_ts = traced_ts # when contact tracing is done, start from that timestep until 0
    
    if min_ts is None:
        min_ts = traced_ts # when contact tracing is traced for e.g. 24 hours, trace until the same timestep on X prev days

    contact_tracing_info_by_simcelltype = {}

    for agent, simcelltype, start_time in contact_tracing_info_arr:
        if min_ts <= start_time <= max_ts:
            if simcelltype not in contact_tracing_info_by_simcelltype:
                contact_tracing_info_by_simcelltype[simcelltype] = set()

            contact_tracing_info_by_simcelltype[simcelltype].add(agent)        

    return contact_tracing_info_by_simcelltype


def get_all_contacts_ids_by_id_and_timesteprange(id, agents_contacts_keys_set, traced_ts, min_ts, max_ts, shuffle=False):
    if max_ts is None:
        max_ts = traced_ts # when contact tracing is done, start from that timestep until 0
    
    if min_ts is None:
        min_ts = traced_ts # when contact tracing is traced for e.g. 24 hours, trace until the same timestep on X prev days

    # contacts_ids = []
    # for pairid, (st_ts, _) in agents_contacts_keys:
    #     if (id in pairid and 
    #         (st_ts >= min_ts) and 
    #         (st_ts <= max_ts)):
    #         contact_id = pairid[1] if pairid[0] == id else pairid[0]

    #         contacts_ids.append(contact_id)

    # contacts_ids = [pairid[1] if pairid[0] == id else pairid[0] for pairid, (st_ts, _) in agents_contacts_keys_set if (id in pairid and (st_ts >= min_ts) and (st_ts <= max_ts))]

    contacts_ids = {agent2 if agent1 == id else agent1
                    for (agent1, agent2), (_, start_time, _) in agents_contacts_keys_set
                    if (agent1 == id or agent2 == id) and min_ts <= start_time <= max_ts}
            
    if len(contacts_ids) == 0:
        return None
    
    contacts_ids = list(contacts_ids) # convert from set

    contacts_ids = np.array(contacts_ids)

    if shuffle:
        np.random.shuffle(contacts_ids)

    return contacts_ids

def get_sus_mort_prog_age_bracket_index(age):
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
        
def split_dicts_by_agentsids(agents_ids, agents, vars_util, agents_partial, vars_util_partial, agents_ids_by_ages=None, agents_ids_by_ages_partial=None, is_itinerary=False, is_dask_task=False, agents_epi=None, agents_epi_partial=None):
    for uid in agents_ids:
        agents_partial[uid] = agents[uid]

        if agents_epi is not None:
            agents_epi_partial[uid] = agents_epi[uid]

        if uid in vars_util.agents_seir_state:
            vars_util_partial.agents_seir_state[uid] = vars_util.agents_seir_state[uid]
        # if is_dask_task:
        #     vars_util_partial.agents_seir_state.append(vars_util.agents_seir_state[uid])

            # if len(vars_util.agents_vaccination_doses) > 0: # e.g. from itinerary, not applicable
            #     vars_util_partial.agents_vaccination_doses.append(vars_util.agents_vaccination_doses[uid])

        if is_itinerary and agents_ids_by_ages is not None and agents_ids_by_ages_partial is not None:
            agents_ids_by_ages_partial[uid] = agents_ids_by_ages[uid]

        if not is_itinerary and uid in vars_util.agents_seir_state_transition_for_day: # although will never have data if is_itinerary
            vars_util_partial.agents_seir_state_transition_for_day[uid] = vars_util.agents_seir_state_transition_for_day[uid]

        if uid in vars_util.agents_infection_type:
            vars_util_partial.agents_infection_type[uid] = vars_util.agents_infection_type[uid]

        if uid in vars_util.agents_infection_severity:
            vars_util_partial.agents_infection_severity[uid] = vars_util.agents_infection_severity[uid]

        if uid in vars_util.agents_vaccination_doses:
            vars_util_partial.agents_vaccination_doses[uid] = vars_util.agents_vaccination_doses[uid]

    return agents_partial, agents_ids_by_ages_partial, vars_util_partial, agents_epi_partial

def split_dicts_by_agentsids_copy(agents_ids, agents, agents_epi, vars_util, agents_partial, agents_epi_partial, vars_util_partial, agents_ids_by_ages=None, agents_ids_by_ages_partial=None, is_itinerary=False, is_dask_full_array_mapping=False):
    for uid in agents_ids:
        agents_partial[uid] = deepcopy(agents[uid])
        agents_epi_partial[uid] = deepcopy(agents_epi[uid])

        if uid in vars_util.agents_seir_state[uid]:
            vars_util_partial.agents_seir_state[uid] = vars_util.agents_seir_state[uid]
        # if is_dask_full_array_mapping:
        #     vars_util_partial.agents_seir_state.append(vars_util.agents_seir_state[uid])

            # if len(vars_util.agents_vaccination_doses) > 0: # e.g. from itinerary, not applicable
            #     vars_util_partial.agents_vaccination_doses.append(copy(vars_util.agents_vaccination_doses[uid]))

        if is_itinerary and agents_ids_by_ages is not None and agents_ids_by_ages_partial is not None:
            agents_ids_by_ages_partial[uid] = agents_ids_by_ages[uid]

        if not is_itinerary and uid in vars_util.agents_seir_state_transition_for_day: # although will never have data if is_itinerary
            vars_util_partial.agents_seir_state_transition_for_day[uid] = vars_util.agents_seir_state_transition_for_day[uid]

        if uid in vars_util.agents_infection_type:
            vars_util_partial.agents_infection_type[uid] = vars_util.agents_infection_type[uid]

        if uid in vars_util.agents_infection_severity:
            vars_util_partial.agents_infection_severity[uid] = vars_util.agents_infection_severity[uid]

        if uid in vars_util.agents_vaccination_doses:
            vars_util_partial.agents_vaccination_doses[uid] = deepcopy(vars_util.agents_vaccination_doses[uid])

    # if not is_dask_full_array_mapping:
    #     vars_util_partial.agents_seir_state = copy(vars_util.agents_seir_state)

    return agents_partial, agents_epi_partial, agents_ids_by_ages_partial, vars_util_partial

def split_agents_epi_by_agentsids(agents_ids_to_send, agents_epi, vars_util, agents_epi_to_send, vars_util_to_send):
    for agent_id in agents_ids_to_send:
        agents_epi_to_send[agent_id] = agents_epi[agent_id]
        vars_util_to_send.agents_seir_state[agent_id] = vars_util.agents_seir_state[agent_id]

        # should the below be uncommented, this should only apply for the Itinerary SimStage
        # if agent_id in self.vars_util.agents_seir_state_transition_for_day:
        #     vars_util_to_send.agents_seir_state_transition_for_day[agent_id] = self.vars_util.agents_seir_state_transition_for_day[agent_id]

        if agent_id in vars_util.agents_infection_type:
            vars_util_to_send.agents_infection_type[agent_id] = vars_util.agents_infection_type[agent_id]

        if agent_id in vars_util.agents_infection_severity:
            vars_util_to_send.agents_infection_severity[agent_id] = vars_util.agents_infection_severity[agent_id]

        if agent_id in vars_util.agents_vaccination_doses:
            vars_util_to_send.agents_vaccination_doses[agent_id] = vars_util.agents_vaccination_doses[agent_id]

    return agents_epi_to_send, vars_util_to_send

def sync_state_info_by_agentsids(agents_ids, agents, agents_epi, vars_util, agents_partial, agents_epi_partial, vars_util_partial, contact_tracing=False):
    # updated_count = 0
    for agentindex, agentid in enumerate(agents_ids):
        curr_agent = agents_partial[agentid]
        curr_agent_epi = agents_epi_partial[agentid]
        
        if not contact_tracing:
            agents[agentid] = curr_agent
            agents_epi[agentid] = curr_agent_epi
        else:
            main_agent = agents_epi[agentid] # may also add handling to update only the updated fields rather than all fields that can be updated
            main_agent["test_day"] = curr_agent_epi["test_day"]
            main_agent["test_result_day"] = curr_agent_epi["test_result_day"]
            main_agent["quarantine_days"] = curr_agent_epi["quarantine_days"]

        if not contact_tracing:
            vars_util.agents_seir_state[agentid] = seirstateutil.agents_seir_state_get(vars_util_partial.agents_seir_state, agentid) #agentindex

            # if agentid in vars_util_partial.agents_seir_state_transition_for_day:
            #     vars_util.agents_seir_state_transition_for_day[agentid] = vars_util_partial.agents_seir_state_transition_for_day[agentid]
        
        if agentid in vars_util_partial.agents_infection_type:
            vars_util.agents_infection_type[agentid] = vars_util_partial.agents_infection_type[agentid]

        if agentid in vars_util_partial.agents_infection_severity:
            vars_util.agents_infection_severity[agentid] = vars_util_partial.agents_infection_severity[agentid]

        if agentid in vars_util_partial.agents_vaccination_doses:
            vars_util.agents_vaccination_doses[agentid] = vars_util_partial.agents_vaccination_doses[agentid]

        # updated_count += 1  

    # print("synced " + str(updated_count) + " agents")
    
    return agents, agents_epi, vars_util

def sync_state_info_by_agentsids_cn(agents_ids, agents_epi, vars_util, agents_epi_partial, vars_util_partial):
    # updated_count = 0
    for _, agentid in enumerate(agents_ids):
        curr_agent_epi = agents_epi_partial[agentid]
        
        agents_epi[agentid] = curr_agent_epi

        if agentid in vars_util_partial.agents_seir_state:
            vars_util.agents_seir_state[agentid] = seirstateutil.agents_seir_state_get(vars_util_partial.agents_seir_state, agentid) # agentindex

        if agentid in vars_util_partial.agents_infection_type:
            vars_util.agents_infection_type[agentid] = vars_util_partial.agents_infection_type[agentid]

        if agentid in vars_util_partial.agents_infection_severity:
            vars_util.agents_infection_severity[agentid] = vars_util_partial.agents_infection_severity[agentid]

        if agentid in vars_util_partial.agents_vaccination_doses:
            vars_util.agents_vaccination_doses[agentid] = vars_util_partial.agents_vaccination_doses[agentid]

        # updated_count += 1  

    # print("synced " + str(updated_count) + " agents")
    
    return agents_epi, vars_util

def sync_state_info_by_agentsids_ct(agents_ids, agents_epi, agents_epi_partial):
    # updated_count = 0
    for _, agentid in enumerate(agents_ids):
        curr_agent_epi = agents_epi_partial[agentid]
        
        agents_epi[agentid] = curr_agent_epi
    
    return agents_epi

def sync_state_info_by_agentsids_agents_epi(agents_ids, agents_epi, vars_util, agents_epi_partial, vars_util_partial):
    # updated_count = 0
    for _, agentid in enumerate(agents_ids):
        curr_agent_epi = agents_epi_partial[agentid]
        
        agents_epi[agentid] = curr_agent_epi

        vars_util.agents_seir_state[agentid] = seirstateutil.agents_seir_state_get(vars_util_partial.agents_seir_state, agentid) #agentindex

        # should this be uncommented it needs to apply only to the Itinerary SimStage
        # if agentid in vars_util_partial.agents_seir_state_transition_for_day:
        #     vars_util.agents_seir_state_transition_for_day[agentid] = vars_util_partial.agents_seir_state_transition_for_day[agentid]
        
        if agentid in vars_util_partial.agents_infection_type:
            vars_util.agents_infection_type[agentid] = vars_util_partial.agents_infection_type[agentid]

        if agentid in vars_util_partial.agents_infection_severity:
            vars_util.agents_infection_severity[agentid] = vars_util_partial.agents_infection_severity[agentid]

        if agentid in vars_util_partial.agents_vaccination_doses:
            vars_util.agents_vaccination_doses[agentid] = vars_util_partial.agents_vaccination_doses[agentid]

        # updated_count += 1  

    # print("synced " + str(updated_count) + " agents")
    
    return agents_epi, vars_util

def sync_state_info_sets(day, vars_util, vars_util_partial):
    if len(vars_util_partial.contact_tracing_agent_ids) > 0:
        vars_util.contact_tracing_agent_ids.update(vars_util_partial.contact_tracing_agent_ids)

    if len(vars_util_partial.directcontacts_by_simcelltype_by_day) > 0:
        current_index = len(vars_util.directcontacts_by_simcelltype_by_day)

        if day not in vars_util.directcontacts_by_simcelltype_by_day_start_marker: # sync_state_info_sets is called multiple times, but start index must only be set the first time
            vars_util.directcontacts_by_simcelltype_by_day_start_marker[day] = current_index

        vars_util.directcontacts_by_simcelltype_by_day.extend(vars_util_partial.directcontacts_by_simcelltype_by_day)

        # for index, dc in enumerate(vars_util_partial.directcontacts_by_simcelltype_by_day):
        #     new_index = current_index + index
            # vars_util.dc_by_sct_by_day_agent1_index.append([dc[1], new_index])
            # vars_util.dc_by_sct_by_day_agent2_index.append([dc[2], new_index])

    return vars_util

# not currently being used
# this was being used in single process scenarios (single-threaded), 
# eventually went for a solution where the indexes are updated at the same point of inserting the contacts (for single process cases only)
# def update_direct_contact_indexes(day, vars_util):
#     current_index = len(vars_util.dc_by_sct_by_day_agent1_index)

#     if day not in vars_util.directcontacts_by_simcelltype_by_day_start_marker: # sync_state_info_sets is called multiple times, but start index must only be set the first time
#         vars_util.directcontacts_by_simcelltype_by_day_start_marker[day] = current_index

#     for new_index in range(current_index, len(vars_util.directcontacts_by_simcelltype_by_day)):
#         dc = vars_util.directcontacts_by_simcelltype_by_day[new_index]
#         vars_util.dc_by_sct_by_day_agent1_index.append([dc[1], new_index])
#         vars_util.dc_by_sct_by_day_agent2_index.append([dc[2], new_index])

#     return vars_util

def sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial):
    for cellid, agents_timesteps in vars_util_partial.cells_agents_timesteps.items():
        if cellid not in vars_util.cells_agents_timesteps:
            vars_util.cells_agents_timesteps[cellid] = []

        vars_util.cells_agents_timesteps[cellid] += agents_timesteps

    return vars_util

# load balancing

def split_residences_by_weight(residences, num_partitions):
    # Sort residences based on their weights in ascending order
    sorted_residences_with_indices = sorted(enumerate(residences), key=lambda x: x[1]['lb_weight'])

    sorted_residences = [residence for _, residence in sorted_residences_with_indices]
    sorted_indices = [index for index, _ in sorted_residences_with_indices]

    process_residences_indices = [[] for i in range(num_partitions)]
    cursor = 0
    for index, _ in enumerate(sorted_residences):
        process_residences_indices[cursor].append(sorted_indices[index])

        cursor += 1

        if cursor == num_partitions:
            cursor = 0

    # process_residences_indices_lengths = [len(pri) for pri in process_residences_indices]
    # print(process_residences_indices_lengths)

    return process_residences_indices

# weights are worker based
def itinerary_load_balancing(residences, num_workers, weights):
    total = np.sum(weights) # get the total sum of the weights

    # generate the num of tasks per worker by dividing each weight by the total and multiplying by the num of residences, converting to int and rounding to the nearest integer
    tasks_per_worker = np.round((weights / total) * len(residences)).astype(int) 

    if sum(tasks_per_worker) != len(residences):
        remaining = len(residences) - sum(tasks_per_worker)
        
        new_added = 0
        while (new_added < remaining):
            for i in range(len(tasks_per_worker)-1, -1, -1):
                if new_added < remaining:
                    tasks_per_worker[i] += 1
                    new_added += 1
                else:
                    break

    sorted_residences_with_indices = sorted(enumerate(residences), key=lambda x: x[1]['lb_weight']) # sort the residences based on the lb_weight (lb_weight is based on the num of residents in each residence)

    sorted_indices = [index for index, _ in sorted_residences_with_indices] # reflect the indices of sorted_residences_with_indices

    process_residences_indices = [[] for i in range(num_workers)] # create a multi-dim array with num_workers indices (node-agnostic) (the values of which will be added down under)

    cursor = 0
    for worker_index, num_tasks_this_worker in enumerate(tasks_per_worker): # add the indices of each residence, for every worker
        process_residences_indices[worker_index].extend(sorted_indices[cursor: cursor + num_tasks_this_worker]) # use extend, as already an array
        cursor += num_tasks_this_worker

    return process_residences_indices

def itinerary_load_balancing_by_node(residences, num_workers, nodes_n_workers, weights):
    total = np.sum(weights) # get the total sum of the weights

    tasks_per_node = np.round((weights / total) * len(residences)).astype(int) # generate the num of tasks per node by dividing each weight by the total and multiplying by the num of residences, converting to int and rounding to the nearest integer

    sorted_residences_with_indices = sorted(enumerate(residences), key=lambda x: x[1]['lb_weight']) # sort the residences based on the lb_weight (lb_weight is based on the num of residents in each residence)

    sorted_indices = [index for index, _ in sorted_residences_with_indices] # reflect the indices of sorted_residences_with_indices

    process_residences_indices = [[] for i in range(num_workers)] # create a multi-dim array with num_workers indices (node-agnostic) (the values of which will be added down under)

    cursor = 0
    worker_index = 0
    for i, (ni, num_workers) in enumerate(nodes_n_workers.items()): # i is the current index, ni is the node of the index, num_workers is the num of workers per node
        num_tasks_this_node = tasks_per_node[i] # get the num of tasks for the current node
        num_tasks_per_worker = split_balanced_partitions(num_tasks_this_node, num_workers) # balance the num of tasks according to the num of workers

        for _, num_tasks_this_worker in enumerate(num_tasks_per_worker): # add the indices of each residence, for every worker
            process_residences_indices[worker_index].extend(sorted_indices[cursor: cursor + num_tasks_this_worker]) # use extend, as already an array
            worker_index += 1
            cursor += num_tasks_this_worker

    return process_residences_indices

def split_balanced_partitions(x, n):
    base_value = x // n
    remainder = x % n
    partitions = [base_value] * n

    while remainder > 0:
        temp_remainder = remainder
        for i in range(remainder):
            if i < len(partitions):
                partitions[i] += 1
                temp_remainder -= 1
            else:
                break
        
        remainder = temp_remainder

    return partitions

def split_cellsagentstimesteps_balanced(cells_agents_timesteps, num_dicts):
    # Sort the keys based on the length of the array in the value in ascending order
    sorted_keys = sorted(cells_agents_timesteps.keys(), key=lambda k: len(cells_agents_timesteps[k]))

    # Initialize empty dictionaries for each partition
    partitions = [customdict.CustomDict() for _ in range(num_dicts)]

    # Distribute keys evenly across partitions
    for i, key in enumerate(sorted_keys):
        partitions[i % num_dicts][key] = cells_agents_timesteps[key]

    return partitions

def binary_search_2d_all_indices(arr, target, col_index=0):
    indices = []
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        mid_value = arr[mid][col_index]
        if mid_value == target:
            indices.append(mid)
            # Search to the left of mid
            i = mid - 1
            while i >= 0 and arr[i][col_index] == target:
                indices.append(i)
                i -= 1
            # Search to the right of mid
            i = mid + 1
            while i < len(arr) and arr[i][col_index] == target:
                indices.append(i)
                i += 1
            return indices
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1

    return indices  # Empty list if no matches found

def yield_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def log_memory_usage(f=None, prepend_text=None, memory_info=None):
    if prepend_text is None:
        prepend_text = ""

    if memory_info is None:
        memory_info = psutil.virtual_memory()
        
    memory_total = memory_info.total / (1024 ** 2)
    memory_available = memory_info.available / (1024 ** 2)
    memory_used_up = memory_info.used / (1024 ** 2)
    memory_cached = 0
    try:
        memory_cached = memory_info.cached / (1024 ** 2)
    except:
        memory_cached = -1

    memory_buffer = 0
    try:
        memory_buffer = memory_info.buffers / (1024 ** 2)
    except:
        memory_buffer = -1
        
    print("{0}Total memory {1}, Used up memory {2}, Available memory {3}, Cached memory {4}, Buffer memory {5}".format(str(prepend_text), str(memory_total), str(memory_used_up), str(memory_available), str(memory_cached), str(memory_buffer)))

    if f is not None:
        f.flush()

def asizeof_formatted(data):
    return round(asizeof.asizeof(data) / (1024 ** 2), 2)

def asizeof_list_formatted(data):
    temp_sum = 0
    for d in data:
        temp_sum += asizeof.asizeof(d)

    return round(temp_sum / (1024 ** 2), 2)
        
# this inefficient in that it takes longer than the actual work done in the worker processes. another strategy will be opted for and this will not be used.
# def split_for_contacttracing(agents, directcontacts_by_simcelltype_by_day, agentids, cells_households, cells_institutions, cells_accommodation):
#     agents_partial = {}
#     dc_by_simcelltype_by_day_partial = set()

#     for agent_id, _ in agentids:
#         temp_dc_by_simcelltype_by_day = {dc for dc in directcontacts_by_simcelltype_by_day if agent_id == dc[2] or agent_id == dc[3]}

#         dc_by_simcelltype_by_day_partial.update(temp_dc_by_simcelltype_by_day)

#         for props in temp_dc_by_simcelltype_by_day:
#             agentid_to_add = props[3] if agent_id == props[2] else props[2]
#             agents_partial[agentid_to_add] = agents[agentid_to_add]

#             res_cell_id =  agents_partial[agentid_to_add]["res_cellid"]

#             residence = None

#             residents_key = ""
#             staff_key = ""

#             if res_cell_id in cells_households:
#                 residents_key = "resident_uids"
#                 staff_key = "staff_uids"
#                 residence = cells_households[res_cell_id] # res_cell_id
#             elif res_cell_id in cells_institutions:
#                 residents_key = "resident_uids"
#                 staff_key = "staff_uids"
#                 residence = cells_institutions[res_cell_id]["place"] # res_cell_id 
#             elif res_cell_id in cells_accommodation:
#                 residents_key = "member_uids"
#                 residence = cells_accommodation[res_cell_id]["place"] # res_cell_id

#             if residence is not None:
#                 resident_ids = residence[residents_key] # self.cells_mp.get(residence, residents_key) 
#                 contact_id_index = np.argwhere(resident_ids == agentid_to_add)
#                 resident_ids = np.delete(resident_ids, contact_id_index)

#                 employees_ids = []

#                 if staff_key != "":
#                     employees_ids = residence[staff_key] # self.cells_mp.get(residence, staff_key)

#                 secondary_contact_ids = []
#                 if employees_ids is None or len(employees_ids) == 0:
#                     secondary_contact_ids = resident_ids
#                 else:
#                     try:
#                         secondary_contact_ids = np.concatenate((resident_ids, employees_ids))
#                     except Exception as e:
#                         print("problemos: " + e)

#                 if secondary_contact_ids is not None and len(secondary_contact_ids) > 0:
#                     for sec_agent_id in secondary_contact_ids:
#                         agents_partial[sec_agent_id] = agents[sec_agent_id]

#     return agents_partial, dc_by_simcelltype_by_day_partial

# itinerary, itinerary_daskmp, contactnetwork, contactnetwork_daskmp, contacttracing = False, False, False, False, False
class MethodType(IntEnum):
    ItineraryMP = 0
    ItineraryDist = 1
    ItineraryDistMP = 2
    ContactNetworkMP = 3
    ContactNetworkDist = 4
    ContactNetworkDistMP = 5
    ContactTracingMP = 6
    ContactTracingDist = 7