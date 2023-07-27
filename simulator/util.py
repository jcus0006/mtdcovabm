import datetime
import numpy as np
import scipy.stats as stats
import powerlaw
import matplotlib.pyplot as plt
import random

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
    if useNp:
        sample = min - 1

        while sample < min or sample > max:
            sample = sample_gamma(gamma_shape, min, max, k, returnInt)
    else:
        scale = (max - min) / (gamma_shape * k)

        trunc_gamma = stats.truncnorm((min - scale) / np.sqrt(gamma_shape),
                              (max - scale) / np.sqrt(gamma_shape),
                              loc=scale, scale=np.sqrt(gamma_shape))
        
        sample = trunc_gamma.rvs()

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

def generate_sociability_rate_powerlaw_dist(temp_agents, agents_ids_by_agebrackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count):
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

                if params["visualise"]:
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
        case "household":
            return "residence"
        case "workplace":
            return "workplace"
        case "accom":
            return "residence"
        case "hospital":
            return "community"
        case "entertainment":
            return "community"
        case "school":
            return "school"
        case "institution":
            return "residence"
        case "transport":
            return "community"
        case "religion":
            return "community"
        case "airport":
            return "community"
        case _:
            return "community"

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

def get_all_contacts_ids_by_id_and_timesteprange(id, agents_contacts_keys, traced_ts, min_ts, max_ts, agents_ids = None, agents_degrees = None, shuffle=False):
    contacts_ids = []

    if max_ts is None:
        max_ts = traced_ts # when contact tracing is done, start from that timestep until 0
    
    if min_ts is None:
        min_ts = traced_ts # when contact tracing is traced for e.g. 24 hours, trace until the same timestep on X prev days

    for pairid, (st_ts, _) in agents_contacts_keys:
        if (id in pairid and 
            (st_ts >= min_ts) and 
            (st_ts <= max_ts)):
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
        
def split_dicts_by_agentsids(agents_ids, agents, agents_ids_by_ages, vars_util, agents_partial, agents_ids_by_ages_partial, vars_util_partial):
    for uid in agents_ids:
        agents_partial[uid] = agents[uid]
        agents_ids_by_ages_partial[uid] = agents_ids_by_ages[uid]

        if uid in vars_util.agents_seir_state_transition_for_day:
            vars_util_partial.agents_seir_state_transition_for_day[uid] = vars_util.agents_seir_state_transition_for_day[uid]

        if uid in vars_util.agents_infection_type:
            vars_util_partial.agents_infection_type[uid] = vars_util.agents_infection_type[uid]

        if uid in vars_util.agents_infection_severity:
            vars_util_partial.agents_infection_severity[uid] = vars_util.agents_infection_severity[uid]

    return agents_partial, agents_ids_by_ages_partial, vars_util_partial

def sync_state_info_by_agentsids(agents_ids, agents, vars_util, agents_partial, vars_util_partial):
    for uid in agents_ids:
        agents[uid] = agents_partial[uid]

        if uid in vars_util_partial.agents_seir_state_transition_for_day:
            vars_util.agents_seir_state[uid] = vars_util_partial.agents_seir_state[uid] # not partial
            vars_util.agents_seir_state_transition_for_day[uid] = vars_util_partial.agents_seir_state_transition_for_day[uid]
            vars_util.agents_infection_type = vars_util_partial.agents_infection_type[uid]
            vars_util.agents_infection_severity = vars_util_partial.agents_infection_severity[uid]

    return agents, vars_util

def sync_state_info_sets(vars_util, vars_util_partial):
    if len(vars_util_partial.contact_tracing_agent_ids) > 0:
        vars_util.contact_tracing_agent_ids.update(vars_util_partial.contact_tracing_agent_ids)

    if len(vars_util_partial.directcontacts_by_simcelltype_by_day) > 0:
        vars_util.directcontacts_by_simcelltype_by_day.update(vars_util_partial.directcontacts_by_simcelltype_by_day)

    return vars_util

def sync_state_info_cells_agents_timesteps(vars_util, vars_util_partial):
    for cellid, agents_timesteps in vars_util_partial.cells_agents_timesteps.items():
        if cellid not in vars_util.cells_agents_timesteps:
            vars_util.cells_agents_timesteps[cellid] = []

        vars_util.cells_agents_timesteps[cellid] += agents_timesteps

    return vars_util