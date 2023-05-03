import datetime
import numpy as np
import scipy.stats as stats
import powerlaw
import matplotlib.pyplot as plt

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

def generate_sociability_rate_powerlaw_dist(temp_agents, agents_ids_by_agebrackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count):
    for agebracket_index, agents_ids_in_bracket in agents_ids_by_agebrackets.items():
        powerlaw_dist_params = powerlaw_distribution_parameters[agebracket_index]

        exponent, xmin = powerlaw_dist_params[2], powerlaw_dist_params[3]
        # exponent, xmin = 2.64, 4.0
        dist = powerlaw.Power_Law(xmin=xmin, parameters=[exponent])

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

    return temp_agents
